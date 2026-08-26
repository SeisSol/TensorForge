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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v10_a = v4_i1 * 12;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __ldcg(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 12;
              int32_t v32_a = v2_lead + v31_a;
              float v40_data = __ldcg(&glb_m3[(v2_lead + v31_a)]);
              int32_t v41_a = 0 + v25_i1;
              r2[v41_a] = v40_data;
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
            if (v2_lead < 12) {
              float v46_data = r0[0];
              float v47_data = s0[0];
              float v49_data = ir1[0];
              ir1[0] = (v49_data + (v46_data * v47_data));
              float v52_data = s0[12];
              float v54_data = ir1[1];
              ir1[1] = (v54_data + (v46_data * v52_data));
              float v57_data = s0[24];
              float v59_data = ir1[2];
              ir1[2] = (v59_data + (v46_data * v57_data));
              float v62_data = s0[36];
              float v64_data = ir1[3];
              ir1[3] = (v64_data + (v46_data * v62_data));
              float v67_data = s0[48];
              float v69_data = ir1[4];
              ir1[4] = (v69_data + (v46_data * v67_data));
              float v72_data = s0[60];
              float v74_data = ir1[5];
              ir1[5] = (v74_data + (v46_data * v72_data));
              float v77_data = s0[72];
              float v79_data = ir1[6];
              ir1[6] = (v79_data + (v46_data * v77_data));
              float v82_data = s0[84];
              float v84_data = ir1[7];
              ir1[7] = (v84_data + (v46_data * v82_data));
            }
            if (v2_lead < 12) {
              float v90_data = r0[1];
              float v91_data = s0[1];
              float v93_data = ir1[0];
              ir1[0] = (v93_data + (v90_data * v91_data));
              float v96_data = s0[13];
              float v98_data = ir1[1];
              ir1[1] = (v98_data + (v90_data * v96_data));
              float v101_data = s0[25];
              float v103_data = ir1[2];
              ir1[2] = (v103_data + (v90_data * v101_data));
              float v106_data = s0[37];
              float v108_data = ir1[3];
              ir1[3] = (v108_data + (v90_data * v106_data));
              float v111_data = s0[49];
              float v113_data = ir1[4];
              ir1[4] = (v113_data + (v90_data * v111_data));
              float v116_data = s0[61];
              float v118_data = ir1[5];
              ir1[5] = (v118_data + (v90_data * v116_data));
              float v121_data = s0[73];
              float v123_data = ir1[6];
              ir1[6] = (v123_data + (v90_data * v121_data));
              float v126_data = s0[85];
              float v128_data = ir1[7];
              ir1[7] = (v128_data + (v90_data * v126_data));
            }
            if (v2_lead < 12) {
              float v134_data = r0[2];
              float v135_data = s0[2];
              float v137_data = ir1[0];
              ir1[0] = (v137_data + (v134_data * v135_data));
              float v140_data = s0[14];
              float v142_data = ir1[1];
              ir1[1] = (v142_data + (v134_data * v140_data));
              float v145_data = s0[26];
              float v147_data = ir1[2];
              ir1[2] = (v147_data + (v134_data * v145_data));
              float v150_data = s0[38];
              float v152_data = ir1[3];
              ir1[3] = (v152_data + (v134_data * v150_data));
              float v155_data = s0[50];
              float v157_data = ir1[4];
              ir1[4] = (v157_data + (v134_data * v155_data));
              float v160_data = s0[62];
              float v162_data = ir1[5];
              ir1[5] = (v162_data + (v134_data * v160_data));
              float v165_data = s0[74];
              float v167_data = ir1[6];
              ir1[6] = (v167_data + (v134_data * v165_data));
              float v170_data = s0[86];
              float v172_data = ir1[7];
              ir1[7] = (v172_data + (v134_data * v170_data));
            }
            if (v2_lead < 12) {
              float v178_data = r0[3];
              float v179_data = s0[3];
              float v181_data = ir1[0];
              ir1[0] = (v181_data + (v178_data * v179_data));
              float v184_data = s0[15];
              float v186_data = ir1[1];
              ir1[1] = (v186_data + (v178_data * v184_data));
              float v189_data = s0[27];
              float v191_data = ir1[2];
              ir1[2] = (v191_data + (v178_data * v189_data));
              float v194_data = s0[39];
              float v196_data = ir1[3];
              ir1[3] = (v196_data + (v178_data * v194_data));
              float v199_data = s0[51];
              float v201_data = ir1[4];
              ir1[4] = (v201_data + (v178_data * v199_data));
              float v204_data = s0[63];
              float v206_data = ir1[5];
              ir1[5] = (v206_data + (v178_data * v204_data));
              float v209_data = s0[75];
              float v211_data = ir1[6];
              ir1[6] = (v211_data + (v178_data * v209_data));
              float v214_data = s0[87];
              float v216_data = ir1[7];
              ir1[7] = (v216_data + (v178_data * v214_data));
            }
            if (v2_lead < 12) {
              float v222_data = r0[4];
              float v223_data = s0[4];
              float v225_data = ir1[0];
              ir1[0] = (v225_data + (v222_data * v223_data));
              float v228_data = s0[16];
              float v230_data = ir1[1];
              ir1[1] = (v230_data + (v222_data * v228_data));
              float v233_data = s0[28];
              float v235_data = ir1[2];
              ir1[2] = (v235_data + (v222_data * v233_data));
              float v238_data = s0[40];
              float v240_data = ir1[3];
              ir1[3] = (v240_data + (v222_data * v238_data));
              float v243_data = s0[52];
              float v245_data = ir1[4];
              ir1[4] = (v245_data + (v222_data * v243_data));
              float v248_data = s0[64];
              float v250_data = ir1[5];
              ir1[5] = (v250_data + (v222_data * v248_data));
              float v253_data = s0[76];
              float v255_data = ir1[6];
              ir1[6] = (v255_data + (v222_data * v253_data));
              float v258_data = s0[88];
              float v260_data = ir1[7];
              ir1[7] = (v260_data + (v222_data * v258_data));
            }
            if (v2_lead < 12) {
              float v266_data = r0[5];
              float v267_data = s0[5];
              float v269_data = ir1[0];
              ir1[0] = (v269_data + (v266_data * v267_data));
              float v272_data = s0[17];
              float v274_data = ir1[1];
              ir1[1] = (v274_data + (v266_data * v272_data));
              float v277_data = s0[29];
              float v279_data = ir1[2];
              ir1[2] = (v279_data + (v266_data * v277_data));
              float v282_data = s0[41];
              float v284_data = ir1[3];
              ir1[3] = (v284_data + (v266_data * v282_data));
              float v287_data = s0[53];
              float v289_data = ir1[4];
              ir1[4] = (v289_data + (v266_data * v287_data));
              float v292_data = s0[65];
              float v294_data = ir1[5];
              ir1[5] = (v294_data + (v266_data * v292_data));
              float v297_data = s0[77];
              float v299_data = ir1[6];
              ir1[6] = (v299_data + (v266_data * v297_data));
              float v302_data = s0[89];
              float v304_data = ir1[7];
              ir1[7] = (v304_data + (v266_data * v302_data));
            }
            if (v2_lead < 12) {
              float v310_data = r0[6];
              float v311_data = s0[6];
              float v313_data = ir1[0];
              ir1[0] = (v313_data + (v310_data * v311_data));
              float v316_data = s0[18];
              float v318_data = ir1[1];
              ir1[1] = (v318_data + (v310_data * v316_data));
              float v321_data = s0[30];
              float v323_data = ir1[2];
              ir1[2] = (v323_data + (v310_data * v321_data));
              float v326_data = s0[42];
              float v328_data = ir1[3];
              ir1[3] = (v328_data + (v310_data * v326_data));
              float v331_data = s0[54];
              float v333_data = ir1[4];
              ir1[4] = (v333_data + (v310_data * v331_data));
              float v336_data = s0[66];
              float v338_data = ir1[5];
              ir1[5] = (v338_data + (v310_data * v336_data));
              float v341_data = s0[78];
              float v343_data = ir1[6];
              ir1[6] = (v343_data + (v310_data * v341_data));
              float v346_data = s0[90];
              float v348_data = ir1[7];
              ir1[7] = (v348_data + (v310_data * v346_data));
            }
            if (v2_lead < 12) {
              float v354_data = r0[7];
              float v355_data = s0[7];
              float v357_data = ir1[0];
              ir1[0] = (v357_data + (v354_data * v355_data));
              float v360_data = s0[19];
              float v362_data = ir1[1];
              ir1[1] = (v362_data + (v354_data * v360_data));
              float v365_data = s0[31];
              float v367_data = ir1[2];
              ir1[2] = (v367_data + (v354_data * v365_data));
              float v370_data = s0[43];
              float v372_data = ir1[3];
              ir1[3] = (v372_data + (v354_data * v370_data));
              float v375_data = s0[55];
              float v377_data = ir1[4];
              ir1[4] = (v377_data + (v354_data * v375_data));
              float v380_data = s0[67];
              float v382_data = ir1[5];
              ir1[5] = (v382_data + (v354_data * v380_data));
              float v385_data = s0[79];
              float v387_data = ir1[6];
              ir1[6] = (v387_data + (v354_data * v385_data));
              float v390_data = s0[91];
              float v392_data = ir1[7];
              ir1[7] = (v392_data + (v354_data * v390_data));
            }
            if (v2_lead < 12) {
              float v398_data = r0[8];
              float v399_data = s0[8];
              float v401_data = ir1[0];
              ir1[0] = (v401_data + (v398_data * v399_data));
              float v404_data = s0[20];
              float v406_data = ir1[1];
              ir1[1] = (v406_data + (v398_data * v404_data));
              float v409_data = s0[32];
              float v411_data = ir1[2];
              ir1[2] = (v411_data + (v398_data * v409_data));
              float v414_data = s0[44];
              float v416_data = ir1[3];
              ir1[3] = (v416_data + (v398_data * v414_data));
              float v419_data = s0[56];
              float v421_data = ir1[4];
              ir1[4] = (v421_data + (v398_data * v419_data));
              float v424_data = s0[68];
              float v426_data = ir1[5];
              ir1[5] = (v426_data + (v398_data * v424_data));
              float v429_data = s0[80];
              float v431_data = ir1[6];
              ir1[6] = (v431_data + (v398_data * v429_data));
              float v434_data = s0[92];
              float v436_data = ir1[7];
              ir1[7] = (v436_data + (v398_data * v434_data));
            }
            if (v2_lead < 12) {
              float v442_data = r0[9];
              float v443_data = s0[9];
              float v445_data = ir1[0];
              ir1[0] = (v445_data + (v442_data * v443_data));
              float v448_data = s0[21];
              float v450_data = ir1[1];
              ir1[1] = (v450_data + (v442_data * v448_data));
              float v453_data = s0[33];
              float v455_data = ir1[2];
              ir1[2] = (v455_data + (v442_data * v453_data));
              float v458_data = s0[45];
              float v460_data = ir1[3];
              ir1[3] = (v460_data + (v442_data * v458_data));
              float v463_data = s0[57];
              float v465_data = ir1[4];
              ir1[4] = (v465_data + (v442_data * v463_data));
              float v468_data = s0[69];
              float v470_data = ir1[5];
              ir1[5] = (v470_data + (v442_data * v468_data));
              float v473_data = s0[81];
              float v475_data = ir1[6];
              ir1[6] = (v475_data + (v442_data * v473_data));
              float v478_data = s0[93];
              float v480_data = ir1[7];
              ir1[7] = (v480_data + (v442_data * v478_data));
            }
            if (v2_lead < 12) {
              float v486_data = r0[10];
              float v487_data = s0[10];
              float v489_data = ir1[0];
              ir1[0] = (v489_data + (v486_data * v487_data));
              float v492_data = s0[22];
              float v494_data = ir1[1];
              ir1[1] = (v494_data + (v486_data * v492_data));
              float v497_data = s0[34];
              float v499_data = ir1[2];
              ir1[2] = (v499_data + (v486_data * v497_data));
              float v502_data = s0[46];
              float v504_data = ir1[3];
              ir1[3] = (v504_data + (v486_data * v502_data));
              float v507_data = s0[58];
              float v509_data = ir1[4];
              ir1[4] = (v509_data + (v486_data * v507_data));
              float v512_data = s0[70];
              float v514_data = ir1[5];
              ir1[5] = (v514_data + (v486_data * v512_data));
              float v517_data = s0[82];
              float v519_data = ir1[6];
              ir1[6] = (v519_data + (v486_data * v517_data));
              float v522_data = s0[94];
              float v524_data = ir1[7];
              ir1[7] = (v524_data + (v486_data * v522_data));
            }
            if (v2_lead < 12) {
              float v530_data = r0[11];
              float v531_data = s0[11];
              float v533_data = ir1[0];
              ir1[0] = (v533_data + (v530_data * v531_data));
              float v536_data = s0[23];
              float v538_data = ir1[1];
              ir1[1] = (v538_data + (v530_data * v536_data));
              float v541_data = s0[35];
              float v543_data = ir1[2];
              ir1[2] = (v543_data + (v530_data * v541_data));
              float v546_data = s0[47];
              float v548_data = ir1[3];
              ir1[3] = (v548_data + (v530_data * v546_data));
              float v551_data = s0[59];
              float v553_data = ir1[4];
              ir1[4] = (v553_data + (v530_data * v551_data));
              float v556_data = s0[71];
              float v558_data = ir1[5];
              ir1[5] = (v558_data + (v530_data * v556_data));
              float v561_data = s0[83];
              float v563_data = ir1[6];
              ir1[6] = (v563_data + (v530_data * v561_data));
              float v566_data = s0[95];
              float v568_data = ir1[7];
              ir1[7] = (v568_data + (v530_data * v566_data));
            }
            if (v2_lead < 12) {
              #pragma unroll
              for (int32_t v574_n1 = 0; v574_n1 < 8; ++v574_n1) {
                int32_t v575_a = 0 + v574_n1;
                float v577_data = ir1[v574_n1];
                int32_t v578_a = 0 + v574_n1;
                r1[v574_n1] = v577_data;
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
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v584_i1 = 0; v584_i1 < 12; ++v584_i1) {
              int32_t v590_a = v584_i1 * 12;
              int32_t v591_a = v2_lead + v590_a;
              float v599_data = __ldcg(&glb_m5[(v2_lead + v590_a)]);
              int32_t v600_a = 0 + v584_i1;
              r4[v600_a] = v599_data;
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
            if (v2_lead < 12) {
              float v605_data = r2[0];
              float v606_data = s1[0];
              float v608_data = ir3[0];
              ir3[0] = (v608_data + (v605_data * v606_data));
              float v611_data = s1[12];
              float v613_data = ir3[1];
              ir3[1] = (v613_data + (v605_data * v611_data));
              float v616_data = s1[24];
              float v618_data = ir3[2];
              ir3[2] = (v618_data + (v605_data * v616_data));
              float v621_data = s1[36];
              float v623_data = ir3[3];
              ir3[3] = (v623_data + (v605_data * v621_data));
              float v626_data = s1[48];
              float v628_data = ir3[4];
              ir3[4] = (v628_data + (v605_data * v626_data));
              float v631_data = s1[60];
              float v633_data = ir3[5];
              ir3[5] = (v633_data + (v605_data * v631_data));
              float v636_data = s1[72];
              float v638_data = ir3[6];
              ir3[6] = (v638_data + (v605_data * v636_data));
              float v641_data = s1[84];
              float v643_data = ir3[7];
              ir3[7] = (v643_data + (v605_data * v641_data));
            }
            if (v2_lead < 12) {
              float v649_data = r2[1];
              float v650_data = s1[1];
              float v652_data = ir3[0];
              ir3[0] = (v652_data + (v649_data * v650_data));
              float v655_data = s1[13];
              float v657_data = ir3[1];
              ir3[1] = (v657_data + (v649_data * v655_data));
              float v660_data = s1[25];
              float v662_data = ir3[2];
              ir3[2] = (v662_data + (v649_data * v660_data));
              float v665_data = s1[37];
              float v667_data = ir3[3];
              ir3[3] = (v667_data + (v649_data * v665_data));
              float v670_data = s1[49];
              float v672_data = ir3[4];
              ir3[4] = (v672_data + (v649_data * v670_data));
              float v675_data = s1[61];
              float v677_data = ir3[5];
              ir3[5] = (v677_data + (v649_data * v675_data));
              float v680_data = s1[73];
              float v682_data = ir3[6];
              ir3[6] = (v682_data + (v649_data * v680_data));
              float v685_data = s1[85];
              float v687_data = ir3[7];
              ir3[7] = (v687_data + (v649_data * v685_data));
            }
            if (v2_lead < 12) {
              float v693_data = r2[2];
              float v694_data = s1[2];
              float v696_data = ir3[0];
              ir3[0] = (v696_data + (v693_data * v694_data));
              float v699_data = s1[14];
              float v701_data = ir3[1];
              ir3[1] = (v701_data + (v693_data * v699_data));
              float v704_data = s1[26];
              float v706_data = ir3[2];
              ir3[2] = (v706_data + (v693_data * v704_data));
              float v709_data = s1[38];
              float v711_data = ir3[3];
              ir3[3] = (v711_data + (v693_data * v709_data));
              float v714_data = s1[50];
              float v716_data = ir3[4];
              ir3[4] = (v716_data + (v693_data * v714_data));
              float v719_data = s1[62];
              float v721_data = ir3[5];
              ir3[5] = (v721_data + (v693_data * v719_data));
              float v724_data = s1[74];
              float v726_data = ir3[6];
              ir3[6] = (v726_data + (v693_data * v724_data));
              float v729_data = s1[86];
              float v731_data = ir3[7];
              ir3[7] = (v731_data + (v693_data * v729_data));
            }
            if (v2_lead < 12) {
              float v737_data = r2[3];
              float v738_data = s1[3];
              float v740_data = ir3[0];
              ir3[0] = (v740_data + (v737_data * v738_data));
              float v743_data = s1[15];
              float v745_data = ir3[1];
              ir3[1] = (v745_data + (v737_data * v743_data));
              float v748_data = s1[27];
              float v750_data = ir3[2];
              ir3[2] = (v750_data + (v737_data * v748_data));
              float v753_data = s1[39];
              float v755_data = ir3[3];
              ir3[3] = (v755_data + (v737_data * v753_data));
              float v758_data = s1[51];
              float v760_data = ir3[4];
              ir3[4] = (v760_data + (v737_data * v758_data));
              float v763_data = s1[63];
              float v765_data = ir3[5];
              ir3[5] = (v765_data + (v737_data * v763_data));
              float v768_data = s1[75];
              float v770_data = ir3[6];
              ir3[6] = (v770_data + (v737_data * v768_data));
              float v773_data = s1[87];
              float v775_data = ir3[7];
              ir3[7] = (v775_data + (v737_data * v773_data));
            }
            if (v2_lead < 12) {
              float v781_data = r2[4];
              float v782_data = s1[4];
              float v784_data = ir3[0];
              ir3[0] = (v784_data + (v781_data * v782_data));
              float v787_data = s1[16];
              float v789_data = ir3[1];
              ir3[1] = (v789_data + (v781_data * v787_data));
              float v792_data = s1[28];
              float v794_data = ir3[2];
              ir3[2] = (v794_data + (v781_data * v792_data));
              float v797_data = s1[40];
              float v799_data = ir3[3];
              ir3[3] = (v799_data + (v781_data * v797_data));
              float v802_data = s1[52];
              float v804_data = ir3[4];
              ir3[4] = (v804_data + (v781_data * v802_data));
              float v807_data = s1[64];
              float v809_data = ir3[5];
              ir3[5] = (v809_data + (v781_data * v807_data));
              float v812_data = s1[76];
              float v814_data = ir3[6];
              ir3[6] = (v814_data + (v781_data * v812_data));
              float v817_data = s1[88];
              float v819_data = ir3[7];
              ir3[7] = (v819_data + (v781_data * v817_data));
            }
            if (v2_lead < 12) {
              float v825_data = r2[5];
              float v826_data = s1[5];
              float v828_data = ir3[0];
              ir3[0] = (v828_data + (v825_data * v826_data));
              float v831_data = s1[17];
              float v833_data = ir3[1];
              ir3[1] = (v833_data + (v825_data * v831_data));
              float v836_data = s1[29];
              float v838_data = ir3[2];
              ir3[2] = (v838_data + (v825_data * v836_data));
              float v841_data = s1[41];
              float v843_data = ir3[3];
              ir3[3] = (v843_data + (v825_data * v841_data));
              float v846_data = s1[53];
              float v848_data = ir3[4];
              ir3[4] = (v848_data + (v825_data * v846_data));
              float v851_data = s1[65];
              float v853_data = ir3[5];
              ir3[5] = (v853_data + (v825_data * v851_data));
              float v856_data = s1[77];
              float v858_data = ir3[6];
              ir3[6] = (v858_data + (v825_data * v856_data));
              float v861_data = s1[89];
              float v863_data = ir3[7];
              ir3[7] = (v863_data + (v825_data * v861_data));
            }
            if (v2_lead < 12) {
              float v869_data = r2[6];
              float v870_data = s1[6];
              float v872_data = ir3[0];
              ir3[0] = (v872_data + (v869_data * v870_data));
              float v875_data = s1[18];
              float v877_data = ir3[1];
              ir3[1] = (v877_data + (v869_data * v875_data));
              float v880_data = s1[30];
              float v882_data = ir3[2];
              ir3[2] = (v882_data + (v869_data * v880_data));
              float v885_data = s1[42];
              float v887_data = ir3[3];
              ir3[3] = (v887_data + (v869_data * v885_data));
              float v890_data = s1[54];
              float v892_data = ir3[4];
              ir3[4] = (v892_data + (v869_data * v890_data));
              float v895_data = s1[66];
              float v897_data = ir3[5];
              ir3[5] = (v897_data + (v869_data * v895_data));
              float v900_data = s1[78];
              float v902_data = ir3[6];
              ir3[6] = (v902_data + (v869_data * v900_data));
              float v905_data = s1[90];
              float v907_data = ir3[7];
              ir3[7] = (v907_data + (v869_data * v905_data));
            }
            if (v2_lead < 12) {
              float v913_data = r2[7];
              float v914_data = s1[7];
              float v916_data = ir3[0];
              ir3[0] = (v916_data + (v913_data * v914_data));
              float v919_data = s1[19];
              float v921_data = ir3[1];
              ir3[1] = (v921_data + (v913_data * v919_data));
              float v924_data = s1[31];
              float v926_data = ir3[2];
              ir3[2] = (v926_data + (v913_data * v924_data));
              float v929_data = s1[43];
              float v931_data = ir3[3];
              ir3[3] = (v931_data + (v913_data * v929_data));
              float v934_data = s1[55];
              float v936_data = ir3[4];
              ir3[4] = (v936_data + (v913_data * v934_data));
              float v939_data = s1[67];
              float v941_data = ir3[5];
              ir3[5] = (v941_data + (v913_data * v939_data));
              float v944_data = s1[79];
              float v946_data = ir3[6];
              ir3[6] = (v946_data + (v913_data * v944_data));
              float v949_data = s1[91];
              float v951_data = ir3[7];
              ir3[7] = (v951_data + (v913_data * v949_data));
            }
            if (v2_lead < 12) {
              float v957_data = r2[8];
              float v958_data = s1[8];
              float v960_data = ir3[0];
              ir3[0] = (v960_data + (v957_data * v958_data));
              float v963_data = s1[20];
              float v965_data = ir3[1];
              ir3[1] = (v965_data + (v957_data * v963_data));
              float v968_data = s1[32];
              float v970_data = ir3[2];
              ir3[2] = (v970_data + (v957_data * v968_data));
              float v973_data = s1[44];
              float v975_data = ir3[3];
              ir3[3] = (v975_data + (v957_data * v973_data));
              float v978_data = s1[56];
              float v980_data = ir3[4];
              ir3[4] = (v980_data + (v957_data * v978_data));
              float v983_data = s1[68];
              float v985_data = ir3[5];
              ir3[5] = (v985_data + (v957_data * v983_data));
              float v988_data = s1[80];
              float v990_data = ir3[6];
              ir3[6] = (v990_data + (v957_data * v988_data));
              float v993_data = s1[92];
              float v995_data = ir3[7];
              ir3[7] = (v995_data + (v957_data * v993_data));
            }
            if (v2_lead < 12) {
              float v1001_data = r2[9];
              float v1002_data = s1[9];
              float v1004_data = ir3[0];
              ir3[0] = (v1004_data + (v1001_data * v1002_data));
              float v1007_data = s1[21];
              float v1009_data = ir3[1];
              ir3[1] = (v1009_data + (v1001_data * v1007_data));
              float v1012_data = s1[33];
              float v1014_data = ir3[2];
              ir3[2] = (v1014_data + (v1001_data * v1012_data));
              float v1017_data = s1[45];
              float v1019_data = ir3[3];
              ir3[3] = (v1019_data + (v1001_data * v1017_data));
              float v1022_data = s1[57];
              float v1024_data = ir3[4];
              ir3[4] = (v1024_data + (v1001_data * v1022_data));
              float v1027_data = s1[69];
              float v1029_data = ir3[5];
              ir3[5] = (v1029_data + (v1001_data * v1027_data));
              float v1032_data = s1[81];
              float v1034_data = ir3[6];
              ir3[6] = (v1034_data + (v1001_data * v1032_data));
              float v1037_data = s1[93];
              float v1039_data = ir3[7];
              ir3[7] = (v1039_data + (v1001_data * v1037_data));
            }
            if (v2_lead < 12) {
              float v1045_data = r2[10];
              float v1046_data = s1[10];
              float v1048_data = ir3[0];
              ir3[0] = (v1048_data + (v1045_data * v1046_data));
              float v1051_data = s1[22];
              float v1053_data = ir3[1];
              ir3[1] = (v1053_data + (v1045_data * v1051_data));
              float v1056_data = s1[34];
              float v1058_data = ir3[2];
              ir3[2] = (v1058_data + (v1045_data * v1056_data));
              float v1061_data = s1[46];
              float v1063_data = ir3[3];
              ir3[3] = (v1063_data + (v1045_data * v1061_data));
              float v1066_data = s1[58];
              float v1068_data = ir3[4];
              ir3[4] = (v1068_data + (v1045_data * v1066_data));
              float v1071_data = s1[70];
              float v1073_data = ir3[5];
              ir3[5] = (v1073_data + (v1045_data * v1071_data));
              float v1076_data = s1[82];
              float v1078_data = ir3[6];
              ir3[6] = (v1078_data + (v1045_data * v1076_data));
              float v1081_data = s1[94];
              float v1083_data = ir3[7];
              ir3[7] = (v1083_data + (v1045_data * v1081_data));
            }
            if (v2_lead < 12) {
              float v1089_data = r2[11];
              float v1090_data = s1[11];
              float v1092_data = ir3[0];
              ir3[0] = (v1092_data + (v1089_data * v1090_data));
              float v1095_data = s1[23];
              float v1097_data = ir3[1];
              ir3[1] = (v1097_data + (v1089_data * v1095_data));
              float v1100_data = s1[35];
              float v1102_data = ir3[2];
              ir3[2] = (v1102_data + (v1089_data * v1100_data));
              float v1105_data = s1[47];
              float v1107_data = ir3[3];
              ir3[3] = (v1107_data + (v1089_data * v1105_data));
              float v1110_data = s1[59];
              float v1112_data = ir3[4];
              ir3[4] = (v1112_data + (v1089_data * v1110_data));
              float v1115_data = s1[71];
              float v1117_data = ir3[5];
              ir3[5] = (v1117_data + (v1089_data * v1115_data));
              float v1120_data = s1[83];
              float v1122_data = ir3[6];
              ir3[6] = (v1122_data + (v1089_data * v1120_data));
              float v1125_data = s1[95];
              float v1127_data = ir3[7];
              ir3[7] = (v1127_data + (v1089_data * v1125_data));
            }
            if (v2_lead < 12) {
              #pragma unroll
              for (int32_t v1133_n1 = 0; v1133_n1 < 8; ++v1133_n1) {
                int32_t v1134_a = 0 + v1133_n1;
                float v1136_data = ir3[v1133_n1];
                int32_t v1137_a = 0 + v1133_n1;
                float v1139_data = r1[v1133_n1];
                int32_t v1141_a = 0 + v1133_n1;
                r3[v1133_n1] = (v1139_data + v1136_data);
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
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v1147_i1 = 0; v1147_i1 < 12; ++v1147_i1) {
              int32_t v1153_a = v1147_i1 * 12;
              int32_t v1154_a = v2_lead + v1153_a;
              float v1162_data = __ldcg(&glb_m7[(v2_lead + v1153_a)]);
              int32_t v1163_a = 0 + v1147_i1;
              r6[v1163_a] = v1162_data;
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
            if (v2_lead < 12) {
              float v1168_data = r4[0];
              float v1169_data = s2[0];
              float v1171_data = ir5[0];
              ir5[0] = (v1171_data + (v1168_data * v1169_data));
              float v1174_data = s2[12];
              float v1176_data = ir5[1];
              ir5[1] = (v1176_data + (v1168_data * v1174_data));
              float v1179_data = s2[24];
              float v1181_data = ir5[2];
              ir5[2] = (v1181_data + (v1168_data * v1179_data));
              float v1184_data = s2[36];
              float v1186_data = ir5[3];
              ir5[3] = (v1186_data + (v1168_data * v1184_data));
              float v1189_data = s2[48];
              float v1191_data = ir5[4];
              ir5[4] = (v1191_data + (v1168_data * v1189_data));
              float v1194_data = s2[60];
              float v1196_data = ir5[5];
              ir5[5] = (v1196_data + (v1168_data * v1194_data));
              float v1199_data = s2[72];
              float v1201_data = ir5[6];
              ir5[6] = (v1201_data + (v1168_data * v1199_data));
              float v1204_data = s2[84];
              float v1206_data = ir5[7];
              ir5[7] = (v1206_data + (v1168_data * v1204_data));
            }
            if (v2_lead < 12) {
              float v1212_data = r4[1];
              float v1213_data = s2[1];
              float v1215_data = ir5[0];
              ir5[0] = (v1215_data + (v1212_data * v1213_data));
              float v1218_data = s2[13];
              float v1220_data = ir5[1];
              ir5[1] = (v1220_data + (v1212_data * v1218_data));
              float v1223_data = s2[25];
              float v1225_data = ir5[2];
              ir5[2] = (v1225_data + (v1212_data * v1223_data));
              float v1228_data = s2[37];
              float v1230_data = ir5[3];
              ir5[3] = (v1230_data + (v1212_data * v1228_data));
              float v1233_data = s2[49];
              float v1235_data = ir5[4];
              ir5[4] = (v1235_data + (v1212_data * v1233_data));
              float v1238_data = s2[61];
              float v1240_data = ir5[5];
              ir5[5] = (v1240_data + (v1212_data * v1238_data));
              float v1243_data = s2[73];
              float v1245_data = ir5[6];
              ir5[6] = (v1245_data + (v1212_data * v1243_data));
              float v1248_data = s2[85];
              float v1250_data = ir5[7];
              ir5[7] = (v1250_data + (v1212_data * v1248_data));
            }
            if (v2_lead < 12) {
              float v1256_data = r4[2];
              float v1257_data = s2[2];
              float v1259_data = ir5[0];
              ir5[0] = (v1259_data + (v1256_data * v1257_data));
              float v1262_data = s2[14];
              float v1264_data = ir5[1];
              ir5[1] = (v1264_data + (v1256_data * v1262_data));
              float v1267_data = s2[26];
              float v1269_data = ir5[2];
              ir5[2] = (v1269_data + (v1256_data * v1267_data));
              float v1272_data = s2[38];
              float v1274_data = ir5[3];
              ir5[3] = (v1274_data + (v1256_data * v1272_data));
              float v1277_data = s2[50];
              float v1279_data = ir5[4];
              ir5[4] = (v1279_data + (v1256_data * v1277_data));
              float v1282_data = s2[62];
              float v1284_data = ir5[5];
              ir5[5] = (v1284_data + (v1256_data * v1282_data));
              float v1287_data = s2[74];
              float v1289_data = ir5[6];
              ir5[6] = (v1289_data + (v1256_data * v1287_data));
              float v1292_data = s2[86];
              float v1294_data = ir5[7];
              ir5[7] = (v1294_data + (v1256_data * v1292_data));
            }
            if (v2_lead < 12) {
              float v1300_data = r4[3];
              float v1301_data = s2[3];
              float v1303_data = ir5[0];
              ir5[0] = (v1303_data + (v1300_data * v1301_data));
              float v1306_data = s2[15];
              float v1308_data = ir5[1];
              ir5[1] = (v1308_data + (v1300_data * v1306_data));
              float v1311_data = s2[27];
              float v1313_data = ir5[2];
              ir5[2] = (v1313_data + (v1300_data * v1311_data));
              float v1316_data = s2[39];
              float v1318_data = ir5[3];
              ir5[3] = (v1318_data + (v1300_data * v1316_data));
              float v1321_data = s2[51];
              float v1323_data = ir5[4];
              ir5[4] = (v1323_data + (v1300_data * v1321_data));
              float v1326_data = s2[63];
              float v1328_data = ir5[5];
              ir5[5] = (v1328_data + (v1300_data * v1326_data));
              float v1331_data = s2[75];
              float v1333_data = ir5[6];
              ir5[6] = (v1333_data + (v1300_data * v1331_data));
              float v1336_data = s2[87];
              float v1338_data = ir5[7];
              ir5[7] = (v1338_data + (v1300_data * v1336_data));
            }
            if (v2_lead < 12) {
              float v1344_data = r4[4];
              float v1345_data = s2[4];
              float v1347_data = ir5[0];
              ir5[0] = (v1347_data + (v1344_data * v1345_data));
              float v1350_data = s2[16];
              float v1352_data = ir5[1];
              ir5[1] = (v1352_data + (v1344_data * v1350_data));
              float v1355_data = s2[28];
              float v1357_data = ir5[2];
              ir5[2] = (v1357_data + (v1344_data * v1355_data));
              float v1360_data = s2[40];
              float v1362_data = ir5[3];
              ir5[3] = (v1362_data + (v1344_data * v1360_data));
              float v1365_data = s2[52];
              float v1367_data = ir5[4];
              ir5[4] = (v1367_data + (v1344_data * v1365_data));
              float v1370_data = s2[64];
              float v1372_data = ir5[5];
              ir5[5] = (v1372_data + (v1344_data * v1370_data));
              float v1375_data = s2[76];
              float v1377_data = ir5[6];
              ir5[6] = (v1377_data + (v1344_data * v1375_data));
              float v1380_data = s2[88];
              float v1382_data = ir5[7];
              ir5[7] = (v1382_data + (v1344_data * v1380_data));
            }
            if (v2_lead < 12) {
              float v1388_data = r4[5];
              float v1389_data = s2[5];
              float v1391_data = ir5[0];
              ir5[0] = (v1391_data + (v1388_data * v1389_data));
              float v1394_data = s2[17];
              float v1396_data = ir5[1];
              ir5[1] = (v1396_data + (v1388_data * v1394_data));
              float v1399_data = s2[29];
              float v1401_data = ir5[2];
              ir5[2] = (v1401_data + (v1388_data * v1399_data));
              float v1404_data = s2[41];
              float v1406_data = ir5[3];
              ir5[3] = (v1406_data + (v1388_data * v1404_data));
              float v1409_data = s2[53];
              float v1411_data = ir5[4];
              ir5[4] = (v1411_data + (v1388_data * v1409_data));
              float v1414_data = s2[65];
              float v1416_data = ir5[5];
              ir5[5] = (v1416_data + (v1388_data * v1414_data));
              float v1419_data = s2[77];
              float v1421_data = ir5[6];
              ir5[6] = (v1421_data + (v1388_data * v1419_data));
              float v1424_data = s2[89];
              float v1426_data = ir5[7];
              ir5[7] = (v1426_data + (v1388_data * v1424_data));
            }
            if (v2_lead < 12) {
              float v1432_data = r4[6];
              float v1433_data = s2[6];
              float v1435_data = ir5[0];
              ir5[0] = (v1435_data + (v1432_data * v1433_data));
              float v1438_data = s2[18];
              float v1440_data = ir5[1];
              ir5[1] = (v1440_data + (v1432_data * v1438_data));
              float v1443_data = s2[30];
              float v1445_data = ir5[2];
              ir5[2] = (v1445_data + (v1432_data * v1443_data));
              float v1448_data = s2[42];
              float v1450_data = ir5[3];
              ir5[3] = (v1450_data + (v1432_data * v1448_data));
              float v1453_data = s2[54];
              float v1455_data = ir5[4];
              ir5[4] = (v1455_data + (v1432_data * v1453_data));
              float v1458_data = s2[66];
              float v1460_data = ir5[5];
              ir5[5] = (v1460_data + (v1432_data * v1458_data));
              float v1463_data = s2[78];
              float v1465_data = ir5[6];
              ir5[6] = (v1465_data + (v1432_data * v1463_data));
              float v1468_data = s2[90];
              float v1470_data = ir5[7];
              ir5[7] = (v1470_data + (v1432_data * v1468_data));
            }
            if (v2_lead < 12) {
              float v1476_data = r4[7];
              float v1477_data = s2[7];
              float v1479_data = ir5[0];
              ir5[0] = (v1479_data + (v1476_data * v1477_data));
              float v1482_data = s2[19];
              float v1484_data = ir5[1];
              ir5[1] = (v1484_data + (v1476_data * v1482_data));
              float v1487_data = s2[31];
              float v1489_data = ir5[2];
              ir5[2] = (v1489_data + (v1476_data * v1487_data));
              float v1492_data = s2[43];
              float v1494_data = ir5[3];
              ir5[3] = (v1494_data + (v1476_data * v1492_data));
              float v1497_data = s2[55];
              float v1499_data = ir5[4];
              ir5[4] = (v1499_data + (v1476_data * v1497_data));
              float v1502_data = s2[67];
              float v1504_data = ir5[5];
              ir5[5] = (v1504_data + (v1476_data * v1502_data));
              float v1507_data = s2[79];
              float v1509_data = ir5[6];
              ir5[6] = (v1509_data + (v1476_data * v1507_data));
              float v1512_data = s2[91];
              float v1514_data = ir5[7];
              ir5[7] = (v1514_data + (v1476_data * v1512_data));
            }
            if (v2_lead < 12) {
              float v1520_data = r4[8];
              float v1521_data = s2[8];
              float v1523_data = ir5[0];
              ir5[0] = (v1523_data + (v1520_data * v1521_data));
              float v1526_data = s2[20];
              float v1528_data = ir5[1];
              ir5[1] = (v1528_data + (v1520_data * v1526_data));
              float v1531_data = s2[32];
              float v1533_data = ir5[2];
              ir5[2] = (v1533_data + (v1520_data * v1531_data));
              float v1536_data = s2[44];
              float v1538_data = ir5[3];
              ir5[3] = (v1538_data + (v1520_data * v1536_data));
              float v1541_data = s2[56];
              float v1543_data = ir5[4];
              ir5[4] = (v1543_data + (v1520_data * v1541_data));
              float v1546_data = s2[68];
              float v1548_data = ir5[5];
              ir5[5] = (v1548_data + (v1520_data * v1546_data));
              float v1551_data = s2[80];
              float v1553_data = ir5[6];
              ir5[6] = (v1553_data + (v1520_data * v1551_data));
              float v1556_data = s2[92];
              float v1558_data = ir5[7];
              ir5[7] = (v1558_data + (v1520_data * v1556_data));
            }
            if (v2_lead < 12) {
              float v1564_data = r4[9];
              float v1565_data = s2[9];
              float v1567_data = ir5[0];
              ir5[0] = (v1567_data + (v1564_data * v1565_data));
              float v1570_data = s2[21];
              float v1572_data = ir5[1];
              ir5[1] = (v1572_data + (v1564_data * v1570_data));
              float v1575_data = s2[33];
              float v1577_data = ir5[2];
              ir5[2] = (v1577_data + (v1564_data * v1575_data));
              float v1580_data = s2[45];
              float v1582_data = ir5[3];
              ir5[3] = (v1582_data + (v1564_data * v1580_data));
              float v1585_data = s2[57];
              float v1587_data = ir5[4];
              ir5[4] = (v1587_data + (v1564_data * v1585_data));
              float v1590_data = s2[69];
              float v1592_data = ir5[5];
              ir5[5] = (v1592_data + (v1564_data * v1590_data));
              float v1595_data = s2[81];
              float v1597_data = ir5[6];
              ir5[6] = (v1597_data + (v1564_data * v1595_data));
              float v1600_data = s2[93];
              float v1602_data = ir5[7];
              ir5[7] = (v1602_data + (v1564_data * v1600_data));
            }
            if (v2_lead < 12) {
              float v1608_data = r4[10];
              float v1609_data = s2[10];
              float v1611_data = ir5[0];
              ir5[0] = (v1611_data + (v1608_data * v1609_data));
              float v1614_data = s2[22];
              float v1616_data = ir5[1];
              ir5[1] = (v1616_data + (v1608_data * v1614_data));
              float v1619_data = s2[34];
              float v1621_data = ir5[2];
              ir5[2] = (v1621_data + (v1608_data * v1619_data));
              float v1624_data = s2[46];
              float v1626_data = ir5[3];
              ir5[3] = (v1626_data + (v1608_data * v1624_data));
              float v1629_data = s2[58];
              float v1631_data = ir5[4];
              ir5[4] = (v1631_data + (v1608_data * v1629_data));
              float v1634_data = s2[70];
              float v1636_data = ir5[5];
              ir5[5] = (v1636_data + (v1608_data * v1634_data));
              float v1639_data = s2[82];
              float v1641_data = ir5[6];
              ir5[6] = (v1641_data + (v1608_data * v1639_data));
              float v1644_data = s2[94];
              float v1646_data = ir5[7];
              ir5[7] = (v1646_data + (v1608_data * v1644_data));
            }
            if (v2_lead < 12) {
              float v1652_data = r4[11];
              float v1653_data = s2[11];
              float v1655_data = ir5[0];
              ir5[0] = (v1655_data + (v1652_data * v1653_data));
              float v1658_data = s2[23];
              float v1660_data = ir5[1];
              ir5[1] = (v1660_data + (v1652_data * v1658_data));
              float v1663_data = s2[35];
              float v1665_data = ir5[2];
              ir5[2] = (v1665_data + (v1652_data * v1663_data));
              float v1668_data = s2[47];
              float v1670_data = ir5[3];
              ir5[3] = (v1670_data + (v1652_data * v1668_data));
              float v1673_data = s2[59];
              float v1675_data = ir5[4];
              ir5[4] = (v1675_data + (v1652_data * v1673_data));
              float v1678_data = s2[71];
              float v1680_data = ir5[5];
              ir5[5] = (v1680_data + (v1652_data * v1678_data));
              float v1683_data = s2[83];
              float v1685_data = ir5[6];
              ir5[6] = (v1685_data + (v1652_data * v1683_data));
              float v1688_data = s2[95];
              float v1690_data = ir5[7];
              ir5[7] = (v1690_data + (v1652_data * v1688_data));
            }
            if (v2_lead < 12) {
              #pragma unroll
              for (int32_t v1696_n1 = 0; v1696_n1 < 8; ++v1696_n1) {
                int32_t v1697_a = 0 + v1696_n1;
                float v1699_data = ir5[v1696_n1];
                int32_t v1700_a = 0 + v1696_n1;
                float v1702_data = r3[v1696_n1];
                int32_t v1704_a = 0 + v1696_n1;
                r5[v1696_n1] = (v1702_data + v1699_data);
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
            if (v2_lead < 12) {
              float v1710_data = r6[0];
              float v1711_data = s3[0];
              float v1713_data = ir7[0];
              ir7[0] = (v1713_data + (v1710_data * v1711_data));
              float v1716_data = s3[12];
              float v1718_data = ir7[1];
              ir7[1] = (v1718_data + (v1710_data * v1716_data));
              float v1721_data = s3[24];
              float v1723_data = ir7[2];
              ir7[2] = (v1723_data + (v1710_data * v1721_data));
              float v1726_data = s3[36];
              float v1728_data = ir7[3];
              ir7[3] = (v1728_data + (v1710_data * v1726_data));
              float v1731_data = s3[48];
              float v1733_data = ir7[4];
              ir7[4] = (v1733_data + (v1710_data * v1731_data));
              float v1736_data = s3[60];
              float v1738_data = ir7[5];
              ir7[5] = (v1738_data + (v1710_data * v1736_data));
              float v1741_data = s3[72];
              float v1743_data = ir7[6];
              ir7[6] = (v1743_data + (v1710_data * v1741_data));
              float v1746_data = s3[84];
              float v1748_data = ir7[7];
              ir7[7] = (v1748_data + (v1710_data * v1746_data));
            }
            if (v2_lead < 12) {
              float v1754_data = r6[1];
              float v1755_data = s3[1];
              float v1757_data = ir7[0];
              ir7[0] = (v1757_data + (v1754_data * v1755_data));
              float v1760_data = s3[13];
              float v1762_data = ir7[1];
              ir7[1] = (v1762_data + (v1754_data * v1760_data));
              float v1765_data = s3[25];
              float v1767_data = ir7[2];
              ir7[2] = (v1767_data + (v1754_data * v1765_data));
              float v1770_data = s3[37];
              float v1772_data = ir7[3];
              ir7[3] = (v1772_data + (v1754_data * v1770_data));
              float v1775_data = s3[49];
              float v1777_data = ir7[4];
              ir7[4] = (v1777_data + (v1754_data * v1775_data));
              float v1780_data = s3[61];
              float v1782_data = ir7[5];
              ir7[5] = (v1782_data + (v1754_data * v1780_data));
              float v1785_data = s3[73];
              float v1787_data = ir7[6];
              ir7[6] = (v1787_data + (v1754_data * v1785_data));
              float v1790_data = s3[85];
              float v1792_data = ir7[7];
              ir7[7] = (v1792_data + (v1754_data * v1790_data));
            }
            if (v2_lead < 12) {
              float v1798_data = r6[2];
              float v1799_data = s3[2];
              float v1801_data = ir7[0];
              ir7[0] = (v1801_data + (v1798_data * v1799_data));
              float v1804_data = s3[14];
              float v1806_data = ir7[1];
              ir7[1] = (v1806_data + (v1798_data * v1804_data));
              float v1809_data = s3[26];
              float v1811_data = ir7[2];
              ir7[2] = (v1811_data + (v1798_data * v1809_data));
              float v1814_data = s3[38];
              float v1816_data = ir7[3];
              ir7[3] = (v1816_data + (v1798_data * v1814_data));
              float v1819_data = s3[50];
              float v1821_data = ir7[4];
              ir7[4] = (v1821_data + (v1798_data * v1819_data));
              float v1824_data = s3[62];
              float v1826_data = ir7[5];
              ir7[5] = (v1826_data + (v1798_data * v1824_data));
              float v1829_data = s3[74];
              float v1831_data = ir7[6];
              ir7[6] = (v1831_data + (v1798_data * v1829_data));
              float v1834_data = s3[86];
              float v1836_data = ir7[7];
              ir7[7] = (v1836_data + (v1798_data * v1834_data));
            }
            if (v2_lead < 12) {
              float v1842_data = r6[3];
              float v1843_data = s3[3];
              float v1845_data = ir7[0];
              ir7[0] = (v1845_data + (v1842_data * v1843_data));
              float v1848_data = s3[15];
              float v1850_data = ir7[1];
              ir7[1] = (v1850_data + (v1842_data * v1848_data));
              float v1853_data = s3[27];
              float v1855_data = ir7[2];
              ir7[2] = (v1855_data + (v1842_data * v1853_data));
              float v1858_data = s3[39];
              float v1860_data = ir7[3];
              ir7[3] = (v1860_data + (v1842_data * v1858_data));
              float v1863_data = s3[51];
              float v1865_data = ir7[4];
              ir7[4] = (v1865_data + (v1842_data * v1863_data));
              float v1868_data = s3[63];
              float v1870_data = ir7[5];
              ir7[5] = (v1870_data + (v1842_data * v1868_data));
              float v1873_data = s3[75];
              float v1875_data = ir7[6];
              ir7[6] = (v1875_data + (v1842_data * v1873_data));
              float v1878_data = s3[87];
              float v1880_data = ir7[7];
              ir7[7] = (v1880_data + (v1842_data * v1878_data));
            }
            if (v2_lead < 12) {
              float v1886_data = r6[4];
              float v1887_data = s3[4];
              float v1889_data = ir7[0];
              ir7[0] = (v1889_data + (v1886_data * v1887_data));
              float v1892_data = s3[16];
              float v1894_data = ir7[1];
              ir7[1] = (v1894_data + (v1886_data * v1892_data));
              float v1897_data = s3[28];
              float v1899_data = ir7[2];
              ir7[2] = (v1899_data + (v1886_data * v1897_data));
              float v1902_data = s3[40];
              float v1904_data = ir7[3];
              ir7[3] = (v1904_data + (v1886_data * v1902_data));
              float v1907_data = s3[52];
              float v1909_data = ir7[4];
              ir7[4] = (v1909_data + (v1886_data * v1907_data));
              float v1912_data = s3[64];
              float v1914_data = ir7[5];
              ir7[5] = (v1914_data + (v1886_data * v1912_data));
              float v1917_data = s3[76];
              float v1919_data = ir7[6];
              ir7[6] = (v1919_data + (v1886_data * v1917_data));
              float v1922_data = s3[88];
              float v1924_data = ir7[7];
              ir7[7] = (v1924_data + (v1886_data * v1922_data));
            }
            if (v2_lead < 12) {
              float v1930_data = r6[5];
              float v1931_data = s3[5];
              float v1933_data = ir7[0];
              ir7[0] = (v1933_data + (v1930_data * v1931_data));
              float v1936_data = s3[17];
              float v1938_data = ir7[1];
              ir7[1] = (v1938_data + (v1930_data * v1936_data));
              float v1941_data = s3[29];
              float v1943_data = ir7[2];
              ir7[2] = (v1943_data + (v1930_data * v1941_data));
              float v1946_data = s3[41];
              float v1948_data = ir7[3];
              ir7[3] = (v1948_data + (v1930_data * v1946_data));
              float v1951_data = s3[53];
              float v1953_data = ir7[4];
              ir7[4] = (v1953_data + (v1930_data * v1951_data));
              float v1956_data = s3[65];
              float v1958_data = ir7[5];
              ir7[5] = (v1958_data + (v1930_data * v1956_data));
              float v1961_data = s3[77];
              float v1963_data = ir7[6];
              ir7[6] = (v1963_data + (v1930_data * v1961_data));
              float v1966_data = s3[89];
              float v1968_data = ir7[7];
              ir7[7] = (v1968_data + (v1930_data * v1966_data));
            }
            if (v2_lead < 12) {
              float v1974_data = r6[6];
              float v1975_data = s3[6];
              float v1977_data = ir7[0];
              ir7[0] = (v1977_data + (v1974_data * v1975_data));
              float v1980_data = s3[18];
              float v1982_data = ir7[1];
              ir7[1] = (v1982_data + (v1974_data * v1980_data));
              float v1985_data = s3[30];
              float v1987_data = ir7[2];
              ir7[2] = (v1987_data + (v1974_data * v1985_data));
              float v1990_data = s3[42];
              float v1992_data = ir7[3];
              ir7[3] = (v1992_data + (v1974_data * v1990_data));
              float v1995_data = s3[54];
              float v1997_data = ir7[4];
              ir7[4] = (v1997_data + (v1974_data * v1995_data));
              float v2000_data = s3[66];
              float v2002_data = ir7[5];
              ir7[5] = (v2002_data + (v1974_data * v2000_data));
              float v2005_data = s3[78];
              float v2007_data = ir7[6];
              ir7[6] = (v2007_data + (v1974_data * v2005_data));
              float v2010_data = s3[90];
              float v2012_data = ir7[7];
              ir7[7] = (v2012_data + (v1974_data * v2010_data));
            }
            if (v2_lead < 12) {
              float v2018_data = r6[7];
              float v2019_data = s3[7];
              float v2021_data = ir7[0];
              ir7[0] = (v2021_data + (v2018_data * v2019_data));
              float v2024_data = s3[19];
              float v2026_data = ir7[1];
              ir7[1] = (v2026_data + (v2018_data * v2024_data));
              float v2029_data = s3[31];
              float v2031_data = ir7[2];
              ir7[2] = (v2031_data + (v2018_data * v2029_data));
              float v2034_data = s3[43];
              float v2036_data = ir7[3];
              ir7[3] = (v2036_data + (v2018_data * v2034_data));
              float v2039_data = s3[55];
              float v2041_data = ir7[4];
              ir7[4] = (v2041_data + (v2018_data * v2039_data));
              float v2044_data = s3[67];
              float v2046_data = ir7[5];
              ir7[5] = (v2046_data + (v2018_data * v2044_data));
              float v2049_data = s3[79];
              float v2051_data = ir7[6];
              ir7[6] = (v2051_data + (v2018_data * v2049_data));
              float v2054_data = s3[91];
              float v2056_data = ir7[7];
              ir7[7] = (v2056_data + (v2018_data * v2054_data));
            }
            if (v2_lead < 12) {
              float v2062_data = r6[8];
              float v2063_data = s3[8];
              float v2065_data = ir7[0];
              ir7[0] = (v2065_data + (v2062_data * v2063_data));
              float v2068_data = s3[20];
              float v2070_data = ir7[1];
              ir7[1] = (v2070_data + (v2062_data * v2068_data));
              float v2073_data = s3[32];
              float v2075_data = ir7[2];
              ir7[2] = (v2075_data + (v2062_data * v2073_data));
              float v2078_data = s3[44];
              float v2080_data = ir7[3];
              ir7[3] = (v2080_data + (v2062_data * v2078_data));
              float v2083_data = s3[56];
              float v2085_data = ir7[4];
              ir7[4] = (v2085_data + (v2062_data * v2083_data));
              float v2088_data = s3[68];
              float v2090_data = ir7[5];
              ir7[5] = (v2090_data + (v2062_data * v2088_data));
              float v2093_data = s3[80];
              float v2095_data = ir7[6];
              ir7[6] = (v2095_data + (v2062_data * v2093_data));
              float v2098_data = s3[92];
              float v2100_data = ir7[7];
              ir7[7] = (v2100_data + (v2062_data * v2098_data));
            }
            if (v2_lead < 12) {
              float v2106_data = r6[9];
              float v2107_data = s3[9];
              float v2109_data = ir7[0];
              ir7[0] = (v2109_data + (v2106_data * v2107_data));
              float v2112_data = s3[21];
              float v2114_data = ir7[1];
              ir7[1] = (v2114_data + (v2106_data * v2112_data));
              float v2117_data = s3[33];
              float v2119_data = ir7[2];
              ir7[2] = (v2119_data + (v2106_data * v2117_data));
              float v2122_data = s3[45];
              float v2124_data = ir7[3];
              ir7[3] = (v2124_data + (v2106_data * v2122_data));
              float v2127_data = s3[57];
              float v2129_data = ir7[4];
              ir7[4] = (v2129_data + (v2106_data * v2127_data));
              float v2132_data = s3[69];
              float v2134_data = ir7[5];
              ir7[5] = (v2134_data + (v2106_data * v2132_data));
              float v2137_data = s3[81];
              float v2139_data = ir7[6];
              ir7[6] = (v2139_data + (v2106_data * v2137_data));
              float v2142_data = s3[93];
              float v2144_data = ir7[7];
              ir7[7] = (v2144_data + (v2106_data * v2142_data));
            }
            if (v2_lead < 12) {
              float v2150_data = r6[10];
              float v2151_data = s3[10];
              float v2153_data = ir7[0];
              ir7[0] = (v2153_data + (v2150_data * v2151_data));
              float v2156_data = s3[22];
              float v2158_data = ir7[1];
              ir7[1] = (v2158_data + (v2150_data * v2156_data));
              float v2161_data = s3[34];
              float v2163_data = ir7[2];
              ir7[2] = (v2163_data + (v2150_data * v2161_data));
              float v2166_data = s3[46];
              float v2168_data = ir7[3];
              ir7[3] = (v2168_data + (v2150_data * v2166_data));
              float v2171_data = s3[58];
              float v2173_data = ir7[4];
              ir7[4] = (v2173_data + (v2150_data * v2171_data));
              float v2176_data = s3[70];
              float v2178_data = ir7[5];
              ir7[5] = (v2178_data + (v2150_data * v2176_data));
              float v2181_data = s3[82];
              float v2183_data = ir7[6];
              ir7[6] = (v2183_data + (v2150_data * v2181_data));
              float v2186_data = s3[94];
              float v2188_data = ir7[7];
              ir7[7] = (v2188_data + (v2150_data * v2186_data));
            }
            if (v2_lead < 12) {
              float v2194_data = r6[11];
              float v2195_data = s3[11];
              float v2197_data = ir7[0];
              ir7[0] = (v2197_data + (v2194_data * v2195_data));
              float v2200_data = s3[23];
              float v2202_data = ir7[1];
              ir7[1] = (v2202_data + (v2194_data * v2200_data));
              float v2205_data = s3[35];
              float v2207_data = ir7[2];
              ir7[2] = (v2207_data + (v2194_data * v2205_data));
              float v2210_data = s3[47];
              float v2212_data = ir7[3];
              ir7[3] = (v2212_data + (v2194_data * v2210_data));
              float v2215_data = s3[59];
              float v2217_data = ir7[4];
              ir7[4] = (v2217_data + (v2194_data * v2215_data));
              float v2220_data = s3[71];
              float v2222_data = ir7[5];
              ir7[5] = (v2222_data + (v2194_data * v2220_data));
              float v2225_data = s3[83];
              float v2227_data = ir7[6];
              ir7[6] = (v2227_data + (v2194_data * v2225_data));
              float v2230_data = s3[95];
              float v2232_data = ir7[7];
              ir7[7] = (v2232_data + (v2194_data * v2230_data));
            }
            if (v2_lead < 12) {
              #pragma unroll
              for (int32_t v2238_n1 = 0; v2238_n1 < 8; ++v2238_n1) {
                int32_t v2239_a = 0 + v2238_n1;
                float v2241_data = ir7[v2238_n1];
                int32_t v2242_a = 0 + v2238_n1;
                float v2244_data = r5[v2238_n1];
                int32_t v2246_a = 0 + v2238_n1;
                r7[v2238_n1] = (v2244_data + v2241_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v2252_i1 = 0; v2252_i1 < 8; ++v2252_i1) {
              int32_t v2253_a = 0 + v2252_i1;
              float v2255_data = r7[v2252_i1];
              int32_t v2262_a = v2_lead + (v2252_i1 * 12);
              glb_m0[v2262_a] = v2255_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

