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
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              int32_t v34_a = v28_i1 * 12;
              int32_t v35_a = v3_lead + v34_a;
              float v43_data = __ldcg(&glb_m3[(v3_lead + v34_a)]);
              int32_t v44_a = 0 + v28_i1;
              r2[v44_a] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v3_lead < 12) {
            float v51_data = r0[0];
            float v52_data = s0[0];
            float v54_data = ir1[0];
            ir1[0] = (v54_data + (v51_data * v52_data));
            float v57_data = s0[12];
            float v59_data = ir1[1];
            ir1[1] = (v59_data + (v51_data * v57_data));
            float v62_data = s0[24];
            float v64_data = ir1[2];
            ir1[2] = (v64_data + (v51_data * v62_data));
            float v67_data = s0[36];
            float v69_data = ir1[3];
            ir1[3] = (v69_data + (v51_data * v67_data));
            float v72_data = s0[48];
            float v74_data = ir1[4];
            ir1[4] = (v74_data + (v51_data * v72_data));
            float v77_data = s0[60];
            float v79_data = ir1[5];
            ir1[5] = (v79_data + (v51_data * v77_data));
            float v82_data = s0[72];
            float v84_data = ir1[6];
            ir1[6] = (v84_data + (v51_data * v82_data));
            float v87_data = s0[84];
            float v89_data = ir1[7];
            ir1[7] = (v89_data + (v51_data * v87_data));
          }
          if (v3_lead < 12) {
            float v95_data = r0[1];
            float v96_data = s0[1];
            float v98_data = ir1[0];
            ir1[0] = (v98_data + (v95_data * v96_data));
            float v101_data = s0[13];
            float v103_data = ir1[1];
            ir1[1] = (v103_data + (v95_data * v101_data));
            float v106_data = s0[25];
            float v108_data = ir1[2];
            ir1[2] = (v108_data + (v95_data * v106_data));
            float v111_data = s0[37];
            float v113_data = ir1[3];
            ir1[3] = (v113_data + (v95_data * v111_data));
            float v116_data = s0[49];
            float v118_data = ir1[4];
            ir1[4] = (v118_data + (v95_data * v116_data));
            float v121_data = s0[61];
            float v123_data = ir1[5];
            ir1[5] = (v123_data + (v95_data * v121_data));
            float v126_data = s0[73];
            float v128_data = ir1[6];
            ir1[6] = (v128_data + (v95_data * v126_data));
            float v131_data = s0[85];
            float v133_data = ir1[7];
            ir1[7] = (v133_data + (v95_data * v131_data));
          }
          if (v3_lead < 12) {
            float v139_data = r0[2];
            float v140_data = s0[2];
            float v142_data = ir1[0];
            ir1[0] = (v142_data + (v139_data * v140_data));
            float v145_data = s0[14];
            float v147_data = ir1[1];
            ir1[1] = (v147_data + (v139_data * v145_data));
            float v150_data = s0[26];
            float v152_data = ir1[2];
            ir1[2] = (v152_data + (v139_data * v150_data));
            float v155_data = s0[38];
            float v157_data = ir1[3];
            ir1[3] = (v157_data + (v139_data * v155_data));
            float v160_data = s0[50];
            float v162_data = ir1[4];
            ir1[4] = (v162_data + (v139_data * v160_data));
            float v165_data = s0[62];
            float v167_data = ir1[5];
            ir1[5] = (v167_data + (v139_data * v165_data));
            float v170_data = s0[74];
            float v172_data = ir1[6];
            ir1[6] = (v172_data + (v139_data * v170_data));
            float v175_data = s0[86];
            float v177_data = ir1[7];
            ir1[7] = (v177_data + (v139_data * v175_data));
          }
          if (v3_lead < 12) {
            float v183_data = r0[3];
            float v184_data = s0[3];
            float v186_data = ir1[0];
            ir1[0] = (v186_data + (v183_data * v184_data));
            float v189_data = s0[15];
            float v191_data = ir1[1];
            ir1[1] = (v191_data + (v183_data * v189_data));
            float v194_data = s0[27];
            float v196_data = ir1[2];
            ir1[2] = (v196_data + (v183_data * v194_data));
            float v199_data = s0[39];
            float v201_data = ir1[3];
            ir1[3] = (v201_data + (v183_data * v199_data));
            float v204_data = s0[51];
            float v206_data = ir1[4];
            ir1[4] = (v206_data + (v183_data * v204_data));
            float v209_data = s0[63];
            float v211_data = ir1[5];
            ir1[5] = (v211_data + (v183_data * v209_data));
            float v214_data = s0[75];
            float v216_data = ir1[6];
            ir1[6] = (v216_data + (v183_data * v214_data));
            float v219_data = s0[87];
            float v221_data = ir1[7];
            ir1[7] = (v221_data + (v183_data * v219_data));
          }
          if (v3_lead < 12) {
            float v227_data = r0[4];
            float v228_data = s0[4];
            float v230_data = ir1[0];
            ir1[0] = (v230_data + (v227_data * v228_data));
            float v233_data = s0[16];
            float v235_data = ir1[1];
            ir1[1] = (v235_data + (v227_data * v233_data));
            float v238_data = s0[28];
            float v240_data = ir1[2];
            ir1[2] = (v240_data + (v227_data * v238_data));
            float v243_data = s0[40];
            float v245_data = ir1[3];
            ir1[3] = (v245_data + (v227_data * v243_data));
            float v248_data = s0[52];
            float v250_data = ir1[4];
            ir1[4] = (v250_data + (v227_data * v248_data));
            float v253_data = s0[64];
            float v255_data = ir1[5];
            ir1[5] = (v255_data + (v227_data * v253_data));
            float v258_data = s0[76];
            float v260_data = ir1[6];
            ir1[6] = (v260_data + (v227_data * v258_data));
            float v263_data = s0[88];
            float v265_data = ir1[7];
            ir1[7] = (v265_data + (v227_data * v263_data));
          }
          if (v3_lead < 12) {
            float v271_data = r0[5];
            float v272_data = s0[5];
            float v274_data = ir1[0];
            ir1[0] = (v274_data + (v271_data * v272_data));
            float v277_data = s0[17];
            float v279_data = ir1[1];
            ir1[1] = (v279_data + (v271_data * v277_data));
            float v282_data = s0[29];
            float v284_data = ir1[2];
            ir1[2] = (v284_data + (v271_data * v282_data));
            float v287_data = s0[41];
            float v289_data = ir1[3];
            ir1[3] = (v289_data + (v271_data * v287_data));
            float v292_data = s0[53];
            float v294_data = ir1[4];
            ir1[4] = (v294_data + (v271_data * v292_data));
            float v297_data = s0[65];
            float v299_data = ir1[5];
            ir1[5] = (v299_data + (v271_data * v297_data));
            float v302_data = s0[77];
            float v304_data = ir1[6];
            ir1[6] = (v304_data + (v271_data * v302_data));
            float v307_data = s0[89];
            float v309_data = ir1[7];
            ir1[7] = (v309_data + (v271_data * v307_data));
          }
          if (v3_lead < 12) {
            float v315_data = r0[6];
            float v316_data = s0[6];
            float v318_data = ir1[0];
            ir1[0] = (v318_data + (v315_data * v316_data));
            float v321_data = s0[18];
            float v323_data = ir1[1];
            ir1[1] = (v323_data + (v315_data * v321_data));
            float v326_data = s0[30];
            float v328_data = ir1[2];
            ir1[2] = (v328_data + (v315_data * v326_data));
            float v331_data = s0[42];
            float v333_data = ir1[3];
            ir1[3] = (v333_data + (v315_data * v331_data));
            float v336_data = s0[54];
            float v338_data = ir1[4];
            ir1[4] = (v338_data + (v315_data * v336_data));
            float v341_data = s0[66];
            float v343_data = ir1[5];
            ir1[5] = (v343_data + (v315_data * v341_data));
            float v346_data = s0[78];
            float v348_data = ir1[6];
            ir1[6] = (v348_data + (v315_data * v346_data));
            float v351_data = s0[90];
            float v353_data = ir1[7];
            ir1[7] = (v353_data + (v315_data * v351_data));
          }
          if (v3_lead < 12) {
            float v359_data = r0[7];
            float v360_data = s0[7];
            float v362_data = ir1[0];
            ir1[0] = (v362_data + (v359_data * v360_data));
            float v365_data = s0[19];
            float v367_data = ir1[1];
            ir1[1] = (v367_data + (v359_data * v365_data));
            float v370_data = s0[31];
            float v372_data = ir1[2];
            ir1[2] = (v372_data + (v359_data * v370_data));
            float v375_data = s0[43];
            float v377_data = ir1[3];
            ir1[3] = (v377_data + (v359_data * v375_data));
            float v380_data = s0[55];
            float v382_data = ir1[4];
            ir1[4] = (v382_data + (v359_data * v380_data));
            float v385_data = s0[67];
            float v387_data = ir1[5];
            ir1[5] = (v387_data + (v359_data * v385_data));
            float v390_data = s0[79];
            float v392_data = ir1[6];
            ir1[6] = (v392_data + (v359_data * v390_data));
            float v395_data = s0[91];
            float v397_data = ir1[7];
            ir1[7] = (v397_data + (v359_data * v395_data));
          }
          if (v3_lead < 12) {
            float v403_data = r0[8];
            float v404_data = s0[8];
            float v406_data = ir1[0];
            ir1[0] = (v406_data + (v403_data * v404_data));
            float v409_data = s0[20];
            float v411_data = ir1[1];
            ir1[1] = (v411_data + (v403_data * v409_data));
            float v414_data = s0[32];
            float v416_data = ir1[2];
            ir1[2] = (v416_data + (v403_data * v414_data));
            float v419_data = s0[44];
            float v421_data = ir1[3];
            ir1[3] = (v421_data + (v403_data * v419_data));
            float v424_data = s0[56];
            float v426_data = ir1[4];
            ir1[4] = (v426_data + (v403_data * v424_data));
            float v429_data = s0[68];
            float v431_data = ir1[5];
            ir1[5] = (v431_data + (v403_data * v429_data));
            float v434_data = s0[80];
            float v436_data = ir1[6];
            ir1[6] = (v436_data + (v403_data * v434_data));
            float v439_data = s0[92];
            float v441_data = ir1[7];
            ir1[7] = (v441_data + (v403_data * v439_data));
          }
          if (v3_lead < 12) {
            float v447_data = r0[9];
            float v448_data = s0[9];
            float v450_data = ir1[0];
            ir1[0] = (v450_data + (v447_data * v448_data));
            float v453_data = s0[21];
            float v455_data = ir1[1];
            ir1[1] = (v455_data + (v447_data * v453_data));
            float v458_data = s0[33];
            float v460_data = ir1[2];
            ir1[2] = (v460_data + (v447_data * v458_data));
            float v463_data = s0[45];
            float v465_data = ir1[3];
            ir1[3] = (v465_data + (v447_data * v463_data));
            float v468_data = s0[57];
            float v470_data = ir1[4];
            ir1[4] = (v470_data + (v447_data * v468_data));
            float v473_data = s0[69];
            float v475_data = ir1[5];
            ir1[5] = (v475_data + (v447_data * v473_data));
            float v478_data = s0[81];
            float v480_data = ir1[6];
            ir1[6] = (v480_data + (v447_data * v478_data));
            float v483_data = s0[93];
            float v485_data = ir1[7];
            ir1[7] = (v485_data + (v447_data * v483_data));
          }
          if (v3_lead < 12) {
            float v491_data = r0[10];
            float v492_data = s0[10];
            float v494_data = ir1[0];
            ir1[0] = (v494_data + (v491_data * v492_data));
            float v497_data = s0[22];
            float v499_data = ir1[1];
            ir1[1] = (v499_data + (v491_data * v497_data));
            float v502_data = s0[34];
            float v504_data = ir1[2];
            ir1[2] = (v504_data + (v491_data * v502_data));
            float v507_data = s0[46];
            float v509_data = ir1[3];
            ir1[3] = (v509_data + (v491_data * v507_data));
            float v512_data = s0[58];
            float v514_data = ir1[4];
            ir1[4] = (v514_data + (v491_data * v512_data));
            float v517_data = s0[70];
            float v519_data = ir1[5];
            ir1[5] = (v519_data + (v491_data * v517_data));
            float v522_data = s0[82];
            float v524_data = ir1[6];
            ir1[6] = (v524_data + (v491_data * v522_data));
            float v527_data = s0[94];
            float v529_data = ir1[7];
            ir1[7] = (v529_data + (v491_data * v527_data));
          }
          if (v3_lead < 12) {
            float v535_data = r0[11];
            float v536_data = s0[11];
            float v538_data = ir1[0];
            ir1[0] = (v538_data + (v535_data * v536_data));
            float v541_data = s0[23];
            float v543_data = ir1[1];
            ir1[1] = (v543_data + (v535_data * v541_data));
            float v546_data = s0[35];
            float v548_data = ir1[2];
            ir1[2] = (v548_data + (v535_data * v546_data));
            float v551_data = s0[47];
            float v553_data = ir1[3];
            ir1[3] = (v553_data + (v535_data * v551_data));
            float v556_data = s0[59];
            float v558_data = ir1[4];
            ir1[4] = (v558_data + (v535_data * v556_data));
            float v561_data = s0[71];
            float v563_data = ir1[5];
            ir1[5] = (v563_data + (v535_data * v561_data));
            float v566_data = s0[83];
            float v568_data = ir1[6];
            ir1[6] = (v568_data + (v535_data * v566_data));
            float v571_data = s0[95];
            float v573_data = ir1[7];
            ir1[7] = (v573_data + (v535_data * v571_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v579_n1 = 0; v579_n1 < 8; ++v579_n1) {
              int32_t v580_a = 0 + v579_n1;
              float v582_data = ir1[v579_n1];
              int32_t v583_a = 0 + v579_n1;
              r1[v579_n1] = v582_data;
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
            for (int32_t v591_i1 = 0; v591_i1 < 12; ++v591_i1) {
              int32_t v597_a = v591_i1 * 12;
              int32_t v598_a = v3_lead + v597_a;
              float v606_data = __ldcg(&glb_m5[(v3_lead + v597_a)]);
              int32_t v607_a = 0 + v591_i1;
              r4[v607_a] = v606_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v3_lead < 12) {
            float v614_data = r2[0];
            float v615_data = s1[0];
            float v617_data = ir3[0];
            ir3[0] = (v617_data + (v614_data * v615_data));
            float v620_data = s1[12];
            float v622_data = ir3[1];
            ir3[1] = (v622_data + (v614_data * v620_data));
            float v625_data = s1[24];
            float v627_data = ir3[2];
            ir3[2] = (v627_data + (v614_data * v625_data));
            float v630_data = s1[36];
            float v632_data = ir3[3];
            ir3[3] = (v632_data + (v614_data * v630_data));
            float v635_data = s1[48];
            float v637_data = ir3[4];
            ir3[4] = (v637_data + (v614_data * v635_data));
            float v640_data = s1[60];
            float v642_data = ir3[5];
            ir3[5] = (v642_data + (v614_data * v640_data));
            float v645_data = s1[72];
            float v647_data = ir3[6];
            ir3[6] = (v647_data + (v614_data * v645_data));
            float v650_data = s1[84];
            float v652_data = ir3[7];
            ir3[7] = (v652_data + (v614_data * v650_data));
          }
          if (v3_lead < 12) {
            float v658_data = r2[1];
            float v659_data = s1[1];
            float v661_data = ir3[0];
            ir3[0] = (v661_data + (v658_data * v659_data));
            float v664_data = s1[13];
            float v666_data = ir3[1];
            ir3[1] = (v666_data + (v658_data * v664_data));
            float v669_data = s1[25];
            float v671_data = ir3[2];
            ir3[2] = (v671_data + (v658_data * v669_data));
            float v674_data = s1[37];
            float v676_data = ir3[3];
            ir3[3] = (v676_data + (v658_data * v674_data));
            float v679_data = s1[49];
            float v681_data = ir3[4];
            ir3[4] = (v681_data + (v658_data * v679_data));
            float v684_data = s1[61];
            float v686_data = ir3[5];
            ir3[5] = (v686_data + (v658_data * v684_data));
            float v689_data = s1[73];
            float v691_data = ir3[6];
            ir3[6] = (v691_data + (v658_data * v689_data));
            float v694_data = s1[85];
            float v696_data = ir3[7];
            ir3[7] = (v696_data + (v658_data * v694_data));
          }
          if (v3_lead < 12) {
            float v702_data = r2[2];
            float v703_data = s1[2];
            float v705_data = ir3[0];
            ir3[0] = (v705_data + (v702_data * v703_data));
            float v708_data = s1[14];
            float v710_data = ir3[1];
            ir3[1] = (v710_data + (v702_data * v708_data));
            float v713_data = s1[26];
            float v715_data = ir3[2];
            ir3[2] = (v715_data + (v702_data * v713_data));
            float v718_data = s1[38];
            float v720_data = ir3[3];
            ir3[3] = (v720_data + (v702_data * v718_data));
            float v723_data = s1[50];
            float v725_data = ir3[4];
            ir3[4] = (v725_data + (v702_data * v723_data));
            float v728_data = s1[62];
            float v730_data = ir3[5];
            ir3[5] = (v730_data + (v702_data * v728_data));
            float v733_data = s1[74];
            float v735_data = ir3[6];
            ir3[6] = (v735_data + (v702_data * v733_data));
            float v738_data = s1[86];
            float v740_data = ir3[7];
            ir3[7] = (v740_data + (v702_data * v738_data));
          }
          if (v3_lead < 12) {
            float v746_data = r2[3];
            float v747_data = s1[3];
            float v749_data = ir3[0];
            ir3[0] = (v749_data + (v746_data * v747_data));
            float v752_data = s1[15];
            float v754_data = ir3[1];
            ir3[1] = (v754_data + (v746_data * v752_data));
            float v757_data = s1[27];
            float v759_data = ir3[2];
            ir3[2] = (v759_data + (v746_data * v757_data));
            float v762_data = s1[39];
            float v764_data = ir3[3];
            ir3[3] = (v764_data + (v746_data * v762_data));
            float v767_data = s1[51];
            float v769_data = ir3[4];
            ir3[4] = (v769_data + (v746_data * v767_data));
            float v772_data = s1[63];
            float v774_data = ir3[5];
            ir3[5] = (v774_data + (v746_data * v772_data));
            float v777_data = s1[75];
            float v779_data = ir3[6];
            ir3[6] = (v779_data + (v746_data * v777_data));
            float v782_data = s1[87];
            float v784_data = ir3[7];
            ir3[7] = (v784_data + (v746_data * v782_data));
          }
          if (v3_lead < 12) {
            float v790_data = r2[4];
            float v791_data = s1[4];
            float v793_data = ir3[0];
            ir3[0] = (v793_data + (v790_data * v791_data));
            float v796_data = s1[16];
            float v798_data = ir3[1];
            ir3[1] = (v798_data + (v790_data * v796_data));
            float v801_data = s1[28];
            float v803_data = ir3[2];
            ir3[2] = (v803_data + (v790_data * v801_data));
            float v806_data = s1[40];
            float v808_data = ir3[3];
            ir3[3] = (v808_data + (v790_data * v806_data));
            float v811_data = s1[52];
            float v813_data = ir3[4];
            ir3[4] = (v813_data + (v790_data * v811_data));
            float v816_data = s1[64];
            float v818_data = ir3[5];
            ir3[5] = (v818_data + (v790_data * v816_data));
            float v821_data = s1[76];
            float v823_data = ir3[6];
            ir3[6] = (v823_data + (v790_data * v821_data));
            float v826_data = s1[88];
            float v828_data = ir3[7];
            ir3[7] = (v828_data + (v790_data * v826_data));
          }
          if (v3_lead < 12) {
            float v834_data = r2[5];
            float v835_data = s1[5];
            float v837_data = ir3[0];
            ir3[0] = (v837_data + (v834_data * v835_data));
            float v840_data = s1[17];
            float v842_data = ir3[1];
            ir3[1] = (v842_data + (v834_data * v840_data));
            float v845_data = s1[29];
            float v847_data = ir3[2];
            ir3[2] = (v847_data + (v834_data * v845_data));
            float v850_data = s1[41];
            float v852_data = ir3[3];
            ir3[3] = (v852_data + (v834_data * v850_data));
            float v855_data = s1[53];
            float v857_data = ir3[4];
            ir3[4] = (v857_data + (v834_data * v855_data));
            float v860_data = s1[65];
            float v862_data = ir3[5];
            ir3[5] = (v862_data + (v834_data * v860_data));
            float v865_data = s1[77];
            float v867_data = ir3[6];
            ir3[6] = (v867_data + (v834_data * v865_data));
            float v870_data = s1[89];
            float v872_data = ir3[7];
            ir3[7] = (v872_data + (v834_data * v870_data));
          }
          if (v3_lead < 12) {
            float v878_data = r2[6];
            float v879_data = s1[6];
            float v881_data = ir3[0];
            ir3[0] = (v881_data + (v878_data * v879_data));
            float v884_data = s1[18];
            float v886_data = ir3[1];
            ir3[1] = (v886_data + (v878_data * v884_data));
            float v889_data = s1[30];
            float v891_data = ir3[2];
            ir3[2] = (v891_data + (v878_data * v889_data));
            float v894_data = s1[42];
            float v896_data = ir3[3];
            ir3[3] = (v896_data + (v878_data * v894_data));
            float v899_data = s1[54];
            float v901_data = ir3[4];
            ir3[4] = (v901_data + (v878_data * v899_data));
            float v904_data = s1[66];
            float v906_data = ir3[5];
            ir3[5] = (v906_data + (v878_data * v904_data));
            float v909_data = s1[78];
            float v911_data = ir3[6];
            ir3[6] = (v911_data + (v878_data * v909_data));
            float v914_data = s1[90];
            float v916_data = ir3[7];
            ir3[7] = (v916_data + (v878_data * v914_data));
          }
          if (v3_lead < 12) {
            float v922_data = r2[7];
            float v923_data = s1[7];
            float v925_data = ir3[0];
            ir3[0] = (v925_data + (v922_data * v923_data));
            float v928_data = s1[19];
            float v930_data = ir3[1];
            ir3[1] = (v930_data + (v922_data * v928_data));
            float v933_data = s1[31];
            float v935_data = ir3[2];
            ir3[2] = (v935_data + (v922_data * v933_data));
            float v938_data = s1[43];
            float v940_data = ir3[3];
            ir3[3] = (v940_data + (v922_data * v938_data));
            float v943_data = s1[55];
            float v945_data = ir3[4];
            ir3[4] = (v945_data + (v922_data * v943_data));
            float v948_data = s1[67];
            float v950_data = ir3[5];
            ir3[5] = (v950_data + (v922_data * v948_data));
            float v953_data = s1[79];
            float v955_data = ir3[6];
            ir3[6] = (v955_data + (v922_data * v953_data));
            float v958_data = s1[91];
            float v960_data = ir3[7];
            ir3[7] = (v960_data + (v922_data * v958_data));
          }
          if (v3_lead < 12) {
            float v966_data = r2[8];
            float v967_data = s1[8];
            float v969_data = ir3[0];
            ir3[0] = (v969_data + (v966_data * v967_data));
            float v972_data = s1[20];
            float v974_data = ir3[1];
            ir3[1] = (v974_data + (v966_data * v972_data));
            float v977_data = s1[32];
            float v979_data = ir3[2];
            ir3[2] = (v979_data + (v966_data * v977_data));
            float v982_data = s1[44];
            float v984_data = ir3[3];
            ir3[3] = (v984_data + (v966_data * v982_data));
            float v987_data = s1[56];
            float v989_data = ir3[4];
            ir3[4] = (v989_data + (v966_data * v987_data));
            float v992_data = s1[68];
            float v994_data = ir3[5];
            ir3[5] = (v994_data + (v966_data * v992_data));
            float v997_data = s1[80];
            float v999_data = ir3[6];
            ir3[6] = (v999_data + (v966_data * v997_data));
            float v1002_data = s1[92];
            float v1004_data = ir3[7];
            ir3[7] = (v1004_data + (v966_data * v1002_data));
          }
          if (v3_lead < 12) {
            float v1010_data = r2[9];
            float v1011_data = s1[9];
            float v1013_data = ir3[0];
            ir3[0] = (v1013_data + (v1010_data * v1011_data));
            float v1016_data = s1[21];
            float v1018_data = ir3[1];
            ir3[1] = (v1018_data + (v1010_data * v1016_data));
            float v1021_data = s1[33];
            float v1023_data = ir3[2];
            ir3[2] = (v1023_data + (v1010_data * v1021_data));
            float v1026_data = s1[45];
            float v1028_data = ir3[3];
            ir3[3] = (v1028_data + (v1010_data * v1026_data));
            float v1031_data = s1[57];
            float v1033_data = ir3[4];
            ir3[4] = (v1033_data + (v1010_data * v1031_data));
            float v1036_data = s1[69];
            float v1038_data = ir3[5];
            ir3[5] = (v1038_data + (v1010_data * v1036_data));
            float v1041_data = s1[81];
            float v1043_data = ir3[6];
            ir3[6] = (v1043_data + (v1010_data * v1041_data));
            float v1046_data = s1[93];
            float v1048_data = ir3[7];
            ir3[7] = (v1048_data + (v1010_data * v1046_data));
          }
          if (v3_lead < 12) {
            float v1054_data = r2[10];
            float v1055_data = s1[10];
            float v1057_data = ir3[0];
            ir3[0] = (v1057_data + (v1054_data * v1055_data));
            float v1060_data = s1[22];
            float v1062_data = ir3[1];
            ir3[1] = (v1062_data + (v1054_data * v1060_data));
            float v1065_data = s1[34];
            float v1067_data = ir3[2];
            ir3[2] = (v1067_data + (v1054_data * v1065_data));
            float v1070_data = s1[46];
            float v1072_data = ir3[3];
            ir3[3] = (v1072_data + (v1054_data * v1070_data));
            float v1075_data = s1[58];
            float v1077_data = ir3[4];
            ir3[4] = (v1077_data + (v1054_data * v1075_data));
            float v1080_data = s1[70];
            float v1082_data = ir3[5];
            ir3[5] = (v1082_data + (v1054_data * v1080_data));
            float v1085_data = s1[82];
            float v1087_data = ir3[6];
            ir3[6] = (v1087_data + (v1054_data * v1085_data));
            float v1090_data = s1[94];
            float v1092_data = ir3[7];
            ir3[7] = (v1092_data + (v1054_data * v1090_data));
          }
          if (v3_lead < 12) {
            float v1098_data = r2[11];
            float v1099_data = s1[11];
            float v1101_data = ir3[0];
            ir3[0] = (v1101_data + (v1098_data * v1099_data));
            float v1104_data = s1[23];
            float v1106_data = ir3[1];
            ir3[1] = (v1106_data + (v1098_data * v1104_data));
            float v1109_data = s1[35];
            float v1111_data = ir3[2];
            ir3[2] = (v1111_data + (v1098_data * v1109_data));
            float v1114_data = s1[47];
            float v1116_data = ir3[3];
            ir3[3] = (v1116_data + (v1098_data * v1114_data));
            float v1119_data = s1[59];
            float v1121_data = ir3[4];
            ir3[4] = (v1121_data + (v1098_data * v1119_data));
            float v1124_data = s1[71];
            float v1126_data = ir3[5];
            ir3[5] = (v1126_data + (v1098_data * v1124_data));
            float v1129_data = s1[83];
            float v1131_data = ir3[6];
            ir3[6] = (v1131_data + (v1098_data * v1129_data));
            float v1134_data = s1[95];
            float v1136_data = ir3[7];
            ir3[7] = (v1136_data + (v1098_data * v1134_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v1142_n1 = 0; v1142_n1 < 8; ++v1142_n1) {
              int32_t v1143_a = 0 + v1142_n1;
              float v1145_data = ir3[v1142_n1];
              int32_t v1146_a = 0 + v1142_n1;
              float v1148_data = r1[v1142_n1];
              int32_t v1150_a = 0 + v1142_n1;
              r3[v1142_n1] = (v1148_data + v1145_data);
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
            for (int32_t v1158_i1 = 0; v1158_i1 < 12; ++v1158_i1) {
              int32_t v1164_a = v1158_i1 * 12;
              int32_t v1165_a = v3_lead + v1164_a;
              float v1173_data = __ldcg(&glb_m7[(v3_lead + v1164_a)]);
              int32_t v1174_a = 0 + v1158_i1;
              r6[v1174_a] = v1173_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v3_lead < 12) {
            float v1181_data = r4[0];
            float v1182_data = s2[0];
            float v1184_data = ir5[0];
            ir5[0] = (v1184_data + (v1181_data * v1182_data));
            float v1187_data = s2[12];
            float v1189_data = ir5[1];
            ir5[1] = (v1189_data + (v1181_data * v1187_data));
            float v1192_data = s2[24];
            float v1194_data = ir5[2];
            ir5[2] = (v1194_data + (v1181_data * v1192_data));
            float v1197_data = s2[36];
            float v1199_data = ir5[3];
            ir5[3] = (v1199_data + (v1181_data * v1197_data));
            float v1202_data = s2[48];
            float v1204_data = ir5[4];
            ir5[4] = (v1204_data + (v1181_data * v1202_data));
            float v1207_data = s2[60];
            float v1209_data = ir5[5];
            ir5[5] = (v1209_data + (v1181_data * v1207_data));
            float v1212_data = s2[72];
            float v1214_data = ir5[6];
            ir5[6] = (v1214_data + (v1181_data * v1212_data));
            float v1217_data = s2[84];
            float v1219_data = ir5[7];
            ir5[7] = (v1219_data + (v1181_data * v1217_data));
          }
          if (v3_lead < 12) {
            float v1225_data = r4[1];
            float v1226_data = s2[1];
            float v1228_data = ir5[0];
            ir5[0] = (v1228_data + (v1225_data * v1226_data));
            float v1231_data = s2[13];
            float v1233_data = ir5[1];
            ir5[1] = (v1233_data + (v1225_data * v1231_data));
            float v1236_data = s2[25];
            float v1238_data = ir5[2];
            ir5[2] = (v1238_data + (v1225_data * v1236_data));
            float v1241_data = s2[37];
            float v1243_data = ir5[3];
            ir5[3] = (v1243_data + (v1225_data * v1241_data));
            float v1246_data = s2[49];
            float v1248_data = ir5[4];
            ir5[4] = (v1248_data + (v1225_data * v1246_data));
            float v1251_data = s2[61];
            float v1253_data = ir5[5];
            ir5[5] = (v1253_data + (v1225_data * v1251_data));
            float v1256_data = s2[73];
            float v1258_data = ir5[6];
            ir5[6] = (v1258_data + (v1225_data * v1256_data));
            float v1261_data = s2[85];
            float v1263_data = ir5[7];
            ir5[7] = (v1263_data + (v1225_data * v1261_data));
          }
          if (v3_lead < 12) {
            float v1269_data = r4[2];
            float v1270_data = s2[2];
            float v1272_data = ir5[0];
            ir5[0] = (v1272_data + (v1269_data * v1270_data));
            float v1275_data = s2[14];
            float v1277_data = ir5[1];
            ir5[1] = (v1277_data + (v1269_data * v1275_data));
            float v1280_data = s2[26];
            float v1282_data = ir5[2];
            ir5[2] = (v1282_data + (v1269_data * v1280_data));
            float v1285_data = s2[38];
            float v1287_data = ir5[3];
            ir5[3] = (v1287_data + (v1269_data * v1285_data));
            float v1290_data = s2[50];
            float v1292_data = ir5[4];
            ir5[4] = (v1292_data + (v1269_data * v1290_data));
            float v1295_data = s2[62];
            float v1297_data = ir5[5];
            ir5[5] = (v1297_data + (v1269_data * v1295_data));
            float v1300_data = s2[74];
            float v1302_data = ir5[6];
            ir5[6] = (v1302_data + (v1269_data * v1300_data));
            float v1305_data = s2[86];
            float v1307_data = ir5[7];
            ir5[7] = (v1307_data + (v1269_data * v1305_data));
          }
          if (v3_lead < 12) {
            float v1313_data = r4[3];
            float v1314_data = s2[3];
            float v1316_data = ir5[0];
            ir5[0] = (v1316_data + (v1313_data * v1314_data));
            float v1319_data = s2[15];
            float v1321_data = ir5[1];
            ir5[1] = (v1321_data + (v1313_data * v1319_data));
            float v1324_data = s2[27];
            float v1326_data = ir5[2];
            ir5[2] = (v1326_data + (v1313_data * v1324_data));
            float v1329_data = s2[39];
            float v1331_data = ir5[3];
            ir5[3] = (v1331_data + (v1313_data * v1329_data));
            float v1334_data = s2[51];
            float v1336_data = ir5[4];
            ir5[4] = (v1336_data + (v1313_data * v1334_data));
            float v1339_data = s2[63];
            float v1341_data = ir5[5];
            ir5[5] = (v1341_data + (v1313_data * v1339_data));
            float v1344_data = s2[75];
            float v1346_data = ir5[6];
            ir5[6] = (v1346_data + (v1313_data * v1344_data));
            float v1349_data = s2[87];
            float v1351_data = ir5[7];
            ir5[7] = (v1351_data + (v1313_data * v1349_data));
          }
          if (v3_lead < 12) {
            float v1357_data = r4[4];
            float v1358_data = s2[4];
            float v1360_data = ir5[0];
            ir5[0] = (v1360_data + (v1357_data * v1358_data));
            float v1363_data = s2[16];
            float v1365_data = ir5[1];
            ir5[1] = (v1365_data + (v1357_data * v1363_data));
            float v1368_data = s2[28];
            float v1370_data = ir5[2];
            ir5[2] = (v1370_data + (v1357_data * v1368_data));
            float v1373_data = s2[40];
            float v1375_data = ir5[3];
            ir5[3] = (v1375_data + (v1357_data * v1373_data));
            float v1378_data = s2[52];
            float v1380_data = ir5[4];
            ir5[4] = (v1380_data + (v1357_data * v1378_data));
            float v1383_data = s2[64];
            float v1385_data = ir5[5];
            ir5[5] = (v1385_data + (v1357_data * v1383_data));
            float v1388_data = s2[76];
            float v1390_data = ir5[6];
            ir5[6] = (v1390_data + (v1357_data * v1388_data));
            float v1393_data = s2[88];
            float v1395_data = ir5[7];
            ir5[7] = (v1395_data + (v1357_data * v1393_data));
          }
          if (v3_lead < 12) {
            float v1401_data = r4[5];
            float v1402_data = s2[5];
            float v1404_data = ir5[0];
            ir5[0] = (v1404_data + (v1401_data * v1402_data));
            float v1407_data = s2[17];
            float v1409_data = ir5[1];
            ir5[1] = (v1409_data + (v1401_data * v1407_data));
            float v1412_data = s2[29];
            float v1414_data = ir5[2];
            ir5[2] = (v1414_data + (v1401_data * v1412_data));
            float v1417_data = s2[41];
            float v1419_data = ir5[3];
            ir5[3] = (v1419_data + (v1401_data * v1417_data));
            float v1422_data = s2[53];
            float v1424_data = ir5[4];
            ir5[4] = (v1424_data + (v1401_data * v1422_data));
            float v1427_data = s2[65];
            float v1429_data = ir5[5];
            ir5[5] = (v1429_data + (v1401_data * v1427_data));
            float v1432_data = s2[77];
            float v1434_data = ir5[6];
            ir5[6] = (v1434_data + (v1401_data * v1432_data));
            float v1437_data = s2[89];
            float v1439_data = ir5[7];
            ir5[7] = (v1439_data + (v1401_data * v1437_data));
          }
          if (v3_lead < 12) {
            float v1445_data = r4[6];
            float v1446_data = s2[6];
            float v1448_data = ir5[0];
            ir5[0] = (v1448_data + (v1445_data * v1446_data));
            float v1451_data = s2[18];
            float v1453_data = ir5[1];
            ir5[1] = (v1453_data + (v1445_data * v1451_data));
            float v1456_data = s2[30];
            float v1458_data = ir5[2];
            ir5[2] = (v1458_data + (v1445_data * v1456_data));
            float v1461_data = s2[42];
            float v1463_data = ir5[3];
            ir5[3] = (v1463_data + (v1445_data * v1461_data));
            float v1466_data = s2[54];
            float v1468_data = ir5[4];
            ir5[4] = (v1468_data + (v1445_data * v1466_data));
            float v1471_data = s2[66];
            float v1473_data = ir5[5];
            ir5[5] = (v1473_data + (v1445_data * v1471_data));
            float v1476_data = s2[78];
            float v1478_data = ir5[6];
            ir5[6] = (v1478_data + (v1445_data * v1476_data));
            float v1481_data = s2[90];
            float v1483_data = ir5[7];
            ir5[7] = (v1483_data + (v1445_data * v1481_data));
          }
          if (v3_lead < 12) {
            float v1489_data = r4[7];
            float v1490_data = s2[7];
            float v1492_data = ir5[0];
            ir5[0] = (v1492_data + (v1489_data * v1490_data));
            float v1495_data = s2[19];
            float v1497_data = ir5[1];
            ir5[1] = (v1497_data + (v1489_data * v1495_data));
            float v1500_data = s2[31];
            float v1502_data = ir5[2];
            ir5[2] = (v1502_data + (v1489_data * v1500_data));
            float v1505_data = s2[43];
            float v1507_data = ir5[3];
            ir5[3] = (v1507_data + (v1489_data * v1505_data));
            float v1510_data = s2[55];
            float v1512_data = ir5[4];
            ir5[4] = (v1512_data + (v1489_data * v1510_data));
            float v1515_data = s2[67];
            float v1517_data = ir5[5];
            ir5[5] = (v1517_data + (v1489_data * v1515_data));
            float v1520_data = s2[79];
            float v1522_data = ir5[6];
            ir5[6] = (v1522_data + (v1489_data * v1520_data));
            float v1525_data = s2[91];
            float v1527_data = ir5[7];
            ir5[7] = (v1527_data + (v1489_data * v1525_data));
          }
          if (v3_lead < 12) {
            float v1533_data = r4[8];
            float v1534_data = s2[8];
            float v1536_data = ir5[0];
            ir5[0] = (v1536_data + (v1533_data * v1534_data));
            float v1539_data = s2[20];
            float v1541_data = ir5[1];
            ir5[1] = (v1541_data + (v1533_data * v1539_data));
            float v1544_data = s2[32];
            float v1546_data = ir5[2];
            ir5[2] = (v1546_data + (v1533_data * v1544_data));
            float v1549_data = s2[44];
            float v1551_data = ir5[3];
            ir5[3] = (v1551_data + (v1533_data * v1549_data));
            float v1554_data = s2[56];
            float v1556_data = ir5[4];
            ir5[4] = (v1556_data + (v1533_data * v1554_data));
            float v1559_data = s2[68];
            float v1561_data = ir5[5];
            ir5[5] = (v1561_data + (v1533_data * v1559_data));
            float v1564_data = s2[80];
            float v1566_data = ir5[6];
            ir5[6] = (v1566_data + (v1533_data * v1564_data));
            float v1569_data = s2[92];
            float v1571_data = ir5[7];
            ir5[7] = (v1571_data + (v1533_data * v1569_data));
          }
          if (v3_lead < 12) {
            float v1577_data = r4[9];
            float v1578_data = s2[9];
            float v1580_data = ir5[0];
            ir5[0] = (v1580_data + (v1577_data * v1578_data));
            float v1583_data = s2[21];
            float v1585_data = ir5[1];
            ir5[1] = (v1585_data + (v1577_data * v1583_data));
            float v1588_data = s2[33];
            float v1590_data = ir5[2];
            ir5[2] = (v1590_data + (v1577_data * v1588_data));
            float v1593_data = s2[45];
            float v1595_data = ir5[3];
            ir5[3] = (v1595_data + (v1577_data * v1593_data));
            float v1598_data = s2[57];
            float v1600_data = ir5[4];
            ir5[4] = (v1600_data + (v1577_data * v1598_data));
            float v1603_data = s2[69];
            float v1605_data = ir5[5];
            ir5[5] = (v1605_data + (v1577_data * v1603_data));
            float v1608_data = s2[81];
            float v1610_data = ir5[6];
            ir5[6] = (v1610_data + (v1577_data * v1608_data));
            float v1613_data = s2[93];
            float v1615_data = ir5[7];
            ir5[7] = (v1615_data + (v1577_data * v1613_data));
          }
          if (v3_lead < 12) {
            float v1621_data = r4[10];
            float v1622_data = s2[10];
            float v1624_data = ir5[0];
            ir5[0] = (v1624_data + (v1621_data * v1622_data));
            float v1627_data = s2[22];
            float v1629_data = ir5[1];
            ir5[1] = (v1629_data + (v1621_data * v1627_data));
            float v1632_data = s2[34];
            float v1634_data = ir5[2];
            ir5[2] = (v1634_data + (v1621_data * v1632_data));
            float v1637_data = s2[46];
            float v1639_data = ir5[3];
            ir5[3] = (v1639_data + (v1621_data * v1637_data));
            float v1642_data = s2[58];
            float v1644_data = ir5[4];
            ir5[4] = (v1644_data + (v1621_data * v1642_data));
            float v1647_data = s2[70];
            float v1649_data = ir5[5];
            ir5[5] = (v1649_data + (v1621_data * v1647_data));
            float v1652_data = s2[82];
            float v1654_data = ir5[6];
            ir5[6] = (v1654_data + (v1621_data * v1652_data));
            float v1657_data = s2[94];
            float v1659_data = ir5[7];
            ir5[7] = (v1659_data + (v1621_data * v1657_data));
          }
          if (v3_lead < 12) {
            float v1665_data = r4[11];
            float v1666_data = s2[11];
            float v1668_data = ir5[0];
            ir5[0] = (v1668_data + (v1665_data * v1666_data));
            float v1671_data = s2[23];
            float v1673_data = ir5[1];
            ir5[1] = (v1673_data + (v1665_data * v1671_data));
            float v1676_data = s2[35];
            float v1678_data = ir5[2];
            ir5[2] = (v1678_data + (v1665_data * v1676_data));
            float v1681_data = s2[47];
            float v1683_data = ir5[3];
            ir5[3] = (v1683_data + (v1665_data * v1681_data));
            float v1686_data = s2[59];
            float v1688_data = ir5[4];
            ir5[4] = (v1688_data + (v1665_data * v1686_data));
            float v1691_data = s2[71];
            float v1693_data = ir5[5];
            ir5[5] = (v1693_data + (v1665_data * v1691_data));
            float v1696_data = s2[83];
            float v1698_data = ir5[6];
            ir5[6] = (v1698_data + (v1665_data * v1696_data));
            float v1701_data = s2[95];
            float v1703_data = ir5[7];
            ir5[7] = (v1703_data + (v1665_data * v1701_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v1709_n1 = 0; v1709_n1 < 8; ++v1709_n1) {
              int32_t v1710_a = 0 + v1709_n1;
              float v1712_data = ir5[v1709_n1];
              int32_t v1713_a = 0 + v1709_n1;
              float v1715_data = r3[v1709_n1];
              int32_t v1717_a = 0 + v1709_n1;
              r5[v1709_n1] = (v1715_data + v1712_data);
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
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v3_lead < 12) {
            float v1726_data = r6[0];
            float v1727_data = s3[0];
            float v1729_data = ir7[0];
            ir7[0] = (v1729_data + (v1726_data * v1727_data));
            float v1732_data = s3[12];
            float v1734_data = ir7[1];
            ir7[1] = (v1734_data + (v1726_data * v1732_data));
            float v1737_data = s3[24];
            float v1739_data = ir7[2];
            ir7[2] = (v1739_data + (v1726_data * v1737_data));
            float v1742_data = s3[36];
            float v1744_data = ir7[3];
            ir7[3] = (v1744_data + (v1726_data * v1742_data));
            float v1747_data = s3[48];
            float v1749_data = ir7[4];
            ir7[4] = (v1749_data + (v1726_data * v1747_data));
            float v1752_data = s3[60];
            float v1754_data = ir7[5];
            ir7[5] = (v1754_data + (v1726_data * v1752_data));
            float v1757_data = s3[72];
            float v1759_data = ir7[6];
            ir7[6] = (v1759_data + (v1726_data * v1757_data));
            float v1762_data = s3[84];
            float v1764_data = ir7[7];
            ir7[7] = (v1764_data + (v1726_data * v1762_data));
          }
          if (v3_lead < 12) {
            float v1770_data = r6[1];
            float v1771_data = s3[1];
            float v1773_data = ir7[0];
            ir7[0] = (v1773_data + (v1770_data * v1771_data));
            float v1776_data = s3[13];
            float v1778_data = ir7[1];
            ir7[1] = (v1778_data + (v1770_data * v1776_data));
            float v1781_data = s3[25];
            float v1783_data = ir7[2];
            ir7[2] = (v1783_data + (v1770_data * v1781_data));
            float v1786_data = s3[37];
            float v1788_data = ir7[3];
            ir7[3] = (v1788_data + (v1770_data * v1786_data));
            float v1791_data = s3[49];
            float v1793_data = ir7[4];
            ir7[4] = (v1793_data + (v1770_data * v1791_data));
            float v1796_data = s3[61];
            float v1798_data = ir7[5];
            ir7[5] = (v1798_data + (v1770_data * v1796_data));
            float v1801_data = s3[73];
            float v1803_data = ir7[6];
            ir7[6] = (v1803_data + (v1770_data * v1801_data));
            float v1806_data = s3[85];
            float v1808_data = ir7[7];
            ir7[7] = (v1808_data + (v1770_data * v1806_data));
          }
          if (v3_lead < 12) {
            float v1814_data = r6[2];
            float v1815_data = s3[2];
            float v1817_data = ir7[0];
            ir7[0] = (v1817_data + (v1814_data * v1815_data));
            float v1820_data = s3[14];
            float v1822_data = ir7[1];
            ir7[1] = (v1822_data + (v1814_data * v1820_data));
            float v1825_data = s3[26];
            float v1827_data = ir7[2];
            ir7[2] = (v1827_data + (v1814_data * v1825_data));
            float v1830_data = s3[38];
            float v1832_data = ir7[3];
            ir7[3] = (v1832_data + (v1814_data * v1830_data));
            float v1835_data = s3[50];
            float v1837_data = ir7[4];
            ir7[4] = (v1837_data + (v1814_data * v1835_data));
            float v1840_data = s3[62];
            float v1842_data = ir7[5];
            ir7[5] = (v1842_data + (v1814_data * v1840_data));
            float v1845_data = s3[74];
            float v1847_data = ir7[6];
            ir7[6] = (v1847_data + (v1814_data * v1845_data));
            float v1850_data = s3[86];
            float v1852_data = ir7[7];
            ir7[7] = (v1852_data + (v1814_data * v1850_data));
          }
          if (v3_lead < 12) {
            float v1858_data = r6[3];
            float v1859_data = s3[3];
            float v1861_data = ir7[0];
            ir7[0] = (v1861_data + (v1858_data * v1859_data));
            float v1864_data = s3[15];
            float v1866_data = ir7[1];
            ir7[1] = (v1866_data + (v1858_data * v1864_data));
            float v1869_data = s3[27];
            float v1871_data = ir7[2];
            ir7[2] = (v1871_data + (v1858_data * v1869_data));
            float v1874_data = s3[39];
            float v1876_data = ir7[3];
            ir7[3] = (v1876_data + (v1858_data * v1874_data));
            float v1879_data = s3[51];
            float v1881_data = ir7[4];
            ir7[4] = (v1881_data + (v1858_data * v1879_data));
            float v1884_data = s3[63];
            float v1886_data = ir7[5];
            ir7[5] = (v1886_data + (v1858_data * v1884_data));
            float v1889_data = s3[75];
            float v1891_data = ir7[6];
            ir7[6] = (v1891_data + (v1858_data * v1889_data));
            float v1894_data = s3[87];
            float v1896_data = ir7[7];
            ir7[7] = (v1896_data + (v1858_data * v1894_data));
          }
          if (v3_lead < 12) {
            float v1902_data = r6[4];
            float v1903_data = s3[4];
            float v1905_data = ir7[0];
            ir7[0] = (v1905_data + (v1902_data * v1903_data));
            float v1908_data = s3[16];
            float v1910_data = ir7[1];
            ir7[1] = (v1910_data + (v1902_data * v1908_data));
            float v1913_data = s3[28];
            float v1915_data = ir7[2];
            ir7[2] = (v1915_data + (v1902_data * v1913_data));
            float v1918_data = s3[40];
            float v1920_data = ir7[3];
            ir7[3] = (v1920_data + (v1902_data * v1918_data));
            float v1923_data = s3[52];
            float v1925_data = ir7[4];
            ir7[4] = (v1925_data + (v1902_data * v1923_data));
            float v1928_data = s3[64];
            float v1930_data = ir7[5];
            ir7[5] = (v1930_data + (v1902_data * v1928_data));
            float v1933_data = s3[76];
            float v1935_data = ir7[6];
            ir7[6] = (v1935_data + (v1902_data * v1933_data));
            float v1938_data = s3[88];
            float v1940_data = ir7[7];
            ir7[7] = (v1940_data + (v1902_data * v1938_data));
          }
          if (v3_lead < 12) {
            float v1946_data = r6[5];
            float v1947_data = s3[5];
            float v1949_data = ir7[0];
            ir7[0] = (v1949_data + (v1946_data * v1947_data));
            float v1952_data = s3[17];
            float v1954_data = ir7[1];
            ir7[1] = (v1954_data + (v1946_data * v1952_data));
            float v1957_data = s3[29];
            float v1959_data = ir7[2];
            ir7[2] = (v1959_data + (v1946_data * v1957_data));
            float v1962_data = s3[41];
            float v1964_data = ir7[3];
            ir7[3] = (v1964_data + (v1946_data * v1962_data));
            float v1967_data = s3[53];
            float v1969_data = ir7[4];
            ir7[4] = (v1969_data + (v1946_data * v1967_data));
            float v1972_data = s3[65];
            float v1974_data = ir7[5];
            ir7[5] = (v1974_data + (v1946_data * v1972_data));
            float v1977_data = s3[77];
            float v1979_data = ir7[6];
            ir7[6] = (v1979_data + (v1946_data * v1977_data));
            float v1982_data = s3[89];
            float v1984_data = ir7[7];
            ir7[7] = (v1984_data + (v1946_data * v1982_data));
          }
          if (v3_lead < 12) {
            float v1990_data = r6[6];
            float v1991_data = s3[6];
            float v1993_data = ir7[0];
            ir7[0] = (v1993_data + (v1990_data * v1991_data));
            float v1996_data = s3[18];
            float v1998_data = ir7[1];
            ir7[1] = (v1998_data + (v1990_data * v1996_data));
            float v2001_data = s3[30];
            float v2003_data = ir7[2];
            ir7[2] = (v2003_data + (v1990_data * v2001_data));
            float v2006_data = s3[42];
            float v2008_data = ir7[3];
            ir7[3] = (v2008_data + (v1990_data * v2006_data));
            float v2011_data = s3[54];
            float v2013_data = ir7[4];
            ir7[4] = (v2013_data + (v1990_data * v2011_data));
            float v2016_data = s3[66];
            float v2018_data = ir7[5];
            ir7[5] = (v2018_data + (v1990_data * v2016_data));
            float v2021_data = s3[78];
            float v2023_data = ir7[6];
            ir7[6] = (v2023_data + (v1990_data * v2021_data));
            float v2026_data = s3[90];
            float v2028_data = ir7[7];
            ir7[7] = (v2028_data + (v1990_data * v2026_data));
          }
          if (v3_lead < 12) {
            float v2034_data = r6[7];
            float v2035_data = s3[7];
            float v2037_data = ir7[0];
            ir7[0] = (v2037_data + (v2034_data * v2035_data));
            float v2040_data = s3[19];
            float v2042_data = ir7[1];
            ir7[1] = (v2042_data + (v2034_data * v2040_data));
            float v2045_data = s3[31];
            float v2047_data = ir7[2];
            ir7[2] = (v2047_data + (v2034_data * v2045_data));
            float v2050_data = s3[43];
            float v2052_data = ir7[3];
            ir7[3] = (v2052_data + (v2034_data * v2050_data));
            float v2055_data = s3[55];
            float v2057_data = ir7[4];
            ir7[4] = (v2057_data + (v2034_data * v2055_data));
            float v2060_data = s3[67];
            float v2062_data = ir7[5];
            ir7[5] = (v2062_data + (v2034_data * v2060_data));
            float v2065_data = s3[79];
            float v2067_data = ir7[6];
            ir7[6] = (v2067_data + (v2034_data * v2065_data));
            float v2070_data = s3[91];
            float v2072_data = ir7[7];
            ir7[7] = (v2072_data + (v2034_data * v2070_data));
          }
          if (v3_lead < 12) {
            float v2078_data = r6[8];
            float v2079_data = s3[8];
            float v2081_data = ir7[0];
            ir7[0] = (v2081_data + (v2078_data * v2079_data));
            float v2084_data = s3[20];
            float v2086_data = ir7[1];
            ir7[1] = (v2086_data + (v2078_data * v2084_data));
            float v2089_data = s3[32];
            float v2091_data = ir7[2];
            ir7[2] = (v2091_data + (v2078_data * v2089_data));
            float v2094_data = s3[44];
            float v2096_data = ir7[3];
            ir7[3] = (v2096_data + (v2078_data * v2094_data));
            float v2099_data = s3[56];
            float v2101_data = ir7[4];
            ir7[4] = (v2101_data + (v2078_data * v2099_data));
            float v2104_data = s3[68];
            float v2106_data = ir7[5];
            ir7[5] = (v2106_data + (v2078_data * v2104_data));
            float v2109_data = s3[80];
            float v2111_data = ir7[6];
            ir7[6] = (v2111_data + (v2078_data * v2109_data));
            float v2114_data = s3[92];
            float v2116_data = ir7[7];
            ir7[7] = (v2116_data + (v2078_data * v2114_data));
          }
          if (v3_lead < 12) {
            float v2122_data = r6[9];
            float v2123_data = s3[9];
            float v2125_data = ir7[0];
            ir7[0] = (v2125_data + (v2122_data * v2123_data));
            float v2128_data = s3[21];
            float v2130_data = ir7[1];
            ir7[1] = (v2130_data + (v2122_data * v2128_data));
            float v2133_data = s3[33];
            float v2135_data = ir7[2];
            ir7[2] = (v2135_data + (v2122_data * v2133_data));
            float v2138_data = s3[45];
            float v2140_data = ir7[3];
            ir7[3] = (v2140_data + (v2122_data * v2138_data));
            float v2143_data = s3[57];
            float v2145_data = ir7[4];
            ir7[4] = (v2145_data + (v2122_data * v2143_data));
            float v2148_data = s3[69];
            float v2150_data = ir7[5];
            ir7[5] = (v2150_data + (v2122_data * v2148_data));
            float v2153_data = s3[81];
            float v2155_data = ir7[6];
            ir7[6] = (v2155_data + (v2122_data * v2153_data));
            float v2158_data = s3[93];
            float v2160_data = ir7[7];
            ir7[7] = (v2160_data + (v2122_data * v2158_data));
          }
          if (v3_lead < 12) {
            float v2166_data = r6[10];
            float v2167_data = s3[10];
            float v2169_data = ir7[0];
            ir7[0] = (v2169_data + (v2166_data * v2167_data));
            float v2172_data = s3[22];
            float v2174_data = ir7[1];
            ir7[1] = (v2174_data + (v2166_data * v2172_data));
            float v2177_data = s3[34];
            float v2179_data = ir7[2];
            ir7[2] = (v2179_data + (v2166_data * v2177_data));
            float v2182_data = s3[46];
            float v2184_data = ir7[3];
            ir7[3] = (v2184_data + (v2166_data * v2182_data));
            float v2187_data = s3[58];
            float v2189_data = ir7[4];
            ir7[4] = (v2189_data + (v2166_data * v2187_data));
            float v2192_data = s3[70];
            float v2194_data = ir7[5];
            ir7[5] = (v2194_data + (v2166_data * v2192_data));
            float v2197_data = s3[82];
            float v2199_data = ir7[6];
            ir7[6] = (v2199_data + (v2166_data * v2197_data));
            float v2202_data = s3[94];
            float v2204_data = ir7[7];
            ir7[7] = (v2204_data + (v2166_data * v2202_data));
          }
          if (v3_lead < 12) {
            float v2210_data = r6[11];
            float v2211_data = s3[11];
            float v2213_data = ir7[0];
            ir7[0] = (v2213_data + (v2210_data * v2211_data));
            float v2216_data = s3[23];
            float v2218_data = ir7[1];
            ir7[1] = (v2218_data + (v2210_data * v2216_data));
            float v2221_data = s3[35];
            float v2223_data = ir7[2];
            ir7[2] = (v2223_data + (v2210_data * v2221_data));
            float v2226_data = s3[47];
            float v2228_data = ir7[3];
            ir7[3] = (v2228_data + (v2210_data * v2226_data));
            float v2231_data = s3[59];
            float v2233_data = ir7[4];
            ir7[4] = (v2233_data + (v2210_data * v2231_data));
            float v2236_data = s3[71];
            float v2238_data = ir7[5];
            ir7[5] = (v2238_data + (v2210_data * v2236_data));
            float v2241_data = s3[83];
            float v2243_data = ir7[6];
            ir7[6] = (v2243_data + (v2210_data * v2241_data));
            float v2246_data = s3[95];
            float v2248_data = ir7[7];
            ir7[7] = (v2248_data + (v2210_data * v2246_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2254_n1 = 0; v2254_n1 < 8; ++v2254_n1) {
              int32_t v2255_a = 0 + v2254_n1;
              float v2257_data = ir7[v2254_n1];
              int32_t v2258_a = 0 + v2254_n1;
              float v2260_data = r5[v2254_n1];
              int32_t v2262_a = 0 + v2254_n1;
              r7[v2254_n1] = (v2260_data + v2257_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2268_i1 = 0; v2268_i1 < 8; ++v2268_i1) {
              int32_t v2269_a = 0 + v2268_i1;
              float v2271_data = r7[v2268_i1];
              int32_t v2278_a = v3_lead + (v2268_i1 * 12);
              glb_m0[v2278_a] = v2271_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

