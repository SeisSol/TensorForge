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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              float v26_data = __ldcg(&glb_m1[(v16_lead + (v18_i1 * 12))]);
              r0[v18_i1] = v26_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              float v43_data = __ldcg(&glb_m3[(v16_lead + (v35_i1 * 12))]);
              r2[v35_i1] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v16_lead < 12) {
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
            float v67_data = s0[37];
            float v69_data = ir1[3];
            ir1[3] = (v69_data + (v51_data * v67_data));
            float v72_data = s0[49];
            float v74_data = ir1[4];
            ir1[4] = (v74_data + (v51_data * v72_data));
            float v77_data = s0[61];
            float v79_data = ir1[5];
            ir1[5] = (v79_data + (v51_data * v77_data));
            float v82_data = s0[74];
            float v84_data = ir1[6];
            ir1[6] = (v84_data + (v51_data * v82_data));
            float v87_data = s0[86];
            float v89_data = ir1[7];
            ir1[7] = (v89_data + (v51_data * v87_data));
          }
          if (v16_lead < 12) {
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
            float v111_data = s0[36];
            float v113_data = ir1[3];
            ir1[3] = (v113_data + (v95_data * v111_data));
            float v116_data = s0[48];
            float v118_data = ir1[4];
            ir1[4] = (v118_data + (v95_data * v116_data));
            float v121_data = s0[60];
            float v123_data = ir1[5];
            ir1[5] = (v123_data + (v95_data * v121_data));
            float v126_data = s0[75];
            float v128_data = ir1[6];
            ir1[6] = (v128_data + (v95_data * v126_data));
            float v131_data = s0[87];
            float v133_data = ir1[7];
            ir1[7] = (v133_data + (v95_data * v131_data));
          }
          if (v16_lead < 12) {
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
            float v155_data = s0[39];
            float v157_data = ir1[3];
            ir1[3] = (v157_data + (v139_data * v155_data));
            float v160_data = s0[51];
            float v162_data = ir1[4];
            ir1[4] = (v162_data + (v139_data * v160_data));
            float v165_data = s0[63];
            float v167_data = ir1[5];
            ir1[5] = (v167_data + (v139_data * v165_data));
            float v170_data = s0[72];
            float v172_data = ir1[6];
            ir1[6] = (v172_data + (v139_data * v170_data));
            float v175_data = s0[84];
            float v177_data = ir1[7];
            ir1[7] = (v177_data + (v139_data * v175_data));
          }
          if (v16_lead < 12) {
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
            float v199_data = s0[38];
            float v201_data = ir1[3];
            ir1[3] = (v201_data + (v183_data * v199_data));
            float v204_data = s0[50];
            float v206_data = ir1[4];
            ir1[4] = (v206_data + (v183_data * v204_data));
            float v209_data = s0[62];
            float v211_data = ir1[5];
            ir1[5] = (v211_data + (v183_data * v209_data));
            float v214_data = s0[73];
            float v216_data = ir1[6];
            ir1[6] = (v216_data + (v183_data * v214_data));
            float v219_data = s0[85];
            float v221_data = ir1[7];
            ir1[7] = (v221_data + (v183_data * v219_data));
          }
          if (v16_lead < 12) {
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
            float v243_data = s0[41];
            float v245_data = ir1[3];
            ir1[3] = (v245_data + (v227_data * v243_data));
            float v248_data = s0[53];
            float v250_data = ir1[4];
            ir1[4] = (v250_data + (v227_data * v248_data));
            float v253_data = s0[66];
            float v255_data = ir1[5];
            ir1[5] = (v255_data + (v227_data * v253_data));
            float v258_data = s0[78];
            float v260_data = ir1[6];
            ir1[6] = (v260_data + (v227_data * v258_data));
            float v263_data = s0[90];
            float v265_data = ir1[7];
            ir1[7] = (v265_data + (v227_data * v263_data));
          }
          if (v16_lead < 12) {
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
            float v287_data = s0[40];
            float v289_data = ir1[3];
            ir1[3] = (v289_data + (v271_data * v287_data));
            float v292_data = s0[52];
            float v294_data = ir1[4];
            ir1[4] = (v294_data + (v271_data * v292_data));
            float v297_data = s0[67];
            float v299_data = ir1[5];
            ir1[5] = (v299_data + (v271_data * v297_data));
            float v302_data = s0[79];
            float v304_data = ir1[6];
            ir1[6] = (v304_data + (v271_data * v302_data));
            float v307_data = s0[91];
            float v309_data = ir1[7];
            ir1[7] = (v309_data + (v271_data * v307_data));
          }
          if (v16_lead < 12) {
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
            float v331_data = s0[43];
            float v333_data = ir1[3];
            ir1[3] = (v333_data + (v315_data * v331_data));
            float v336_data = s0[55];
            float v338_data = ir1[4];
            ir1[4] = (v338_data + (v315_data * v336_data));
            float v341_data = s0[64];
            float v343_data = ir1[5];
            ir1[5] = (v343_data + (v315_data * v341_data));
            float v346_data = s0[76];
            float v348_data = ir1[6];
            ir1[6] = (v348_data + (v315_data * v346_data));
            float v351_data = s0[88];
            float v353_data = ir1[7];
            ir1[7] = (v353_data + (v315_data * v351_data));
          }
          if (v16_lead < 12) {
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
            float v375_data = s0[42];
            float v377_data = ir1[3];
            ir1[3] = (v377_data + (v359_data * v375_data));
            float v380_data = s0[54];
            float v382_data = ir1[4];
            ir1[4] = (v382_data + (v359_data * v380_data));
            float v385_data = s0[65];
            float v387_data = ir1[5];
            ir1[5] = (v387_data + (v359_data * v385_data));
            float v390_data = s0[77];
            float v392_data = ir1[6];
            ir1[6] = (v392_data + (v359_data * v390_data));
            float v395_data = s0[89];
            float v397_data = ir1[7];
            ir1[7] = (v397_data + (v359_data * v395_data));
          }
          if (v16_lead < 12) {
            float v403_data = r0[8];
            float v404_data = s0[8];
            float v406_data = ir1[0];
            ir1[0] = (v406_data + (v403_data * v404_data));
            float v409_data = s0[20];
            float v411_data = ir1[1];
            ir1[1] = (v411_data + (v403_data * v409_data));
            float v414_data = s0[33];
            float v416_data = ir1[2];
            ir1[2] = (v416_data + (v403_data * v414_data));
            float v419_data = s0[45];
            float v421_data = ir1[3];
            ir1[3] = (v421_data + (v403_data * v419_data));
            float v424_data = s0[57];
            float v426_data = ir1[4];
            ir1[4] = (v426_data + (v403_data * v424_data));
            float v429_data = s0[70];
            float v431_data = ir1[5];
            ir1[5] = (v431_data + (v403_data * v429_data));
            float v434_data = s0[82];
            float v436_data = ir1[6];
            ir1[6] = (v436_data + (v403_data * v434_data));
            float v439_data = s0[94];
            float v441_data = ir1[7];
            ir1[7] = (v441_data + (v403_data * v439_data));
          }
          if (v16_lead < 12) {
            float v447_data = r0[9];
            float v448_data = s0[9];
            float v450_data = ir1[0];
            ir1[0] = (v450_data + (v447_data * v448_data));
            float v453_data = s0[21];
            float v455_data = ir1[1];
            ir1[1] = (v455_data + (v447_data * v453_data));
            float v458_data = s0[32];
            float v460_data = ir1[2];
            ir1[2] = (v460_data + (v447_data * v458_data));
            float v463_data = s0[44];
            float v465_data = ir1[3];
            ir1[3] = (v465_data + (v447_data * v463_data));
            float v468_data = s0[56];
            float v470_data = ir1[4];
            ir1[4] = (v470_data + (v447_data * v468_data));
            float v473_data = s0[71];
            float v475_data = ir1[5];
            ir1[5] = (v475_data + (v447_data * v473_data));
            float v478_data = s0[83];
            float v480_data = ir1[6];
            ir1[6] = (v480_data + (v447_data * v478_data));
            float v483_data = s0[95];
            float v485_data = ir1[7];
            ir1[7] = (v485_data + (v447_data * v483_data));
          }
          if (v16_lead < 12) {
            float v491_data = r0[10];
            float v492_data = s0[10];
            float v494_data = ir1[0];
            ir1[0] = (v494_data + (v491_data * v492_data));
            float v497_data = s0[22];
            float v499_data = ir1[1];
            ir1[1] = (v499_data + (v491_data * v497_data));
            float v502_data = s0[35];
            float v504_data = ir1[2];
            ir1[2] = (v504_data + (v491_data * v502_data));
            float v507_data = s0[47];
            float v509_data = ir1[3];
            ir1[3] = (v509_data + (v491_data * v507_data));
            float v512_data = s0[59];
            float v514_data = ir1[4];
            ir1[4] = (v514_data + (v491_data * v512_data));
            float v517_data = s0[68];
            float v519_data = ir1[5];
            ir1[5] = (v519_data + (v491_data * v517_data));
            float v522_data = s0[80];
            float v524_data = ir1[6];
            ir1[6] = (v524_data + (v491_data * v522_data));
            float v527_data = s0[92];
            float v529_data = ir1[7];
            ir1[7] = (v529_data + (v491_data * v527_data));
          }
          if (v16_lead < 12) {
            float v535_data = r0[11];
            float v536_data = s0[11];
            float v538_data = ir1[0];
            ir1[0] = (v538_data + (v535_data * v536_data));
            float v541_data = s0[23];
            float v543_data = ir1[1];
            ir1[1] = (v543_data + (v535_data * v541_data));
            float v546_data = s0[34];
            float v548_data = ir1[2];
            ir1[2] = (v548_data + (v535_data * v546_data));
            float v551_data = s0[46];
            float v553_data = ir1[3];
            ir1[3] = (v553_data + (v535_data * v551_data));
            float v556_data = s0[58];
            float v558_data = ir1[4];
            ir1[4] = (v558_data + (v535_data * v556_data));
            float v561_data = s0[69];
            float v563_data = ir1[5];
            ir1[5] = (v563_data + (v535_data * v561_data));
            float v566_data = s0[81];
            float v568_data = ir1[6];
            ir1[6] = (v568_data + (v535_data * v566_data));
            float v571_data = s0[93];
            float v573_data = ir1[7];
            ir1[7] = (v573_data + (v535_data * v571_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v579_n1 = 0; v579_n1 < 8; ++v579_n1) {
              float v581_data = ir1[v579_n1];
              r1[v579_n1] = v581_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v590_i1 = 0; v590_i1 < 12; ++v590_i1) {
              float v598_data = __ldcg(&glb_m5[(v16_lead + (v590_i1 * 12))]);
              r4[v590_i1] = v598_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v16_lead < 12) {
            float v606_data = r2[0];
            float v607_data = s1[0];
            float v609_data = ir3[0];
            ir3[0] = (v609_data + (v606_data * v607_data));
            float v612_data = s1[12];
            float v614_data = ir3[1];
            ir3[1] = (v614_data + (v606_data * v612_data));
            float v617_data = s1[24];
            float v619_data = ir3[2];
            ir3[2] = (v619_data + (v606_data * v617_data));
            float v622_data = s1[37];
            float v624_data = ir3[3];
            ir3[3] = (v624_data + (v606_data * v622_data));
            float v627_data = s1[49];
            float v629_data = ir3[4];
            ir3[4] = (v629_data + (v606_data * v627_data));
            float v632_data = s1[61];
            float v634_data = ir3[5];
            ir3[5] = (v634_data + (v606_data * v632_data));
            float v637_data = s1[74];
            float v639_data = ir3[6];
            ir3[6] = (v639_data + (v606_data * v637_data));
            float v642_data = s1[86];
            float v644_data = ir3[7];
            ir3[7] = (v644_data + (v606_data * v642_data));
          }
          if (v16_lead < 12) {
            float v650_data = r2[1];
            float v651_data = s1[1];
            float v653_data = ir3[0];
            ir3[0] = (v653_data + (v650_data * v651_data));
            float v656_data = s1[13];
            float v658_data = ir3[1];
            ir3[1] = (v658_data + (v650_data * v656_data));
            float v661_data = s1[25];
            float v663_data = ir3[2];
            ir3[2] = (v663_data + (v650_data * v661_data));
            float v666_data = s1[36];
            float v668_data = ir3[3];
            ir3[3] = (v668_data + (v650_data * v666_data));
            float v671_data = s1[48];
            float v673_data = ir3[4];
            ir3[4] = (v673_data + (v650_data * v671_data));
            float v676_data = s1[60];
            float v678_data = ir3[5];
            ir3[5] = (v678_data + (v650_data * v676_data));
            float v681_data = s1[75];
            float v683_data = ir3[6];
            ir3[6] = (v683_data + (v650_data * v681_data));
            float v686_data = s1[87];
            float v688_data = ir3[7];
            ir3[7] = (v688_data + (v650_data * v686_data));
          }
          if (v16_lead < 12) {
            float v694_data = r2[2];
            float v695_data = s1[2];
            float v697_data = ir3[0];
            ir3[0] = (v697_data + (v694_data * v695_data));
            float v700_data = s1[14];
            float v702_data = ir3[1];
            ir3[1] = (v702_data + (v694_data * v700_data));
            float v705_data = s1[26];
            float v707_data = ir3[2];
            ir3[2] = (v707_data + (v694_data * v705_data));
            float v710_data = s1[39];
            float v712_data = ir3[3];
            ir3[3] = (v712_data + (v694_data * v710_data));
            float v715_data = s1[51];
            float v717_data = ir3[4];
            ir3[4] = (v717_data + (v694_data * v715_data));
            float v720_data = s1[63];
            float v722_data = ir3[5];
            ir3[5] = (v722_data + (v694_data * v720_data));
            float v725_data = s1[72];
            float v727_data = ir3[6];
            ir3[6] = (v727_data + (v694_data * v725_data));
            float v730_data = s1[84];
            float v732_data = ir3[7];
            ir3[7] = (v732_data + (v694_data * v730_data));
          }
          if (v16_lead < 12) {
            float v738_data = r2[3];
            float v739_data = s1[3];
            float v741_data = ir3[0];
            ir3[0] = (v741_data + (v738_data * v739_data));
            float v744_data = s1[15];
            float v746_data = ir3[1];
            ir3[1] = (v746_data + (v738_data * v744_data));
            float v749_data = s1[27];
            float v751_data = ir3[2];
            ir3[2] = (v751_data + (v738_data * v749_data));
            float v754_data = s1[38];
            float v756_data = ir3[3];
            ir3[3] = (v756_data + (v738_data * v754_data));
            float v759_data = s1[50];
            float v761_data = ir3[4];
            ir3[4] = (v761_data + (v738_data * v759_data));
            float v764_data = s1[62];
            float v766_data = ir3[5];
            ir3[5] = (v766_data + (v738_data * v764_data));
            float v769_data = s1[73];
            float v771_data = ir3[6];
            ir3[6] = (v771_data + (v738_data * v769_data));
            float v774_data = s1[85];
            float v776_data = ir3[7];
            ir3[7] = (v776_data + (v738_data * v774_data));
          }
          if (v16_lead < 12) {
            float v782_data = r2[4];
            float v783_data = s1[4];
            float v785_data = ir3[0];
            ir3[0] = (v785_data + (v782_data * v783_data));
            float v788_data = s1[16];
            float v790_data = ir3[1];
            ir3[1] = (v790_data + (v782_data * v788_data));
            float v793_data = s1[28];
            float v795_data = ir3[2];
            ir3[2] = (v795_data + (v782_data * v793_data));
            float v798_data = s1[41];
            float v800_data = ir3[3];
            ir3[3] = (v800_data + (v782_data * v798_data));
            float v803_data = s1[53];
            float v805_data = ir3[4];
            ir3[4] = (v805_data + (v782_data * v803_data));
            float v808_data = s1[66];
            float v810_data = ir3[5];
            ir3[5] = (v810_data + (v782_data * v808_data));
            float v813_data = s1[78];
            float v815_data = ir3[6];
            ir3[6] = (v815_data + (v782_data * v813_data));
            float v818_data = s1[90];
            float v820_data = ir3[7];
            ir3[7] = (v820_data + (v782_data * v818_data));
          }
          if (v16_lead < 12) {
            float v826_data = r2[5];
            float v827_data = s1[5];
            float v829_data = ir3[0];
            ir3[0] = (v829_data + (v826_data * v827_data));
            float v832_data = s1[17];
            float v834_data = ir3[1];
            ir3[1] = (v834_data + (v826_data * v832_data));
            float v837_data = s1[29];
            float v839_data = ir3[2];
            ir3[2] = (v839_data + (v826_data * v837_data));
            float v842_data = s1[40];
            float v844_data = ir3[3];
            ir3[3] = (v844_data + (v826_data * v842_data));
            float v847_data = s1[52];
            float v849_data = ir3[4];
            ir3[4] = (v849_data + (v826_data * v847_data));
            float v852_data = s1[67];
            float v854_data = ir3[5];
            ir3[5] = (v854_data + (v826_data * v852_data));
            float v857_data = s1[79];
            float v859_data = ir3[6];
            ir3[6] = (v859_data + (v826_data * v857_data));
            float v862_data = s1[91];
            float v864_data = ir3[7];
            ir3[7] = (v864_data + (v826_data * v862_data));
          }
          if (v16_lead < 12) {
            float v870_data = r2[6];
            float v871_data = s1[6];
            float v873_data = ir3[0];
            ir3[0] = (v873_data + (v870_data * v871_data));
            float v876_data = s1[18];
            float v878_data = ir3[1];
            ir3[1] = (v878_data + (v870_data * v876_data));
            float v881_data = s1[30];
            float v883_data = ir3[2];
            ir3[2] = (v883_data + (v870_data * v881_data));
            float v886_data = s1[43];
            float v888_data = ir3[3];
            ir3[3] = (v888_data + (v870_data * v886_data));
            float v891_data = s1[55];
            float v893_data = ir3[4];
            ir3[4] = (v893_data + (v870_data * v891_data));
            float v896_data = s1[64];
            float v898_data = ir3[5];
            ir3[5] = (v898_data + (v870_data * v896_data));
            float v901_data = s1[76];
            float v903_data = ir3[6];
            ir3[6] = (v903_data + (v870_data * v901_data));
            float v906_data = s1[88];
            float v908_data = ir3[7];
            ir3[7] = (v908_data + (v870_data * v906_data));
          }
          if (v16_lead < 12) {
            float v914_data = r2[7];
            float v915_data = s1[7];
            float v917_data = ir3[0];
            ir3[0] = (v917_data + (v914_data * v915_data));
            float v920_data = s1[19];
            float v922_data = ir3[1];
            ir3[1] = (v922_data + (v914_data * v920_data));
            float v925_data = s1[31];
            float v927_data = ir3[2];
            ir3[2] = (v927_data + (v914_data * v925_data));
            float v930_data = s1[42];
            float v932_data = ir3[3];
            ir3[3] = (v932_data + (v914_data * v930_data));
            float v935_data = s1[54];
            float v937_data = ir3[4];
            ir3[4] = (v937_data + (v914_data * v935_data));
            float v940_data = s1[65];
            float v942_data = ir3[5];
            ir3[5] = (v942_data + (v914_data * v940_data));
            float v945_data = s1[77];
            float v947_data = ir3[6];
            ir3[6] = (v947_data + (v914_data * v945_data));
            float v950_data = s1[89];
            float v952_data = ir3[7];
            ir3[7] = (v952_data + (v914_data * v950_data));
          }
          if (v16_lead < 12) {
            float v958_data = r2[8];
            float v959_data = s1[8];
            float v961_data = ir3[0];
            ir3[0] = (v961_data + (v958_data * v959_data));
            float v964_data = s1[20];
            float v966_data = ir3[1];
            ir3[1] = (v966_data + (v958_data * v964_data));
            float v969_data = s1[33];
            float v971_data = ir3[2];
            ir3[2] = (v971_data + (v958_data * v969_data));
            float v974_data = s1[45];
            float v976_data = ir3[3];
            ir3[3] = (v976_data + (v958_data * v974_data));
            float v979_data = s1[57];
            float v981_data = ir3[4];
            ir3[4] = (v981_data + (v958_data * v979_data));
            float v984_data = s1[70];
            float v986_data = ir3[5];
            ir3[5] = (v986_data + (v958_data * v984_data));
            float v989_data = s1[82];
            float v991_data = ir3[6];
            ir3[6] = (v991_data + (v958_data * v989_data));
            float v994_data = s1[94];
            float v996_data = ir3[7];
            ir3[7] = (v996_data + (v958_data * v994_data));
          }
          if (v16_lead < 12) {
            float v1002_data = r2[9];
            float v1003_data = s1[9];
            float v1005_data = ir3[0];
            ir3[0] = (v1005_data + (v1002_data * v1003_data));
            float v1008_data = s1[21];
            float v1010_data = ir3[1];
            ir3[1] = (v1010_data + (v1002_data * v1008_data));
            float v1013_data = s1[32];
            float v1015_data = ir3[2];
            ir3[2] = (v1015_data + (v1002_data * v1013_data));
            float v1018_data = s1[44];
            float v1020_data = ir3[3];
            ir3[3] = (v1020_data + (v1002_data * v1018_data));
            float v1023_data = s1[56];
            float v1025_data = ir3[4];
            ir3[4] = (v1025_data + (v1002_data * v1023_data));
            float v1028_data = s1[71];
            float v1030_data = ir3[5];
            ir3[5] = (v1030_data + (v1002_data * v1028_data));
            float v1033_data = s1[83];
            float v1035_data = ir3[6];
            ir3[6] = (v1035_data + (v1002_data * v1033_data));
            float v1038_data = s1[95];
            float v1040_data = ir3[7];
            ir3[7] = (v1040_data + (v1002_data * v1038_data));
          }
          if (v16_lead < 12) {
            float v1046_data = r2[10];
            float v1047_data = s1[10];
            float v1049_data = ir3[0];
            ir3[0] = (v1049_data + (v1046_data * v1047_data));
            float v1052_data = s1[22];
            float v1054_data = ir3[1];
            ir3[1] = (v1054_data + (v1046_data * v1052_data));
            float v1057_data = s1[35];
            float v1059_data = ir3[2];
            ir3[2] = (v1059_data + (v1046_data * v1057_data));
            float v1062_data = s1[47];
            float v1064_data = ir3[3];
            ir3[3] = (v1064_data + (v1046_data * v1062_data));
            float v1067_data = s1[59];
            float v1069_data = ir3[4];
            ir3[4] = (v1069_data + (v1046_data * v1067_data));
            float v1072_data = s1[68];
            float v1074_data = ir3[5];
            ir3[5] = (v1074_data + (v1046_data * v1072_data));
            float v1077_data = s1[80];
            float v1079_data = ir3[6];
            ir3[6] = (v1079_data + (v1046_data * v1077_data));
            float v1082_data = s1[92];
            float v1084_data = ir3[7];
            ir3[7] = (v1084_data + (v1046_data * v1082_data));
          }
          if (v16_lead < 12) {
            float v1090_data = r2[11];
            float v1091_data = s1[11];
            float v1093_data = ir3[0];
            ir3[0] = (v1093_data + (v1090_data * v1091_data));
            float v1096_data = s1[23];
            float v1098_data = ir3[1];
            ir3[1] = (v1098_data + (v1090_data * v1096_data));
            float v1101_data = s1[34];
            float v1103_data = ir3[2];
            ir3[2] = (v1103_data + (v1090_data * v1101_data));
            float v1106_data = s1[46];
            float v1108_data = ir3[3];
            ir3[3] = (v1108_data + (v1090_data * v1106_data));
            float v1111_data = s1[58];
            float v1113_data = ir3[4];
            ir3[4] = (v1113_data + (v1090_data * v1111_data));
            float v1116_data = s1[69];
            float v1118_data = ir3[5];
            ir3[5] = (v1118_data + (v1090_data * v1116_data));
            float v1121_data = s1[81];
            float v1123_data = ir3[6];
            ir3[6] = (v1123_data + (v1090_data * v1121_data));
            float v1126_data = s1[93];
            float v1128_data = ir3[7];
            ir3[7] = (v1128_data + (v1090_data * v1126_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v1134_n1 = 0; v1134_n1 < 8; ++v1134_n1) {
              float v1136_data = ir3[v1134_n1];
              float v1138_data = r1[v1134_n1];
              r3[v1134_n1] = (v1138_data + v1136_data);
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          // s2 = load{g>s}(glb_m6[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v1148_i1 = 0; v1148_i1 < 12; ++v1148_i1) {
              float v1156_data = __ldcg(&glb_m7[(v16_lead + (v1148_i1 * 12))]);
              r6[v1148_i1] = v1156_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v16_lead < 12) {
            float v1164_data = r4[0];
            float v1165_data = s2[0];
            float v1167_data = ir5[0];
            ir5[0] = (v1167_data + (v1164_data * v1165_data));
            float v1170_data = s2[12];
            float v1172_data = ir5[1];
            ir5[1] = (v1172_data + (v1164_data * v1170_data));
            float v1175_data = s2[24];
            float v1177_data = ir5[2];
            ir5[2] = (v1177_data + (v1164_data * v1175_data));
            float v1180_data = s2[37];
            float v1182_data = ir5[3];
            ir5[3] = (v1182_data + (v1164_data * v1180_data));
            float v1185_data = s2[49];
            float v1187_data = ir5[4];
            ir5[4] = (v1187_data + (v1164_data * v1185_data));
            float v1190_data = s2[61];
            float v1192_data = ir5[5];
            ir5[5] = (v1192_data + (v1164_data * v1190_data));
            float v1195_data = s2[74];
            float v1197_data = ir5[6];
            ir5[6] = (v1197_data + (v1164_data * v1195_data));
            float v1200_data = s2[86];
            float v1202_data = ir5[7];
            ir5[7] = (v1202_data + (v1164_data * v1200_data));
          }
          if (v16_lead < 12) {
            float v1208_data = r4[1];
            float v1209_data = s2[1];
            float v1211_data = ir5[0];
            ir5[0] = (v1211_data + (v1208_data * v1209_data));
            float v1214_data = s2[13];
            float v1216_data = ir5[1];
            ir5[1] = (v1216_data + (v1208_data * v1214_data));
            float v1219_data = s2[25];
            float v1221_data = ir5[2];
            ir5[2] = (v1221_data + (v1208_data * v1219_data));
            float v1224_data = s2[36];
            float v1226_data = ir5[3];
            ir5[3] = (v1226_data + (v1208_data * v1224_data));
            float v1229_data = s2[48];
            float v1231_data = ir5[4];
            ir5[4] = (v1231_data + (v1208_data * v1229_data));
            float v1234_data = s2[60];
            float v1236_data = ir5[5];
            ir5[5] = (v1236_data + (v1208_data * v1234_data));
            float v1239_data = s2[75];
            float v1241_data = ir5[6];
            ir5[6] = (v1241_data + (v1208_data * v1239_data));
            float v1244_data = s2[87];
            float v1246_data = ir5[7];
            ir5[7] = (v1246_data + (v1208_data * v1244_data));
          }
          if (v16_lead < 12) {
            float v1252_data = r4[2];
            float v1253_data = s2[2];
            float v1255_data = ir5[0];
            ir5[0] = (v1255_data + (v1252_data * v1253_data));
            float v1258_data = s2[14];
            float v1260_data = ir5[1];
            ir5[1] = (v1260_data + (v1252_data * v1258_data));
            float v1263_data = s2[26];
            float v1265_data = ir5[2];
            ir5[2] = (v1265_data + (v1252_data * v1263_data));
            float v1268_data = s2[39];
            float v1270_data = ir5[3];
            ir5[3] = (v1270_data + (v1252_data * v1268_data));
            float v1273_data = s2[51];
            float v1275_data = ir5[4];
            ir5[4] = (v1275_data + (v1252_data * v1273_data));
            float v1278_data = s2[63];
            float v1280_data = ir5[5];
            ir5[5] = (v1280_data + (v1252_data * v1278_data));
            float v1283_data = s2[72];
            float v1285_data = ir5[6];
            ir5[6] = (v1285_data + (v1252_data * v1283_data));
            float v1288_data = s2[84];
            float v1290_data = ir5[7];
            ir5[7] = (v1290_data + (v1252_data * v1288_data));
          }
          if (v16_lead < 12) {
            float v1296_data = r4[3];
            float v1297_data = s2[3];
            float v1299_data = ir5[0];
            ir5[0] = (v1299_data + (v1296_data * v1297_data));
            float v1302_data = s2[15];
            float v1304_data = ir5[1];
            ir5[1] = (v1304_data + (v1296_data * v1302_data));
            float v1307_data = s2[27];
            float v1309_data = ir5[2];
            ir5[2] = (v1309_data + (v1296_data * v1307_data));
            float v1312_data = s2[38];
            float v1314_data = ir5[3];
            ir5[3] = (v1314_data + (v1296_data * v1312_data));
            float v1317_data = s2[50];
            float v1319_data = ir5[4];
            ir5[4] = (v1319_data + (v1296_data * v1317_data));
            float v1322_data = s2[62];
            float v1324_data = ir5[5];
            ir5[5] = (v1324_data + (v1296_data * v1322_data));
            float v1327_data = s2[73];
            float v1329_data = ir5[6];
            ir5[6] = (v1329_data + (v1296_data * v1327_data));
            float v1332_data = s2[85];
            float v1334_data = ir5[7];
            ir5[7] = (v1334_data + (v1296_data * v1332_data));
          }
          if (v16_lead < 12) {
            float v1340_data = r4[4];
            float v1341_data = s2[4];
            float v1343_data = ir5[0];
            ir5[0] = (v1343_data + (v1340_data * v1341_data));
            float v1346_data = s2[16];
            float v1348_data = ir5[1];
            ir5[1] = (v1348_data + (v1340_data * v1346_data));
            float v1351_data = s2[28];
            float v1353_data = ir5[2];
            ir5[2] = (v1353_data + (v1340_data * v1351_data));
            float v1356_data = s2[41];
            float v1358_data = ir5[3];
            ir5[3] = (v1358_data + (v1340_data * v1356_data));
            float v1361_data = s2[53];
            float v1363_data = ir5[4];
            ir5[4] = (v1363_data + (v1340_data * v1361_data));
            float v1366_data = s2[66];
            float v1368_data = ir5[5];
            ir5[5] = (v1368_data + (v1340_data * v1366_data));
            float v1371_data = s2[78];
            float v1373_data = ir5[6];
            ir5[6] = (v1373_data + (v1340_data * v1371_data));
            float v1376_data = s2[90];
            float v1378_data = ir5[7];
            ir5[7] = (v1378_data + (v1340_data * v1376_data));
          }
          if (v16_lead < 12) {
            float v1384_data = r4[5];
            float v1385_data = s2[5];
            float v1387_data = ir5[0];
            ir5[0] = (v1387_data + (v1384_data * v1385_data));
            float v1390_data = s2[17];
            float v1392_data = ir5[1];
            ir5[1] = (v1392_data + (v1384_data * v1390_data));
            float v1395_data = s2[29];
            float v1397_data = ir5[2];
            ir5[2] = (v1397_data + (v1384_data * v1395_data));
            float v1400_data = s2[40];
            float v1402_data = ir5[3];
            ir5[3] = (v1402_data + (v1384_data * v1400_data));
            float v1405_data = s2[52];
            float v1407_data = ir5[4];
            ir5[4] = (v1407_data + (v1384_data * v1405_data));
            float v1410_data = s2[67];
            float v1412_data = ir5[5];
            ir5[5] = (v1412_data + (v1384_data * v1410_data));
            float v1415_data = s2[79];
            float v1417_data = ir5[6];
            ir5[6] = (v1417_data + (v1384_data * v1415_data));
            float v1420_data = s2[91];
            float v1422_data = ir5[7];
            ir5[7] = (v1422_data + (v1384_data * v1420_data));
          }
          if (v16_lead < 12) {
            float v1428_data = r4[6];
            float v1429_data = s2[6];
            float v1431_data = ir5[0];
            ir5[0] = (v1431_data + (v1428_data * v1429_data));
            float v1434_data = s2[18];
            float v1436_data = ir5[1];
            ir5[1] = (v1436_data + (v1428_data * v1434_data));
            float v1439_data = s2[30];
            float v1441_data = ir5[2];
            ir5[2] = (v1441_data + (v1428_data * v1439_data));
            float v1444_data = s2[43];
            float v1446_data = ir5[3];
            ir5[3] = (v1446_data + (v1428_data * v1444_data));
            float v1449_data = s2[55];
            float v1451_data = ir5[4];
            ir5[4] = (v1451_data + (v1428_data * v1449_data));
            float v1454_data = s2[64];
            float v1456_data = ir5[5];
            ir5[5] = (v1456_data + (v1428_data * v1454_data));
            float v1459_data = s2[76];
            float v1461_data = ir5[6];
            ir5[6] = (v1461_data + (v1428_data * v1459_data));
            float v1464_data = s2[88];
            float v1466_data = ir5[7];
            ir5[7] = (v1466_data + (v1428_data * v1464_data));
          }
          if (v16_lead < 12) {
            float v1472_data = r4[7];
            float v1473_data = s2[7];
            float v1475_data = ir5[0];
            ir5[0] = (v1475_data + (v1472_data * v1473_data));
            float v1478_data = s2[19];
            float v1480_data = ir5[1];
            ir5[1] = (v1480_data + (v1472_data * v1478_data));
            float v1483_data = s2[31];
            float v1485_data = ir5[2];
            ir5[2] = (v1485_data + (v1472_data * v1483_data));
            float v1488_data = s2[42];
            float v1490_data = ir5[3];
            ir5[3] = (v1490_data + (v1472_data * v1488_data));
            float v1493_data = s2[54];
            float v1495_data = ir5[4];
            ir5[4] = (v1495_data + (v1472_data * v1493_data));
            float v1498_data = s2[65];
            float v1500_data = ir5[5];
            ir5[5] = (v1500_data + (v1472_data * v1498_data));
            float v1503_data = s2[77];
            float v1505_data = ir5[6];
            ir5[6] = (v1505_data + (v1472_data * v1503_data));
            float v1508_data = s2[89];
            float v1510_data = ir5[7];
            ir5[7] = (v1510_data + (v1472_data * v1508_data));
          }
          if (v16_lead < 12) {
            float v1516_data = r4[8];
            float v1517_data = s2[8];
            float v1519_data = ir5[0];
            ir5[0] = (v1519_data + (v1516_data * v1517_data));
            float v1522_data = s2[20];
            float v1524_data = ir5[1];
            ir5[1] = (v1524_data + (v1516_data * v1522_data));
            float v1527_data = s2[33];
            float v1529_data = ir5[2];
            ir5[2] = (v1529_data + (v1516_data * v1527_data));
            float v1532_data = s2[45];
            float v1534_data = ir5[3];
            ir5[3] = (v1534_data + (v1516_data * v1532_data));
            float v1537_data = s2[57];
            float v1539_data = ir5[4];
            ir5[4] = (v1539_data + (v1516_data * v1537_data));
            float v1542_data = s2[70];
            float v1544_data = ir5[5];
            ir5[5] = (v1544_data + (v1516_data * v1542_data));
            float v1547_data = s2[82];
            float v1549_data = ir5[6];
            ir5[6] = (v1549_data + (v1516_data * v1547_data));
            float v1552_data = s2[94];
            float v1554_data = ir5[7];
            ir5[7] = (v1554_data + (v1516_data * v1552_data));
          }
          if (v16_lead < 12) {
            float v1560_data = r4[9];
            float v1561_data = s2[9];
            float v1563_data = ir5[0];
            ir5[0] = (v1563_data + (v1560_data * v1561_data));
            float v1566_data = s2[21];
            float v1568_data = ir5[1];
            ir5[1] = (v1568_data + (v1560_data * v1566_data));
            float v1571_data = s2[32];
            float v1573_data = ir5[2];
            ir5[2] = (v1573_data + (v1560_data * v1571_data));
            float v1576_data = s2[44];
            float v1578_data = ir5[3];
            ir5[3] = (v1578_data + (v1560_data * v1576_data));
            float v1581_data = s2[56];
            float v1583_data = ir5[4];
            ir5[4] = (v1583_data + (v1560_data * v1581_data));
            float v1586_data = s2[71];
            float v1588_data = ir5[5];
            ir5[5] = (v1588_data + (v1560_data * v1586_data));
            float v1591_data = s2[83];
            float v1593_data = ir5[6];
            ir5[6] = (v1593_data + (v1560_data * v1591_data));
            float v1596_data = s2[95];
            float v1598_data = ir5[7];
            ir5[7] = (v1598_data + (v1560_data * v1596_data));
          }
          if (v16_lead < 12) {
            float v1604_data = r4[10];
            float v1605_data = s2[10];
            float v1607_data = ir5[0];
            ir5[0] = (v1607_data + (v1604_data * v1605_data));
            float v1610_data = s2[22];
            float v1612_data = ir5[1];
            ir5[1] = (v1612_data + (v1604_data * v1610_data));
            float v1615_data = s2[35];
            float v1617_data = ir5[2];
            ir5[2] = (v1617_data + (v1604_data * v1615_data));
            float v1620_data = s2[47];
            float v1622_data = ir5[3];
            ir5[3] = (v1622_data + (v1604_data * v1620_data));
            float v1625_data = s2[59];
            float v1627_data = ir5[4];
            ir5[4] = (v1627_data + (v1604_data * v1625_data));
            float v1630_data = s2[68];
            float v1632_data = ir5[5];
            ir5[5] = (v1632_data + (v1604_data * v1630_data));
            float v1635_data = s2[80];
            float v1637_data = ir5[6];
            ir5[6] = (v1637_data + (v1604_data * v1635_data));
            float v1640_data = s2[92];
            float v1642_data = ir5[7];
            ir5[7] = (v1642_data + (v1604_data * v1640_data));
          }
          if (v16_lead < 12) {
            float v1648_data = r4[11];
            float v1649_data = s2[11];
            float v1651_data = ir5[0];
            ir5[0] = (v1651_data + (v1648_data * v1649_data));
            float v1654_data = s2[23];
            float v1656_data = ir5[1];
            ir5[1] = (v1656_data + (v1648_data * v1654_data));
            float v1659_data = s2[34];
            float v1661_data = ir5[2];
            ir5[2] = (v1661_data + (v1648_data * v1659_data));
            float v1664_data = s2[46];
            float v1666_data = ir5[3];
            ir5[3] = (v1666_data + (v1648_data * v1664_data));
            float v1669_data = s2[58];
            float v1671_data = ir5[4];
            ir5[4] = (v1671_data + (v1648_data * v1669_data));
            float v1674_data = s2[69];
            float v1676_data = ir5[5];
            ir5[5] = (v1676_data + (v1648_data * v1674_data));
            float v1679_data = s2[81];
            float v1681_data = ir5[6];
            ir5[6] = (v1681_data + (v1648_data * v1679_data));
            float v1684_data = s2[93];
            float v1686_data = ir5[7];
            ir5[7] = (v1686_data + (v1648_data * v1684_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v1692_n1 = 0; v1692_n1 < 8; ++v1692_n1) {
              float v1694_data = ir5[v1692_n1];
              float v1696_data = r3[v1692_n1];
              r5[v1692_n1] = (v1696_data + v1694_data);
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          // s3 = load{g>s}(glb_m8[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v16_lead < 12) {
            float v1707_data = r6[0];
            float v1708_data = s3[0];
            float v1710_data = ir7[0];
            ir7[0] = (v1710_data + (v1707_data * v1708_data));
            float v1713_data = s3[12];
            float v1715_data = ir7[1];
            ir7[1] = (v1715_data + (v1707_data * v1713_data));
            float v1718_data = s3[24];
            float v1720_data = ir7[2];
            ir7[2] = (v1720_data + (v1707_data * v1718_data));
            float v1723_data = s3[37];
            float v1725_data = ir7[3];
            ir7[3] = (v1725_data + (v1707_data * v1723_data));
            float v1728_data = s3[49];
            float v1730_data = ir7[4];
            ir7[4] = (v1730_data + (v1707_data * v1728_data));
            float v1733_data = s3[61];
            float v1735_data = ir7[5];
            ir7[5] = (v1735_data + (v1707_data * v1733_data));
            float v1738_data = s3[74];
            float v1740_data = ir7[6];
            ir7[6] = (v1740_data + (v1707_data * v1738_data));
            float v1743_data = s3[86];
            float v1745_data = ir7[7];
            ir7[7] = (v1745_data + (v1707_data * v1743_data));
          }
          if (v16_lead < 12) {
            float v1751_data = r6[1];
            float v1752_data = s3[1];
            float v1754_data = ir7[0];
            ir7[0] = (v1754_data + (v1751_data * v1752_data));
            float v1757_data = s3[13];
            float v1759_data = ir7[1];
            ir7[1] = (v1759_data + (v1751_data * v1757_data));
            float v1762_data = s3[25];
            float v1764_data = ir7[2];
            ir7[2] = (v1764_data + (v1751_data * v1762_data));
            float v1767_data = s3[36];
            float v1769_data = ir7[3];
            ir7[3] = (v1769_data + (v1751_data * v1767_data));
            float v1772_data = s3[48];
            float v1774_data = ir7[4];
            ir7[4] = (v1774_data + (v1751_data * v1772_data));
            float v1777_data = s3[60];
            float v1779_data = ir7[5];
            ir7[5] = (v1779_data + (v1751_data * v1777_data));
            float v1782_data = s3[75];
            float v1784_data = ir7[6];
            ir7[6] = (v1784_data + (v1751_data * v1782_data));
            float v1787_data = s3[87];
            float v1789_data = ir7[7];
            ir7[7] = (v1789_data + (v1751_data * v1787_data));
          }
          if (v16_lead < 12) {
            float v1795_data = r6[2];
            float v1796_data = s3[2];
            float v1798_data = ir7[0];
            ir7[0] = (v1798_data + (v1795_data * v1796_data));
            float v1801_data = s3[14];
            float v1803_data = ir7[1];
            ir7[1] = (v1803_data + (v1795_data * v1801_data));
            float v1806_data = s3[26];
            float v1808_data = ir7[2];
            ir7[2] = (v1808_data + (v1795_data * v1806_data));
            float v1811_data = s3[39];
            float v1813_data = ir7[3];
            ir7[3] = (v1813_data + (v1795_data * v1811_data));
            float v1816_data = s3[51];
            float v1818_data = ir7[4];
            ir7[4] = (v1818_data + (v1795_data * v1816_data));
            float v1821_data = s3[63];
            float v1823_data = ir7[5];
            ir7[5] = (v1823_data + (v1795_data * v1821_data));
            float v1826_data = s3[72];
            float v1828_data = ir7[6];
            ir7[6] = (v1828_data + (v1795_data * v1826_data));
            float v1831_data = s3[84];
            float v1833_data = ir7[7];
            ir7[7] = (v1833_data + (v1795_data * v1831_data));
          }
          if (v16_lead < 12) {
            float v1839_data = r6[3];
            float v1840_data = s3[3];
            float v1842_data = ir7[0];
            ir7[0] = (v1842_data + (v1839_data * v1840_data));
            float v1845_data = s3[15];
            float v1847_data = ir7[1];
            ir7[1] = (v1847_data + (v1839_data * v1845_data));
            float v1850_data = s3[27];
            float v1852_data = ir7[2];
            ir7[2] = (v1852_data + (v1839_data * v1850_data));
            float v1855_data = s3[38];
            float v1857_data = ir7[3];
            ir7[3] = (v1857_data + (v1839_data * v1855_data));
            float v1860_data = s3[50];
            float v1862_data = ir7[4];
            ir7[4] = (v1862_data + (v1839_data * v1860_data));
            float v1865_data = s3[62];
            float v1867_data = ir7[5];
            ir7[5] = (v1867_data + (v1839_data * v1865_data));
            float v1870_data = s3[73];
            float v1872_data = ir7[6];
            ir7[6] = (v1872_data + (v1839_data * v1870_data));
            float v1875_data = s3[85];
            float v1877_data = ir7[7];
            ir7[7] = (v1877_data + (v1839_data * v1875_data));
          }
          if (v16_lead < 12) {
            float v1883_data = r6[4];
            float v1884_data = s3[4];
            float v1886_data = ir7[0];
            ir7[0] = (v1886_data + (v1883_data * v1884_data));
            float v1889_data = s3[16];
            float v1891_data = ir7[1];
            ir7[1] = (v1891_data + (v1883_data * v1889_data));
            float v1894_data = s3[28];
            float v1896_data = ir7[2];
            ir7[2] = (v1896_data + (v1883_data * v1894_data));
            float v1899_data = s3[41];
            float v1901_data = ir7[3];
            ir7[3] = (v1901_data + (v1883_data * v1899_data));
            float v1904_data = s3[53];
            float v1906_data = ir7[4];
            ir7[4] = (v1906_data + (v1883_data * v1904_data));
            float v1909_data = s3[66];
            float v1911_data = ir7[5];
            ir7[5] = (v1911_data + (v1883_data * v1909_data));
            float v1914_data = s3[78];
            float v1916_data = ir7[6];
            ir7[6] = (v1916_data + (v1883_data * v1914_data));
            float v1919_data = s3[90];
            float v1921_data = ir7[7];
            ir7[7] = (v1921_data + (v1883_data * v1919_data));
          }
          if (v16_lead < 12) {
            float v1927_data = r6[5];
            float v1928_data = s3[5];
            float v1930_data = ir7[0];
            ir7[0] = (v1930_data + (v1927_data * v1928_data));
            float v1933_data = s3[17];
            float v1935_data = ir7[1];
            ir7[1] = (v1935_data + (v1927_data * v1933_data));
            float v1938_data = s3[29];
            float v1940_data = ir7[2];
            ir7[2] = (v1940_data + (v1927_data * v1938_data));
            float v1943_data = s3[40];
            float v1945_data = ir7[3];
            ir7[3] = (v1945_data + (v1927_data * v1943_data));
            float v1948_data = s3[52];
            float v1950_data = ir7[4];
            ir7[4] = (v1950_data + (v1927_data * v1948_data));
            float v1953_data = s3[67];
            float v1955_data = ir7[5];
            ir7[5] = (v1955_data + (v1927_data * v1953_data));
            float v1958_data = s3[79];
            float v1960_data = ir7[6];
            ir7[6] = (v1960_data + (v1927_data * v1958_data));
            float v1963_data = s3[91];
            float v1965_data = ir7[7];
            ir7[7] = (v1965_data + (v1927_data * v1963_data));
          }
          if (v16_lead < 12) {
            float v1971_data = r6[6];
            float v1972_data = s3[6];
            float v1974_data = ir7[0];
            ir7[0] = (v1974_data + (v1971_data * v1972_data));
            float v1977_data = s3[18];
            float v1979_data = ir7[1];
            ir7[1] = (v1979_data + (v1971_data * v1977_data));
            float v1982_data = s3[30];
            float v1984_data = ir7[2];
            ir7[2] = (v1984_data + (v1971_data * v1982_data));
            float v1987_data = s3[43];
            float v1989_data = ir7[3];
            ir7[3] = (v1989_data + (v1971_data * v1987_data));
            float v1992_data = s3[55];
            float v1994_data = ir7[4];
            ir7[4] = (v1994_data + (v1971_data * v1992_data));
            float v1997_data = s3[64];
            float v1999_data = ir7[5];
            ir7[5] = (v1999_data + (v1971_data * v1997_data));
            float v2002_data = s3[76];
            float v2004_data = ir7[6];
            ir7[6] = (v2004_data + (v1971_data * v2002_data));
            float v2007_data = s3[88];
            float v2009_data = ir7[7];
            ir7[7] = (v2009_data + (v1971_data * v2007_data));
          }
          if (v16_lead < 12) {
            float v2015_data = r6[7];
            float v2016_data = s3[7];
            float v2018_data = ir7[0];
            ir7[0] = (v2018_data + (v2015_data * v2016_data));
            float v2021_data = s3[19];
            float v2023_data = ir7[1];
            ir7[1] = (v2023_data + (v2015_data * v2021_data));
            float v2026_data = s3[31];
            float v2028_data = ir7[2];
            ir7[2] = (v2028_data + (v2015_data * v2026_data));
            float v2031_data = s3[42];
            float v2033_data = ir7[3];
            ir7[3] = (v2033_data + (v2015_data * v2031_data));
            float v2036_data = s3[54];
            float v2038_data = ir7[4];
            ir7[4] = (v2038_data + (v2015_data * v2036_data));
            float v2041_data = s3[65];
            float v2043_data = ir7[5];
            ir7[5] = (v2043_data + (v2015_data * v2041_data));
            float v2046_data = s3[77];
            float v2048_data = ir7[6];
            ir7[6] = (v2048_data + (v2015_data * v2046_data));
            float v2051_data = s3[89];
            float v2053_data = ir7[7];
            ir7[7] = (v2053_data + (v2015_data * v2051_data));
          }
          if (v16_lead < 12) {
            float v2059_data = r6[8];
            float v2060_data = s3[8];
            float v2062_data = ir7[0];
            ir7[0] = (v2062_data + (v2059_data * v2060_data));
            float v2065_data = s3[20];
            float v2067_data = ir7[1];
            ir7[1] = (v2067_data + (v2059_data * v2065_data));
            float v2070_data = s3[33];
            float v2072_data = ir7[2];
            ir7[2] = (v2072_data + (v2059_data * v2070_data));
            float v2075_data = s3[45];
            float v2077_data = ir7[3];
            ir7[3] = (v2077_data + (v2059_data * v2075_data));
            float v2080_data = s3[57];
            float v2082_data = ir7[4];
            ir7[4] = (v2082_data + (v2059_data * v2080_data));
            float v2085_data = s3[70];
            float v2087_data = ir7[5];
            ir7[5] = (v2087_data + (v2059_data * v2085_data));
            float v2090_data = s3[82];
            float v2092_data = ir7[6];
            ir7[6] = (v2092_data + (v2059_data * v2090_data));
            float v2095_data = s3[94];
            float v2097_data = ir7[7];
            ir7[7] = (v2097_data + (v2059_data * v2095_data));
          }
          if (v16_lead < 12) {
            float v2103_data = r6[9];
            float v2104_data = s3[9];
            float v2106_data = ir7[0];
            ir7[0] = (v2106_data + (v2103_data * v2104_data));
            float v2109_data = s3[21];
            float v2111_data = ir7[1];
            ir7[1] = (v2111_data + (v2103_data * v2109_data));
            float v2114_data = s3[32];
            float v2116_data = ir7[2];
            ir7[2] = (v2116_data + (v2103_data * v2114_data));
            float v2119_data = s3[44];
            float v2121_data = ir7[3];
            ir7[3] = (v2121_data + (v2103_data * v2119_data));
            float v2124_data = s3[56];
            float v2126_data = ir7[4];
            ir7[4] = (v2126_data + (v2103_data * v2124_data));
            float v2129_data = s3[71];
            float v2131_data = ir7[5];
            ir7[5] = (v2131_data + (v2103_data * v2129_data));
            float v2134_data = s3[83];
            float v2136_data = ir7[6];
            ir7[6] = (v2136_data + (v2103_data * v2134_data));
            float v2139_data = s3[95];
            float v2141_data = ir7[7];
            ir7[7] = (v2141_data + (v2103_data * v2139_data));
          }
          if (v16_lead < 12) {
            float v2147_data = r6[10];
            float v2148_data = s3[10];
            float v2150_data = ir7[0];
            ir7[0] = (v2150_data + (v2147_data * v2148_data));
            float v2153_data = s3[22];
            float v2155_data = ir7[1];
            ir7[1] = (v2155_data + (v2147_data * v2153_data));
            float v2158_data = s3[35];
            float v2160_data = ir7[2];
            ir7[2] = (v2160_data + (v2147_data * v2158_data));
            float v2163_data = s3[47];
            float v2165_data = ir7[3];
            ir7[3] = (v2165_data + (v2147_data * v2163_data));
            float v2168_data = s3[59];
            float v2170_data = ir7[4];
            ir7[4] = (v2170_data + (v2147_data * v2168_data));
            float v2173_data = s3[68];
            float v2175_data = ir7[5];
            ir7[5] = (v2175_data + (v2147_data * v2173_data));
            float v2178_data = s3[80];
            float v2180_data = ir7[6];
            ir7[6] = (v2180_data + (v2147_data * v2178_data));
            float v2183_data = s3[92];
            float v2185_data = ir7[7];
            ir7[7] = (v2185_data + (v2147_data * v2183_data));
          }
          if (v16_lead < 12) {
            float v2191_data = r6[11];
            float v2192_data = s3[11];
            float v2194_data = ir7[0];
            ir7[0] = (v2194_data + (v2191_data * v2192_data));
            float v2197_data = s3[23];
            float v2199_data = ir7[1];
            ir7[1] = (v2199_data + (v2191_data * v2197_data));
            float v2202_data = s3[34];
            float v2204_data = ir7[2];
            ir7[2] = (v2204_data + (v2191_data * v2202_data));
            float v2207_data = s3[46];
            float v2209_data = ir7[3];
            ir7[3] = (v2209_data + (v2191_data * v2207_data));
            float v2212_data = s3[58];
            float v2214_data = ir7[4];
            ir7[4] = (v2214_data + (v2191_data * v2212_data));
            float v2217_data = s3[69];
            float v2219_data = ir7[5];
            ir7[5] = (v2219_data + (v2191_data * v2217_data));
            float v2222_data = s3[81];
            float v2224_data = ir7[6];
            ir7[6] = (v2224_data + (v2191_data * v2222_data));
            float v2227_data = s3[93];
            float v2229_data = ir7[7];
            ir7[7] = (v2229_data + (v2191_data * v2227_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v2235_n1 = 0; v2235_n1 < 8; ++v2235_n1) {
              float v2237_data = ir7[v2235_n1];
              float v2239_data = r5[v2235_n1];
              r7[v2235_n1] = (v2239_data + v2237_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v2246_i1 = 0; v2246_i1 < 8; ++v2246_i1) {
              float v2248_data = r7[v2246_i1];
              glb_m0[(v16_lead + (v2246_i1 * 12))] = v2248_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

