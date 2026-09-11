// === base name ===
kernel_426add61a397acc2

// === header ===
void launcher_kernel_426add61a397acc2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_426add61a397acc2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_426add61a397acc2, block.x * block.y * block.z, 768 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_426add61a397acc2, cudaFuncAttributeMaxDynamicSharedMemorySize, 768 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_426add61a397acc2<<<grid,block,768 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_426add61a397acc2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v5_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v21_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v28_lead = v21_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 10; v23_i1 < 13; ++v23_i1) {
              float v31_data = __ldcg(&glb_m1[(v28_lead + (v23_i1 * 32))]);
              r0[(v22_i0 + (v23_i1 - 10))] = v31_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[1]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v41_data = r0[0];
          float v42_data = s0[114];
          float v44_data = ir1[0];
          ir1[0] = (v44_data + (v41_data * v42_data));
          float v49_data = r0[1];
          float v50_data = s0[115];
          float v52_data = ir1[0];
          ir1[0] = (v52_data + (v49_data * v50_data));
          float v57_data = r0[2];
          float v58_data = s0[116];
          float v60_data = ir1[0];
          ir1[0] = (v60_data + (v57_data * v58_data));
          #pragma unroll
          for (int32_t v65_n0 = 0; v65_n0 < 1; ++v65_n0) {
            #pragma unroll
            for (int32_t v66_n1 = 0; v66_n1 < 1; ++v66_n1) {
              int32_t v67_a = v65_n0 + v66_n1;
              float v68_data = ir1[v67_a];
              r1[v67_a] = v68_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
            int32_t v81_lead = v21_lead + (v73_i0 * 32);
            #pragma unroll
            for (int32_t v74_i1 = 0; v74_i1 < 1; ++v74_i1) {
              float v76_data = r1[(v73_i0 + v74_i1)];
              glb_m0[(v81_lead + ((v74_i1 + 8) * 32))] = v76_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v89_i0 = 0; v89_i0 < 1; ++v89_i0) {
            int32_t v95_lead = v21_lead + (v89_i0 * 32);
            #pragma unroll
            for (int32_t v90_i1 = 0; v90_i1 < 13; ++v90_i1) {
              float v98_data = glb_m0[(v95_lead + (v90_i1 * 32))];
              r2[(v89_i0 + v90_i1)] = v98_data;
            }
          }
          __syncwarp();
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v107_data = r2[0];
          float v108_data = s1[0];
          float v110_data = ir3[0];
          ir3[0] = (v110_data + (v107_data * v108_data));
          float v113_data = s1[13];
          float v115_data = ir3[1];
          ir3[1] = (v115_data + (v107_data * v113_data));
          float v118_data = s1[26];
          float v120_data = ir3[2];
          ir3[2] = (v120_data + (v107_data * v118_data));
          float v123_data = s1[39];
          float v125_data = ir3[3];
          ir3[3] = (v125_data + (v107_data * v123_data));
          float v128_data = s1[52];
          float v130_data = ir3[4];
          ir3[4] = (v130_data + (v107_data * v128_data));
          float v133_data = s1[65];
          float v135_data = ir3[5];
          ir3[5] = (v135_data + (v107_data * v133_data));
          float v138_data = s1[78];
          float v140_data = ir3[6];
          ir3[6] = (v140_data + (v107_data * v138_data));
          float v143_data = s1[91];
          float v145_data = ir3[7];
          ir3[7] = (v145_data + (v107_data * v143_data));
          float v148_data = s1[104];
          float v150_data = ir3[8];
          ir3[8] = (v150_data + (v107_data * v148_data));
          float v153_data = s1[117];
          float v155_data = ir3[9];
          ir3[9] = (v155_data + (v107_data * v153_data));
          float v158_data = s1[130];
          float v160_data = ir3[10];
          ir3[10] = (v160_data + (v107_data * v158_data));
          float v163_data = s1[143];
          float v165_data = ir3[11];
          ir3[11] = (v165_data + (v107_data * v163_data));
          float v168_data = s1[156];
          float v170_data = ir3[12];
          ir3[12] = (v170_data + (v107_data * v168_data));
          float v175_data = r2[1];
          float v176_data = s1[1];
          float v178_data = ir3[0];
          ir3[0] = (v178_data + (v175_data * v176_data));
          float v181_data = s1[14];
          float v183_data = ir3[1];
          ir3[1] = (v183_data + (v175_data * v181_data));
          float v186_data = s1[27];
          float v188_data = ir3[2];
          ir3[2] = (v188_data + (v175_data * v186_data));
          float v191_data = s1[40];
          float v193_data = ir3[3];
          ir3[3] = (v193_data + (v175_data * v191_data));
          float v196_data = s1[53];
          float v198_data = ir3[4];
          ir3[4] = (v198_data + (v175_data * v196_data));
          float v201_data = s1[66];
          float v203_data = ir3[5];
          ir3[5] = (v203_data + (v175_data * v201_data));
          float v206_data = s1[79];
          float v208_data = ir3[6];
          ir3[6] = (v208_data + (v175_data * v206_data));
          float v211_data = s1[92];
          float v213_data = ir3[7];
          ir3[7] = (v213_data + (v175_data * v211_data));
          float v216_data = s1[105];
          float v218_data = ir3[8];
          ir3[8] = (v218_data + (v175_data * v216_data));
          float v221_data = s1[118];
          float v223_data = ir3[9];
          ir3[9] = (v223_data + (v175_data * v221_data));
          float v226_data = s1[131];
          float v228_data = ir3[10];
          ir3[10] = (v228_data + (v175_data * v226_data));
          float v231_data = s1[144];
          float v233_data = ir3[11];
          ir3[11] = (v233_data + (v175_data * v231_data));
          float v236_data = s1[157];
          float v238_data = ir3[12];
          ir3[12] = (v238_data + (v175_data * v236_data));
          float v243_data = r2[2];
          float v244_data = s1[2];
          float v246_data = ir3[0];
          ir3[0] = (v246_data + (v243_data * v244_data));
          float v249_data = s1[15];
          float v251_data = ir3[1];
          ir3[1] = (v251_data + (v243_data * v249_data));
          float v254_data = s1[28];
          float v256_data = ir3[2];
          ir3[2] = (v256_data + (v243_data * v254_data));
          float v259_data = s1[41];
          float v261_data = ir3[3];
          ir3[3] = (v261_data + (v243_data * v259_data));
          float v264_data = s1[54];
          float v266_data = ir3[4];
          ir3[4] = (v266_data + (v243_data * v264_data));
          float v269_data = s1[67];
          float v271_data = ir3[5];
          ir3[5] = (v271_data + (v243_data * v269_data));
          float v274_data = s1[80];
          float v276_data = ir3[6];
          ir3[6] = (v276_data + (v243_data * v274_data));
          float v279_data = s1[93];
          float v281_data = ir3[7];
          ir3[7] = (v281_data + (v243_data * v279_data));
          float v284_data = s1[106];
          float v286_data = ir3[8];
          ir3[8] = (v286_data + (v243_data * v284_data));
          float v289_data = s1[119];
          float v291_data = ir3[9];
          ir3[9] = (v291_data + (v243_data * v289_data));
          float v294_data = s1[132];
          float v296_data = ir3[10];
          ir3[10] = (v296_data + (v243_data * v294_data));
          float v299_data = s1[145];
          float v301_data = ir3[11];
          ir3[11] = (v301_data + (v243_data * v299_data));
          float v304_data = s1[158];
          float v306_data = ir3[12];
          ir3[12] = (v306_data + (v243_data * v304_data));
          float v311_data = r2[3];
          float v312_data = s1[3];
          float v314_data = ir3[0];
          ir3[0] = (v314_data + (v311_data * v312_data));
          float v317_data = s1[16];
          float v319_data = ir3[1];
          ir3[1] = (v319_data + (v311_data * v317_data));
          float v322_data = s1[29];
          float v324_data = ir3[2];
          ir3[2] = (v324_data + (v311_data * v322_data));
          float v327_data = s1[42];
          float v329_data = ir3[3];
          ir3[3] = (v329_data + (v311_data * v327_data));
          float v332_data = s1[55];
          float v334_data = ir3[4];
          ir3[4] = (v334_data + (v311_data * v332_data));
          float v337_data = s1[68];
          float v339_data = ir3[5];
          ir3[5] = (v339_data + (v311_data * v337_data));
          float v342_data = s1[81];
          float v344_data = ir3[6];
          ir3[6] = (v344_data + (v311_data * v342_data));
          float v347_data = s1[94];
          float v349_data = ir3[7];
          ir3[7] = (v349_data + (v311_data * v347_data));
          float v352_data = s1[107];
          float v354_data = ir3[8];
          ir3[8] = (v354_data + (v311_data * v352_data));
          float v357_data = s1[120];
          float v359_data = ir3[9];
          ir3[9] = (v359_data + (v311_data * v357_data));
          float v362_data = s1[133];
          float v364_data = ir3[10];
          ir3[10] = (v364_data + (v311_data * v362_data));
          float v367_data = s1[146];
          float v369_data = ir3[11];
          ir3[11] = (v369_data + (v311_data * v367_data));
          float v372_data = s1[159];
          float v374_data = ir3[12];
          ir3[12] = (v374_data + (v311_data * v372_data));
          float v379_data = r2[4];
          float v380_data = s1[4];
          float v382_data = ir3[0];
          ir3[0] = (v382_data + (v379_data * v380_data));
          float v385_data = s1[17];
          float v387_data = ir3[1];
          ir3[1] = (v387_data + (v379_data * v385_data));
          float v390_data = s1[30];
          float v392_data = ir3[2];
          ir3[2] = (v392_data + (v379_data * v390_data));
          float v395_data = s1[43];
          float v397_data = ir3[3];
          ir3[3] = (v397_data + (v379_data * v395_data));
          float v400_data = s1[56];
          float v402_data = ir3[4];
          ir3[4] = (v402_data + (v379_data * v400_data));
          float v405_data = s1[69];
          float v407_data = ir3[5];
          ir3[5] = (v407_data + (v379_data * v405_data));
          float v410_data = s1[82];
          float v412_data = ir3[6];
          ir3[6] = (v412_data + (v379_data * v410_data));
          float v415_data = s1[95];
          float v417_data = ir3[7];
          ir3[7] = (v417_data + (v379_data * v415_data));
          float v420_data = s1[108];
          float v422_data = ir3[8];
          ir3[8] = (v422_data + (v379_data * v420_data));
          float v425_data = s1[121];
          float v427_data = ir3[9];
          ir3[9] = (v427_data + (v379_data * v425_data));
          float v430_data = s1[134];
          float v432_data = ir3[10];
          ir3[10] = (v432_data + (v379_data * v430_data));
          float v435_data = s1[147];
          float v437_data = ir3[11];
          ir3[11] = (v437_data + (v379_data * v435_data));
          float v440_data = s1[160];
          float v442_data = ir3[12];
          ir3[12] = (v442_data + (v379_data * v440_data));
          float v447_data = r2[5];
          float v448_data = s1[5];
          float v450_data = ir3[0];
          ir3[0] = (v450_data + (v447_data * v448_data));
          float v453_data = s1[18];
          float v455_data = ir3[1];
          ir3[1] = (v455_data + (v447_data * v453_data));
          float v458_data = s1[31];
          float v460_data = ir3[2];
          ir3[2] = (v460_data + (v447_data * v458_data));
          float v463_data = s1[44];
          float v465_data = ir3[3];
          ir3[3] = (v465_data + (v447_data * v463_data));
          float v468_data = s1[57];
          float v470_data = ir3[4];
          ir3[4] = (v470_data + (v447_data * v468_data));
          float v473_data = s1[70];
          float v475_data = ir3[5];
          ir3[5] = (v475_data + (v447_data * v473_data));
          float v478_data = s1[83];
          float v480_data = ir3[6];
          ir3[6] = (v480_data + (v447_data * v478_data));
          float v483_data = s1[96];
          float v485_data = ir3[7];
          ir3[7] = (v485_data + (v447_data * v483_data));
          float v488_data = s1[109];
          float v490_data = ir3[8];
          ir3[8] = (v490_data + (v447_data * v488_data));
          float v493_data = s1[122];
          float v495_data = ir3[9];
          ir3[9] = (v495_data + (v447_data * v493_data));
          float v498_data = s1[135];
          float v500_data = ir3[10];
          ir3[10] = (v500_data + (v447_data * v498_data));
          float v503_data = s1[148];
          float v505_data = ir3[11];
          ir3[11] = (v505_data + (v447_data * v503_data));
          float v508_data = s1[161];
          float v510_data = ir3[12];
          ir3[12] = (v510_data + (v447_data * v508_data));
          float v515_data = r2[6];
          float v516_data = s1[6];
          float v518_data = ir3[0];
          ir3[0] = (v518_data + (v515_data * v516_data));
          float v521_data = s1[19];
          float v523_data = ir3[1];
          ir3[1] = (v523_data + (v515_data * v521_data));
          float v526_data = s1[32];
          float v528_data = ir3[2];
          ir3[2] = (v528_data + (v515_data * v526_data));
          float v531_data = s1[45];
          float v533_data = ir3[3];
          ir3[3] = (v533_data + (v515_data * v531_data));
          float v536_data = s1[58];
          float v538_data = ir3[4];
          ir3[4] = (v538_data + (v515_data * v536_data));
          float v541_data = s1[71];
          float v543_data = ir3[5];
          ir3[5] = (v543_data + (v515_data * v541_data));
          float v546_data = s1[84];
          float v548_data = ir3[6];
          ir3[6] = (v548_data + (v515_data * v546_data));
          float v551_data = s1[97];
          float v553_data = ir3[7];
          ir3[7] = (v553_data + (v515_data * v551_data));
          float v556_data = s1[110];
          float v558_data = ir3[8];
          ir3[8] = (v558_data + (v515_data * v556_data));
          float v561_data = s1[123];
          float v563_data = ir3[9];
          ir3[9] = (v563_data + (v515_data * v561_data));
          float v566_data = s1[136];
          float v568_data = ir3[10];
          ir3[10] = (v568_data + (v515_data * v566_data));
          float v571_data = s1[149];
          float v573_data = ir3[11];
          ir3[11] = (v573_data + (v515_data * v571_data));
          float v576_data = s1[162];
          float v578_data = ir3[12];
          ir3[12] = (v578_data + (v515_data * v576_data));
          float v583_data = r2[7];
          float v584_data = s1[7];
          float v586_data = ir3[0];
          ir3[0] = (v586_data + (v583_data * v584_data));
          float v589_data = s1[20];
          float v591_data = ir3[1];
          ir3[1] = (v591_data + (v583_data * v589_data));
          float v594_data = s1[33];
          float v596_data = ir3[2];
          ir3[2] = (v596_data + (v583_data * v594_data));
          float v599_data = s1[46];
          float v601_data = ir3[3];
          ir3[3] = (v601_data + (v583_data * v599_data));
          float v604_data = s1[59];
          float v606_data = ir3[4];
          ir3[4] = (v606_data + (v583_data * v604_data));
          float v609_data = s1[72];
          float v611_data = ir3[5];
          ir3[5] = (v611_data + (v583_data * v609_data));
          float v614_data = s1[85];
          float v616_data = ir3[6];
          ir3[6] = (v616_data + (v583_data * v614_data));
          float v619_data = s1[98];
          float v621_data = ir3[7];
          ir3[7] = (v621_data + (v583_data * v619_data));
          float v624_data = s1[111];
          float v626_data = ir3[8];
          ir3[8] = (v626_data + (v583_data * v624_data));
          float v629_data = s1[124];
          float v631_data = ir3[9];
          ir3[9] = (v631_data + (v583_data * v629_data));
          float v634_data = s1[137];
          float v636_data = ir3[10];
          ir3[10] = (v636_data + (v583_data * v634_data));
          float v639_data = s1[150];
          float v641_data = ir3[11];
          ir3[11] = (v641_data + (v583_data * v639_data));
          float v644_data = s1[163];
          float v646_data = ir3[12];
          ir3[12] = (v646_data + (v583_data * v644_data));
          float v651_data = r2[8];
          float v652_data = s1[8];
          float v654_data = ir3[0];
          ir3[0] = (v654_data + (v651_data * v652_data));
          float v657_data = s1[21];
          float v659_data = ir3[1];
          ir3[1] = (v659_data + (v651_data * v657_data));
          float v662_data = s1[34];
          float v664_data = ir3[2];
          ir3[2] = (v664_data + (v651_data * v662_data));
          float v667_data = s1[47];
          float v669_data = ir3[3];
          ir3[3] = (v669_data + (v651_data * v667_data));
          float v672_data = s1[60];
          float v674_data = ir3[4];
          ir3[4] = (v674_data + (v651_data * v672_data));
          float v677_data = s1[73];
          float v679_data = ir3[5];
          ir3[5] = (v679_data + (v651_data * v677_data));
          float v682_data = s1[86];
          float v684_data = ir3[6];
          ir3[6] = (v684_data + (v651_data * v682_data));
          float v687_data = s1[99];
          float v689_data = ir3[7];
          ir3[7] = (v689_data + (v651_data * v687_data));
          float v692_data = s1[112];
          float v694_data = ir3[8];
          ir3[8] = (v694_data + (v651_data * v692_data));
          float v697_data = s1[125];
          float v699_data = ir3[9];
          ir3[9] = (v699_data + (v651_data * v697_data));
          float v702_data = s1[138];
          float v704_data = ir3[10];
          ir3[10] = (v704_data + (v651_data * v702_data));
          float v707_data = s1[151];
          float v709_data = ir3[11];
          ir3[11] = (v709_data + (v651_data * v707_data));
          float v712_data = s1[164];
          float v714_data = ir3[12];
          ir3[12] = (v714_data + (v651_data * v712_data));
          float v719_data = r2[9];
          float v720_data = s1[9];
          float v722_data = ir3[0];
          ir3[0] = (v722_data + (v719_data * v720_data));
          float v725_data = s1[22];
          float v727_data = ir3[1];
          ir3[1] = (v727_data + (v719_data * v725_data));
          float v730_data = s1[35];
          float v732_data = ir3[2];
          ir3[2] = (v732_data + (v719_data * v730_data));
          float v735_data = s1[48];
          float v737_data = ir3[3];
          ir3[3] = (v737_data + (v719_data * v735_data));
          float v740_data = s1[61];
          float v742_data = ir3[4];
          ir3[4] = (v742_data + (v719_data * v740_data));
          float v745_data = s1[74];
          float v747_data = ir3[5];
          ir3[5] = (v747_data + (v719_data * v745_data));
          float v750_data = s1[87];
          float v752_data = ir3[6];
          ir3[6] = (v752_data + (v719_data * v750_data));
          float v755_data = s1[100];
          float v757_data = ir3[7];
          ir3[7] = (v757_data + (v719_data * v755_data));
          float v760_data = s1[113];
          float v762_data = ir3[8];
          ir3[8] = (v762_data + (v719_data * v760_data));
          float v765_data = s1[126];
          float v767_data = ir3[9];
          ir3[9] = (v767_data + (v719_data * v765_data));
          float v770_data = s1[139];
          float v772_data = ir3[10];
          ir3[10] = (v772_data + (v719_data * v770_data));
          float v775_data = s1[152];
          float v777_data = ir3[11];
          ir3[11] = (v777_data + (v719_data * v775_data));
          float v780_data = s1[165];
          float v782_data = ir3[12];
          ir3[12] = (v782_data + (v719_data * v780_data));
          float v787_data = r2[10];
          float v788_data = s1[10];
          float v790_data = ir3[0];
          ir3[0] = (v790_data + (v787_data * v788_data));
          float v793_data = s1[23];
          float v795_data = ir3[1];
          ir3[1] = (v795_data + (v787_data * v793_data));
          float v798_data = s1[36];
          float v800_data = ir3[2];
          ir3[2] = (v800_data + (v787_data * v798_data));
          float v803_data = s1[49];
          float v805_data = ir3[3];
          ir3[3] = (v805_data + (v787_data * v803_data));
          float v808_data = s1[62];
          float v810_data = ir3[4];
          ir3[4] = (v810_data + (v787_data * v808_data));
          float v813_data = s1[75];
          float v815_data = ir3[5];
          ir3[5] = (v815_data + (v787_data * v813_data));
          float v818_data = s1[88];
          float v820_data = ir3[6];
          ir3[6] = (v820_data + (v787_data * v818_data));
          float v823_data = s1[101];
          float v825_data = ir3[7];
          ir3[7] = (v825_data + (v787_data * v823_data));
          float v828_data = s1[114];
          float v830_data = ir3[8];
          ir3[8] = (v830_data + (v787_data * v828_data));
          float v833_data = s1[127];
          float v835_data = ir3[9];
          ir3[9] = (v835_data + (v787_data * v833_data));
          float v838_data = s1[140];
          float v840_data = ir3[10];
          ir3[10] = (v840_data + (v787_data * v838_data));
          float v843_data = s1[153];
          float v845_data = ir3[11];
          ir3[11] = (v845_data + (v787_data * v843_data));
          float v848_data = s1[166];
          float v850_data = ir3[12];
          ir3[12] = (v850_data + (v787_data * v848_data));
          float v855_data = r2[11];
          float v856_data = s1[11];
          float v858_data = ir3[0];
          ir3[0] = (v858_data + (v855_data * v856_data));
          float v861_data = s1[24];
          float v863_data = ir3[1];
          ir3[1] = (v863_data + (v855_data * v861_data));
          float v866_data = s1[37];
          float v868_data = ir3[2];
          ir3[2] = (v868_data + (v855_data * v866_data));
          float v871_data = s1[50];
          float v873_data = ir3[3];
          ir3[3] = (v873_data + (v855_data * v871_data));
          float v876_data = s1[63];
          float v878_data = ir3[4];
          ir3[4] = (v878_data + (v855_data * v876_data));
          float v881_data = s1[76];
          float v883_data = ir3[5];
          ir3[5] = (v883_data + (v855_data * v881_data));
          float v886_data = s1[89];
          float v888_data = ir3[6];
          ir3[6] = (v888_data + (v855_data * v886_data));
          float v891_data = s1[102];
          float v893_data = ir3[7];
          ir3[7] = (v893_data + (v855_data * v891_data));
          float v896_data = s1[115];
          float v898_data = ir3[8];
          ir3[8] = (v898_data + (v855_data * v896_data));
          float v901_data = s1[128];
          float v903_data = ir3[9];
          ir3[9] = (v903_data + (v855_data * v901_data));
          float v906_data = s1[141];
          float v908_data = ir3[10];
          ir3[10] = (v908_data + (v855_data * v906_data));
          float v911_data = s1[154];
          float v913_data = ir3[11];
          ir3[11] = (v913_data + (v855_data * v911_data));
          float v916_data = s1[167];
          float v918_data = ir3[12];
          ir3[12] = (v918_data + (v855_data * v916_data));
          float v923_data = r2[12];
          float v924_data = s1[12];
          float v926_data = ir3[0];
          ir3[0] = (v926_data + (v923_data * v924_data));
          float v929_data = s1[25];
          float v931_data = ir3[1];
          ir3[1] = (v931_data + (v923_data * v929_data));
          float v934_data = s1[38];
          float v936_data = ir3[2];
          ir3[2] = (v936_data + (v923_data * v934_data));
          float v939_data = s1[51];
          float v941_data = ir3[3];
          ir3[3] = (v941_data + (v923_data * v939_data));
          float v944_data = s1[64];
          float v946_data = ir3[4];
          ir3[4] = (v946_data + (v923_data * v944_data));
          float v949_data = s1[77];
          float v951_data = ir3[5];
          ir3[5] = (v951_data + (v923_data * v949_data));
          float v954_data = s1[90];
          float v956_data = ir3[6];
          ir3[6] = (v956_data + (v923_data * v954_data));
          float v959_data = s1[103];
          float v961_data = ir3[7];
          ir3[7] = (v961_data + (v923_data * v959_data));
          float v964_data = s1[116];
          float v966_data = ir3[8];
          ir3[8] = (v966_data + (v923_data * v964_data));
          float v969_data = s1[129];
          float v971_data = ir3[9];
          ir3[9] = (v971_data + (v923_data * v969_data));
          float v974_data = s1[142];
          float v976_data = ir3[10];
          ir3[10] = (v976_data + (v923_data * v974_data));
          float v979_data = s1[155];
          float v981_data = ir3[11];
          ir3[11] = (v981_data + (v923_data * v979_data));
          float v984_data = s1[168];
          float v986_data = ir3[12];
          ir3[12] = (v986_data + (v923_data * v984_data));
          #pragma unroll
          for (int32_t v991_n0 = 0; v991_n0 < 1; ++v991_n0) {
            #pragma unroll
            for (int32_t v992_n1 = 0; v992_n1 < 13; ++v992_n1) {
              int32_t v993_a = v991_n0 + v992_n1;
              float v994_data = ir3[v993_a];
              r3[v993_a] = v994_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v999_i0 = 0; v999_i0 < 1; ++v999_i0) {
            int32_t v1007_lead = v21_lead + (v999_i0 * 32);
            #pragma unroll
            for (int32_t v1000_i1 = 0; v1000_i1 < 13; ++v1000_i1) {
              float v1002_data = r3[(v999_i0 + v1000_i1)];
              glb_m3[(v1007_lead + (v1000_i1 * 32))] = v1002_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

