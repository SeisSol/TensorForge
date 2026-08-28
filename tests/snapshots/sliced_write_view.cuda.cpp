// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_939857c66e, block.x * block.y * block.z, 1536 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_939857c66e, cudaFuncAttributeMaxDynamicSharedMemorySize, 1536 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_939857c66e<<<grid,block,1536 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          alignas(8) float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v8_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
            int32_t v14_lead = v9_i0 * 32;
            int32_t v15_lead = v8_lead + v14_lead;
            int32_t v22_lead = v8_lead + v14_lead;
            #pragma unroll
            for (int32_t v10_i1 = 10; v10_i1 < 13; ++v10_i1) {
              int32_t v16_a = v10_i1 * 32;
              int32_t v17_a = v15_lead + v16_a;
              float v25_data = __ldcg(&glb_m1[(v22_lead + v16_a)]);
              int32_t v27_a = v9_i0 + (v10_i1 - 10);
              r0[v27_a] = v25_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
              __pipeline_commit();
            }
            if (threadIdx.x < 9) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[1]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v36_data = r0[0];
          float v37_data = s0[114];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v44_data = r0[1];
          float v45_data = s0[115];
          float v47_data = ir1[0];
          ir1[0] = (v47_data + (v44_data * v45_data));
          float v52_data = r0[2];
          float v53_data = s0[116];
          float v55_data = ir1[0];
          ir1[0] = (v55_data + (v52_data * v53_data));
          #pragma unroll
          for (int32_t v60_n0 = 0; v60_n0 < 1; ++v60_n0) {
            #pragma unroll
            for (int32_t v61_n1 = 0; v61_n1 < 1; ++v61_n1) {
              int32_t v62_a = v60_n0 + v61_n1;
              int32_t v63_a = v60_n0 + v61_n1;
              float v64_data = ir1[v63_a];
              r1[v63_a] = v64_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v69_i0 = 0; v69_i0 < 1; ++v69_i0) {
            int32_t v78_lead = v8_lead + (v69_i0 * 32);
            #pragma unroll
            for (int32_t v70_i1 = 0; v70_i1 < 1; ++v70_i1) {
              int32_t v71_a = v69_i0 + v70_i1;
              float v73_data = r1[(v69_i0 + v70_i1)];
              glb_m0[(v78_lead + ((v70_i1 + 8) * 32))] = v73_data;
            }
          }
          alignas(16) float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v86_i0 = 0; v86_i0 < 1; ++v86_i0) {
            int32_t v91_lead = v86_i0 * 32;
            int32_t v92_lead = v8_lead + v91_lead;
            int32_t v99_lead = v8_lead + v91_lead;
            #pragma unroll
            for (int32_t v87_i1 = 0; v87_i1 < 13; ++v87_i1) {
              int32_t v93_a = v87_i1 * 32;
              int32_t v94_a = v92_lead + v93_a;
              float v102_data = glb_m0[(v99_lead + v93_a)];
              int32_t v103_a = v86_i0 + v87_i1;
              r2[v103_a] = v102_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
              __pipeline_commit();
            }
            if (threadIdx.x < 9) {
              __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
              __pipeline_commit();
            }
          }
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) float r3[13]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v112_data = r2[0];
          float v113_data = s1[0];
          float v115_data = ir3[0];
          ir3[0] = (v115_data + (v112_data * v113_data));
          float v118_data = s1[13];
          float v120_data = ir3[1];
          ir3[1] = (v120_data + (v112_data * v118_data));
          float v123_data = s1[26];
          float v125_data = ir3[2];
          ir3[2] = (v125_data + (v112_data * v123_data));
          float v128_data = s1[39];
          float v130_data = ir3[3];
          ir3[3] = (v130_data + (v112_data * v128_data));
          float v133_data = s1[52];
          float v135_data = ir3[4];
          ir3[4] = (v135_data + (v112_data * v133_data));
          float v138_data = s1[65];
          float v140_data = ir3[5];
          ir3[5] = (v140_data + (v112_data * v138_data));
          float v143_data = s1[78];
          float v145_data = ir3[6];
          ir3[6] = (v145_data + (v112_data * v143_data));
          float v148_data = s1[91];
          float v150_data = ir3[7];
          ir3[7] = (v150_data + (v112_data * v148_data));
          float v153_data = s1[104];
          float v155_data = ir3[8];
          ir3[8] = (v155_data + (v112_data * v153_data));
          float v158_data = s1[117];
          float v160_data = ir3[9];
          ir3[9] = (v160_data + (v112_data * v158_data));
          float v163_data = s1[130];
          float v165_data = ir3[10];
          ir3[10] = (v165_data + (v112_data * v163_data));
          float v168_data = s1[143];
          float v170_data = ir3[11];
          ir3[11] = (v170_data + (v112_data * v168_data));
          float v173_data = s1[156];
          float v175_data = ir3[12];
          ir3[12] = (v175_data + (v112_data * v173_data));
          float v180_data = r2[1];
          float v181_data = s1[1];
          float v183_data = ir3[0];
          ir3[0] = (v183_data + (v180_data * v181_data));
          float v186_data = s1[14];
          float v188_data = ir3[1];
          ir3[1] = (v188_data + (v180_data * v186_data));
          float v191_data = s1[27];
          float v193_data = ir3[2];
          ir3[2] = (v193_data + (v180_data * v191_data));
          float v196_data = s1[40];
          float v198_data = ir3[3];
          ir3[3] = (v198_data + (v180_data * v196_data));
          float v201_data = s1[53];
          float v203_data = ir3[4];
          ir3[4] = (v203_data + (v180_data * v201_data));
          float v206_data = s1[66];
          float v208_data = ir3[5];
          ir3[5] = (v208_data + (v180_data * v206_data));
          float v211_data = s1[79];
          float v213_data = ir3[6];
          ir3[6] = (v213_data + (v180_data * v211_data));
          float v216_data = s1[92];
          float v218_data = ir3[7];
          ir3[7] = (v218_data + (v180_data * v216_data));
          float v221_data = s1[105];
          float v223_data = ir3[8];
          ir3[8] = (v223_data + (v180_data * v221_data));
          float v226_data = s1[118];
          float v228_data = ir3[9];
          ir3[9] = (v228_data + (v180_data * v226_data));
          float v231_data = s1[131];
          float v233_data = ir3[10];
          ir3[10] = (v233_data + (v180_data * v231_data));
          float v236_data = s1[144];
          float v238_data = ir3[11];
          ir3[11] = (v238_data + (v180_data * v236_data));
          float v241_data = s1[157];
          float v243_data = ir3[12];
          ir3[12] = (v243_data + (v180_data * v241_data));
          float v248_data = r2[2];
          float v249_data = s1[2];
          float v251_data = ir3[0];
          ir3[0] = (v251_data + (v248_data * v249_data));
          float v254_data = s1[15];
          float v256_data = ir3[1];
          ir3[1] = (v256_data + (v248_data * v254_data));
          float v259_data = s1[28];
          float v261_data = ir3[2];
          ir3[2] = (v261_data + (v248_data * v259_data));
          float v264_data = s1[41];
          float v266_data = ir3[3];
          ir3[3] = (v266_data + (v248_data * v264_data));
          float v269_data = s1[54];
          float v271_data = ir3[4];
          ir3[4] = (v271_data + (v248_data * v269_data));
          float v274_data = s1[67];
          float v276_data = ir3[5];
          ir3[5] = (v276_data + (v248_data * v274_data));
          float v279_data = s1[80];
          float v281_data = ir3[6];
          ir3[6] = (v281_data + (v248_data * v279_data));
          float v284_data = s1[93];
          float v286_data = ir3[7];
          ir3[7] = (v286_data + (v248_data * v284_data));
          float v289_data = s1[106];
          float v291_data = ir3[8];
          ir3[8] = (v291_data + (v248_data * v289_data));
          float v294_data = s1[119];
          float v296_data = ir3[9];
          ir3[9] = (v296_data + (v248_data * v294_data));
          float v299_data = s1[132];
          float v301_data = ir3[10];
          ir3[10] = (v301_data + (v248_data * v299_data));
          float v304_data = s1[145];
          float v306_data = ir3[11];
          ir3[11] = (v306_data + (v248_data * v304_data));
          float v309_data = s1[158];
          float v311_data = ir3[12];
          ir3[12] = (v311_data + (v248_data * v309_data));
          float v316_data = r2[3];
          float v317_data = s1[3];
          float v319_data = ir3[0];
          ir3[0] = (v319_data + (v316_data * v317_data));
          float v322_data = s1[16];
          float v324_data = ir3[1];
          ir3[1] = (v324_data + (v316_data * v322_data));
          float v327_data = s1[29];
          float v329_data = ir3[2];
          ir3[2] = (v329_data + (v316_data * v327_data));
          float v332_data = s1[42];
          float v334_data = ir3[3];
          ir3[3] = (v334_data + (v316_data * v332_data));
          float v337_data = s1[55];
          float v339_data = ir3[4];
          ir3[4] = (v339_data + (v316_data * v337_data));
          float v342_data = s1[68];
          float v344_data = ir3[5];
          ir3[5] = (v344_data + (v316_data * v342_data));
          float v347_data = s1[81];
          float v349_data = ir3[6];
          ir3[6] = (v349_data + (v316_data * v347_data));
          float v352_data = s1[94];
          float v354_data = ir3[7];
          ir3[7] = (v354_data + (v316_data * v352_data));
          float v357_data = s1[107];
          float v359_data = ir3[8];
          ir3[8] = (v359_data + (v316_data * v357_data));
          float v362_data = s1[120];
          float v364_data = ir3[9];
          ir3[9] = (v364_data + (v316_data * v362_data));
          float v367_data = s1[133];
          float v369_data = ir3[10];
          ir3[10] = (v369_data + (v316_data * v367_data));
          float v372_data = s1[146];
          float v374_data = ir3[11];
          ir3[11] = (v374_data + (v316_data * v372_data));
          float v377_data = s1[159];
          float v379_data = ir3[12];
          ir3[12] = (v379_data + (v316_data * v377_data));
          float v384_data = r2[4];
          float v385_data = s1[4];
          float v387_data = ir3[0];
          ir3[0] = (v387_data + (v384_data * v385_data));
          float v390_data = s1[17];
          float v392_data = ir3[1];
          ir3[1] = (v392_data + (v384_data * v390_data));
          float v395_data = s1[30];
          float v397_data = ir3[2];
          ir3[2] = (v397_data + (v384_data * v395_data));
          float v400_data = s1[43];
          float v402_data = ir3[3];
          ir3[3] = (v402_data + (v384_data * v400_data));
          float v405_data = s1[56];
          float v407_data = ir3[4];
          ir3[4] = (v407_data + (v384_data * v405_data));
          float v410_data = s1[69];
          float v412_data = ir3[5];
          ir3[5] = (v412_data + (v384_data * v410_data));
          float v415_data = s1[82];
          float v417_data = ir3[6];
          ir3[6] = (v417_data + (v384_data * v415_data));
          float v420_data = s1[95];
          float v422_data = ir3[7];
          ir3[7] = (v422_data + (v384_data * v420_data));
          float v425_data = s1[108];
          float v427_data = ir3[8];
          ir3[8] = (v427_data + (v384_data * v425_data));
          float v430_data = s1[121];
          float v432_data = ir3[9];
          ir3[9] = (v432_data + (v384_data * v430_data));
          float v435_data = s1[134];
          float v437_data = ir3[10];
          ir3[10] = (v437_data + (v384_data * v435_data));
          float v440_data = s1[147];
          float v442_data = ir3[11];
          ir3[11] = (v442_data + (v384_data * v440_data));
          float v445_data = s1[160];
          float v447_data = ir3[12];
          ir3[12] = (v447_data + (v384_data * v445_data));
          float v452_data = r2[5];
          float v453_data = s1[5];
          float v455_data = ir3[0];
          ir3[0] = (v455_data + (v452_data * v453_data));
          float v458_data = s1[18];
          float v460_data = ir3[1];
          ir3[1] = (v460_data + (v452_data * v458_data));
          float v463_data = s1[31];
          float v465_data = ir3[2];
          ir3[2] = (v465_data + (v452_data * v463_data));
          float v468_data = s1[44];
          float v470_data = ir3[3];
          ir3[3] = (v470_data + (v452_data * v468_data));
          float v473_data = s1[57];
          float v475_data = ir3[4];
          ir3[4] = (v475_data + (v452_data * v473_data));
          float v478_data = s1[70];
          float v480_data = ir3[5];
          ir3[5] = (v480_data + (v452_data * v478_data));
          float v483_data = s1[83];
          float v485_data = ir3[6];
          ir3[6] = (v485_data + (v452_data * v483_data));
          float v488_data = s1[96];
          float v490_data = ir3[7];
          ir3[7] = (v490_data + (v452_data * v488_data));
          float v493_data = s1[109];
          float v495_data = ir3[8];
          ir3[8] = (v495_data + (v452_data * v493_data));
          float v498_data = s1[122];
          float v500_data = ir3[9];
          ir3[9] = (v500_data + (v452_data * v498_data));
          float v503_data = s1[135];
          float v505_data = ir3[10];
          ir3[10] = (v505_data + (v452_data * v503_data));
          float v508_data = s1[148];
          float v510_data = ir3[11];
          ir3[11] = (v510_data + (v452_data * v508_data));
          float v513_data = s1[161];
          float v515_data = ir3[12];
          ir3[12] = (v515_data + (v452_data * v513_data));
          float v520_data = r2[6];
          float v521_data = s1[6];
          float v523_data = ir3[0];
          ir3[0] = (v523_data + (v520_data * v521_data));
          float v526_data = s1[19];
          float v528_data = ir3[1];
          ir3[1] = (v528_data + (v520_data * v526_data));
          float v531_data = s1[32];
          float v533_data = ir3[2];
          ir3[2] = (v533_data + (v520_data * v531_data));
          float v536_data = s1[45];
          float v538_data = ir3[3];
          ir3[3] = (v538_data + (v520_data * v536_data));
          float v541_data = s1[58];
          float v543_data = ir3[4];
          ir3[4] = (v543_data + (v520_data * v541_data));
          float v546_data = s1[71];
          float v548_data = ir3[5];
          ir3[5] = (v548_data + (v520_data * v546_data));
          float v551_data = s1[84];
          float v553_data = ir3[6];
          ir3[6] = (v553_data + (v520_data * v551_data));
          float v556_data = s1[97];
          float v558_data = ir3[7];
          ir3[7] = (v558_data + (v520_data * v556_data));
          float v561_data = s1[110];
          float v563_data = ir3[8];
          ir3[8] = (v563_data + (v520_data * v561_data));
          float v566_data = s1[123];
          float v568_data = ir3[9];
          ir3[9] = (v568_data + (v520_data * v566_data));
          float v571_data = s1[136];
          float v573_data = ir3[10];
          ir3[10] = (v573_data + (v520_data * v571_data));
          float v576_data = s1[149];
          float v578_data = ir3[11];
          ir3[11] = (v578_data + (v520_data * v576_data));
          float v581_data = s1[162];
          float v583_data = ir3[12];
          ir3[12] = (v583_data + (v520_data * v581_data));
          float v588_data = r2[7];
          float v589_data = s1[7];
          float v591_data = ir3[0];
          ir3[0] = (v591_data + (v588_data * v589_data));
          float v594_data = s1[20];
          float v596_data = ir3[1];
          ir3[1] = (v596_data + (v588_data * v594_data));
          float v599_data = s1[33];
          float v601_data = ir3[2];
          ir3[2] = (v601_data + (v588_data * v599_data));
          float v604_data = s1[46];
          float v606_data = ir3[3];
          ir3[3] = (v606_data + (v588_data * v604_data));
          float v609_data = s1[59];
          float v611_data = ir3[4];
          ir3[4] = (v611_data + (v588_data * v609_data));
          float v614_data = s1[72];
          float v616_data = ir3[5];
          ir3[5] = (v616_data + (v588_data * v614_data));
          float v619_data = s1[85];
          float v621_data = ir3[6];
          ir3[6] = (v621_data + (v588_data * v619_data));
          float v624_data = s1[98];
          float v626_data = ir3[7];
          ir3[7] = (v626_data + (v588_data * v624_data));
          float v629_data = s1[111];
          float v631_data = ir3[8];
          ir3[8] = (v631_data + (v588_data * v629_data));
          float v634_data = s1[124];
          float v636_data = ir3[9];
          ir3[9] = (v636_data + (v588_data * v634_data));
          float v639_data = s1[137];
          float v641_data = ir3[10];
          ir3[10] = (v641_data + (v588_data * v639_data));
          float v644_data = s1[150];
          float v646_data = ir3[11];
          ir3[11] = (v646_data + (v588_data * v644_data));
          float v649_data = s1[163];
          float v651_data = ir3[12];
          ir3[12] = (v651_data + (v588_data * v649_data));
          float v656_data = r2[8];
          float v657_data = s1[8];
          float v659_data = ir3[0];
          ir3[0] = (v659_data + (v656_data * v657_data));
          float v662_data = s1[21];
          float v664_data = ir3[1];
          ir3[1] = (v664_data + (v656_data * v662_data));
          float v667_data = s1[34];
          float v669_data = ir3[2];
          ir3[2] = (v669_data + (v656_data * v667_data));
          float v672_data = s1[47];
          float v674_data = ir3[3];
          ir3[3] = (v674_data + (v656_data * v672_data));
          float v677_data = s1[60];
          float v679_data = ir3[4];
          ir3[4] = (v679_data + (v656_data * v677_data));
          float v682_data = s1[73];
          float v684_data = ir3[5];
          ir3[5] = (v684_data + (v656_data * v682_data));
          float v687_data = s1[86];
          float v689_data = ir3[6];
          ir3[6] = (v689_data + (v656_data * v687_data));
          float v692_data = s1[99];
          float v694_data = ir3[7];
          ir3[7] = (v694_data + (v656_data * v692_data));
          float v697_data = s1[112];
          float v699_data = ir3[8];
          ir3[8] = (v699_data + (v656_data * v697_data));
          float v702_data = s1[125];
          float v704_data = ir3[9];
          ir3[9] = (v704_data + (v656_data * v702_data));
          float v707_data = s1[138];
          float v709_data = ir3[10];
          ir3[10] = (v709_data + (v656_data * v707_data));
          float v712_data = s1[151];
          float v714_data = ir3[11];
          ir3[11] = (v714_data + (v656_data * v712_data));
          float v717_data = s1[164];
          float v719_data = ir3[12];
          ir3[12] = (v719_data + (v656_data * v717_data));
          float v724_data = r2[9];
          float v725_data = s1[9];
          float v727_data = ir3[0];
          ir3[0] = (v727_data + (v724_data * v725_data));
          float v730_data = s1[22];
          float v732_data = ir3[1];
          ir3[1] = (v732_data + (v724_data * v730_data));
          float v735_data = s1[35];
          float v737_data = ir3[2];
          ir3[2] = (v737_data + (v724_data * v735_data));
          float v740_data = s1[48];
          float v742_data = ir3[3];
          ir3[3] = (v742_data + (v724_data * v740_data));
          float v745_data = s1[61];
          float v747_data = ir3[4];
          ir3[4] = (v747_data + (v724_data * v745_data));
          float v750_data = s1[74];
          float v752_data = ir3[5];
          ir3[5] = (v752_data + (v724_data * v750_data));
          float v755_data = s1[87];
          float v757_data = ir3[6];
          ir3[6] = (v757_data + (v724_data * v755_data));
          float v760_data = s1[100];
          float v762_data = ir3[7];
          ir3[7] = (v762_data + (v724_data * v760_data));
          float v765_data = s1[113];
          float v767_data = ir3[8];
          ir3[8] = (v767_data + (v724_data * v765_data));
          float v770_data = s1[126];
          float v772_data = ir3[9];
          ir3[9] = (v772_data + (v724_data * v770_data));
          float v775_data = s1[139];
          float v777_data = ir3[10];
          ir3[10] = (v777_data + (v724_data * v775_data));
          float v780_data = s1[152];
          float v782_data = ir3[11];
          ir3[11] = (v782_data + (v724_data * v780_data));
          float v785_data = s1[165];
          float v787_data = ir3[12];
          ir3[12] = (v787_data + (v724_data * v785_data));
          float v792_data = r2[10];
          float v793_data = s1[10];
          float v795_data = ir3[0];
          ir3[0] = (v795_data + (v792_data * v793_data));
          float v798_data = s1[23];
          float v800_data = ir3[1];
          ir3[1] = (v800_data + (v792_data * v798_data));
          float v803_data = s1[36];
          float v805_data = ir3[2];
          ir3[2] = (v805_data + (v792_data * v803_data));
          float v808_data = s1[49];
          float v810_data = ir3[3];
          ir3[3] = (v810_data + (v792_data * v808_data));
          float v813_data = s1[62];
          float v815_data = ir3[4];
          ir3[4] = (v815_data + (v792_data * v813_data));
          float v818_data = s1[75];
          float v820_data = ir3[5];
          ir3[5] = (v820_data + (v792_data * v818_data));
          float v823_data = s1[88];
          float v825_data = ir3[6];
          ir3[6] = (v825_data + (v792_data * v823_data));
          float v828_data = s1[101];
          float v830_data = ir3[7];
          ir3[7] = (v830_data + (v792_data * v828_data));
          float v833_data = s1[114];
          float v835_data = ir3[8];
          ir3[8] = (v835_data + (v792_data * v833_data));
          float v838_data = s1[127];
          float v840_data = ir3[9];
          ir3[9] = (v840_data + (v792_data * v838_data));
          float v843_data = s1[140];
          float v845_data = ir3[10];
          ir3[10] = (v845_data + (v792_data * v843_data));
          float v848_data = s1[153];
          float v850_data = ir3[11];
          ir3[11] = (v850_data + (v792_data * v848_data));
          float v853_data = s1[166];
          float v855_data = ir3[12];
          ir3[12] = (v855_data + (v792_data * v853_data));
          float v860_data = r2[11];
          float v861_data = s1[11];
          float v863_data = ir3[0];
          ir3[0] = (v863_data + (v860_data * v861_data));
          float v866_data = s1[24];
          float v868_data = ir3[1];
          ir3[1] = (v868_data + (v860_data * v866_data));
          float v871_data = s1[37];
          float v873_data = ir3[2];
          ir3[2] = (v873_data + (v860_data * v871_data));
          float v876_data = s1[50];
          float v878_data = ir3[3];
          ir3[3] = (v878_data + (v860_data * v876_data));
          float v881_data = s1[63];
          float v883_data = ir3[4];
          ir3[4] = (v883_data + (v860_data * v881_data));
          float v886_data = s1[76];
          float v888_data = ir3[5];
          ir3[5] = (v888_data + (v860_data * v886_data));
          float v891_data = s1[89];
          float v893_data = ir3[6];
          ir3[6] = (v893_data + (v860_data * v891_data));
          float v896_data = s1[102];
          float v898_data = ir3[7];
          ir3[7] = (v898_data + (v860_data * v896_data));
          float v901_data = s1[115];
          float v903_data = ir3[8];
          ir3[8] = (v903_data + (v860_data * v901_data));
          float v906_data = s1[128];
          float v908_data = ir3[9];
          ir3[9] = (v908_data + (v860_data * v906_data));
          float v911_data = s1[141];
          float v913_data = ir3[10];
          ir3[10] = (v913_data + (v860_data * v911_data));
          float v916_data = s1[154];
          float v918_data = ir3[11];
          ir3[11] = (v918_data + (v860_data * v916_data));
          float v921_data = s1[167];
          float v923_data = ir3[12];
          ir3[12] = (v923_data + (v860_data * v921_data));
          float v928_data = r2[12];
          float v929_data = s1[12];
          float v931_data = ir3[0];
          ir3[0] = (v931_data + (v928_data * v929_data));
          float v934_data = s1[25];
          float v936_data = ir3[1];
          ir3[1] = (v936_data + (v928_data * v934_data));
          float v939_data = s1[38];
          float v941_data = ir3[2];
          ir3[2] = (v941_data + (v928_data * v939_data));
          float v944_data = s1[51];
          float v946_data = ir3[3];
          ir3[3] = (v946_data + (v928_data * v944_data));
          float v949_data = s1[64];
          float v951_data = ir3[4];
          ir3[4] = (v951_data + (v928_data * v949_data));
          float v954_data = s1[77];
          float v956_data = ir3[5];
          ir3[5] = (v956_data + (v928_data * v954_data));
          float v959_data = s1[90];
          float v961_data = ir3[6];
          ir3[6] = (v961_data + (v928_data * v959_data));
          float v964_data = s1[103];
          float v966_data = ir3[7];
          ir3[7] = (v966_data + (v928_data * v964_data));
          float v969_data = s1[116];
          float v971_data = ir3[8];
          ir3[8] = (v971_data + (v928_data * v969_data));
          float v974_data = s1[129];
          float v976_data = ir3[9];
          ir3[9] = (v976_data + (v928_data * v974_data));
          float v979_data = s1[142];
          float v981_data = ir3[10];
          ir3[10] = (v981_data + (v928_data * v979_data));
          float v984_data = s1[155];
          float v986_data = ir3[11];
          ir3[11] = (v986_data + (v928_data * v984_data));
          float v989_data = s1[168];
          float v991_data = ir3[12];
          ir3[12] = (v991_data + (v928_data * v989_data));
          #pragma unroll
          for (int32_t v996_n0 = 0; v996_n0 < 1; ++v996_n0) {
            #pragma unroll
            for (int32_t v997_n1 = 0; v997_n1 < 13; ++v997_n1) {
              int32_t v998_a = v996_n0 + v997_n1;
              int32_t v999_a = v996_n0 + v997_n1;
              float v1000_data = ir3[v999_a];
              r3[v999_a] = v1000_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v1005_i0 = 0; v1005_i0 < 1; ++v1005_i0) {
            int32_t v1014_lead = v8_lead + (v1005_i0 * 32);
            #pragma unroll
            for (int32_t v1006_i1 = 0; v1006_i1 < 13; ++v1006_i1) {
              int32_t v1007_a = v1005_i0 + v1006_i1;
              float v1009_data = r3[(v1005_i0 + v1006_i1)];
              glb_m3[(v1014_lead + (v1006_i1 * 32))] = v1009_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

