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
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 - 10);
              r0[v14_a] = v12_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], cuda::aligned_size_t<4>(4), pipeline);
            }
            if (threadIdx.x < 9) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[1]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 32), (0, 1)] [(10, 13)]
            float ir1[1]{};
            float v18_data = r0[0];
            float v19_data = s0[114];
            float v21_data = ir1[0];
            ir1[0] = (v21_data + (v18_data * v19_data));
            float v26_data = r0[1];
            float v27_data = s0[115];
            float v29_data = ir1[0];
            ir1[0] = (v29_data + (v26_data * v27_data));
            float v34_data = r0[2];
            float v35_data = s0[116];
            float v37_data = ir1[0];
            ir1[0] = (v37_data + (v34_data * v35_data));
            #pragma unroll
            for (int32_t v42_n0 = 0; v42_n0 < 1; ++v42_n0) {
              #pragma unroll
              for (int32_t v43_n1 = 0; v43_n1 < 1; ++v43_n1) {
                int32_t v44_a = v42_n0 + v43_n1;
                int32_t v45_a = v42_n0 + v43_n1;
                float v46_data = ir1[v45_a];
                int32_t v47_a = v42_n0 + v43_n1;
                r1[v45_a] = v46_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v51_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v52_i0 = 0; v52_i0 < 1; ++v52_i0) {
            int32_t v61_lead = v51_lead + (v52_i0 * 32);
            #pragma unroll
            for (int32_t v53_i1 = 0; v53_i1 < 1; ++v53_i1) {
              int32_t v54_a = v52_i0 + v53_i1;
              float v56_data = r1[(v52_i0 + v53_i1)];
              int32_t v64_a = v61_lead + ((v53_i1 + 8) * 32);
              glb_m0[v64_a] = v56_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          int32_t v67_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v74_lead = v67_lead + (v68_i0 * 32);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 13; ++v69_i1) {
              int32_t v76_a = v74_lead + (v69_i1 * 32);
              float v77_data;
              {
                v77_data = glb_m0[v76_a];
              }
              int32_t v78_a = v68_i0 + v69_i1;
              r2[v78_a] = v77_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], cuda::aligned_size_t<4>(4), pipeline);
            }
            if (threadIdx.x < 9) {
              cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r3[13]{};
          __syncwarp();
          {
            // r3 = +(r2 * s1) + None
            // [(0, 32), (0, 13)] [(0, 13)]
            float ir3[13]{};
            float v82_data = r2[0];
            float v83_data = s1[0];
            float v85_data = ir3[0];
            ir3[0] = (v85_data + (v82_data * v83_data));
            float v88_data = s1[13];
            float v90_data = ir3[1];
            ir3[1] = (v90_data + (v82_data * v88_data));
            float v93_data = s1[26];
            float v95_data = ir3[2];
            ir3[2] = (v95_data + (v82_data * v93_data));
            float v98_data = s1[39];
            float v100_data = ir3[3];
            ir3[3] = (v100_data + (v82_data * v98_data));
            float v103_data = s1[52];
            float v105_data = ir3[4];
            ir3[4] = (v105_data + (v82_data * v103_data));
            float v108_data = s1[65];
            float v110_data = ir3[5];
            ir3[5] = (v110_data + (v82_data * v108_data));
            float v113_data = s1[78];
            float v115_data = ir3[6];
            ir3[6] = (v115_data + (v82_data * v113_data));
            float v118_data = s1[91];
            float v120_data = ir3[7];
            ir3[7] = (v120_data + (v82_data * v118_data));
            float v123_data = s1[104];
            float v125_data = ir3[8];
            ir3[8] = (v125_data + (v82_data * v123_data));
            float v128_data = s1[117];
            float v130_data = ir3[9];
            ir3[9] = (v130_data + (v82_data * v128_data));
            float v133_data = s1[130];
            float v135_data = ir3[10];
            ir3[10] = (v135_data + (v82_data * v133_data));
            float v138_data = s1[143];
            float v140_data = ir3[11];
            ir3[11] = (v140_data + (v82_data * v138_data));
            float v143_data = s1[156];
            float v145_data = ir3[12];
            ir3[12] = (v145_data + (v82_data * v143_data));
            float v150_data = r2[1];
            float v151_data = s1[1];
            float v153_data = ir3[0];
            ir3[0] = (v153_data + (v150_data * v151_data));
            float v156_data = s1[14];
            float v158_data = ir3[1];
            ir3[1] = (v158_data + (v150_data * v156_data));
            float v161_data = s1[27];
            float v163_data = ir3[2];
            ir3[2] = (v163_data + (v150_data * v161_data));
            float v166_data = s1[40];
            float v168_data = ir3[3];
            ir3[3] = (v168_data + (v150_data * v166_data));
            float v171_data = s1[53];
            float v173_data = ir3[4];
            ir3[4] = (v173_data + (v150_data * v171_data));
            float v176_data = s1[66];
            float v178_data = ir3[5];
            ir3[5] = (v178_data + (v150_data * v176_data));
            float v181_data = s1[79];
            float v183_data = ir3[6];
            ir3[6] = (v183_data + (v150_data * v181_data));
            float v186_data = s1[92];
            float v188_data = ir3[7];
            ir3[7] = (v188_data + (v150_data * v186_data));
            float v191_data = s1[105];
            float v193_data = ir3[8];
            ir3[8] = (v193_data + (v150_data * v191_data));
            float v196_data = s1[118];
            float v198_data = ir3[9];
            ir3[9] = (v198_data + (v150_data * v196_data));
            float v201_data = s1[131];
            float v203_data = ir3[10];
            ir3[10] = (v203_data + (v150_data * v201_data));
            float v206_data = s1[144];
            float v208_data = ir3[11];
            ir3[11] = (v208_data + (v150_data * v206_data));
            float v211_data = s1[157];
            float v213_data = ir3[12];
            ir3[12] = (v213_data + (v150_data * v211_data));
            float v218_data = r2[2];
            float v219_data = s1[2];
            float v221_data = ir3[0];
            ir3[0] = (v221_data + (v218_data * v219_data));
            float v224_data = s1[15];
            float v226_data = ir3[1];
            ir3[1] = (v226_data + (v218_data * v224_data));
            float v229_data = s1[28];
            float v231_data = ir3[2];
            ir3[2] = (v231_data + (v218_data * v229_data));
            float v234_data = s1[41];
            float v236_data = ir3[3];
            ir3[3] = (v236_data + (v218_data * v234_data));
            float v239_data = s1[54];
            float v241_data = ir3[4];
            ir3[4] = (v241_data + (v218_data * v239_data));
            float v244_data = s1[67];
            float v246_data = ir3[5];
            ir3[5] = (v246_data + (v218_data * v244_data));
            float v249_data = s1[80];
            float v251_data = ir3[6];
            ir3[6] = (v251_data + (v218_data * v249_data));
            float v254_data = s1[93];
            float v256_data = ir3[7];
            ir3[7] = (v256_data + (v218_data * v254_data));
            float v259_data = s1[106];
            float v261_data = ir3[8];
            ir3[8] = (v261_data + (v218_data * v259_data));
            float v264_data = s1[119];
            float v266_data = ir3[9];
            ir3[9] = (v266_data + (v218_data * v264_data));
            float v269_data = s1[132];
            float v271_data = ir3[10];
            ir3[10] = (v271_data + (v218_data * v269_data));
            float v274_data = s1[145];
            float v276_data = ir3[11];
            ir3[11] = (v276_data + (v218_data * v274_data));
            float v279_data = s1[158];
            float v281_data = ir3[12];
            ir3[12] = (v281_data + (v218_data * v279_data));
            float v286_data = r2[3];
            float v287_data = s1[3];
            float v289_data = ir3[0];
            ir3[0] = (v289_data + (v286_data * v287_data));
            float v292_data = s1[16];
            float v294_data = ir3[1];
            ir3[1] = (v294_data + (v286_data * v292_data));
            float v297_data = s1[29];
            float v299_data = ir3[2];
            ir3[2] = (v299_data + (v286_data * v297_data));
            float v302_data = s1[42];
            float v304_data = ir3[3];
            ir3[3] = (v304_data + (v286_data * v302_data));
            float v307_data = s1[55];
            float v309_data = ir3[4];
            ir3[4] = (v309_data + (v286_data * v307_data));
            float v312_data = s1[68];
            float v314_data = ir3[5];
            ir3[5] = (v314_data + (v286_data * v312_data));
            float v317_data = s1[81];
            float v319_data = ir3[6];
            ir3[6] = (v319_data + (v286_data * v317_data));
            float v322_data = s1[94];
            float v324_data = ir3[7];
            ir3[7] = (v324_data + (v286_data * v322_data));
            float v327_data = s1[107];
            float v329_data = ir3[8];
            ir3[8] = (v329_data + (v286_data * v327_data));
            float v332_data = s1[120];
            float v334_data = ir3[9];
            ir3[9] = (v334_data + (v286_data * v332_data));
            float v337_data = s1[133];
            float v339_data = ir3[10];
            ir3[10] = (v339_data + (v286_data * v337_data));
            float v342_data = s1[146];
            float v344_data = ir3[11];
            ir3[11] = (v344_data + (v286_data * v342_data));
            float v347_data = s1[159];
            float v349_data = ir3[12];
            ir3[12] = (v349_data + (v286_data * v347_data));
            float v354_data = r2[4];
            float v355_data = s1[4];
            float v357_data = ir3[0];
            ir3[0] = (v357_data + (v354_data * v355_data));
            float v360_data = s1[17];
            float v362_data = ir3[1];
            ir3[1] = (v362_data + (v354_data * v360_data));
            float v365_data = s1[30];
            float v367_data = ir3[2];
            ir3[2] = (v367_data + (v354_data * v365_data));
            float v370_data = s1[43];
            float v372_data = ir3[3];
            ir3[3] = (v372_data + (v354_data * v370_data));
            float v375_data = s1[56];
            float v377_data = ir3[4];
            ir3[4] = (v377_data + (v354_data * v375_data));
            float v380_data = s1[69];
            float v382_data = ir3[5];
            ir3[5] = (v382_data + (v354_data * v380_data));
            float v385_data = s1[82];
            float v387_data = ir3[6];
            ir3[6] = (v387_data + (v354_data * v385_data));
            float v390_data = s1[95];
            float v392_data = ir3[7];
            ir3[7] = (v392_data + (v354_data * v390_data));
            float v395_data = s1[108];
            float v397_data = ir3[8];
            ir3[8] = (v397_data + (v354_data * v395_data));
            float v400_data = s1[121];
            float v402_data = ir3[9];
            ir3[9] = (v402_data + (v354_data * v400_data));
            float v405_data = s1[134];
            float v407_data = ir3[10];
            ir3[10] = (v407_data + (v354_data * v405_data));
            float v410_data = s1[147];
            float v412_data = ir3[11];
            ir3[11] = (v412_data + (v354_data * v410_data));
            float v415_data = s1[160];
            float v417_data = ir3[12];
            ir3[12] = (v417_data + (v354_data * v415_data));
            float v422_data = r2[5];
            float v423_data = s1[5];
            float v425_data = ir3[0];
            ir3[0] = (v425_data + (v422_data * v423_data));
            float v428_data = s1[18];
            float v430_data = ir3[1];
            ir3[1] = (v430_data + (v422_data * v428_data));
            float v433_data = s1[31];
            float v435_data = ir3[2];
            ir3[2] = (v435_data + (v422_data * v433_data));
            float v438_data = s1[44];
            float v440_data = ir3[3];
            ir3[3] = (v440_data + (v422_data * v438_data));
            float v443_data = s1[57];
            float v445_data = ir3[4];
            ir3[4] = (v445_data + (v422_data * v443_data));
            float v448_data = s1[70];
            float v450_data = ir3[5];
            ir3[5] = (v450_data + (v422_data * v448_data));
            float v453_data = s1[83];
            float v455_data = ir3[6];
            ir3[6] = (v455_data + (v422_data * v453_data));
            float v458_data = s1[96];
            float v460_data = ir3[7];
            ir3[7] = (v460_data + (v422_data * v458_data));
            float v463_data = s1[109];
            float v465_data = ir3[8];
            ir3[8] = (v465_data + (v422_data * v463_data));
            float v468_data = s1[122];
            float v470_data = ir3[9];
            ir3[9] = (v470_data + (v422_data * v468_data));
            float v473_data = s1[135];
            float v475_data = ir3[10];
            ir3[10] = (v475_data + (v422_data * v473_data));
            float v478_data = s1[148];
            float v480_data = ir3[11];
            ir3[11] = (v480_data + (v422_data * v478_data));
            float v483_data = s1[161];
            float v485_data = ir3[12];
            ir3[12] = (v485_data + (v422_data * v483_data));
            float v490_data = r2[6];
            float v491_data = s1[6];
            float v493_data = ir3[0];
            ir3[0] = (v493_data + (v490_data * v491_data));
            float v496_data = s1[19];
            float v498_data = ir3[1];
            ir3[1] = (v498_data + (v490_data * v496_data));
            float v501_data = s1[32];
            float v503_data = ir3[2];
            ir3[2] = (v503_data + (v490_data * v501_data));
            float v506_data = s1[45];
            float v508_data = ir3[3];
            ir3[3] = (v508_data + (v490_data * v506_data));
            float v511_data = s1[58];
            float v513_data = ir3[4];
            ir3[4] = (v513_data + (v490_data * v511_data));
            float v516_data = s1[71];
            float v518_data = ir3[5];
            ir3[5] = (v518_data + (v490_data * v516_data));
            float v521_data = s1[84];
            float v523_data = ir3[6];
            ir3[6] = (v523_data + (v490_data * v521_data));
            float v526_data = s1[97];
            float v528_data = ir3[7];
            ir3[7] = (v528_data + (v490_data * v526_data));
            float v531_data = s1[110];
            float v533_data = ir3[8];
            ir3[8] = (v533_data + (v490_data * v531_data));
            float v536_data = s1[123];
            float v538_data = ir3[9];
            ir3[9] = (v538_data + (v490_data * v536_data));
            float v541_data = s1[136];
            float v543_data = ir3[10];
            ir3[10] = (v543_data + (v490_data * v541_data));
            float v546_data = s1[149];
            float v548_data = ir3[11];
            ir3[11] = (v548_data + (v490_data * v546_data));
            float v551_data = s1[162];
            float v553_data = ir3[12];
            ir3[12] = (v553_data + (v490_data * v551_data));
            float v558_data = r2[7];
            float v559_data = s1[7];
            float v561_data = ir3[0];
            ir3[0] = (v561_data + (v558_data * v559_data));
            float v564_data = s1[20];
            float v566_data = ir3[1];
            ir3[1] = (v566_data + (v558_data * v564_data));
            float v569_data = s1[33];
            float v571_data = ir3[2];
            ir3[2] = (v571_data + (v558_data * v569_data));
            float v574_data = s1[46];
            float v576_data = ir3[3];
            ir3[3] = (v576_data + (v558_data * v574_data));
            float v579_data = s1[59];
            float v581_data = ir3[4];
            ir3[4] = (v581_data + (v558_data * v579_data));
            float v584_data = s1[72];
            float v586_data = ir3[5];
            ir3[5] = (v586_data + (v558_data * v584_data));
            float v589_data = s1[85];
            float v591_data = ir3[6];
            ir3[6] = (v591_data + (v558_data * v589_data));
            float v594_data = s1[98];
            float v596_data = ir3[7];
            ir3[7] = (v596_data + (v558_data * v594_data));
            float v599_data = s1[111];
            float v601_data = ir3[8];
            ir3[8] = (v601_data + (v558_data * v599_data));
            float v604_data = s1[124];
            float v606_data = ir3[9];
            ir3[9] = (v606_data + (v558_data * v604_data));
            float v609_data = s1[137];
            float v611_data = ir3[10];
            ir3[10] = (v611_data + (v558_data * v609_data));
            float v614_data = s1[150];
            float v616_data = ir3[11];
            ir3[11] = (v616_data + (v558_data * v614_data));
            float v619_data = s1[163];
            float v621_data = ir3[12];
            ir3[12] = (v621_data + (v558_data * v619_data));
            float v626_data = r2[8];
            float v627_data = s1[8];
            float v629_data = ir3[0];
            ir3[0] = (v629_data + (v626_data * v627_data));
            float v632_data = s1[21];
            float v634_data = ir3[1];
            ir3[1] = (v634_data + (v626_data * v632_data));
            float v637_data = s1[34];
            float v639_data = ir3[2];
            ir3[2] = (v639_data + (v626_data * v637_data));
            float v642_data = s1[47];
            float v644_data = ir3[3];
            ir3[3] = (v644_data + (v626_data * v642_data));
            float v647_data = s1[60];
            float v649_data = ir3[4];
            ir3[4] = (v649_data + (v626_data * v647_data));
            float v652_data = s1[73];
            float v654_data = ir3[5];
            ir3[5] = (v654_data + (v626_data * v652_data));
            float v657_data = s1[86];
            float v659_data = ir3[6];
            ir3[6] = (v659_data + (v626_data * v657_data));
            float v662_data = s1[99];
            float v664_data = ir3[7];
            ir3[7] = (v664_data + (v626_data * v662_data));
            float v667_data = s1[112];
            float v669_data = ir3[8];
            ir3[8] = (v669_data + (v626_data * v667_data));
            float v672_data = s1[125];
            float v674_data = ir3[9];
            ir3[9] = (v674_data + (v626_data * v672_data));
            float v677_data = s1[138];
            float v679_data = ir3[10];
            ir3[10] = (v679_data + (v626_data * v677_data));
            float v682_data = s1[151];
            float v684_data = ir3[11];
            ir3[11] = (v684_data + (v626_data * v682_data));
            float v687_data = s1[164];
            float v689_data = ir3[12];
            ir3[12] = (v689_data + (v626_data * v687_data));
            float v694_data = r2[9];
            float v695_data = s1[9];
            float v697_data = ir3[0];
            ir3[0] = (v697_data + (v694_data * v695_data));
            float v700_data = s1[22];
            float v702_data = ir3[1];
            ir3[1] = (v702_data + (v694_data * v700_data));
            float v705_data = s1[35];
            float v707_data = ir3[2];
            ir3[2] = (v707_data + (v694_data * v705_data));
            float v710_data = s1[48];
            float v712_data = ir3[3];
            ir3[3] = (v712_data + (v694_data * v710_data));
            float v715_data = s1[61];
            float v717_data = ir3[4];
            ir3[4] = (v717_data + (v694_data * v715_data));
            float v720_data = s1[74];
            float v722_data = ir3[5];
            ir3[5] = (v722_data + (v694_data * v720_data));
            float v725_data = s1[87];
            float v727_data = ir3[6];
            ir3[6] = (v727_data + (v694_data * v725_data));
            float v730_data = s1[100];
            float v732_data = ir3[7];
            ir3[7] = (v732_data + (v694_data * v730_data));
            float v735_data = s1[113];
            float v737_data = ir3[8];
            ir3[8] = (v737_data + (v694_data * v735_data));
            float v740_data = s1[126];
            float v742_data = ir3[9];
            ir3[9] = (v742_data + (v694_data * v740_data));
            float v745_data = s1[139];
            float v747_data = ir3[10];
            ir3[10] = (v747_data + (v694_data * v745_data));
            float v750_data = s1[152];
            float v752_data = ir3[11];
            ir3[11] = (v752_data + (v694_data * v750_data));
            float v755_data = s1[165];
            float v757_data = ir3[12];
            ir3[12] = (v757_data + (v694_data * v755_data));
            float v762_data = r2[10];
            float v763_data = s1[10];
            float v765_data = ir3[0];
            ir3[0] = (v765_data + (v762_data * v763_data));
            float v768_data = s1[23];
            float v770_data = ir3[1];
            ir3[1] = (v770_data + (v762_data * v768_data));
            float v773_data = s1[36];
            float v775_data = ir3[2];
            ir3[2] = (v775_data + (v762_data * v773_data));
            float v778_data = s1[49];
            float v780_data = ir3[3];
            ir3[3] = (v780_data + (v762_data * v778_data));
            float v783_data = s1[62];
            float v785_data = ir3[4];
            ir3[4] = (v785_data + (v762_data * v783_data));
            float v788_data = s1[75];
            float v790_data = ir3[5];
            ir3[5] = (v790_data + (v762_data * v788_data));
            float v793_data = s1[88];
            float v795_data = ir3[6];
            ir3[6] = (v795_data + (v762_data * v793_data));
            float v798_data = s1[101];
            float v800_data = ir3[7];
            ir3[7] = (v800_data + (v762_data * v798_data));
            float v803_data = s1[114];
            float v805_data = ir3[8];
            ir3[8] = (v805_data + (v762_data * v803_data));
            float v808_data = s1[127];
            float v810_data = ir3[9];
            ir3[9] = (v810_data + (v762_data * v808_data));
            float v813_data = s1[140];
            float v815_data = ir3[10];
            ir3[10] = (v815_data + (v762_data * v813_data));
            float v818_data = s1[153];
            float v820_data = ir3[11];
            ir3[11] = (v820_data + (v762_data * v818_data));
            float v823_data = s1[166];
            float v825_data = ir3[12];
            ir3[12] = (v825_data + (v762_data * v823_data));
            float v830_data = r2[11];
            float v831_data = s1[11];
            float v833_data = ir3[0];
            ir3[0] = (v833_data + (v830_data * v831_data));
            float v836_data = s1[24];
            float v838_data = ir3[1];
            ir3[1] = (v838_data + (v830_data * v836_data));
            float v841_data = s1[37];
            float v843_data = ir3[2];
            ir3[2] = (v843_data + (v830_data * v841_data));
            float v846_data = s1[50];
            float v848_data = ir3[3];
            ir3[3] = (v848_data + (v830_data * v846_data));
            float v851_data = s1[63];
            float v853_data = ir3[4];
            ir3[4] = (v853_data + (v830_data * v851_data));
            float v856_data = s1[76];
            float v858_data = ir3[5];
            ir3[5] = (v858_data + (v830_data * v856_data));
            float v861_data = s1[89];
            float v863_data = ir3[6];
            ir3[6] = (v863_data + (v830_data * v861_data));
            float v866_data = s1[102];
            float v868_data = ir3[7];
            ir3[7] = (v868_data + (v830_data * v866_data));
            float v871_data = s1[115];
            float v873_data = ir3[8];
            ir3[8] = (v873_data + (v830_data * v871_data));
            float v876_data = s1[128];
            float v878_data = ir3[9];
            ir3[9] = (v878_data + (v830_data * v876_data));
            float v881_data = s1[141];
            float v883_data = ir3[10];
            ir3[10] = (v883_data + (v830_data * v881_data));
            float v886_data = s1[154];
            float v888_data = ir3[11];
            ir3[11] = (v888_data + (v830_data * v886_data));
            float v891_data = s1[167];
            float v893_data = ir3[12];
            ir3[12] = (v893_data + (v830_data * v891_data));
            float v898_data = r2[12];
            float v899_data = s1[12];
            float v901_data = ir3[0];
            ir3[0] = (v901_data + (v898_data * v899_data));
            float v904_data = s1[25];
            float v906_data = ir3[1];
            ir3[1] = (v906_data + (v898_data * v904_data));
            float v909_data = s1[38];
            float v911_data = ir3[2];
            ir3[2] = (v911_data + (v898_data * v909_data));
            float v914_data = s1[51];
            float v916_data = ir3[3];
            ir3[3] = (v916_data + (v898_data * v914_data));
            float v919_data = s1[64];
            float v921_data = ir3[4];
            ir3[4] = (v921_data + (v898_data * v919_data));
            float v924_data = s1[77];
            float v926_data = ir3[5];
            ir3[5] = (v926_data + (v898_data * v924_data));
            float v929_data = s1[90];
            float v931_data = ir3[6];
            ir3[6] = (v931_data + (v898_data * v929_data));
            float v934_data = s1[103];
            float v936_data = ir3[7];
            ir3[7] = (v936_data + (v898_data * v934_data));
            float v939_data = s1[116];
            float v941_data = ir3[8];
            ir3[8] = (v941_data + (v898_data * v939_data));
            float v944_data = s1[129];
            float v946_data = ir3[9];
            ir3[9] = (v946_data + (v898_data * v944_data));
            float v949_data = s1[142];
            float v951_data = ir3[10];
            ir3[10] = (v951_data + (v898_data * v949_data));
            float v954_data = s1[155];
            float v956_data = ir3[11];
            ir3[11] = (v956_data + (v898_data * v954_data));
            float v959_data = s1[168];
            float v961_data = ir3[12];
            ir3[12] = (v961_data + (v898_data * v959_data));
            #pragma unroll
            for (int32_t v966_n0 = 0; v966_n0 < 1; ++v966_n0) {
              #pragma unroll
              for (int32_t v967_n1 = 0; v967_n1 < 13; ++v967_n1) {
                int32_t v968_a = v966_n0 + v967_n1;
                int32_t v969_a = v966_n0 + v967_n1;
                float v970_data = ir3[v969_a];
                int32_t v971_a = v966_n0 + v967_n1;
                r3[v969_a] = v970_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r3);
          int32_t v975_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v976_i0 = 0; v976_i0 < 1; ++v976_i0) {
            int32_t v985_lead = v975_lead + (v976_i0 * 32);
            #pragma unroll
            for (int32_t v977_i1 = 0; v977_i1 < 13; ++v977_i1) {
              int32_t v978_a = v976_i0 + v977_i1;
              float v980_data = r3[(v976_i0 + v977_i1)];
              int32_t v987_a = v985_lead + (v977_i1 * 32);
              glb_m3[v987_a] = v980_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

