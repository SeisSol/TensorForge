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
                float v45_data = ir1[v44_a];
                int32_t v46_a = v42_n0 + v43_n1;
                r1[v46_a] = v45_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v49_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v50_i0 = 0; v50_i0 < 1; ++v50_i0) {
            int32_t v58_lead = v49_lead + (v50_i0 * 32);
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 1; ++v51_i1) {
              int32_t v52_a = v50_i0 + v51_i1;
              float v53_data = r1[v52_a];
              int32_t v61_a = v58_lead + ((v51_i1 + 8) * 32);
              glb_m0[v61_a] = v53_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          int32_t v64_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v65_i0 = 0; v65_i0 < 1; ++v65_i0) {
            int32_t v71_lead = v64_lead + (v65_i0 * 32);
            #pragma unroll
            for (int32_t v66_i1 = 0; v66_i1 < 13; ++v66_i1) {
              int32_t v73_a = v71_lead + (v66_i1 * 32);
              float v74_data;
              {
                v74_data = glb_m0[v73_a];
              }
              int32_t v75_a = v65_i0 + v66_i1;
              r2[v75_a] = v74_data;
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
            float v79_data = r2[0];
            float v80_data = s1[0];
            float v82_data = ir3[0];
            ir3[0] = (v82_data + (v79_data * v80_data));
            float v85_data = s1[13];
            float v87_data = ir3[1];
            ir3[1] = (v87_data + (v79_data * v85_data));
            float v90_data = s1[26];
            float v92_data = ir3[2];
            ir3[2] = (v92_data + (v79_data * v90_data));
            float v95_data = s1[39];
            float v97_data = ir3[3];
            ir3[3] = (v97_data + (v79_data * v95_data));
            float v100_data = s1[52];
            float v102_data = ir3[4];
            ir3[4] = (v102_data + (v79_data * v100_data));
            float v105_data = s1[65];
            float v107_data = ir3[5];
            ir3[5] = (v107_data + (v79_data * v105_data));
            float v110_data = s1[78];
            float v112_data = ir3[6];
            ir3[6] = (v112_data + (v79_data * v110_data));
            float v115_data = s1[91];
            float v117_data = ir3[7];
            ir3[7] = (v117_data + (v79_data * v115_data));
            float v120_data = s1[104];
            float v122_data = ir3[8];
            ir3[8] = (v122_data + (v79_data * v120_data));
            float v125_data = s1[117];
            float v127_data = ir3[9];
            ir3[9] = (v127_data + (v79_data * v125_data));
            float v130_data = s1[130];
            float v132_data = ir3[10];
            ir3[10] = (v132_data + (v79_data * v130_data));
            float v135_data = s1[143];
            float v137_data = ir3[11];
            ir3[11] = (v137_data + (v79_data * v135_data));
            float v140_data = s1[156];
            float v142_data = ir3[12];
            ir3[12] = (v142_data + (v79_data * v140_data));
            float v147_data = r2[1];
            float v148_data = s1[1];
            float v150_data = ir3[0];
            ir3[0] = (v150_data + (v147_data * v148_data));
            float v153_data = s1[14];
            float v155_data = ir3[1];
            ir3[1] = (v155_data + (v147_data * v153_data));
            float v158_data = s1[27];
            float v160_data = ir3[2];
            ir3[2] = (v160_data + (v147_data * v158_data));
            float v163_data = s1[40];
            float v165_data = ir3[3];
            ir3[3] = (v165_data + (v147_data * v163_data));
            float v168_data = s1[53];
            float v170_data = ir3[4];
            ir3[4] = (v170_data + (v147_data * v168_data));
            float v173_data = s1[66];
            float v175_data = ir3[5];
            ir3[5] = (v175_data + (v147_data * v173_data));
            float v178_data = s1[79];
            float v180_data = ir3[6];
            ir3[6] = (v180_data + (v147_data * v178_data));
            float v183_data = s1[92];
            float v185_data = ir3[7];
            ir3[7] = (v185_data + (v147_data * v183_data));
            float v188_data = s1[105];
            float v190_data = ir3[8];
            ir3[8] = (v190_data + (v147_data * v188_data));
            float v193_data = s1[118];
            float v195_data = ir3[9];
            ir3[9] = (v195_data + (v147_data * v193_data));
            float v198_data = s1[131];
            float v200_data = ir3[10];
            ir3[10] = (v200_data + (v147_data * v198_data));
            float v203_data = s1[144];
            float v205_data = ir3[11];
            ir3[11] = (v205_data + (v147_data * v203_data));
            float v208_data = s1[157];
            float v210_data = ir3[12];
            ir3[12] = (v210_data + (v147_data * v208_data));
            float v215_data = r2[2];
            float v216_data = s1[2];
            float v218_data = ir3[0];
            ir3[0] = (v218_data + (v215_data * v216_data));
            float v221_data = s1[15];
            float v223_data = ir3[1];
            ir3[1] = (v223_data + (v215_data * v221_data));
            float v226_data = s1[28];
            float v228_data = ir3[2];
            ir3[2] = (v228_data + (v215_data * v226_data));
            float v231_data = s1[41];
            float v233_data = ir3[3];
            ir3[3] = (v233_data + (v215_data * v231_data));
            float v236_data = s1[54];
            float v238_data = ir3[4];
            ir3[4] = (v238_data + (v215_data * v236_data));
            float v241_data = s1[67];
            float v243_data = ir3[5];
            ir3[5] = (v243_data + (v215_data * v241_data));
            float v246_data = s1[80];
            float v248_data = ir3[6];
            ir3[6] = (v248_data + (v215_data * v246_data));
            float v251_data = s1[93];
            float v253_data = ir3[7];
            ir3[7] = (v253_data + (v215_data * v251_data));
            float v256_data = s1[106];
            float v258_data = ir3[8];
            ir3[8] = (v258_data + (v215_data * v256_data));
            float v261_data = s1[119];
            float v263_data = ir3[9];
            ir3[9] = (v263_data + (v215_data * v261_data));
            float v266_data = s1[132];
            float v268_data = ir3[10];
            ir3[10] = (v268_data + (v215_data * v266_data));
            float v271_data = s1[145];
            float v273_data = ir3[11];
            ir3[11] = (v273_data + (v215_data * v271_data));
            float v276_data = s1[158];
            float v278_data = ir3[12];
            ir3[12] = (v278_data + (v215_data * v276_data));
            float v283_data = r2[3];
            float v284_data = s1[3];
            float v286_data = ir3[0];
            ir3[0] = (v286_data + (v283_data * v284_data));
            float v289_data = s1[16];
            float v291_data = ir3[1];
            ir3[1] = (v291_data + (v283_data * v289_data));
            float v294_data = s1[29];
            float v296_data = ir3[2];
            ir3[2] = (v296_data + (v283_data * v294_data));
            float v299_data = s1[42];
            float v301_data = ir3[3];
            ir3[3] = (v301_data + (v283_data * v299_data));
            float v304_data = s1[55];
            float v306_data = ir3[4];
            ir3[4] = (v306_data + (v283_data * v304_data));
            float v309_data = s1[68];
            float v311_data = ir3[5];
            ir3[5] = (v311_data + (v283_data * v309_data));
            float v314_data = s1[81];
            float v316_data = ir3[6];
            ir3[6] = (v316_data + (v283_data * v314_data));
            float v319_data = s1[94];
            float v321_data = ir3[7];
            ir3[7] = (v321_data + (v283_data * v319_data));
            float v324_data = s1[107];
            float v326_data = ir3[8];
            ir3[8] = (v326_data + (v283_data * v324_data));
            float v329_data = s1[120];
            float v331_data = ir3[9];
            ir3[9] = (v331_data + (v283_data * v329_data));
            float v334_data = s1[133];
            float v336_data = ir3[10];
            ir3[10] = (v336_data + (v283_data * v334_data));
            float v339_data = s1[146];
            float v341_data = ir3[11];
            ir3[11] = (v341_data + (v283_data * v339_data));
            float v344_data = s1[159];
            float v346_data = ir3[12];
            ir3[12] = (v346_data + (v283_data * v344_data));
            float v351_data = r2[4];
            float v352_data = s1[4];
            float v354_data = ir3[0];
            ir3[0] = (v354_data + (v351_data * v352_data));
            float v357_data = s1[17];
            float v359_data = ir3[1];
            ir3[1] = (v359_data + (v351_data * v357_data));
            float v362_data = s1[30];
            float v364_data = ir3[2];
            ir3[2] = (v364_data + (v351_data * v362_data));
            float v367_data = s1[43];
            float v369_data = ir3[3];
            ir3[3] = (v369_data + (v351_data * v367_data));
            float v372_data = s1[56];
            float v374_data = ir3[4];
            ir3[4] = (v374_data + (v351_data * v372_data));
            float v377_data = s1[69];
            float v379_data = ir3[5];
            ir3[5] = (v379_data + (v351_data * v377_data));
            float v382_data = s1[82];
            float v384_data = ir3[6];
            ir3[6] = (v384_data + (v351_data * v382_data));
            float v387_data = s1[95];
            float v389_data = ir3[7];
            ir3[7] = (v389_data + (v351_data * v387_data));
            float v392_data = s1[108];
            float v394_data = ir3[8];
            ir3[8] = (v394_data + (v351_data * v392_data));
            float v397_data = s1[121];
            float v399_data = ir3[9];
            ir3[9] = (v399_data + (v351_data * v397_data));
            float v402_data = s1[134];
            float v404_data = ir3[10];
            ir3[10] = (v404_data + (v351_data * v402_data));
            float v407_data = s1[147];
            float v409_data = ir3[11];
            ir3[11] = (v409_data + (v351_data * v407_data));
            float v412_data = s1[160];
            float v414_data = ir3[12];
            ir3[12] = (v414_data + (v351_data * v412_data));
            float v419_data = r2[5];
            float v420_data = s1[5];
            float v422_data = ir3[0];
            ir3[0] = (v422_data + (v419_data * v420_data));
            float v425_data = s1[18];
            float v427_data = ir3[1];
            ir3[1] = (v427_data + (v419_data * v425_data));
            float v430_data = s1[31];
            float v432_data = ir3[2];
            ir3[2] = (v432_data + (v419_data * v430_data));
            float v435_data = s1[44];
            float v437_data = ir3[3];
            ir3[3] = (v437_data + (v419_data * v435_data));
            float v440_data = s1[57];
            float v442_data = ir3[4];
            ir3[4] = (v442_data + (v419_data * v440_data));
            float v445_data = s1[70];
            float v447_data = ir3[5];
            ir3[5] = (v447_data + (v419_data * v445_data));
            float v450_data = s1[83];
            float v452_data = ir3[6];
            ir3[6] = (v452_data + (v419_data * v450_data));
            float v455_data = s1[96];
            float v457_data = ir3[7];
            ir3[7] = (v457_data + (v419_data * v455_data));
            float v460_data = s1[109];
            float v462_data = ir3[8];
            ir3[8] = (v462_data + (v419_data * v460_data));
            float v465_data = s1[122];
            float v467_data = ir3[9];
            ir3[9] = (v467_data + (v419_data * v465_data));
            float v470_data = s1[135];
            float v472_data = ir3[10];
            ir3[10] = (v472_data + (v419_data * v470_data));
            float v475_data = s1[148];
            float v477_data = ir3[11];
            ir3[11] = (v477_data + (v419_data * v475_data));
            float v480_data = s1[161];
            float v482_data = ir3[12];
            ir3[12] = (v482_data + (v419_data * v480_data));
            float v487_data = r2[6];
            float v488_data = s1[6];
            float v490_data = ir3[0];
            ir3[0] = (v490_data + (v487_data * v488_data));
            float v493_data = s1[19];
            float v495_data = ir3[1];
            ir3[1] = (v495_data + (v487_data * v493_data));
            float v498_data = s1[32];
            float v500_data = ir3[2];
            ir3[2] = (v500_data + (v487_data * v498_data));
            float v503_data = s1[45];
            float v505_data = ir3[3];
            ir3[3] = (v505_data + (v487_data * v503_data));
            float v508_data = s1[58];
            float v510_data = ir3[4];
            ir3[4] = (v510_data + (v487_data * v508_data));
            float v513_data = s1[71];
            float v515_data = ir3[5];
            ir3[5] = (v515_data + (v487_data * v513_data));
            float v518_data = s1[84];
            float v520_data = ir3[6];
            ir3[6] = (v520_data + (v487_data * v518_data));
            float v523_data = s1[97];
            float v525_data = ir3[7];
            ir3[7] = (v525_data + (v487_data * v523_data));
            float v528_data = s1[110];
            float v530_data = ir3[8];
            ir3[8] = (v530_data + (v487_data * v528_data));
            float v533_data = s1[123];
            float v535_data = ir3[9];
            ir3[9] = (v535_data + (v487_data * v533_data));
            float v538_data = s1[136];
            float v540_data = ir3[10];
            ir3[10] = (v540_data + (v487_data * v538_data));
            float v543_data = s1[149];
            float v545_data = ir3[11];
            ir3[11] = (v545_data + (v487_data * v543_data));
            float v548_data = s1[162];
            float v550_data = ir3[12];
            ir3[12] = (v550_data + (v487_data * v548_data));
            float v555_data = r2[7];
            float v556_data = s1[7];
            float v558_data = ir3[0];
            ir3[0] = (v558_data + (v555_data * v556_data));
            float v561_data = s1[20];
            float v563_data = ir3[1];
            ir3[1] = (v563_data + (v555_data * v561_data));
            float v566_data = s1[33];
            float v568_data = ir3[2];
            ir3[2] = (v568_data + (v555_data * v566_data));
            float v571_data = s1[46];
            float v573_data = ir3[3];
            ir3[3] = (v573_data + (v555_data * v571_data));
            float v576_data = s1[59];
            float v578_data = ir3[4];
            ir3[4] = (v578_data + (v555_data * v576_data));
            float v581_data = s1[72];
            float v583_data = ir3[5];
            ir3[5] = (v583_data + (v555_data * v581_data));
            float v586_data = s1[85];
            float v588_data = ir3[6];
            ir3[6] = (v588_data + (v555_data * v586_data));
            float v591_data = s1[98];
            float v593_data = ir3[7];
            ir3[7] = (v593_data + (v555_data * v591_data));
            float v596_data = s1[111];
            float v598_data = ir3[8];
            ir3[8] = (v598_data + (v555_data * v596_data));
            float v601_data = s1[124];
            float v603_data = ir3[9];
            ir3[9] = (v603_data + (v555_data * v601_data));
            float v606_data = s1[137];
            float v608_data = ir3[10];
            ir3[10] = (v608_data + (v555_data * v606_data));
            float v611_data = s1[150];
            float v613_data = ir3[11];
            ir3[11] = (v613_data + (v555_data * v611_data));
            float v616_data = s1[163];
            float v618_data = ir3[12];
            ir3[12] = (v618_data + (v555_data * v616_data));
            float v623_data = r2[8];
            float v624_data = s1[8];
            float v626_data = ir3[0];
            ir3[0] = (v626_data + (v623_data * v624_data));
            float v629_data = s1[21];
            float v631_data = ir3[1];
            ir3[1] = (v631_data + (v623_data * v629_data));
            float v634_data = s1[34];
            float v636_data = ir3[2];
            ir3[2] = (v636_data + (v623_data * v634_data));
            float v639_data = s1[47];
            float v641_data = ir3[3];
            ir3[3] = (v641_data + (v623_data * v639_data));
            float v644_data = s1[60];
            float v646_data = ir3[4];
            ir3[4] = (v646_data + (v623_data * v644_data));
            float v649_data = s1[73];
            float v651_data = ir3[5];
            ir3[5] = (v651_data + (v623_data * v649_data));
            float v654_data = s1[86];
            float v656_data = ir3[6];
            ir3[6] = (v656_data + (v623_data * v654_data));
            float v659_data = s1[99];
            float v661_data = ir3[7];
            ir3[7] = (v661_data + (v623_data * v659_data));
            float v664_data = s1[112];
            float v666_data = ir3[8];
            ir3[8] = (v666_data + (v623_data * v664_data));
            float v669_data = s1[125];
            float v671_data = ir3[9];
            ir3[9] = (v671_data + (v623_data * v669_data));
            float v674_data = s1[138];
            float v676_data = ir3[10];
            ir3[10] = (v676_data + (v623_data * v674_data));
            float v679_data = s1[151];
            float v681_data = ir3[11];
            ir3[11] = (v681_data + (v623_data * v679_data));
            float v684_data = s1[164];
            float v686_data = ir3[12];
            ir3[12] = (v686_data + (v623_data * v684_data));
            float v691_data = r2[9];
            float v692_data = s1[9];
            float v694_data = ir3[0];
            ir3[0] = (v694_data + (v691_data * v692_data));
            float v697_data = s1[22];
            float v699_data = ir3[1];
            ir3[1] = (v699_data + (v691_data * v697_data));
            float v702_data = s1[35];
            float v704_data = ir3[2];
            ir3[2] = (v704_data + (v691_data * v702_data));
            float v707_data = s1[48];
            float v709_data = ir3[3];
            ir3[3] = (v709_data + (v691_data * v707_data));
            float v712_data = s1[61];
            float v714_data = ir3[4];
            ir3[4] = (v714_data + (v691_data * v712_data));
            float v717_data = s1[74];
            float v719_data = ir3[5];
            ir3[5] = (v719_data + (v691_data * v717_data));
            float v722_data = s1[87];
            float v724_data = ir3[6];
            ir3[6] = (v724_data + (v691_data * v722_data));
            float v727_data = s1[100];
            float v729_data = ir3[7];
            ir3[7] = (v729_data + (v691_data * v727_data));
            float v732_data = s1[113];
            float v734_data = ir3[8];
            ir3[8] = (v734_data + (v691_data * v732_data));
            float v737_data = s1[126];
            float v739_data = ir3[9];
            ir3[9] = (v739_data + (v691_data * v737_data));
            float v742_data = s1[139];
            float v744_data = ir3[10];
            ir3[10] = (v744_data + (v691_data * v742_data));
            float v747_data = s1[152];
            float v749_data = ir3[11];
            ir3[11] = (v749_data + (v691_data * v747_data));
            float v752_data = s1[165];
            float v754_data = ir3[12];
            ir3[12] = (v754_data + (v691_data * v752_data));
            float v759_data = r2[10];
            float v760_data = s1[10];
            float v762_data = ir3[0];
            ir3[0] = (v762_data + (v759_data * v760_data));
            float v765_data = s1[23];
            float v767_data = ir3[1];
            ir3[1] = (v767_data + (v759_data * v765_data));
            float v770_data = s1[36];
            float v772_data = ir3[2];
            ir3[2] = (v772_data + (v759_data * v770_data));
            float v775_data = s1[49];
            float v777_data = ir3[3];
            ir3[3] = (v777_data + (v759_data * v775_data));
            float v780_data = s1[62];
            float v782_data = ir3[4];
            ir3[4] = (v782_data + (v759_data * v780_data));
            float v785_data = s1[75];
            float v787_data = ir3[5];
            ir3[5] = (v787_data + (v759_data * v785_data));
            float v790_data = s1[88];
            float v792_data = ir3[6];
            ir3[6] = (v792_data + (v759_data * v790_data));
            float v795_data = s1[101];
            float v797_data = ir3[7];
            ir3[7] = (v797_data + (v759_data * v795_data));
            float v800_data = s1[114];
            float v802_data = ir3[8];
            ir3[8] = (v802_data + (v759_data * v800_data));
            float v805_data = s1[127];
            float v807_data = ir3[9];
            ir3[9] = (v807_data + (v759_data * v805_data));
            float v810_data = s1[140];
            float v812_data = ir3[10];
            ir3[10] = (v812_data + (v759_data * v810_data));
            float v815_data = s1[153];
            float v817_data = ir3[11];
            ir3[11] = (v817_data + (v759_data * v815_data));
            float v820_data = s1[166];
            float v822_data = ir3[12];
            ir3[12] = (v822_data + (v759_data * v820_data));
            float v827_data = r2[11];
            float v828_data = s1[11];
            float v830_data = ir3[0];
            ir3[0] = (v830_data + (v827_data * v828_data));
            float v833_data = s1[24];
            float v835_data = ir3[1];
            ir3[1] = (v835_data + (v827_data * v833_data));
            float v838_data = s1[37];
            float v840_data = ir3[2];
            ir3[2] = (v840_data + (v827_data * v838_data));
            float v843_data = s1[50];
            float v845_data = ir3[3];
            ir3[3] = (v845_data + (v827_data * v843_data));
            float v848_data = s1[63];
            float v850_data = ir3[4];
            ir3[4] = (v850_data + (v827_data * v848_data));
            float v853_data = s1[76];
            float v855_data = ir3[5];
            ir3[5] = (v855_data + (v827_data * v853_data));
            float v858_data = s1[89];
            float v860_data = ir3[6];
            ir3[6] = (v860_data + (v827_data * v858_data));
            float v863_data = s1[102];
            float v865_data = ir3[7];
            ir3[7] = (v865_data + (v827_data * v863_data));
            float v868_data = s1[115];
            float v870_data = ir3[8];
            ir3[8] = (v870_data + (v827_data * v868_data));
            float v873_data = s1[128];
            float v875_data = ir3[9];
            ir3[9] = (v875_data + (v827_data * v873_data));
            float v878_data = s1[141];
            float v880_data = ir3[10];
            ir3[10] = (v880_data + (v827_data * v878_data));
            float v883_data = s1[154];
            float v885_data = ir3[11];
            ir3[11] = (v885_data + (v827_data * v883_data));
            float v888_data = s1[167];
            float v890_data = ir3[12];
            ir3[12] = (v890_data + (v827_data * v888_data));
            float v895_data = r2[12];
            float v896_data = s1[12];
            float v898_data = ir3[0];
            ir3[0] = (v898_data + (v895_data * v896_data));
            float v901_data = s1[25];
            float v903_data = ir3[1];
            ir3[1] = (v903_data + (v895_data * v901_data));
            float v906_data = s1[38];
            float v908_data = ir3[2];
            ir3[2] = (v908_data + (v895_data * v906_data));
            float v911_data = s1[51];
            float v913_data = ir3[3];
            ir3[3] = (v913_data + (v895_data * v911_data));
            float v916_data = s1[64];
            float v918_data = ir3[4];
            ir3[4] = (v918_data + (v895_data * v916_data));
            float v921_data = s1[77];
            float v923_data = ir3[5];
            ir3[5] = (v923_data + (v895_data * v921_data));
            float v926_data = s1[90];
            float v928_data = ir3[6];
            ir3[6] = (v928_data + (v895_data * v926_data));
            float v931_data = s1[103];
            float v933_data = ir3[7];
            ir3[7] = (v933_data + (v895_data * v931_data));
            float v936_data = s1[116];
            float v938_data = ir3[8];
            ir3[8] = (v938_data + (v895_data * v936_data));
            float v941_data = s1[129];
            float v943_data = ir3[9];
            ir3[9] = (v943_data + (v895_data * v941_data));
            float v946_data = s1[142];
            float v948_data = ir3[10];
            ir3[10] = (v948_data + (v895_data * v946_data));
            float v951_data = s1[155];
            float v953_data = ir3[11];
            ir3[11] = (v953_data + (v895_data * v951_data));
            float v956_data = s1[168];
            float v958_data = ir3[12];
            ir3[12] = (v958_data + (v895_data * v956_data));
            #pragma unroll
            for (int32_t v963_n0 = 0; v963_n0 < 1; ++v963_n0) {
              #pragma unroll
              for (int32_t v964_n1 = 0; v964_n1 < 13; ++v964_n1) {
                int32_t v965_a = v963_n0 + v964_n1;
                float v966_data = ir3[v965_a];
                int32_t v967_a = v963_n0 + v964_n1;
                r3[v967_a] = v966_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r3);
          int32_t v970_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v971_i0 = 0; v971_i0 < 1; ++v971_i0) {
            int32_t v979_lead = v970_lead + (v971_i0 * 32);
            #pragma unroll
            for (int32_t v972_i1 = 0; v972_i1 < 13; ++v972_i1) {
              int32_t v973_a = v971_i0 + v972_i1;
              float v974_data = r3[v973_a];
              int32_t v981_a = v979_lead + (v972_i1 * 32);
              glb_m3[v981_a] = v974_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

