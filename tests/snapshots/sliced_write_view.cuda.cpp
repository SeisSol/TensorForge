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
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v34_data = r0[0];
          float v35_data = s0[114];
          float v37_data = ir1[0];
          ir1[0] = (v37_data + (v34_data * v35_data));
          float v42_data = r0[1];
          float v43_data = s0[115];
          float v45_data = ir1[0];
          ir1[0] = (v45_data + (v42_data * v43_data));
          float v50_data = r0[2];
          float v51_data = s0[116];
          float v53_data = ir1[0];
          ir1[0] = (v53_data + (v50_data * v51_data));
          #pragma unroll
          for (int32_t v58_n0 = 0; v58_n0 < 1; ++v58_n0) {
            #pragma unroll
            for (int32_t v59_n1 = 0; v59_n1 < 1; ++v59_n1) {
              int32_t v60_a = v58_n0 + v59_n1;
              int32_t v61_a = v58_n0 + v59_n1;
              float v62_data = ir1[v61_a];
              int32_t v63_a = v58_n0 + v59_n1;
              r1[v61_a] = v62_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v77_lead = v8_lead + (v68_i0 * 32);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 1; ++v69_i1) {
              int32_t v70_a = v68_i0 + v69_i1;
              float v72_data = r1[(v68_i0 + v69_i1)];
              int32_t v80_a = v77_lead + ((v69_i1 + 8) * 32);
              glb_m0[v80_a] = v72_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v85_i0 = 0; v85_i0 < 1; ++v85_i0) {
            int32_t v90_lead = v85_i0 * 32;
            int32_t v91_lead = v8_lead + v90_lead;
            int32_t v98_lead = v8_lead + v90_lead;
            #pragma unroll
            for (int32_t v86_i1 = 0; v86_i1 < 13; ++v86_i1) {
              int32_t v92_a = v86_i1 * 32;
              int32_t v93_a = v91_lead + v92_a;
              float v101_data = glb_m0[(v98_lead + v92_a)];
              int32_t v102_a = v85_i0 + v86_i1;
              r2[v102_a] = v101_data;
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
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v109_data = r2[0];
          float v110_data = s1[0];
          float v112_data = ir3[0];
          ir3[0] = (v112_data + (v109_data * v110_data));
          float v115_data = s1[13];
          float v117_data = ir3[1];
          ir3[1] = (v117_data + (v109_data * v115_data));
          float v120_data = s1[26];
          float v122_data = ir3[2];
          ir3[2] = (v122_data + (v109_data * v120_data));
          float v125_data = s1[39];
          float v127_data = ir3[3];
          ir3[3] = (v127_data + (v109_data * v125_data));
          float v130_data = s1[52];
          float v132_data = ir3[4];
          ir3[4] = (v132_data + (v109_data * v130_data));
          float v135_data = s1[65];
          float v137_data = ir3[5];
          ir3[5] = (v137_data + (v109_data * v135_data));
          float v140_data = s1[78];
          float v142_data = ir3[6];
          ir3[6] = (v142_data + (v109_data * v140_data));
          float v145_data = s1[91];
          float v147_data = ir3[7];
          ir3[7] = (v147_data + (v109_data * v145_data));
          float v150_data = s1[104];
          float v152_data = ir3[8];
          ir3[8] = (v152_data + (v109_data * v150_data));
          float v155_data = s1[117];
          float v157_data = ir3[9];
          ir3[9] = (v157_data + (v109_data * v155_data));
          float v160_data = s1[130];
          float v162_data = ir3[10];
          ir3[10] = (v162_data + (v109_data * v160_data));
          float v165_data = s1[143];
          float v167_data = ir3[11];
          ir3[11] = (v167_data + (v109_data * v165_data));
          float v170_data = s1[156];
          float v172_data = ir3[12];
          ir3[12] = (v172_data + (v109_data * v170_data));
          float v177_data = r2[1];
          float v178_data = s1[1];
          float v180_data = ir3[0];
          ir3[0] = (v180_data + (v177_data * v178_data));
          float v183_data = s1[14];
          float v185_data = ir3[1];
          ir3[1] = (v185_data + (v177_data * v183_data));
          float v188_data = s1[27];
          float v190_data = ir3[2];
          ir3[2] = (v190_data + (v177_data * v188_data));
          float v193_data = s1[40];
          float v195_data = ir3[3];
          ir3[3] = (v195_data + (v177_data * v193_data));
          float v198_data = s1[53];
          float v200_data = ir3[4];
          ir3[4] = (v200_data + (v177_data * v198_data));
          float v203_data = s1[66];
          float v205_data = ir3[5];
          ir3[5] = (v205_data + (v177_data * v203_data));
          float v208_data = s1[79];
          float v210_data = ir3[6];
          ir3[6] = (v210_data + (v177_data * v208_data));
          float v213_data = s1[92];
          float v215_data = ir3[7];
          ir3[7] = (v215_data + (v177_data * v213_data));
          float v218_data = s1[105];
          float v220_data = ir3[8];
          ir3[8] = (v220_data + (v177_data * v218_data));
          float v223_data = s1[118];
          float v225_data = ir3[9];
          ir3[9] = (v225_data + (v177_data * v223_data));
          float v228_data = s1[131];
          float v230_data = ir3[10];
          ir3[10] = (v230_data + (v177_data * v228_data));
          float v233_data = s1[144];
          float v235_data = ir3[11];
          ir3[11] = (v235_data + (v177_data * v233_data));
          float v238_data = s1[157];
          float v240_data = ir3[12];
          ir3[12] = (v240_data + (v177_data * v238_data));
          float v245_data = r2[2];
          float v246_data = s1[2];
          float v248_data = ir3[0];
          ir3[0] = (v248_data + (v245_data * v246_data));
          float v251_data = s1[15];
          float v253_data = ir3[1];
          ir3[1] = (v253_data + (v245_data * v251_data));
          float v256_data = s1[28];
          float v258_data = ir3[2];
          ir3[2] = (v258_data + (v245_data * v256_data));
          float v261_data = s1[41];
          float v263_data = ir3[3];
          ir3[3] = (v263_data + (v245_data * v261_data));
          float v266_data = s1[54];
          float v268_data = ir3[4];
          ir3[4] = (v268_data + (v245_data * v266_data));
          float v271_data = s1[67];
          float v273_data = ir3[5];
          ir3[5] = (v273_data + (v245_data * v271_data));
          float v276_data = s1[80];
          float v278_data = ir3[6];
          ir3[6] = (v278_data + (v245_data * v276_data));
          float v281_data = s1[93];
          float v283_data = ir3[7];
          ir3[7] = (v283_data + (v245_data * v281_data));
          float v286_data = s1[106];
          float v288_data = ir3[8];
          ir3[8] = (v288_data + (v245_data * v286_data));
          float v291_data = s1[119];
          float v293_data = ir3[9];
          ir3[9] = (v293_data + (v245_data * v291_data));
          float v296_data = s1[132];
          float v298_data = ir3[10];
          ir3[10] = (v298_data + (v245_data * v296_data));
          float v301_data = s1[145];
          float v303_data = ir3[11];
          ir3[11] = (v303_data + (v245_data * v301_data));
          float v306_data = s1[158];
          float v308_data = ir3[12];
          ir3[12] = (v308_data + (v245_data * v306_data));
          float v313_data = r2[3];
          float v314_data = s1[3];
          float v316_data = ir3[0];
          ir3[0] = (v316_data + (v313_data * v314_data));
          float v319_data = s1[16];
          float v321_data = ir3[1];
          ir3[1] = (v321_data + (v313_data * v319_data));
          float v324_data = s1[29];
          float v326_data = ir3[2];
          ir3[2] = (v326_data + (v313_data * v324_data));
          float v329_data = s1[42];
          float v331_data = ir3[3];
          ir3[3] = (v331_data + (v313_data * v329_data));
          float v334_data = s1[55];
          float v336_data = ir3[4];
          ir3[4] = (v336_data + (v313_data * v334_data));
          float v339_data = s1[68];
          float v341_data = ir3[5];
          ir3[5] = (v341_data + (v313_data * v339_data));
          float v344_data = s1[81];
          float v346_data = ir3[6];
          ir3[6] = (v346_data + (v313_data * v344_data));
          float v349_data = s1[94];
          float v351_data = ir3[7];
          ir3[7] = (v351_data + (v313_data * v349_data));
          float v354_data = s1[107];
          float v356_data = ir3[8];
          ir3[8] = (v356_data + (v313_data * v354_data));
          float v359_data = s1[120];
          float v361_data = ir3[9];
          ir3[9] = (v361_data + (v313_data * v359_data));
          float v364_data = s1[133];
          float v366_data = ir3[10];
          ir3[10] = (v366_data + (v313_data * v364_data));
          float v369_data = s1[146];
          float v371_data = ir3[11];
          ir3[11] = (v371_data + (v313_data * v369_data));
          float v374_data = s1[159];
          float v376_data = ir3[12];
          ir3[12] = (v376_data + (v313_data * v374_data));
          float v381_data = r2[4];
          float v382_data = s1[4];
          float v384_data = ir3[0];
          ir3[0] = (v384_data + (v381_data * v382_data));
          float v387_data = s1[17];
          float v389_data = ir3[1];
          ir3[1] = (v389_data + (v381_data * v387_data));
          float v392_data = s1[30];
          float v394_data = ir3[2];
          ir3[2] = (v394_data + (v381_data * v392_data));
          float v397_data = s1[43];
          float v399_data = ir3[3];
          ir3[3] = (v399_data + (v381_data * v397_data));
          float v402_data = s1[56];
          float v404_data = ir3[4];
          ir3[4] = (v404_data + (v381_data * v402_data));
          float v407_data = s1[69];
          float v409_data = ir3[5];
          ir3[5] = (v409_data + (v381_data * v407_data));
          float v412_data = s1[82];
          float v414_data = ir3[6];
          ir3[6] = (v414_data + (v381_data * v412_data));
          float v417_data = s1[95];
          float v419_data = ir3[7];
          ir3[7] = (v419_data + (v381_data * v417_data));
          float v422_data = s1[108];
          float v424_data = ir3[8];
          ir3[8] = (v424_data + (v381_data * v422_data));
          float v427_data = s1[121];
          float v429_data = ir3[9];
          ir3[9] = (v429_data + (v381_data * v427_data));
          float v432_data = s1[134];
          float v434_data = ir3[10];
          ir3[10] = (v434_data + (v381_data * v432_data));
          float v437_data = s1[147];
          float v439_data = ir3[11];
          ir3[11] = (v439_data + (v381_data * v437_data));
          float v442_data = s1[160];
          float v444_data = ir3[12];
          ir3[12] = (v444_data + (v381_data * v442_data));
          float v449_data = r2[5];
          float v450_data = s1[5];
          float v452_data = ir3[0];
          ir3[0] = (v452_data + (v449_data * v450_data));
          float v455_data = s1[18];
          float v457_data = ir3[1];
          ir3[1] = (v457_data + (v449_data * v455_data));
          float v460_data = s1[31];
          float v462_data = ir3[2];
          ir3[2] = (v462_data + (v449_data * v460_data));
          float v465_data = s1[44];
          float v467_data = ir3[3];
          ir3[3] = (v467_data + (v449_data * v465_data));
          float v470_data = s1[57];
          float v472_data = ir3[4];
          ir3[4] = (v472_data + (v449_data * v470_data));
          float v475_data = s1[70];
          float v477_data = ir3[5];
          ir3[5] = (v477_data + (v449_data * v475_data));
          float v480_data = s1[83];
          float v482_data = ir3[6];
          ir3[6] = (v482_data + (v449_data * v480_data));
          float v485_data = s1[96];
          float v487_data = ir3[7];
          ir3[7] = (v487_data + (v449_data * v485_data));
          float v490_data = s1[109];
          float v492_data = ir3[8];
          ir3[8] = (v492_data + (v449_data * v490_data));
          float v495_data = s1[122];
          float v497_data = ir3[9];
          ir3[9] = (v497_data + (v449_data * v495_data));
          float v500_data = s1[135];
          float v502_data = ir3[10];
          ir3[10] = (v502_data + (v449_data * v500_data));
          float v505_data = s1[148];
          float v507_data = ir3[11];
          ir3[11] = (v507_data + (v449_data * v505_data));
          float v510_data = s1[161];
          float v512_data = ir3[12];
          ir3[12] = (v512_data + (v449_data * v510_data));
          float v517_data = r2[6];
          float v518_data = s1[6];
          float v520_data = ir3[0];
          ir3[0] = (v520_data + (v517_data * v518_data));
          float v523_data = s1[19];
          float v525_data = ir3[1];
          ir3[1] = (v525_data + (v517_data * v523_data));
          float v528_data = s1[32];
          float v530_data = ir3[2];
          ir3[2] = (v530_data + (v517_data * v528_data));
          float v533_data = s1[45];
          float v535_data = ir3[3];
          ir3[3] = (v535_data + (v517_data * v533_data));
          float v538_data = s1[58];
          float v540_data = ir3[4];
          ir3[4] = (v540_data + (v517_data * v538_data));
          float v543_data = s1[71];
          float v545_data = ir3[5];
          ir3[5] = (v545_data + (v517_data * v543_data));
          float v548_data = s1[84];
          float v550_data = ir3[6];
          ir3[6] = (v550_data + (v517_data * v548_data));
          float v553_data = s1[97];
          float v555_data = ir3[7];
          ir3[7] = (v555_data + (v517_data * v553_data));
          float v558_data = s1[110];
          float v560_data = ir3[8];
          ir3[8] = (v560_data + (v517_data * v558_data));
          float v563_data = s1[123];
          float v565_data = ir3[9];
          ir3[9] = (v565_data + (v517_data * v563_data));
          float v568_data = s1[136];
          float v570_data = ir3[10];
          ir3[10] = (v570_data + (v517_data * v568_data));
          float v573_data = s1[149];
          float v575_data = ir3[11];
          ir3[11] = (v575_data + (v517_data * v573_data));
          float v578_data = s1[162];
          float v580_data = ir3[12];
          ir3[12] = (v580_data + (v517_data * v578_data));
          float v585_data = r2[7];
          float v586_data = s1[7];
          float v588_data = ir3[0];
          ir3[0] = (v588_data + (v585_data * v586_data));
          float v591_data = s1[20];
          float v593_data = ir3[1];
          ir3[1] = (v593_data + (v585_data * v591_data));
          float v596_data = s1[33];
          float v598_data = ir3[2];
          ir3[2] = (v598_data + (v585_data * v596_data));
          float v601_data = s1[46];
          float v603_data = ir3[3];
          ir3[3] = (v603_data + (v585_data * v601_data));
          float v606_data = s1[59];
          float v608_data = ir3[4];
          ir3[4] = (v608_data + (v585_data * v606_data));
          float v611_data = s1[72];
          float v613_data = ir3[5];
          ir3[5] = (v613_data + (v585_data * v611_data));
          float v616_data = s1[85];
          float v618_data = ir3[6];
          ir3[6] = (v618_data + (v585_data * v616_data));
          float v621_data = s1[98];
          float v623_data = ir3[7];
          ir3[7] = (v623_data + (v585_data * v621_data));
          float v626_data = s1[111];
          float v628_data = ir3[8];
          ir3[8] = (v628_data + (v585_data * v626_data));
          float v631_data = s1[124];
          float v633_data = ir3[9];
          ir3[9] = (v633_data + (v585_data * v631_data));
          float v636_data = s1[137];
          float v638_data = ir3[10];
          ir3[10] = (v638_data + (v585_data * v636_data));
          float v641_data = s1[150];
          float v643_data = ir3[11];
          ir3[11] = (v643_data + (v585_data * v641_data));
          float v646_data = s1[163];
          float v648_data = ir3[12];
          ir3[12] = (v648_data + (v585_data * v646_data));
          float v653_data = r2[8];
          float v654_data = s1[8];
          float v656_data = ir3[0];
          ir3[0] = (v656_data + (v653_data * v654_data));
          float v659_data = s1[21];
          float v661_data = ir3[1];
          ir3[1] = (v661_data + (v653_data * v659_data));
          float v664_data = s1[34];
          float v666_data = ir3[2];
          ir3[2] = (v666_data + (v653_data * v664_data));
          float v669_data = s1[47];
          float v671_data = ir3[3];
          ir3[3] = (v671_data + (v653_data * v669_data));
          float v674_data = s1[60];
          float v676_data = ir3[4];
          ir3[4] = (v676_data + (v653_data * v674_data));
          float v679_data = s1[73];
          float v681_data = ir3[5];
          ir3[5] = (v681_data + (v653_data * v679_data));
          float v684_data = s1[86];
          float v686_data = ir3[6];
          ir3[6] = (v686_data + (v653_data * v684_data));
          float v689_data = s1[99];
          float v691_data = ir3[7];
          ir3[7] = (v691_data + (v653_data * v689_data));
          float v694_data = s1[112];
          float v696_data = ir3[8];
          ir3[8] = (v696_data + (v653_data * v694_data));
          float v699_data = s1[125];
          float v701_data = ir3[9];
          ir3[9] = (v701_data + (v653_data * v699_data));
          float v704_data = s1[138];
          float v706_data = ir3[10];
          ir3[10] = (v706_data + (v653_data * v704_data));
          float v709_data = s1[151];
          float v711_data = ir3[11];
          ir3[11] = (v711_data + (v653_data * v709_data));
          float v714_data = s1[164];
          float v716_data = ir3[12];
          ir3[12] = (v716_data + (v653_data * v714_data));
          float v721_data = r2[9];
          float v722_data = s1[9];
          float v724_data = ir3[0];
          ir3[0] = (v724_data + (v721_data * v722_data));
          float v727_data = s1[22];
          float v729_data = ir3[1];
          ir3[1] = (v729_data + (v721_data * v727_data));
          float v732_data = s1[35];
          float v734_data = ir3[2];
          ir3[2] = (v734_data + (v721_data * v732_data));
          float v737_data = s1[48];
          float v739_data = ir3[3];
          ir3[3] = (v739_data + (v721_data * v737_data));
          float v742_data = s1[61];
          float v744_data = ir3[4];
          ir3[4] = (v744_data + (v721_data * v742_data));
          float v747_data = s1[74];
          float v749_data = ir3[5];
          ir3[5] = (v749_data + (v721_data * v747_data));
          float v752_data = s1[87];
          float v754_data = ir3[6];
          ir3[6] = (v754_data + (v721_data * v752_data));
          float v757_data = s1[100];
          float v759_data = ir3[7];
          ir3[7] = (v759_data + (v721_data * v757_data));
          float v762_data = s1[113];
          float v764_data = ir3[8];
          ir3[8] = (v764_data + (v721_data * v762_data));
          float v767_data = s1[126];
          float v769_data = ir3[9];
          ir3[9] = (v769_data + (v721_data * v767_data));
          float v772_data = s1[139];
          float v774_data = ir3[10];
          ir3[10] = (v774_data + (v721_data * v772_data));
          float v777_data = s1[152];
          float v779_data = ir3[11];
          ir3[11] = (v779_data + (v721_data * v777_data));
          float v782_data = s1[165];
          float v784_data = ir3[12];
          ir3[12] = (v784_data + (v721_data * v782_data));
          float v789_data = r2[10];
          float v790_data = s1[10];
          float v792_data = ir3[0];
          ir3[0] = (v792_data + (v789_data * v790_data));
          float v795_data = s1[23];
          float v797_data = ir3[1];
          ir3[1] = (v797_data + (v789_data * v795_data));
          float v800_data = s1[36];
          float v802_data = ir3[2];
          ir3[2] = (v802_data + (v789_data * v800_data));
          float v805_data = s1[49];
          float v807_data = ir3[3];
          ir3[3] = (v807_data + (v789_data * v805_data));
          float v810_data = s1[62];
          float v812_data = ir3[4];
          ir3[4] = (v812_data + (v789_data * v810_data));
          float v815_data = s1[75];
          float v817_data = ir3[5];
          ir3[5] = (v817_data + (v789_data * v815_data));
          float v820_data = s1[88];
          float v822_data = ir3[6];
          ir3[6] = (v822_data + (v789_data * v820_data));
          float v825_data = s1[101];
          float v827_data = ir3[7];
          ir3[7] = (v827_data + (v789_data * v825_data));
          float v830_data = s1[114];
          float v832_data = ir3[8];
          ir3[8] = (v832_data + (v789_data * v830_data));
          float v835_data = s1[127];
          float v837_data = ir3[9];
          ir3[9] = (v837_data + (v789_data * v835_data));
          float v840_data = s1[140];
          float v842_data = ir3[10];
          ir3[10] = (v842_data + (v789_data * v840_data));
          float v845_data = s1[153];
          float v847_data = ir3[11];
          ir3[11] = (v847_data + (v789_data * v845_data));
          float v850_data = s1[166];
          float v852_data = ir3[12];
          ir3[12] = (v852_data + (v789_data * v850_data));
          float v857_data = r2[11];
          float v858_data = s1[11];
          float v860_data = ir3[0];
          ir3[0] = (v860_data + (v857_data * v858_data));
          float v863_data = s1[24];
          float v865_data = ir3[1];
          ir3[1] = (v865_data + (v857_data * v863_data));
          float v868_data = s1[37];
          float v870_data = ir3[2];
          ir3[2] = (v870_data + (v857_data * v868_data));
          float v873_data = s1[50];
          float v875_data = ir3[3];
          ir3[3] = (v875_data + (v857_data * v873_data));
          float v878_data = s1[63];
          float v880_data = ir3[4];
          ir3[4] = (v880_data + (v857_data * v878_data));
          float v883_data = s1[76];
          float v885_data = ir3[5];
          ir3[5] = (v885_data + (v857_data * v883_data));
          float v888_data = s1[89];
          float v890_data = ir3[6];
          ir3[6] = (v890_data + (v857_data * v888_data));
          float v893_data = s1[102];
          float v895_data = ir3[7];
          ir3[7] = (v895_data + (v857_data * v893_data));
          float v898_data = s1[115];
          float v900_data = ir3[8];
          ir3[8] = (v900_data + (v857_data * v898_data));
          float v903_data = s1[128];
          float v905_data = ir3[9];
          ir3[9] = (v905_data + (v857_data * v903_data));
          float v908_data = s1[141];
          float v910_data = ir3[10];
          ir3[10] = (v910_data + (v857_data * v908_data));
          float v913_data = s1[154];
          float v915_data = ir3[11];
          ir3[11] = (v915_data + (v857_data * v913_data));
          float v918_data = s1[167];
          float v920_data = ir3[12];
          ir3[12] = (v920_data + (v857_data * v918_data));
          float v925_data = r2[12];
          float v926_data = s1[12];
          float v928_data = ir3[0];
          ir3[0] = (v928_data + (v925_data * v926_data));
          float v931_data = s1[25];
          float v933_data = ir3[1];
          ir3[1] = (v933_data + (v925_data * v931_data));
          float v936_data = s1[38];
          float v938_data = ir3[2];
          ir3[2] = (v938_data + (v925_data * v936_data));
          float v941_data = s1[51];
          float v943_data = ir3[3];
          ir3[3] = (v943_data + (v925_data * v941_data));
          float v946_data = s1[64];
          float v948_data = ir3[4];
          ir3[4] = (v948_data + (v925_data * v946_data));
          float v951_data = s1[77];
          float v953_data = ir3[5];
          ir3[5] = (v953_data + (v925_data * v951_data));
          float v956_data = s1[90];
          float v958_data = ir3[6];
          ir3[6] = (v958_data + (v925_data * v956_data));
          float v961_data = s1[103];
          float v963_data = ir3[7];
          ir3[7] = (v963_data + (v925_data * v961_data));
          float v966_data = s1[116];
          float v968_data = ir3[8];
          ir3[8] = (v968_data + (v925_data * v966_data));
          float v971_data = s1[129];
          float v973_data = ir3[9];
          ir3[9] = (v973_data + (v925_data * v971_data));
          float v976_data = s1[142];
          float v978_data = ir3[10];
          ir3[10] = (v978_data + (v925_data * v976_data));
          float v981_data = s1[155];
          float v983_data = ir3[11];
          ir3[11] = (v983_data + (v925_data * v981_data));
          float v986_data = s1[168];
          float v988_data = ir3[12];
          ir3[12] = (v988_data + (v925_data * v986_data));
          #pragma unroll
          for (int32_t v993_n0 = 0; v993_n0 < 1; ++v993_n0) {
            #pragma unroll
            for (int32_t v994_n1 = 0; v994_n1 < 13; ++v994_n1) {
              int32_t v995_a = v993_n0 + v994_n1;
              int32_t v996_a = v993_n0 + v994_n1;
              float v997_data = ir3[v996_a];
              int32_t v998_a = v993_n0 + v994_n1;
              r3[v996_a] = v997_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v1003_i0 = 0; v1003_i0 < 1; ++v1003_i0) {
            int32_t v1012_lead = v8_lead + (v1003_i0 * 32);
            #pragma unroll
            for (int32_t v1004_i1 = 0; v1004_i1 < 13; ++v1004_i1) {
              int32_t v1005_a = v1003_i0 + v1004_i1;
              float v1007_data = r3[(v1003_i0 + v1004_i1)];
              int32_t v1014_a = v1012_lead + (v1004_i1 * 32);
              glb_m3[v1014_a] = v1007_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

