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
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v10_a = v4_i1 * 32;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __ldcg(&glb_m1[(v16_lead + v10_a)]);
              int32_t v21_a = v3_i0 + (v4_i1 - 10);
              r0[v21_a] = v19_data;
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
            float v25_data = r0[0];
            float v26_data = s0[114];
            float v28_data = ir1[0];
            ir1[0] = (v28_data + (v25_data * v26_data));
            float v33_data = r0[1];
            float v34_data = s0[115];
            float v36_data = ir1[0];
            ir1[0] = (v36_data + (v33_data * v34_data));
            float v41_data = r0[2];
            float v42_data = s0[116];
            float v44_data = ir1[0];
            ir1[0] = (v44_data + (v41_data * v42_data));
            #pragma unroll
            for (int32_t v49_n0 = 0; v49_n0 < 1; ++v49_n0) {
              #pragma unroll
              for (int32_t v50_n1 = 0; v50_n1 < 1; ++v50_n1) {
                int32_t v51_a = v49_n0 + v50_n1;
                int32_t v52_a = v49_n0 + v50_n1;
                float v53_data = ir1[v52_a];
                int32_t v54_a = v49_n0 + v50_n1;
                r1[v52_a] = v53_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v58_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v59_i0 = 0; v59_i0 < 1; ++v59_i0) {
            int32_t v68_lead = v58_lead + (v59_i0 * 32);
            #pragma unroll
            for (int32_t v60_i1 = 0; v60_i1 < 1; ++v60_i1) {
              int32_t v61_a = v59_i0 + v60_i1;
              float v63_data = r1[(v59_i0 + v60_i1)];
              int32_t v71_a = v68_lead + ((v60_i1 + 8) * 32);
              glb_m0[v71_a] = v63_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          int32_t v74_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v75_i0 = 0; v75_i0 < 1; ++v75_i0) {
            int32_t v80_lead = v75_i0 * 32;
            int32_t v81_lead = v74_lead + v80_lead;
            int32_t v88_lead = v74_lead + v80_lead;
            #pragma unroll
            for (int32_t v76_i1 = 0; v76_i1 < 13; ++v76_i1) {
              int32_t v82_a = v76_i1 * 32;
              int32_t v83_a = v81_lead + v82_a;
              float v91_data = glb_m0[(v88_lead + v82_a)];
              int32_t v92_a = v75_i0 + v76_i1;
              r2[v92_a] = v91_data;
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
            float v96_data = r2[0];
            float v97_data = s1[0];
            float v99_data = ir3[0];
            ir3[0] = (v99_data + (v96_data * v97_data));
            float v102_data = s1[13];
            float v104_data = ir3[1];
            ir3[1] = (v104_data + (v96_data * v102_data));
            float v107_data = s1[26];
            float v109_data = ir3[2];
            ir3[2] = (v109_data + (v96_data * v107_data));
            float v112_data = s1[39];
            float v114_data = ir3[3];
            ir3[3] = (v114_data + (v96_data * v112_data));
            float v117_data = s1[52];
            float v119_data = ir3[4];
            ir3[4] = (v119_data + (v96_data * v117_data));
            float v122_data = s1[65];
            float v124_data = ir3[5];
            ir3[5] = (v124_data + (v96_data * v122_data));
            float v127_data = s1[78];
            float v129_data = ir3[6];
            ir3[6] = (v129_data + (v96_data * v127_data));
            float v132_data = s1[91];
            float v134_data = ir3[7];
            ir3[7] = (v134_data + (v96_data * v132_data));
            float v137_data = s1[104];
            float v139_data = ir3[8];
            ir3[8] = (v139_data + (v96_data * v137_data));
            float v142_data = s1[117];
            float v144_data = ir3[9];
            ir3[9] = (v144_data + (v96_data * v142_data));
            float v147_data = s1[130];
            float v149_data = ir3[10];
            ir3[10] = (v149_data + (v96_data * v147_data));
            float v152_data = s1[143];
            float v154_data = ir3[11];
            ir3[11] = (v154_data + (v96_data * v152_data));
            float v157_data = s1[156];
            float v159_data = ir3[12];
            ir3[12] = (v159_data + (v96_data * v157_data));
            float v164_data = r2[1];
            float v165_data = s1[1];
            float v167_data = ir3[0];
            ir3[0] = (v167_data + (v164_data * v165_data));
            float v170_data = s1[14];
            float v172_data = ir3[1];
            ir3[1] = (v172_data + (v164_data * v170_data));
            float v175_data = s1[27];
            float v177_data = ir3[2];
            ir3[2] = (v177_data + (v164_data * v175_data));
            float v180_data = s1[40];
            float v182_data = ir3[3];
            ir3[3] = (v182_data + (v164_data * v180_data));
            float v185_data = s1[53];
            float v187_data = ir3[4];
            ir3[4] = (v187_data + (v164_data * v185_data));
            float v190_data = s1[66];
            float v192_data = ir3[5];
            ir3[5] = (v192_data + (v164_data * v190_data));
            float v195_data = s1[79];
            float v197_data = ir3[6];
            ir3[6] = (v197_data + (v164_data * v195_data));
            float v200_data = s1[92];
            float v202_data = ir3[7];
            ir3[7] = (v202_data + (v164_data * v200_data));
            float v205_data = s1[105];
            float v207_data = ir3[8];
            ir3[8] = (v207_data + (v164_data * v205_data));
            float v210_data = s1[118];
            float v212_data = ir3[9];
            ir3[9] = (v212_data + (v164_data * v210_data));
            float v215_data = s1[131];
            float v217_data = ir3[10];
            ir3[10] = (v217_data + (v164_data * v215_data));
            float v220_data = s1[144];
            float v222_data = ir3[11];
            ir3[11] = (v222_data + (v164_data * v220_data));
            float v225_data = s1[157];
            float v227_data = ir3[12];
            ir3[12] = (v227_data + (v164_data * v225_data));
            float v232_data = r2[2];
            float v233_data = s1[2];
            float v235_data = ir3[0];
            ir3[0] = (v235_data + (v232_data * v233_data));
            float v238_data = s1[15];
            float v240_data = ir3[1];
            ir3[1] = (v240_data + (v232_data * v238_data));
            float v243_data = s1[28];
            float v245_data = ir3[2];
            ir3[2] = (v245_data + (v232_data * v243_data));
            float v248_data = s1[41];
            float v250_data = ir3[3];
            ir3[3] = (v250_data + (v232_data * v248_data));
            float v253_data = s1[54];
            float v255_data = ir3[4];
            ir3[4] = (v255_data + (v232_data * v253_data));
            float v258_data = s1[67];
            float v260_data = ir3[5];
            ir3[5] = (v260_data + (v232_data * v258_data));
            float v263_data = s1[80];
            float v265_data = ir3[6];
            ir3[6] = (v265_data + (v232_data * v263_data));
            float v268_data = s1[93];
            float v270_data = ir3[7];
            ir3[7] = (v270_data + (v232_data * v268_data));
            float v273_data = s1[106];
            float v275_data = ir3[8];
            ir3[8] = (v275_data + (v232_data * v273_data));
            float v278_data = s1[119];
            float v280_data = ir3[9];
            ir3[9] = (v280_data + (v232_data * v278_data));
            float v283_data = s1[132];
            float v285_data = ir3[10];
            ir3[10] = (v285_data + (v232_data * v283_data));
            float v288_data = s1[145];
            float v290_data = ir3[11];
            ir3[11] = (v290_data + (v232_data * v288_data));
            float v293_data = s1[158];
            float v295_data = ir3[12];
            ir3[12] = (v295_data + (v232_data * v293_data));
            float v300_data = r2[3];
            float v301_data = s1[3];
            float v303_data = ir3[0];
            ir3[0] = (v303_data + (v300_data * v301_data));
            float v306_data = s1[16];
            float v308_data = ir3[1];
            ir3[1] = (v308_data + (v300_data * v306_data));
            float v311_data = s1[29];
            float v313_data = ir3[2];
            ir3[2] = (v313_data + (v300_data * v311_data));
            float v316_data = s1[42];
            float v318_data = ir3[3];
            ir3[3] = (v318_data + (v300_data * v316_data));
            float v321_data = s1[55];
            float v323_data = ir3[4];
            ir3[4] = (v323_data + (v300_data * v321_data));
            float v326_data = s1[68];
            float v328_data = ir3[5];
            ir3[5] = (v328_data + (v300_data * v326_data));
            float v331_data = s1[81];
            float v333_data = ir3[6];
            ir3[6] = (v333_data + (v300_data * v331_data));
            float v336_data = s1[94];
            float v338_data = ir3[7];
            ir3[7] = (v338_data + (v300_data * v336_data));
            float v341_data = s1[107];
            float v343_data = ir3[8];
            ir3[8] = (v343_data + (v300_data * v341_data));
            float v346_data = s1[120];
            float v348_data = ir3[9];
            ir3[9] = (v348_data + (v300_data * v346_data));
            float v351_data = s1[133];
            float v353_data = ir3[10];
            ir3[10] = (v353_data + (v300_data * v351_data));
            float v356_data = s1[146];
            float v358_data = ir3[11];
            ir3[11] = (v358_data + (v300_data * v356_data));
            float v361_data = s1[159];
            float v363_data = ir3[12];
            ir3[12] = (v363_data + (v300_data * v361_data));
            float v368_data = r2[4];
            float v369_data = s1[4];
            float v371_data = ir3[0];
            ir3[0] = (v371_data + (v368_data * v369_data));
            float v374_data = s1[17];
            float v376_data = ir3[1];
            ir3[1] = (v376_data + (v368_data * v374_data));
            float v379_data = s1[30];
            float v381_data = ir3[2];
            ir3[2] = (v381_data + (v368_data * v379_data));
            float v384_data = s1[43];
            float v386_data = ir3[3];
            ir3[3] = (v386_data + (v368_data * v384_data));
            float v389_data = s1[56];
            float v391_data = ir3[4];
            ir3[4] = (v391_data + (v368_data * v389_data));
            float v394_data = s1[69];
            float v396_data = ir3[5];
            ir3[5] = (v396_data + (v368_data * v394_data));
            float v399_data = s1[82];
            float v401_data = ir3[6];
            ir3[6] = (v401_data + (v368_data * v399_data));
            float v404_data = s1[95];
            float v406_data = ir3[7];
            ir3[7] = (v406_data + (v368_data * v404_data));
            float v409_data = s1[108];
            float v411_data = ir3[8];
            ir3[8] = (v411_data + (v368_data * v409_data));
            float v414_data = s1[121];
            float v416_data = ir3[9];
            ir3[9] = (v416_data + (v368_data * v414_data));
            float v419_data = s1[134];
            float v421_data = ir3[10];
            ir3[10] = (v421_data + (v368_data * v419_data));
            float v424_data = s1[147];
            float v426_data = ir3[11];
            ir3[11] = (v426_data + (v368_data * v424_data));
            float v429_data = s1[160];
            float v431_data = ir3[12];
            ir3[12] = (v431_data + (v368_data * v429_data));
            float v436_data = r2[5];
            float v437_data = s1[5];
            float v439_data = ir3[0];
            ir3[0] = (v439_data + (v436_data * v437_data));
            float v442_data = s1[18];
            float v444_data = ir3[1];
            ir3[1] = (v444_data + (v436_data * v442_data));
            float v447_data = s1[31];
            float v449_data = ir3[2];
            ir3[2] = (v449_data + (v436_data * v447_data));
            float v452_data = s1[44];
            float v454_data = ir3[3];
            ir3[3] = (v454_data + (v436_data * v452_data));
            float v457_data = s1[57];
            float v459_data = ir3[4];
            ir3[4] = (v459_data + (v436_data * v457_data));
            float v462_data = s1[70];
            float v464_data = ir3[5];
            ir3[5] = (v464_data + (v436_data * v462_data));
            float v467_data = s1[83];
            float v469_data = ir3[6];
            ir3[6] = (v469_data + (v436_data * v467_data));
            float v472_data = s1[96];
            float v474_data = ir3[7];
            ir3[7] = (v474_data + (v436_data * v472_data));
            float v477_data = s1[109];
            float v479_data = ir3[8];
            ir3[8] = (v479_data + (v436_data * v477_data));
            float v482_data = s1[122];
            float v484_data = ir3[9];
            ir3[9] = (v484_data + (v436_data * v482_data));
            float v487_data = s1[135];
            float v489_data = ir3[10];
            ir3[10] = (v489_data + (v436_data * v487_data));
            float v492_data = s1[148];
            float v494_data = ir3[11];
            ir3[11] = (v494_data + (v436_data * v492_data));
            float v497_data = s1[161];
            float v499_data = ir3[12];
            ir3[12] = (v499_data + (v436_data * v497_data));
            float v504_data = r2[6];
            float v505_data = s1[6];
            float v507_data = ir3[0];
            ir3[0] = (v507_data + (v504_data * v505_data));
            float v510_data = s1[19];
            float v512_data = ir3[1];
            ir3[1] = (v512_data + (v504_data * v510_data));
            float v515_data = s1[32];
            float v517_data = ir3[2];
            ir3[2] = (v517_data + (v504_data * v515_data));
            float v520_data = s1[45];
            float v522_data = ir3[3];
            ir3[3] = (v522_data + (v504_data * v520_data));
            float v525_data = s1[58];
            float v527_data = ir3[4];
            ir3[4] = (v527_data + (v504_data * v525_data));
            float v530_data = s1[71];
            float v532_data = ir3[5];
            ir3[5] = (v532_data + (v504_data * v530_data));
            float v535_data = s1[84];
            float v537_data = ir3[6];
            ir3[6] = (v537_data + (v504_data * v535_data));
            float v540_data = s1[97];
            float v542_data = ir3[7];
            ir3[7] = (v542_data + (v504_data * v540_data));
            float v545_data = s1[110];
            float v547_data = ir3[8];
            ir3[8] = (v547_data + (v504_data * v545_data));
            float v550_data = s1[123];
            float v552_data = ir3[9];
            ir3[9] = (v552_data + (v504_data * v550_data));
            float v555_data = s1[136];
            float v557_data = ir3[10];
            ir3[10] = (v557_data + (v504_data * v555_data));
            float v560_data = s1[149];
            float v562_data = ir3[11];
            ir3[11] = (v562_data + (v504_data * v560_data));
            float v565_data = s1[162];
            float v567_data = ir3[12];
            ir3[12] = (v567_data + (v504_data * v565_data));
            float v572_data = r2[7];
            float v573_data = s1[7];
            float v575_data = ir3[0];
            ir3[0] = (v575_data + (v572_data * v573_data));
            float v578_data = s1[20];
            float v580_data = ir3[1];
            ir3[1] = (v580_data + (v572_data * v578_data));
            float v583_data = s1[33];
            float v585_data = ir3[2];
            ir3[2] = (v585_data + (v572_data * v583_data));
            float v588_data = s1[46];
            float v590_data = ir3[3];
            ir3[3] = (v590_data + (v572_data * v588_data));
            float v593_data = s1[59];
            float v595_data = ir3[4];
            ir3[4] = (v595_data + (v572_data * v593_data));
            float v598_data = s1[72];
            float v600_data = ir3[5];
            ir3[5] = (v600_data + (v572_data * v598_data));
            float v603_data = s1[85];
            float v605_data = ir3[6];
            ir3[6] = (v605_data + (v572_data * v603_data));
            float v608_data = s1[98];
            float v610_data = ir3[7];
            ir3[7] = (v610_data + (v572_data * v608_data));
            float v613_data = s1[111];
            float v615_data = ir3[8];
            ir3[8] = (v615_data + (v572_data * v613_data));
            float v618_data = s1[124];
            float v620_data = ir3[9];
            ir3[9] = (v620_data + (v572_data * v618_data));
            float v623_data = s1[137];
            float v625_data = ir3[10];
            ir3[10] = (v625_data + (v572_data * v623_data));
            float v628_data = s1[150];
            float v630_data = ir3[11];
            ir3[11] = (v630_data + (v572_data * v628_data));
            float v633_data = s1[163];
            float v635_data = ir3[12];
            ir3[12] = (v635_data + (v572_data * v633_data));
            float v640_data = r2[8];
            float v641_data = s1[8];
            float v643_data = ir3[0];
            ir3[0] = (v643_data + (v640_data * v641_data));
            float v646_data = s1[21];
            float v648_data = ir3[1];
            ir3[1] = (v648_data + (v640_data * v646_data));
            float v651_data = s1[34];
            float v653_data = ir3[2];
            ir3[2] = (v653_data + (v640_data * v651_data));
            float v656_data = s1[47];
            float v658_data = ir3[3];
            ir3[3] = (v658_data + (v640_data * v656_data));
            float v661_data = s1[60];
            float v663_data = ir3[4];
            ir3[4] = (v663_data + (v640_data * v661_data));
            float v666_data = s1[73];
            float v668_data = ir3[5];
            ir3[5] = (v668_data + (v640_data * v666_data));
            float v671_data = s1[86];
            float v673_data = ir3[6];
            ir3[6] = (v673_data + (v640_data * v671_data));
            float v676_data = s1[99];
            float v678_data = ir3[7];
            ir3[7] = (v678_data + (v640_data * v676_data));
            float v681_data = s1[112];
            float v683_data = ir3[8];
            ir3[8] = (v683_data + (v640_data * v681_data));
            float v686_data = s1[125];
            float v688_data = ir3[9];
            ir3[9] = (v688_data + (v640_data * v686_data));
            float v691_data = s1[138];
            float v693_data = ir3[10];
            ir3[10] = (v693_data + (v640_data * v691_data));
            float v696_data = s1[151];
            float v698_data = ir3[11];
            ir3[11] = (v698_data + (v640_data * v696_data));
            float v701_data = s1[164];
            float v703_data = ir3[12];
            ir3[12] = (v703_data + (v640_data * v701_data));
            float v708_data = r2[9];
            float v709_data = s1[9];
            float v711_data = ir3[0];
            ir3[0] = (v711_data + (v708_data * v709_data));
            float v714_data = s1[22];
            float v716_data = ir3[1];
            ir3[1] = (v716_data + (v708_data * v714_data));
            float v719_data = s1[35];
            float v721_data = ir3[2];
            ir3[2] = (v721_data + (v708_data * v719_data));
            float v724_data = s1[48];
            float v726_data = ir3[3];
            ir3[3] = (v726_data + (v708_data * v724_data));
            float v729_data = s1[61];
            float v731_data = ir3[4];
            ir3[4] = (v731_data + (v708_data * v729_data));
            float v734_data = s1[74];
            float v736_data = ir3[5];
            ir3[5] = (v736_data + (v708_data * v734_data));
            float v739_data = s1[87];
            float v741_data = ir3[6];
            ir3[6] = (v741_data + (v708_data * v739_data));
            float v744_data = s1[100];
            float v746_data = ir3[7];
            ir3[7] = (v746_data + (v708_data * v744_data));
            float v749_data = s1[113];
            float v751_data = ir3[8];
            ir3[8] = (v751_data + (v708_data * v749_data));
            float v754_data = s1[126];
            float v756_data = ir3[9];
            ir3[9] = (v756_data + (v708_data * v754_data));
            float v759_data = s1[139];
            float v761_data = ir3[10];
            ir3[10] = (v761_data + (v708_data * v759_data));
            float v764_data = s1[152];
            float v766_data = ir3[11];
            ir3[11] = (v766_data + (v708_data * v764_data));
            float v769_data = s1[165];
            float v771_data = ir3[12];
            ir3[12] = (v771_data + (v708_data * v769_data));
            float v776_data = r2[10];
            float v777_data = s1[10];
            float v779_data = ir3[0];
            ir3[0] = (v779_data + (v776_data * v777_data));
            float v782_data = s1[23];
            float v784_data = ir3[1];
            ir3[1] = (v784_data + (v776_data * v782_data));
            float v787_data = s1[36];
            float v789_data = ir3[2];
            ir3[2] = (v789_data + (v776_data * v787_data));
            float v792_data = s1[49];
            float v794_data = ir3[3];
            ir3[3] = (v794_data + (v776_data * v792_data));
            float v797_data = s1[62];
            float v799_data = ir3[4];
            ir3[4] = (v799_data + (v776_data * v797_data));
            float v802_data = s1[75];
            float v804_data = ir3[5];
            ir3[5] = (v804_data + (v776_data * v802_data));
            float v807_data = s1[88];
            float v809_data = ir3[6];
            ir3[6] = (v809_data + (v776_data * v807_data));
            float v812_data = s1[101];
            float v814_data = ir3[7];
            ir3[7] = (v814_data + (v776_data * v812_data));
            float v817_data = s1[114];
            float v819_data = ir3[8];
            ir3[8] = (v819_data + (v776_data * v817_data));
            float v822_data = s1[127];
            float v824_data = ir3[9];
            ir3[9] = (v824_data + (v776_data * v822_data));
            float v827_data = s1[140];
            float v829_data = ir3[10];
            ir3[10] = (v829_data + (v776_data * v827_data));
            float v832_data = s1[153];
            float v834_data = ir3[11];
            ir3[11] = (v834_data + (v776_data * v832_data));
            float v837_data = s1[166];
            float v839_data = ir3[12];
            ir3[12] = (v839_data + (v776_data * v837_data));
            float v844_data = r2[11];
            float v845_data = s1[11];
            float v847_data = ir3[0];
            ir3[0] = (v847_data + (v844_data * v845_data));
            float v850_data = s1[24];
            float v852_data = ir3[1];
            ir3[1] = (v852_data + (v844_data * v850_data));
            float v855_data = s1[37];
            float v857_data = ir3[2];
            ir3[2] = (v857_data + (v844_data * v855_data));
            float v860_data = s1[50];
            float v862_data = ir3[3];
            ir3[3] = (v862_data + (v844_data * v860_data));
            float v865_data = s1[63];
            float v867_data = ir3[4];
            ir3[4] = (v867_data + (v844_data * v865_data));
            float v870_data = s1[76];
            float v872_data = ir3[5];
            ir3[5] = (v872_data + (v844_data * v870_data));
            float v875_data = s1[89];
            float v877_data = ir3[6];
            ir3[6] = (v877_data + (v844_data * v875_data));
            float v880_data = s1[102];
            float v882_data = ir3[7];
            ir3[7] = (v882_data + (v844_data * v880_data));
            float v885_data = s1[115];
            float v887_data = ir3[8];
            ir3[8] = (v887_data + (v844_data * v885_data));
            float v890_data = s1[128];
            float v892_data = ir3[9];
            ir3[9] = (v892_data + (v844_data * v890_data));
            float v895_data = s1[141];
            float v897_data = ir3[10];
            ir3[10] = (v897_data + (v844_data * v895_data));
            float v900_data = s1[154];
            float v902_data = ir3[11];
            ir3[11] = (v902_data + (v844_data * v900_data));
            float v905_data = s1[167];
            float v907_data = ir3[12];
            ir3[12] = (v907_data + (v844_data * v905_data));
            float v912_data = r2[12];
            float v913_data = s1[12];
            float v915_data = ir3[0];
            ir3[0] = (v915_data + (v912_data * v913_data));
            float v918_data = s1[25];
            float v920_data = ir3[1];
            ir3[1] = (v920_data + (v912_data * v918_data));
            float v923_data = s1[38];
            float v925_data = ir3[2];
            ir3[2] = (v925_data + (v912_data * v923_data));
            float v928_data = s1[51];
            float v930_data = ir3[3];
            ir3[3] = (v930_data + (v912_data * v928_data));
            float v933_data = s1[64];
            float v935_data = ir3[4];
            ir3[4] = (v935_data + (v912_data * v933_data));
            float v938_data = s1[77];
            float v940_data = ir3[5];
            ir3[5] = (v940_data + (v912_data * v938_data));
            float v943_data = s1[90];
            float v945_data = ir3[6];
            ir3[6] = (v945_data + (v912_data * v943_data));
            float v948_data = s1[103];
            float v950_data = ir3[7];
            ir3[7] = (v950_data + (v912_data * v948_data));
            float v953_data = s1[116];
            float v955_data = ir3[8];
            ir3[8] = (v955_data + (v912_data * v953_data));
            float v958_data = s1[129];
            float v960_data = ir3[9];
            ir3[9] = (v960_data + (v912_data * v958_data));
            float v963_data = s1[142];
            float v965_data = ir3[10];
            ir3[10] = (v965_data + (v912_data * v963_data));
            float v968_data = s1[155];
            float v970_data = ir3[11];
            ir3[11] = (v970_data + (v912_data * v968_data));
            float v973_data = s1[168];
            float v975_data = ir3[12];
            ir3[12] = (v975_data + (v912_data * v973_data));
            #pragma unroll
            for (int32_t v980_n0 = 0; v980_n0 < 1; ++v980_n0) {
              #pragma unroll
              for (int32_t v981_n1 = 0; v981_n1 < 13; ++v981_n1) {
                int32_t v982_a = v980_n0 + v981_n1;
                int32_t v983_a = v980_n0 + v981_n1;
                float v984_data = ir3[v983_a];
                int32_t v985_a = v980_n0 + v981_n1;
                r3[v983_a] = v984_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r3);
          int32_t v989_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v990_i0 = 0; v990_i0 < 1; ++v990_i0) {
            int32_t v999_lead = v989_lead + (v990_i0 * 32);
            #pragma unroll
            for (int32_t v991_i1 = 0; v991_i1 < 13; ++v991_i1) {
              int32_t v992_a = v990_i0 + v991_i1;
              float v994_data = r3[(v990_i0 + v991_i1)];
              int32_t v1001_a = v999_lead + (v991_i1 * 32);
              glb_m3[v1001_a] = v994_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

