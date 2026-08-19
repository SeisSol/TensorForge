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
                float v46_data = ir1[(v42_n0 + v43_n1)];
                int32_t v47_a = v42_n0 + v43_n1;
                r1[v47_a] = v46_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v50_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
            int32_t v60_lead = v50_lead + (v51_i0 * 32);
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 1; ++v52_i1) {
              int32_t v53_a = v51_i0 + v52_i1;
              float v55_data = r1[(v51_i0 + v52_i1)];
              int32_t v63_a = v60_lead + ((v52_i1 + 8) * 32);
              glb_m0[v63_a] = v55_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          int32_t v66_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
            int32_t v73_lead = v66_lead + (v67_i0 * 32);
            #pragma unroll
            for (int32_t v68_i1 = 0; v68_i1 < 13; ++v68_i1) {
              int32_t v75_a = v73_lead + (v68_i1 * 32);
              float v76_data;
              {
                v76_data = glb_m0[v75_a];
              }
              int32_t v77_a = v67_i0 + v68_i1;
              r2[v77_a] = v76_data;
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
            float v81_data = r2[0];
            float v82_data = s1[0];
            float v84_data = ir3[0];
            ir3[0] = (v84_data + (v81_data * v82_data));
            float v87_data = s1[13];
            float v89_data = ir3[1];
            ir3[1] = (v89_data + (v81_data * v87_data));
            float v92_data = s1[26];
            float v94_data = ir3[2];
            ir3[2] = (v94_data + (v81_data * v92_data));
            float v97_data = s1[39];
            float v99_data = ir3[3];
            ir3[3] = (v99_data + (v81_data * v97_data));
            float v102_data = s1[52];
            float v104_data = ir3[4];
            ir3[4] = (v104_data + (v81_data * v102_data));
            float v107_data = s1[65];
            float v109_data = ir3[5];
            ir3[5] = (v109_data + (v81_data * v107_data));
            float v112_data = s1[78];
            float v114_data = ir3[6];
            ir3[6] = (v114_data + (v81_data * v112_data));
            float v117_data = s1[91];
            float v119_data = ir3[7];
            ir3[7] = (v119_data + (v81_data * v117_data));
            float v122_data = s1[104];
            float v124_data = ir3[8];
            ir3[8] = (v124_data + (v81_data * v122_data));
            float v127_data = s1[117];
            float v129_data = ir3[9];
            ir3[9] = (v129_data + (v81_data * v127_data));
            float v132_data = s1[130];
            float v134_data = ir3[10];
            ir3[10] = (v134_data + (v81_data * v132_data));
            float v137_data = s1[143];
            float v139_data = ir3[11];
            ir3[11] = (v139_data + (v81_data * v137_data));
            float v142_data = s1[156];
            float v144_data = ir3[12];
            ir3[12] = (v144_data + (v81_data * v142_data));
            float v149_data = r2[1];
            float v150_data = s1[1];
            float v152_data = ir3[0];
            ir3[0] = (v152_data + (v149_data * v150_data));
            float v155_data = s1[14];
            float v157_data = ir3[1];
            ir3[1] = (v157_data + (v149_data * v155_data));
            float v160_data = s1[27];
            float v162_data = ir3[2];
            ir3[2] = (v162_data + (v149_data * v160_data));
            float v165_data = s1[40];
            float v167_data = ir3[3];
            ir3[3] = (v167_data + (v149_data * v165_data));
            float v170_data = s1[53];
            float v172_data = ir3[4];
            ir3[4] = (v172_data + (v149_data * v170_data));
            float v175_data = s1[66];
            float v177_data = ir3[5];
            ir3[5] = (v177_data + (v149_data * v175_data));
            float v180_data = s1[79];
            float v182_data = ir3[6];
            ir3[6] = (v182_data + (v149_data * v180_data));
            float v185_data = s1[92];
            float v187_data = ir3[7];
            ir3[7] = (v187_data + (v149_data * v185_data));
            float v190_data = s1[105];
            float v192_data = ir3[8];
            ir3[8] = (v192_data + (v149_data * v190_data));
            float v195_data = s1[118];
            float v197_data = ir3[9];
            ir3[9] = (v197_data + (v149_data * v195_data));
            float v200_data = s1[131];
            float v202_data = ir3[10];
            ir3[10] = (v202_data + (v149_data * v200_data));
            float v205_data = s1[144];
            float v207_data = ir3[11];
            ir3[11] = (v207_data + (v149_data * v205_data));
            float v210_data = s1[157];
            float v212_data = ir3[12];
            ir3[12] = (v212_data + (v149_data * v210_data));
            float v217_data = r2[2];
            float v218_data = s1[2];
            float v220_data = ir3[0];
            ir3[0] = (v220_data + (v217_data * v218_data));
            float v223_data = s1[15];
            float v225_data = ir3[1];
            ir3[1] = (v225_data + (v217_data * v223_data));
            float v228_data = s1[28];
            float v230_data = ir3[2];
            ir3[2] = (v230_data + (v217_data * v228_data));
            float v233_data = s1[41];
            float v235_data = ir3[3];
            ir3[3] = (v235_data + (v217_data * v233_data));
            float v238_data = s1[54];
            float v240_data = ir3[4];
            ir3[4] = (v240_data + (v217_data * v238_data));
            float v243_data = s1[67];
            float v245_data = ir3[5];
            ir3[5] = (v245_data + (v217_data * v243_data));
            float v248_data = s1[80];
            float v250_data = ir3[6];
            ir3[6] = (v250_data + (v217_data * v248_data));
            float v253_data = s1[93];
            float v255_data = ir3[7];
            ir3[7] = (v255_data + (v217_data * v253_data));
            float v258_data = s1[106];
            float v260_data = ir3[8];
            ir3[8] = (v260_data + (v217_data * v258_data));
            float v263_data = s1[119];
            float v265_data = ir3[9];
            ir3[9] = (v265_data + (v217_data * v263_data));
            float v268_data = s1[132];
            float v270_data = ir3[10];
            ir3[10] = (v270_data + (v217_data * v268_data));
            float v273_data = s1[145];
            float v275_data = ir3[11];
            ir3[11] = (v275_data + (v217_data * v273_data));
            float v278_data = s1[158];
            float v280_data = ir3[12];
            ir3[12] = (v280_data + (v217_data * v278_data));
            float v285_data = r2[3];
            float v286_data = s1[3];
            float v288_data = ir3[0];
            ir3[0] = (v288_data + (v285_data * v286_data));
            float v291_data = s1[16];
            float v293_data = ir3[1];
            ir3[1] = (v293_data + (v285_data * v291_data));
            float v296_data = s1[29];
            float v298_data = ir3[2];
            ir3[2] = (v298_data + (v285_data * v296_data));
            float v301_data = s1[42];
            float v303_data = ir3[3];
            ir3[3] = (v303_data + (v285_data * v301_data));
            float v306_data = s1[55];
            float v308_data = ir3[4];
            ir3[4] = (v308_data + (v285_data * v306_data));
            float v311_data = s1[68];
            float v313_data = ir3[5];
            ir3[5] = (v313_data + (v285_data * v311_data));
            float v316_data = s1[81];
            float v318_data = ir3[6];
            ir3[6] = (v318_data + (v285_data * v316_data));
            float v321_data = s1[94];
            float v323_data = ir3[7];
            ir3[7] = (v323_data + (v285_data * v321_data));
            float v326_data = s1[107];
            float v328_data = ir3[8];
            ir3[8] = (v328_data + (v285_data * v326_data));
            float v331_data = s1[120];
            float v333_data = ir3[9];
            ir3[9] = (v333_data + (v285_data * v331_data));
            float v336_data = s1[133];
            float v338_data = ir3[10];
            ir3[10] = (v338_data + (v285_data * v336_data));
            float v341_data = s1[146];
            float v343_data = ir3[11];
            ir3[11] = (v343_data + (v285_data * v341_data));
            float v346_data = s1[159];
            float v348_data = ir3[12];
            ir3[12] = (v348_data + (v285_data * v346_data));
            float v353_data = r2[4];
            float v354_data = s1[4];
            float v356_data = ir3[0];
            ir3[0] = (v356_data + (v353_data * v354_data));
            float v359_data = s1[17];
            float v361_data = ir3[1];
            ir3[1] = (v361_data + (v353_data * v359_data));
            float v364_data = s1[30];
            float v366_data = ir3[2];
            ir3[2] = (v366_data + (v353_data * v364_data));
            float v369_data = s1[43];
            float v371_data = ir3[3];
            ir3[3] = (v371_data + (v353_data * v369_data));
            float v374_data = s1[56];
            float v376_data = ir3[4];
            ir3[4] = (v376_data + (v353_data * v374_data));
            float v379_data = s1[69];
            float v381_data = ir3[5];
            ir3[5] = (v381_data + (v353_data * v379_data));
            float v384_data = s1[82];
            float v386_data = ir3[6];
            ir3[6] = (v386_data + (v353_data * v384_data));
            float v389_data = s1[95];
            float v391_data = ir3[7];
            ir3[7] = (v391_data + (v353_data * v389_data));
            float v394_data = s1[108];
            float v396_data = ir3[8];
            ir3[8] = (v396_data + (v353_data * v394_data));
            float v399_data = s1[121];
            float v401_data = ir3[9];
            ir3[9] = (v401_data + (v353_data * v399_data));
            float v404_data = s1[134];
            float v406_data = ir3[10];
            ir3[10] = (v406_data + (v353_data * v404_data));
            float v409_data = s1[147];
            float v411_data = ir3[11];
            ir3[11] = (v411_data + (v353_data * v409_data));
            float v414_data = s1[160];
            float v416_data = ir3[12];
            ir3[12] = (v416_data + (v353_data * v414_data));
            float v421_data = r2[5];
            float v422_data = s1[5];
            float v424_data = ir3[0];
            ir3[0] = (v424_data + (v421_data * v422_data));
            float v427_data = s1[18];
            float v429_data = ir3[1];
            ir3[1] = (v429_data + (v421_data * v427_data));
            float v432_data = s1[31];
            float v434_data = ir3[2];
            ir3[2] = (v434_data + (v421_data * v432_data));
            float v437_data = s1[44];
            float v439_data = ir3[3];
            ir3[3] = (v439_data + (v421_data * v437_data));
            float v442_data = s1[57];
            float v444_data = ir3[4];
            ir3[4] = (v444_data + (v421_data * v442_data));
            float v447_data = s1[70];
            float v449_data = ir3[5];
            ir3[5] = (v449_data + (v421_data * v447_data));
            float v452_data = s1[83];
            float v454_data = ir3[6];
            ir3[6] = (v454_data + (v421_data * v452_data));
            float v457_data = s1[96];
            float v459_data = ir3[7];
            ir3[7] = (v459_data + (v421_data * v457_data));
            float v462_data = s1[109];
            float v464_data = ir3[8];
            ir3[8] = (v464_data + (v421_data * v462_data));
            float v467_data = s1[122];
            float v469_data = ir3[9];
            ir3[9] = (v469_data + (v421_data * v467_data));
            float v472_data = s1[135];
            float v474_data = ir3[10];
            ir3[10] = (v474_data + (v421_data * v472_data));
            float v477_data = s1[148];
            float v479_data = ir3[11];
            ir3[11] = (v479_data + (v421_data * v477_data));
            float v482_data = s1[161];
            float v484_data = ir3[12];
            ir3[12] = (v484_data + (v421_data * v482_data));
            float v489_data = r2[6];
            float v490_data = s1[6];
            float v492_data = ir3[0];
            ir3[0] = (v492_data + (v489_data * v490_data));
            float v495_data = s1[19];
            float v497_data = ir3[1];
            ir3[1] = (v497_data + (v489_data * v495_data));
            float v500_data = s1[32];
            float v502_data = ir3[2];
            ir3[2] = (v502_data + (v489_data * v500_data));
            float v505_data = s1[45];
            float v507_data = ir3[3];
            ir3[3] = (v507_data + (v489_data * v505_data));
            float v510_data = s1[58];
            float v512_data = ir3[4];
            ir3[4] = (v512_data + (v489_data * v510_data));
            float v515_data = s1[71];
            float v517_data = ir3[5];
            ir3[5] = (v517_data + (v489_data * v515_data));
            float v520_data = s1[84];
            float v522_data = ir3[6];
            ir3[6] = (v522_data + (v489_data * v520_data));
            float v525_data = s1[97];
            float v527_data = ir3[7];
            ir3[7] = (v527_data + (v489_data * v525_data));
            float v530_data = s1[110];
            float v532_data = ir3[8];
            ir3[8] = (v532_data + (v489_data * v530_data));
            float v535_data = s1[123];
            float v537_data = ir3[9];
            ir3[9] = (v537_data + (v489_data * v535_data));
            float v540_data = s1[136];
            float v542_data = ir3[10];
            ir3[10] = (v542_data + (v489_data * v540_data));
            float v545_data = s1[149];
            float v547_data = ir3[11];
            ir3[11] = (v547_data + (v489_data * v545_data));
            float v550_data = s1[162];
            float v552_data = ir3[12];
            ir3[12] = (v552_data + (v489_data * v550_data));
            float v557_data = r2[7];
            float v558_data = s1[7];
            float v560_data = ir3[0];
            ir3[0] = (v560_data + (v557_data * v558_data));
            float v563_data = s1[20];
            float v565_data = ir3[1];
            ir3[1] = (v565_data + (v557_data * v563_data));
            float v568_data = s1[33];
            float v570_data = ir3[2];
            ir3[2] = (v570_data + (v557_data * v568_data));
            float v573_data = s1[46];
            float v575_data = ir3[3];
            ir3[3] = (v575_data + (v557_data * v573_data));
            float v578_data = s1[59];
            float v580_data = ir3[4];
            ir3[4] = (v580_data + (v557_data * v578_data));
            float v583_data = s1[72];
            float v585_data = ir3[5];
            ir3[5] = (v585_data + (v557_data * v583_data));
            float v588_data = s1[85];
            float v590_data = ir3[6];
            ir3[6] = (v590_data + (v557_data * v588_data));
            float v593_data = s1[98];
            float v595_data = ir3[7];
            ir3[7] = (v595_data + (v557_data * v593_data));
            float v598_data = s1[111];
            float v600_data = ir3[8];
            ir3[8] = (v600_data + (v557_data * v598_data));
            float v603_data = s1[124];
            float v605_data = ir3[9];
            ir3[9] = (v605_data + (v557_data * v603_data));
            float v608_data = s1[137];
            float v610_data = ir3[10];
            ir3[10] = (v610_data + (v557_data * v608_data));
            float v613_data = s1[150];
            float v615_data = ir3[11];
            ir3[11] = (v615_data + (v557_data * v613_data));
            float v618_data = s1[163];
            float v620_data = ir3[12];
            ir3[12] = (v620_data + (v557_data * v618_data));
            float v625_data = r2[8];
            float v626_data = s1[8];
            float v628_data = ir3[0];
            ir3[0] = (v628_data + (v625_data * v626_data));
            float v631_data = s1[21];
            float v633_data = ir3[1];
            ir3[1] = (v633_data + (v625_data * v631_data));
            float v636_data = s1[34];
            float v638_data = ir3[2];
            ir3[2] = (v638_data + (v625_data * v636_data));
            float v641_data = s1[47];
            float v643_data = ir3[3];
            ir3[3] = (v643_data + (v625_data * v641_data));
            float v646_data = s1[60];
            float v648_data = ir3[4];
            ir3[4] = (v648_data + (v625_data * v646_data));
            float v651_data = s1[73];
            float v653_data = ir3[5];
            ir3[5] = (v653_data + (v625_data * v651_data));
            float v656_data = s1[86];
            float v658_data = ir3[6];
            ir3[6] = (v658_data + (v625_data * v656_data));
            float v661_data = s1[99];
            float v663_data = ir3[7];
            ir3[7] = (v663_data + (v625_data * v661_data));
            float v666_data = s1[112];
            float v668_data = ir3[8];
            ir3[8] = (v668_data + (v625_data * v666_data));
            float v671_data = s1[125];
            float v673_data = ir3[9];
            ir3[9] = (v673_data + (v625_data * v671_data));
            float v676_data = s1[138];
            float v678_data = ir3[10];
            ir3[10] = (v678_data + (v625_data * v676_data));
            float v681_data = s1[151];
            float v683_data = ir3[11];
            ir3[11] = (v683_data + (v625_data * v681_data));
            float v686_data = s1[164];
            float v688_data = ir3[12];
            ir3[12] = (v688_data + (v625_data * v686_data));
            float v693_data = r2[9];
            float v694_data = s1[9];
            float v696_data = ir3[0];
            ir3[0] = (v696_data + (v693_data * v694_data));
            float v699_data = s1[22];
            float v701_data = ir3[1];
            ir3[1] = (v701_data + (v693_data * v699_data));
            float v704_data = s1[35];
            float v706_data = ir3[2];
            ir3[2] = (v706_data + (v693_data * v704_data));
            float v709_data = s1[48];
            float v711_data = ir3[3];
            ir3[3] = (v711_data + (v693_data * v709_data));
            float v714_data = s1[61];
            float v716_data = ir3[4];
            ir3[4] = (v716_data + (v693_data * v714_data));
            float v719_data = s1[74];
            float v721_data = ir3[5];
            ir3[5] = (v721_data + (v693_data * v719_data));
            float v724_data = s1[87];
            float v726_data = ir3[6];
            ir3[6] = (v726_data + (v693_data * v724_data));
            float v729_data = s1[100];
            float v731_data = ir3[7];
            ir3[7] = (v731_data + (v693_data * v729_data));
            float v734_data = s1[113];
            float v736_data = ir3[8];
            ir3[8] = (v736_data + (v693_data * v734_data));
            float v739_data = s1[126];
            float v741_data = ir3[9];
            ir3[9] = (v741_data + (v693_data * v739_data));
            float v744_data = s1[139];
            float v746_data = ir3[10];
            ir3[10] = (v746_data + (v693_data * v744_data));
            float v749_data = s1[152];
            float v751_data = ir3[11];
            ir3[11] = (v751_data + (v693_data * v749_data));
            float v754_data = s1[165];
            float v756_data = ir3[12];
            ir3[12] = (v756_data + (v693_data * v754_data));
            float v761_data = r2[10];
            float v762_data = s1[10];
            float v764_data = ir3[0];
            ir3[0] = (v764_data + (v761_data * v762_data));
            float v767_data = s1[23];
            float v769_data = ir3[1];
            ir3[1] = (v769_data + (v761_data * v767_data));
            float v772_data = s1[36];
            float v774_data = ir3[2];
            ir3[2] = (v774_data + (v761_data * v772_data));
            float v777_data = s1[49];
            float v779_data = ir3[3];
            ir3[3] = (v779_data + (v761_data * v777_data));
            float v782_data = s1[62];
            float v784_data = ir3[4];
            ir3[4] = (v784_data + (v761_data * v782_data));
            float v787_data = s1[75];
            float v789_data = ir3[5];
            ir3[5] = (v789_data + (v761_data * v787_data));
            float v792_data = s1[88];
            float v794_data = ir3[6];
            ir3[6] = (v794_data + (v761_data * v792_data));
            float v797_data = s1[101];
            float v799_data = ir3[7];
            ir3[7] = (v799_data + (v761_data * v797_data));
            float v802_data = s1[114];
            float v804_data = ir3[8];
            ir3[8] = (v804_data + (v761_data * v802_data));
            float v807_data = s1[127];
            float v809_data = ir3[9];
            ir3[9] = (v809_data + (v761_data * v807_data));
            float v812_data = s1[140];
            float v814_data = ir3[10];
            ir3[10] = (v814_data + (v761_data * v812_data));
            float v817_data = s1[153];
            float v819_data = ir3[11];
            ir3[11] = (v819_data + (v761_data * v817_data));
            float v822_data = s1[166];
            float v824_data = ir3[12];
            ir3[12] = (v824_data + (v761_data * v822_data));
            float v829_data = r2[11];
            float v830_data = s1[11];
            float v832_data = ir3[0];
            ir3[0] = (v832_data + (v829_data * v830_data));
            float v835_data = s1[24];
            float v837_data = ir3[1];
            ir3[1] = (v837_data + (v829_data * v835_data));
            float v840_data = s1[37];
            float v842_data = ir3[2];
            ir3[2] = (v842_data + (v829_data * v840_data));
            float v845_data = s1[50];
            float v847_data = ir3[3];
            ir3[3] = (v847_data + (v829_data * v845_data));
            float v850_data = s1[63];
            float v852_data = ir3[4];
            ir3[4] = (v852_data + (v829_data * v850_data));
            float v855_data = s1[76];
            float v857_data = ir3[5];
            ir3[5] = (v857_data + (v829_data * v855_data));
            float v860_data = s1[89];
            float v862_data = ir3[6];
            ir3[6] = (v862_data + (v829_data * v860_data));
            float v865_data = s1[102];
            float v867_data = ir3[7];
            ir3[7] = (v867_data + (v829_data * v865_data));
            float v870_data = s1[115];
            float v872_data = ir3[8];
            ir3[8] = (v872_data + (v829_data * v870_data));
            float v875_data = s1[128];
            float v877_data = ir3[9];
            ir3[9] = (v877_data + (v829_data * v875_data));
            float v880_data = s1[141];
            float v882_data = ir3[10];
            ir3[10] = (v882_data + (v829_data * v880_data));
            float v885_data = s1[154];
            float v887_data = ir3[11];
            ir3[11] = (v887_data + (v829_data * v885_data));
            float v890_data = s1[167];
            float v892_data = ir3[12];
            ir3[12] = (v892_data + (v829_data * v890_data));
            float v897_data = r2[12];
            float v898_data = s1[12];
            float v900_data = ir3[0];
            ir3[0] = (v900_data + (v897_data * v898_data));
            float v903_data = s1[25];
            float v905_data = ir3[1];
            ir3[1] = (v905_data + (v897_data * v903_data));
            float v908_data = s1[38];
            float v910_data = ir3[2];
            ir3[2] = (v910_data + (v897_data * v908_data));
            float v913_data = s1[51];
            float v915_data = ir3[3];
            ir3[3] = (v915_data + (v897_data * v913_data));
            float v918_data = s1[64];
            float v920_data = ir3[4];
            ir3[4] = (v920_data + (v897_data * v918_data));
            float v923_data = s1[77];
            float v925_data = ir3[5];
            ir3[5] = (v925_data + (v897_data * v923_data));
            float v928_data = s1[90];
            float v930_data = ir3[6];
            ir3[6] = (v930_data + (v897_data * v928_data));
            float v933_data = s1[103];
            float v935_data = ir3[7];
            ir3[7] = (v935_data + (v897_data * v933_data));
            float v938_data = s1[116];
            float v940_data = ir3[8];
            ir3[8] = (v940_data + (v897_data * v938_data));
            float v943_data = s1[129];
            float v945_data = ir3[9];
            ir3[9] = (v945_data + (v897_data * v943_data));
            float v948_data = s1[142];
            float v950_data = ir3[10];
            ir3[10] = (v950_data + (v897_data * v948_data));
            float v953_data = s1[155];
            float v955_data = ir3[11];
            ir3[11] = (v955_data + (v897_data * v953_data));
            float v958_data = s1[168];
            float v960_data = ir3[12];
            ir3[12] = (v960_data + (v897_data * v958_data));
            #pragma unroll
            for (int32_t v965_n0 = 0; v965_n0 < 1; ++v965_n0) {
              #pragma unroll
              for (int32_t v966_n1 = 0; v966_n1 < 13; ++v966_n1) {
                int32_t v967_a = v965_n0 + v966_n1;
                float v969_data = ir3[(v965_n0 + v966_n1)];
                int32_t v970_a = v965_n0 + v966_n1;
                r3[v970_a] = v969_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r3);
          int32_t v973_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v974_i0 = 0; v974_i0 < 1; ++v974_i0) {
            int32_t v983_lead = v973_lead + (v974_i0 * 32);
            #pragma unroll
            for (int32_t v975_i1 = 0; v975_i1 < 13; ++v975_i1) {
              int32_t v976_a = v974_i0 + v975_i1;
              float v978_data = r3[(v974_i0 + v975_i1)];
              int32_t v985_a = v983_lead + (v975_i1 * 32);
              glb_m3[v985_a] = v978_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

