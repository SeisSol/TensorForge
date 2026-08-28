// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 3072 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_7ab185b978, cudaFuncAttributeMaxDynamicSharedMemorySize, 3072 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7ab185b978<<<grid,block,3072 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×9(32×9) {0..32}×{0..9} pointer_based
    // m1 16×9(16×9) {0..16}×{0..9} pointer_based
    // m2 16×9(16×9) {0..16}×{0..9} pointer_based
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based
    // m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[384 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v18_lead = v13_i0 * 32;
            int32_t v19_lead = v12_lead + v18_lead;
            int32_t v26_lead = v12_lead + v18_lead;
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 9; ++v14_i1) {
              int32_t v20_a = v14_i1 * 32;
              int32_t v21_a = v19_lead + v20_a;
              float v29_data = __ldcg(&glb_m0[(v26_lead + v20_a)]);
              r0[(v13_i0 + v14_i1)] = v29_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
              int32_t v42_a = v36_i1 * 16;
              int32_t v43_a = v12_lead + v42_a;
              float v51_data = __ldcg(&glb_m1[(v12_lead + v42_a)]);
              r2[v36_i1] = v51_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v57_data = r0[0];
          float v58_data = r1[0];
          r1[0] = (v58_data + v57_data);
          float v60_data = r0[1];
          float v61_data = r1[1];
          r1[1] = (v61_data + v60_data);
          float v63_data = r0[2];
          float v64_data = r1[2];
          r1[2] = (v64_data + v63_data);
          float v66_data = r0[3];
          float v67_data = r1[3];
          r1[3] = (v67_data + v66_data);
          float v69_data = r0[4];
          float v70_data = r1[4];
          r1[4] = (v70_data + v69_data);
          float v72_data = r0[5];
          float v73_data = r1[5];
          r1[5] = (v73_data + v72_data);
          float v75_data = r0[6];
          float v76_data = r1[6];
          r1[6] = (v76_data + v75_data);
          float v78_data = r0[7];
          float v79_data = r1[7];
          r1[7] = (v79_data + v78_data);
          float v81_data = r0[8];
          float v82_data = r1[8];
          r1[8] = (v82_data + v81_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v88_i0 = 0; v88_i0 < 1; ++v88_i0) {
            int32_t v97_lead = v12_lead + (v88_i0 * 32);
            #pragma unroll
            for (int32_t v89_i1 = 0; v89_i1 < 9; ++v89_i1) {
              int32_t v90_a = v88_i0 + v89_i1;
              float v92_data = r1[(v88_i0 + v89_i1)];
              s0[(v97_lead + (v89_i1 * 32))] = v92_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v105_i1 = 0; v105_i1 < 9; ++v105_i1) {
              int32_t v111_a = v105_i1 * 16;
              int32_t v112_a = v12_lead + v111_a;
              float v120_data = __ldcg(&glb_m2[(v12_lead + v111_a)]);
              r4[v105_i1] = v120_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v12_lead < 16) {
            float v128_data = r2[0];
            float v129_data = ir3[0];
            ir3[0] = (v129_data + v128_data);
            float v131_data = r2[1];
            float v132_data = ir3[1];
            ir3[1] = (v132_data + v131_data);
            float v134_data = r2[2];
            float v135_data = ir3[2];
            ir3[2] = (v135_data + v134_data);
            float v137_data = r2[3];
            float v138_data = ir3[3];
            ir3[3] = (v138_data + v137_data);
            float v140_data = r2[4];
            float v141_data = ir3[4];
            ir3[4] = (v141_data + v140_data);
            float v143_data = r2[5];
            float v144_data = ir3[5];
            ir3[5] = (v144_data + v143_data);
            float v146_data = r2[6];
            float v147_data = ir3[6];
            ir3[6] = (v147_data + v146_data);
            float v149_data = r2[7];
            float v150_data = ir3[7];
            ir3[7] = (v150_data + v149_data);
            float v152_data = r2[8];
            float v153_data = ir3[8];
            ir3[8] = (v153_data + v152_data);
          }
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v159_n1 = 0; v159_n1 < 9; ++v159_n1) {
              int32_t v160_a = 0 + v159_n1;
              float v162_data = ir3[v159_n1];
              int32_t v168_a = v159_n1 * 32;
              int32_t v169_a = v12_lead + v168_a;
              float v177_data = s0[(v12_lead + v168_a)];
              r3[v159_n1] = (v177_data + v162_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v184_i1 = 0; v184_i1 < 9; ++v184_i1) {
              int32_t v185_a = 0 + v184_i1;
              float v187_data = r3[v184_i1];
              s0[(v12_lead + (v184_i1 * 32))] = v187_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v12_lead < 16) {
            float v201_data = r4[0];
            float v202_data = ir5[0];
            ir5[0] = (v202_data + v201_data);
            float v204_data = r4[1];
            float v205_data = ir5[1];
            ir5[1] = (v205_data + v204_data);
            float v207_data = r4[2];
            float v208_data = ir5[2];
            ir5[2] = (v208_data + v207_data);
            float v210_data = r4[3];
            float v211_data = ir5[3];
            ir5[3] = (v211_data + v210_data);
            float v213_data = r4[4];
            float v214_data = ir5[4];
            ir5[4] = (v214_data + v213_data);
            float v216_data = r4[5];
            float v217_data = ir5[5];
            ir5[5] = (v217_data + v216_data);
            float v219_data = r4[6];
            float v220_data = ir5[6];
            ir5[6] = (v220_data + v219_data);
            float v222_data = r4[7];
            float v223_data = ir5[7];
            ir5[7] = (v223_data + v222_data);
            float v225_data = r4[8];
            float v226_data = ir5[8];
            ir5[8] = (v226_data + v225_data);
          }
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v232_n1 = 0; v232_n1 < 9; ++v232_n1) {
              int32_t v233_a = 0 + v232_n1;
              float v235_data = ir5[v232_n1];
              int32_t v241_a = v232_n1 * 32;
              int32_t v242_a = v12_lead + v241_a;
              float v250_data = s0[(v12_lead + v241_a)];
              r5[v232_n1] = (v250_data + v235_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v257_i1 = 0; v257_i1 < 9; ++v257_i1) {
              int32_t v258_a = 0 + v257_i1;
              float v260_data = r5[v257_i1];
              s0[(v12_lead + (v257_i1 * 32))] = v260_data;
            }
          }
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
            __pipeline_commit();
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          int32_t v282_a = v12_lead + 0;
          float v289_data = s0[v12_lead];
          float v290_data = s1[0];
          float v292_data = ir6[0];
          ir6[0] = (v292_data + (v289_data * v290_data));
          int32_t v299_a = v12_lead + 0;
          float v306_data = s0[v12_lead];
          float v307_data = s1[9];
          float v309_data = ir6[1];
          ir6[1] = (v309_data + (v306_data * v307_data));
          int32_t v316_a = v12_lead + 0;
          float v323_data = s0[v12_lead];
          float v324_data = s1[18];
          float v326_data = ir6[2];
          ir6[2] = (v326_data + (v323_data * v324_data));
          int32_t v333_a = v12_lead + 0;
          float v340_data = s0[v12_lead];
          float v341_data = s1[27];
          float v343_data = ir6[3];
          ir6[3] = (v343_data + (v340_data * v341_data));
          int32_t v350_a = v12_lead + 0;
          float v357_data = s0[v12_lead];
          float v358_data = s1[36];
          float v360_data = ir6[4];
          ir6[4] = (v360_data + (v357_data * v358_data));
          int32_t v367_a = v12_lead + 0;
          float v374_data = s0[v12_lead];
          float v375_data = s1[45];
          float v377_data = ir6[5];
          ir6[5] = (v377_data + (v374_data * v375_data));
          int32_t v384_a = v12_lead + 0;
          float v391_data = s0[v12_lead];
          float v392_data = s1[54];
          float v394_data = ir6[6];
          ir6[6] = (v394_data + (v391_data * v392_data));
          int32_t v401_a = v12_lead + 0;
          float v408_data = s0[v12_lead];
          float v409_data = s1[63];
          float v411_data = ir6[7];
          ir6[7] = (v411_data + (v408_data * v409_data));
          int32_t v418_a = v12_lead + 0;
          float v425_data = s0[v12_lead];
          float v426_data = s1[72];
          float v428_data = ir6[8];
          ir6[8] = (v428_data + (v425_data * v426_data));
          int32_t v438_a = v12_lead + 32;
          float v445_data = s0[(v12_lead + 32)];
          float v446_data = s1[1];
          float v448_data = ir6[0];
          ir6[0] = (v448_data + (v445_data * v446_data));
          int32_t v455_a = v12_lead + 32;
          float v462_data = s0[(v12_lead + 32)];
          float v463_data = s1[10];
          float v465_data = ir6[1];
          ir6[1] = (v465_data + (v462_data * v463_data));
          int32_t v472_a = v12_lead + 32;
          float v479_data = s0[(v12_lead + 32)];
          float v480_data = s1[19];
          float v482_data = ir6[2];
          ir6[2] = (v482_data + (v479_data * v480_data));
          int32_t v489_a = v12_lead + 32;
          float v496_data = s0[(v12_lead + 32)];
          float v497_data = s1[28];
          float v499_data = ir6[3];
          ir6[3] = (v499_data + (v496_data * v497_data));
          int32_t v506_a = v12_lead + 32;
          float v513_data = s0[(v12_lead + 32)];
          float v514_data = s1[37];
          float v516_data = ir6[4];
          ir6[4] = (v516_data + (v513_data * v514_data));
          int32_t v523_a = v12_lead + 32;
          float v530_data = s0[(v12_lead + 32)];
          float v531_data = s1[46];
          float v533_data = ir6[5];
          ir6[5] = (v533_data + (v530_data * v531_data));
          int32_t v540_a = v12_lead + 32;
          float v547_data = s0[(v12_lead + 32)];
          float v548_data = s1[55];
          float v550_data = ir6[6];
          ir6[6] = (v550_data + (v547_data * v548_data));
          int32_t v557_a = v12_lead + 32;
          float v564_data = s0[(v12_lead + 32)];
          float v565_data = s1[64];
          float v567_data = ir6[7];
          ir6[7] = (v567_data + (v564_data * v565_data));
          int32_t v574_a = v12_lead + 32;
          float v581_data = s0[(v12_lead + 32)];
          float v582_data = s1[73];
          float v584_data = ir6[8];
          ir6[8] = (v584_data + (v581_data * v582_data));
          int32_t v594_a = v12_lead + 64;
          float v601_data = s0[(v12_lead + 64)];
          float v602_data = s1[2];
          float v604_data = ir6[0];
          ir6[0] = (v604_data + (v601_data * v602_data));
          int32_t v611_a = v12_lead + 64;
          float v618_data = s0[(v12_lead + 64)];
          float v619_data = s1[11];
          float v621_data = ir6[1];
          ir6[1] = (v621_data + (v618_data * v619_data));
          int32_t v628_a = v12_lead + 64;
          float v635_data = s0[(v12_lead + 64)];
          float v636_data = s1[20];
          float v638_data = ir6[2];
          ir6[2] = (v638_data + (v635_data * v636_data));
          int32_t v645_a = v12_lead + 64;
          float v652_data = s0[(v12_lead + 64)];
          float v653_data = s1[29];
          float v655_data = ir6[3];
          ir6[3] = (v655_data + (v652_data * v653_data));
          int32_t v662_a = v12_lead + 64;
          float v669_data = s0[(v12_lead + 64)];
          float v670_data = s1[38];
          float v672_data = ir6[4];
          ir6[4] = (v672_data + (v669_data * v670_data));
          int32_t v679_a = v12_lead + 64;
          float v686_data = s0[(v12_lead + 64)];
          float v687_data = s1[47];
          float v689_data = ir6[5];
          ir6[5] = (v689_data + (v686_data * v687_data));
          int32_t v696_a = v12_lead + 64;
          float v703_data = s0[(v12_lead + 64)];
          float v704_data = s1[56];
          float v706_data = ir6[6];
          ir6[6] = (v706_data + (v703_data * v704_data));
          int32_t v713_a = v12_lead + 64;
          float v720_data = s0[(v12_lead + 64)];
          float v721_data = s1[65];
          float v723_data = ir6[7];
          ir6[7] = (v723_data + (v720_data * v721_data));
          int32_t v730_a = v12_lead + 64;
          float v737_data = s0[(v12_lead + 64)];
          float v738_data = s1[74];
          float v740_data = ir6[8];
          ir6[8] = (v740_data + (v737_data * v738_data));
          int32_t v750_a = v12_lead + 96;
          float v757_data = s0[(v12_lead + 96)];
          float v758_data = s1[3];
          float v760_data = ir6[0];
          ir6[0] = (v760_data + (v757_data * v758_data));
          int32_t v767_a = v12_lead + 96;
          float v774_data = s0[(v12_lead + 96)];
          float v775_data = s1[12];
          float v777_data = ir6[1];
          ir6[1] = (v777_data + (v774_data * v775_data));
          int32_t v784_a = v12_lead + 96;
          float v791_data = s0[(v12_lead + 96)];
          float v792_data = s1[21];
          float v794_data = ir6[2];
          ir6[2] = (v794_data + (v791_data * v792_data));
          int32_t v801_a = v12_lead + 96;
          float v808_data = s0[(v12_lead + 96)];
          float v809_data = s1[30];
          float v811_data = ir6[3];
          ir6[3] = (v811_data + (v808_data * v809_data));
          int32_t v818_a = v12_lead + 96;
          float v825_data = s0[(v12_lead + 96)];
          float v826_data = s1[39];
          float v828_data = ir6[4];
          ir6[4] = (v828_data + (v825_data * v826_data));
          int32_t v835_a = v12_lead + 96;
          float v842_data = s0[(v12_lead + 96)];
          float v843_data = s1[48];
          float v845_data = ir6[5];
          ir6[5] = (v845_data + (v842_data * v843_data));
          int32_t v852_a = v12_lead + 96;
          float v859_data = s0[(v12_lead + 96)];
          float v860_data = s1[57];
          float v862_data = ir6[6];
          ir6[6] = (v862_data + (v859_data * v860_data));
          int32_t v869_a = v12_lead + 96;
          float v876_data = s0[(v12_lead + 96)];
          float v877_data = s1[66];
          float v879_data = ir6[7];
          ir6[7] = (v879_data + (v876_data * v877_data));
          int32_t v886_a = v12_lead + 96;
          float v893_data = s0[(v12_lead + 96)];
          float v894_data = s1[75];
          float v896_data = ir6[8];
          ir6[8] = (v896_data + (v893_data * v894_data));
          int32_t v906_a = v12_lead + 128;
          float v913_data = s0[(v12_lead + 128)];
          float v914_data = s1[4];
          float v916_data = ir6[0];
          ir6[0] = (v916_data + (v913_data * v914_data));
          int32_t v923_a = v12_lead + 128;
          float v930_data = s0[(v12_lead + 128)];
          float v931_data = s1[13];
          float v933_data = ir6[1];
          ir6[1] = (v933_data + (v930_data * v931_data));
          int32_t v940_a = v12_lead + 128;
          float v947_data = s0[(v12_lead + 128)];
          float v948_data = s1[22];
          float v950_data = ir6[2];
          ir6[2] = (v950_data + (v947_data * v948_data));
          int32_t v957_a = v12_lead + 128;
          float v964_data = s0[(v12_lead + 128)];
          float v965_data = s1[31];
          float v967_data = ir6[3];
          ir6[3] = (v967_data + (v964_data * v965_data));
          int32_t v974_a = v12_lead + 128;
          float v981_data = s0[(v12_lead + 128)];
          float v982_data = s1[40];
          float v984_data = ir6[4];
          ir6[4] = (v984_data + (v981_data * v982_data));
          int32_t v991_a = v12_lead + 128;
          float v998_data = s0[(v12_lead + 128)];
          float v999_data = s1[49];
          float v1001_data = ir6[5];
          ir6[5] = (v1001_data + (v998_data * v999_data));
          int32_t v1008_a = v12_lead + 128;
          float v1015_data = s0[(v12_lead + 128)];
          float v1016_data = s1[58];
          float v1018_data = ir6[6];
          ir6[6] = (v1018_data + (v1015_data * v1016_data));
          int32_t v1025_a = v12_lead + 128;
          float v1032_data = s0[(v12_lead + 128)];
          float v1033_data = s1[67];
          float v1035_data = ir6[7];
          ir6[7] = (v1035_data + (v1032_data * v1033_data));
          int32_t v1042_a = v12_lead + 128;
          float v1049_data = s0[(v12_lead + 128)];
          float v1050_data = s1[76];
          float v1052_data = ir6[8];
          ir6[8] = (v1052_data + (v1049_data * v1050_data));
          int32_t v1062_a = v12_lead + 160;
          float v1069_data = s0[(v12_lead + 160)];
          float v1070_data = s1[5];
          float v1072_data = ir6[0];
          ir6[0] = (v1072_data + (v1069_data * v1070_data));
          int32_t v1079_a = v12_lead + 160;
          float v1086_data = s0[(v12_lead + 160)];
          float v1087_data = s1[14];
          float v1089_data = ir6[1];
          ir6[1] = (v1089_data + (v1086_data * v1087_data));
          int32_t v1096_a = v12_lead + 160;
          float v1103_data = s0[(v12_lead + 160)];
          float v1104_data = s1[23];
          float v1106_data = ir6[2];
          ir6[2] = (v1106_data + (v1103_data * v1104_data));
          int32_t v1113_a = v12_lead + 160;
          float v1120_data = s0[(v12_lead + 160)];
          float v1121_data = s1[32];
          float v1123_data = ir6[3];
          ir6[3] = (v1123_data + (v1120_data * v1121_data));
          int32_t v1130_a = v12_lead + 160;
          float v1137_data = s0[(v12_lead + 160)];
          float v1138_data = s1[41];
          float v1140_data = ir6[4];
          ir6[4] = (v1140_data + (v1137_data * v1138_data));
          int32_t v1147_a = v12_lead + 160;
          float v1154_data = s0[(v12_lead + 160)];
          float v1155_data = s1[50];
          float v1157_data = ir6[5];
          ir6[5] = (v1157_data + (v1154_data * v1155_data));
          int32_t v1164_a = v12_lead + 160;
          float v1171_data = s0[(v12_lead + 160)];
          float v1172_data = s1[59];
          float v1174_data = ir6[6];
          ir6[6] = (v1174_data + (v1171_data * v1172_data));
          int32_t v1181_a = v12_lead + 160;
          float v1188_data = s0[(v12_lead + 160)];
          float v1189_data = s1[68];
          float v1191_data = ir6[7];
          ir6[7] = (v1191_data + (v1188_data * v1189_data));
          int32_t v1198_a = v12_lead + 160;
          float v1205_data = s0[(v12_lead + 160)];
          float v1206_data = s1[77];
          float v1208_data = ir6[8];
          ir6[8] = (v1208_data + (v1205_data * v1206_data));
          int32_t v1218_a = v12_lead + 192;
          float v1225_data = s0[(v12_lead + 192)];
          float v1226_data = s1[6];
          float v1228_data = ir6[0];
          ir6[0] = (v1228_data + (v1225_data * v1226_data));
          int32_t v1235_a = v12_lead + 192;
          float v1242_data = s0[(v12_lead + 192)];
          float v1243_data = s1[15];
          float v1245_data = ir6[1];
          ir6[1] = (v1245_data + (v1242_data * v1243_data));
          int32_t v1252_a = v12_lead + 192;
          float v1259_data = s0[(v12_lead + 192)];
          float v1260_data = s1[24];
          float v1262_data = ir6[2];
          ir6[2] = (v1262_data + (v1259_data * v1260_data));
          int32_t v1269_a = v12_lead + 192;
          float v1276_data = s0[(v12_lead + 192)];
          float v1277_data = s1[33];
          float v1279_data = ir6[3];
          ir6[3] = (v1279_data + (v1276_data * v1277_data));
          int32_t v1286_a = v12_lead + 192;
          float v1293_data = s0[(v12_lead + 192)];
          float v1294_data = s1[42];
          float v1296_data = ir6[4];
          ir6[4] = (v1296_data + (v1293_data * v1294_data));
          int32_t v1303_a = v12_lead + 192;
          float v1310_data = s0[(v12_lead + 192)];
          float v1311_data = s1[51];
          float v1313_data = ir6[5];
          ir6[5] = (v1313_data + (v1310_data * v1311_data));
          int32_t v1320_a = v12_lead + 192;
          float v1327_data = s0[(v12_lead + 192)];
          float v1328_data = s1[60];
          float v1330_data = ir6[6];
          ir6[6] = (v1330_data + (v1327_data * v1328_data));
          int32_t v1337_a = v12_lead + 192;
          float v1344_data = s0[(v12_lead + 192)];
          float v1345_data = s1[69];
          float v1347_data = ir6[7];
          ir6[7] = (v1347_data + (v1344_data * v1345_data));
          int32_t v1354_a = v12_lead + 192;
          float v1361_data = s0[(v12_lead + 192)];
          float v1362_data = s1[78];
          float v1364_data = ir6[8];
          ir6[8] = (v1364_data + (v1361_data * v1362_data));
          int32_t v1374_a = v12_lead + 224;
          float v1381_data = s0[(v12_lead + 224)];
          float v1382_data = s1[7];
          float v1384_data = ir6[0];
          ir6[0] = (v1384_data + (v1381_data * v1382_data));
          int32_t v1391_a = v12_lead + 224;
          float v1398_data = s0[(v12_lead + 224)];
          float v1399_data = s1[16];
          float v1401_data = ir6[1];
          ir6[1] = (v1401_data + (v1398_data * v1399_data));
          int32_t v1408_a = v12_lead + 224;
          float v1415_data = s0[(v12_lead + 224)];
          float v1416_data = s1[25];
          float v1418_data = ir6[2];
          ir6[2] = (v1418_data + (v1415_data * v1416_data));
          int32_t v1425_a = v12_lead + 224;
          float v1432_data = s0[(v12_lead + 224)];
          float v1433_data = s1[34];
          float v1435_data = ir6[3];
          ir6[3] = (v1435_data + (v1432_data * v1433_data));
          int32_t v1442_a = v12_lead + 224;
          float v1449_data = s0[(v12_lead + 224)];
          float v1450_data = s1[43];
          float v1452_data = ir6[4];
          ir6[4] = (v1452_data + (v1449_data * v1450_data));
          int32_t v1459_a = v12_lead + 224;
          float v1466_data = s0[(v12_lead + 224)];
          float v1467_data = s1[52];
          float v1469_data = ir6[5];
          ir6[5] = (v1469_data + (v1466_data * v1467_data));
          int32_t v1476_a = v12_lead + 224;
          float v1483_data = s0[(v12_lead + 224)];
          float v1484_data = s1[61];
          float v1486_data = ir6[6];
          ir6[6] = (v1486_data + (v1483_data * v1484_data));
          int32_t v1493_a = v12_lead + 224;
          float v1500_data = s0[(v12_lead + 224)];
          float v1501_data = s1[70];
          float v1503_data = ir6[7];
          ir6[7] = (v1503_data + (v1500_data * v1501_data));
          int32_t v1510_a = v12_lead + 224;
          float v1517_data = s0[(v12_lead + 224)];
          float v1518_data = s1[79];
          float v1520_data = ir6[8];
          ir6[8] = (v1520_data + (v1517_data * v1518_data));
          int32_t v1530_a = v12_lead + 256;
          float v1537_data = s0[(v12_lead + 256)];
          float v1538_data = s1[8];
          float v1540_data = ir6[0];
          ir6[0] = (v1540_data + (v1537_data * v1538_data));
          int32_t v1547_a = v12_lead + 256;
          float v1554_data = s0[(v12_lead + 256)];
          float v1555_data = s1[17];
          float v1557_data = ir6[1];
          ir6[1] = (v1557_data + (v1554_data * v1555_data));
          int32_t v1564_a = v12_lead + 256;
          float v1571_data = s0[(v12_lead + 256)];
          float v1572_data = s1[26];
          float v1574_data = ir6[2];
          ir6[2] = (v1574_data + (v1571_data * v1572_data));
          int32_t v1581_a = v12_lead + 256;
          float v1588_data = s0[(v12_lead + 256)];
          float v1589_data = s1[35];
          float v1591_data = ir6[3];
          ir6[3] = (v1591_data + (v1588_data * v1589_data));
          int32_t v1598_a = v12_lead + 256;
          float v1605_data = s0[(v12_lead + 256)];
          float v1606_data = s1[44];
          float v1608_data = ir6[4];
          ir6[4] = (v1608_data + (v1605_data * v1606_data));
          int32_t v1615_a = v12_lead + 256;
          float v1622_data = s0[(v12_lead + 256)];
          float v1623_data = s1[53];
          float v1625_data = ir6[5];
          ir6[5] = (v1625_data + (v1622_data * v1623_data));
          int32_t v1632_a = v12_lead + 256;
          float v1639_data = s0[(v12_lead + 256)];
          float v1640_data = s1[62];
          float v1642_data = ir6[6];
          ir6[6] = (v1642_data + (v1639_data * v1640_data));
          int32_t v1649_a = v12_lead + 256;
          float v1656_data = s0[(v12_lead + 256)];
          float v1657_data = s1[71];
          float v1659_data = ir6[7];
          ir6[7] = (v1659_data + (v1656_data * v1657_data));
          int32_t v1666_a = v12_lead + 256;
          float v1673_data = s0[(v12_lead + 256)];
          float v1674_data = s1[80];
          float v1676_data = ir6[8];
          ir6[8] = (v1676_data + (v1673_data * v1674_data));
          #pragma unroll
          for (int32_t v1681_n0 = 0; v1681_n0 < 1; ++v1681_n0) {
            #pragma unroll
            for (int32_t v1682_n1 = 0; v1682_n1 < 9; ++v1682_n1) {
              int32_t v1683_a = v1681_n0 + v1682_n1;
              int32_t v1684_a = v1681_n0 + v1682_n1;
              float v1685_data = ir6[v1684_a];
              r6[v1684_a] = v1685_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1690_i0 = 0; v1690_i0 < 1; ++v1690_i0) {
            int32_t v1699_lead = v12_lead + (v1690_i0 * 32);
            #pragma unroll
            for (int32_t v1691_i1 = 0; v1691_i1 < 9; ++v1691_i1) {
              int32_t v1692_a = v1690_i0 + v1691_i1;
              float v1694_data = r6[(v1690_i0 + v1691_i1)];
              glb_m3[(v1699_lead + (v1691_i1 * 32))] = v1694_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

