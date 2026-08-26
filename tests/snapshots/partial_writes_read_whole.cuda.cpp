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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __ldcg(&glb_m0[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              int32_t v33_a = v27_i1 * 16;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __ldcg(&glb_m1[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v48_data = r0[0];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + v48_data);
          float v51_data = r0[1];
          float v52_data = ir1[1];
          ir1[1] = (v52_data + v51_data);
          float v54_data = r0[2];
          float v55_data = ir1[2];
          ir1[2] = (v55_data + v54_data);
          float v57_data = r0[3];
          float v58_data = ir1[3];
          ir1[3] = (v58_data + v57_data);
          float v60_data = r0[4];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + v60_data);
          float v63_data = r0[5];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + v63_data);
          float v66_data = r0[6];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + v66_data);
          float v69_data = r0[7];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + v69_data);
          float v72_data = r0[8];
          float v73_data = ir1[8];
          ir1[8] = (v73_data + v72_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v78_i0 = 0; v78_i0 < 1; ++v78_i0) {
            int32_t v87_lead = v3_lead + (v78_i0 * 32);
            #pragma unroll
            for (int32_t v79_i1 = 0; v79_i1 < 9; ++v79_i1) {
              int32_t v80_a = v78_i0 + v79_i1;
              float v82_data = r1[(v78_i0 + v79_i1)];
              int32_t v89_a = v87_lead + (v79_i1 * 32);
              s0[v89_a] = v82_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v95_i1 = 0; v95_i1 < 9; ++v95_i1) {
              int32_t v101_a = v95_i1 * 16;
              int32_t v102_a = v3_lead + v101_a;
              float v110_data = __ldcg(&glb_m2[(v3_lead + v101_a)]);
              int32_t v111_a = 0 + v95_i1;
              r4[v111_a] = v110_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          {
            // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir3[9]{};
            if (v3_lead < 16) {
              float v117_data = r2[0];
              float v118_data = ir3[0];
              ir3[0] = (v118_data + v117_data);
              float v120_data = r2[1];
              float v121_data = ir3[1];
              ir3[1] = (v121_data + v120_data);
              float v123_data = r2[2];
              float v124_data = ir3[2];
              ir3[2] = (v124_data + v123_data);
              float v126_data = r2[3];
              float v127_data = ir3[3];
              ir3[3] = (v127_data + v126_data);
              float v129_data = r2[4];
              float v130_data = ir3[4];
              ir3[4] = (v130_data + v129_data);
              float v132_data = r2[5];
              float v133_data = ir3[5];
              ir3[5] = (v133_data + v132_data);
              float v135_data = r2[6];
              float v136_data = ir3[6];
              ir3[6] = (v136_data + v135_data);
              float v138_data = r2[7];
              float v139_data = ir3[7];
              ir3[7] = (v139_data + v138_data);
              float v141_data = r2[8];
              float v142_data = ir3[8];
              ir3[8] = (v142_data + v141_data);
            }
            if (v3_lead < 16) {
              #pragma unroll
              for (int32_t v148_n1 = 0; v148_n1 < 9; ++v148_n1) {
                int32_t v149_a = 0 + v148_n1;
                float v151_data = ir3[v148_n1];
                int32_t v157_a = v148_n1 * 32;
                int32_t v158_a = v3_lead + v157_a;
                float v166_data = s0[(v3_lead + v157_a)];
                int32_t v168_a = 0 + v148_n1;
                r3[v148_n1] = (v166_data + v151_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v174_i1 = 0; v174_i1 < 9; ++v174_i1) {
              int32_t v175_a = 0 + v174_i1;
              float v177_data = r3[v174_i1];
              int32_t v184_a = v3_lead + (v174_i1 * 32);
              s0[v184_a] = v177_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          {
            // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir5[9]{};
            if (v3_lead < 16) {
              float v190_data = r4[0];
              float v191_data = ir5[0];
              ir5[0] = (v191_data + v190_data);
              float v193_data = r4[1];
              float v194_data = ir5[1];
              ir5[1] = (v194_data + v193_data);
              float v196_data = r4[2];
              float v197_data = ir5[2];
              ir5[2] = (v197_data + v196_data);
              float v199_data = r4[3];
              float v200_data = ir5[3];
              ir5[3] = (v200_data + v199_data);
              float v202_data = r4[4];
              float v203_data = ir5[4];
              ir5[4] = (v203_data + v202_data);
              float v205_data = r4[5];
              float v206_data = ir5[5];
              ir5[5] = (v206_data + v205_data);
              float v208_data = r4[6];
              float v209_data = ir5[6];
              ir5[6] = (v209_data + v208_data);
              float v211_data = r4[7];
              float v212_data = ir5[7];
              ir5[7] = (v212_data + v211_data);
              float v214_data = r4[8];
              float v215_data = ir5[8];
              ir5[8] = (v215_data + v214_data);
            }
            if (v3_lead < 16) {
              #pragma unroll
              for (int32_t v221_n1 = 0; v221_n1 < 9; ++v221_n1) {
                int32_t v222_a = 0 + v221_n1;
                float v224_data = ir5[v221_n1];
                int32_t v230_a = v221_n1 * 32;
                int32_t v231_a = v3_lead + v230_a;
                float v239_data = s0[(v3_lead + v230_a)];
                int32_t v241_a = 0 + v221_n1;
                r5[v221_n1] = (v239_data + v224_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v247_i1 = 0; v247_i1 < 9; ++v247_i1) {
              int32_t v248_a = 0 + v247_i1;
              float v250_data = r5[v247_i1];
              int32_t v257_a = v3_lead + (v247_i1 * 32);
              s0[v257_a] = v250_data;
            }
          }
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<4>(4), pipeline);
          cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<4>(4), pipeline);
          if (threadIdx.x < 17) {
            cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], cuda::aligned_size_t<4>(4), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r6[9]{};
          __syncwarp();
          {
            // r6 = +(s0 * s1) + None
            // [(0, 32), (0, 9)] [(0, 9)]
            float ir6[9]{};
            int32_t v267_a = v3_lead + 0;
            float v274_data = s0[v3_lead];
            float v275_data = s1[0];
            float v277_data = ir6[0];
            ir6[0] = (v277_data + (v274_data * v275_data));
            int32_t v284_a = v3_lead + 0;
            float v291_data = s0[v3_lead];
            float v292_data = s1[9];
            float v294_data = ir6[1];
            ir6[1] = (v294_data + (v291_data * v292_data));
            int32_t v301_a = v3_lead + 0;
            float v308_data = s0[v3_lead];
            float v309_data = s1[18];
            float v311_data = ir6[2];
            ir6[2] = (v311_data + (v308_data * v309_data));
            int32_t v318_a = v3_lead + 0;
            float v325_data = s0[v3_lead];
            float v326_data = s1[27];
            float v328_data = ir6[3];
            ir6[3] = (v328_data + (v325_data * v326_data));
            int32_t v335_a = v3_lead + 0;
            float v342_data = s0[v3_lead];
            float v343_data = s1[36];
            float v345_data = ir6[4];
            ir6[4] = (v345_data + (v342_data * v343_data));
            int32_t v352_a = v3_lead + 0;
            float v359_data = s0[v3_lead];
            float v360_data = s1[45];
            float v362_data = ir6[5];
            ir6[5] = (v362_data + (v359_data * v360_data));
            int32_t v369_a = v3_lead + 0;
            float v376_data = s0[v3_lead];
            float v377_data = s1[54];
            float v379_data = ir6[6];
            ir6[6] = (v379_data + (v376_data * v377_data));
            int32_t v386_a = v3_lead + 0;
            float v393_data = s0[v3_lead];
            float v394_data = s1[63];
            float v396_data = ir6[7];
            ir6[7] = (v396_data + (v393_data * v394_data));
            int32_t v403_a = v3_lead + 0;
            float v410_data = s0[v3_lead];
            float v411_data = s1[72];
            float v413_data = ir6[8];
            ir6[8] = (v413_data + (v410_data * v411_data));
            int32_t v423_a = v3_lead + 32;
            float v430_data = s0[(v3_lead + 32)];
            float v431_data = s1[1];
            float v433_data = ir6[0];
            ir6[0] = (v433_data + (v430_data * v431_data));
            int32_t v440_a = v3_lead + 32;
            float v447_data = s0[(v3_lead + 32)];
            float v448_data = s1[10];
            float v450_data = ir6[1];
            ir6[1] = (v450_data + (v447_data * v448_data));
            int32_t v457_a = v3_lead + 32;
            float v464_data = s0[(v3_lead + 32)];
            float v465_data = s1[19];
            float v467_data = ir6[2];
            ir6[2] = (v467_data + (v464_data * v465_data));
            int32_t v474_a = v3_lead + 32;
            float v481_data = s0[(v3_lead + 32)];
            float v482_data = s1[28];
            float v484_data = ir6[3];
            ir6[3] = (v484_data + (v481_data * v482_data));
            int32_t v491_a = v3_lead + 32;
            float v498_data = s0[(v3_lead + 32)];
            float v499_data = s1[37];
            float v501_data = ir6[4];
            ir6[4] = (v501_data + (v498_data * v499_data));
            int32_t v508_a = v3_lead + 32;
            float v515_data = s0[(v3_lead + 32)];
            float v516_data = s1[46];
            float v518_data = ir6[5];
            ir6[5] = (v518_data + (v515_data * v516_data));
            int32_t v525_a = v3_lead + 32;
            float v532_data = s0[(v3_lead + 32)];
            float v533_data = s1[55];
            float v535_data = ir6[6];
            ir6[6] = (v535_data + (v532_data * v533_data));
            int32_t v542_a = v3_lead + 32;
            float v549_data = s0[(v3_lead + 32)];
            float v550_data = s1[64];
            float v552_data = ir6[7];
            ir6[7] = (v552_data + (v549_data * v550_data));
            int32_t v559_a = v3_lead + 32;
            float v566_data = s0[(v3_lead + 32)];
            float v567_data = s1[73];
            float v569_data = ir6[8];
            ir6[8] = (v569_data + (v566_data * v567_data));
            int32_t v579_a = v3_lead + 64;
            float v586_data = s0[(v3_lead + 64)];
            float v587_data = s1[2];
            float v589_data = ir6[0];
            ir6[0] = (v589_data + (v586_data * v587_data));
            int32_t v596_a = v3_lead + 64;
            float v603_data = s0[(v3_lead + 64)];
            float v604_data = s1[11];
            float v606_data = ir6[1];
            ir6[1] = (v606_data + (v603_data * v604_data));
            int32_t v613_a = v3_lead + 64;
            float v620_data = s0[(v3_lead + 64)];
            float v621_data = s1[20];
            float v623_data = ir6[2];
            ir6[2] = (v623_data + (v620_data * v621_data));
            int32_t v630_a = v3_lead + 64;
            float v637_data = s0[(v3_lead + 64)];
            float v638_data = s1[29];
            float v640_data = ir6[3];
            ir6[3] = (v640_data + (v637_data * v638_data));
            int32_t v647_a = v3_lead + 64;
            float v654_data = s0[(v3_lead + 64)];
            float v655_data = s1[38];
            float v657_data = ir6[4];
            ir6[4] = (v657_data + (v654_data * v655_data));
            int32_t v664_a = v3_lead + 64;
            float v671_data = s0[(v3_lead + 64)];
            float v672_data = s1[47];
            float v674_data = ir6[5];
            ir6[5] = (v674_data + (v671_data * v672_data));
            int32_t v681_a = v3_lead + 64;
            float v688_data = s0[(v3_lead + 64)];
            float v689_data = s1[56];
            float v691_data = ir6[6];
            ir6[6] = (v691_data + (v688_data * v689_data));
            int32_t v698_a = v3_lead + 64;
            float v705_data = s0[(v3_lead + 64)];
            float v706_data = s1[65];
            float v708_data = ir6[7];
            ir6[7] = (v708_data + (v705_data * v706_data));
            int32_t v715_a = v3_lead + 64;
            float v722_data = s0[(v3_lead + 64)];
            float v723_data = s1[74];
            float v725_data = ir6[8];
            ir6[8] = (v725_data + (v722_data * v723_data));
            int32_t v735_a = v3_lead + 96;
            float v742_data = s0[(v3_lead + 96)];
            float v743_data = s1[3];
            float v745_data = ir6[0];
            ir6[0] = (v745_data + (v742_data * v743_data));
            int32_t v752_a = v3_lead + 96;
            float v759_data = s0[(v3_lead + 96)];
            float v760_data = s1[12];
            float v762_data = ir6[1];
            ir6[1] = (v762_data + (v759_data * v760_data));
            int32_t v769_a = v3_lead + 96;
            float v776_data = s0[(v3_lead + 96)];
            float v777_data = s1[21];
            float v779_data = ir6[2];
            ir6[2] = (v779_data + (v776_data * v777_data));
            int32_t v786_a = v3_lead + 96;
            float v793_data = s0[(v3_lead + 96)];
            float v794_data = s1[30];
            float v796_data = ir6[3];
            ir6[3] = (v796_data + (v793_data * v794_data));
            int32_t v803_a = v3_lead + 96;
            float v810_data = s0[(v3_lead + 96)];
            float v811_data = s1[39];
            float v813_data = ir6[4];
            ir6[4] = (v813_data + (v810_data * v811_data));
            int32_t v820_a = v3_lead + 96;
            float v827_data = s0[(v3_lead + 96)];
            float v828_data = s1[48];
            float v830_data = ir6[5];
            ir6[5] = (v830_data + (v827_data * v828_data));
            int32_t v837_a = v3_lead + 96;
            float v844_data = s0[(v3_lead + 96)];
            float v845_data = s1[57];
            float v847_data = ir6[6];
            ir6[6] = (v847_data + (v844_data * v845_data));
            int32_t v854_a = v3_lead + 96;
            float v861_data = s0[(v3_lead + 96)];
            float v862_data = s1[66];
            float v864_data = ir6[7];
            ir6[7] = (v864_data + (v861_data * v862_data));
            int32_t v871_a = v3_lead + 96;
            float v878_data = s0[(v3_lead + 96)];
            float v879_data = s1[75];
            float v881_data = ir6[8];
            ir6[8] = (v881_data + (v878_data * v879_data));
            int32_t v891_a = v3_lead + 128;
            float v898_data = s0[(v3_lead + 128)];
            float v899_data = s1[4];
            float v901_data = ir6[0];
            ir6[0] = (v901_data + (v898_data * v899_data));
            int32_t v908_a = v3_lead + 128;
            float v915_data = s0[(v3_lead + 128)];
            float v916_data = s1[13];
            float v918_data = ir6[1];
            ir6[1] = (v918_data + (v915_data * v916_data));
            int32_t v925_a = v3_lead + 128;
            float v932_data = s0[(v3_lead + 128)];
            float v933_data = s1[22];
            float v935_data = ir6[2];
            ir6[2] = (v935_data + (v932_data * v933_data));
            int32_t v942_a = v3_lead + 128;
            float v949_data = s0[(v3_lead + 128)];
            float v950_data = s1[31];
            float v952_data = ir6[3];
            ir6[3] = (v952_data + (v949_data * v950_data));
            int32_t v959_a = v3_lead + 128;
            float v966_data = s0[(v3_lead + 128)];
            float v967_data = s1[40];
            float v969_data = ir6[4];
            ir6[4] = (v969_data + (v966_data * v967_data));
            int32_t v976_a = v3_lead + 128;
            float v983_data = s0[(v3_lead + 128)];
            float v984_data = s1[49];
            float v986_data = ir6[5];
            ir6[5] = (v986_data + (v983_data * v984_data));
            int32_t v993_a = v3_lead + 128;
            float v1000_data = s0[(v3_lead + 128)];
            float v1001_data = s1[58];
            float v1003_data = ir6[6];
            ir6[6] = (v1003_data + (v1000_data * v1001_data));
            int32_t v1010_a = v3_lead + 128;
            float v1017_data = s0[(v3_lead + 128)];
            float v1018_data = s1[67];
            float v1020_data = ir6[7];
            ir6[7] = (v1020_data + (v1017_data * v1018_data));
            int32_t v1027_a = v3_lead + 128;
            float v1034_data = s0[(v3_lead + 128)];
            float v1035_data = s1[76];
            float v1037_data = ir6[8];
            ir6[8] = (v1037_data + (v1034_data * v1035_data));
            int32_t v1047_a = v3_lead + 160;
            float v1054_data = s0[(v3_lead + 160)];
            float v1055_data = s1[5];
            float v1057_data = ir6[0];
            ir6[0] = (v1057_data + (v1054_data * v1055_data));
            int32_t v1064_a = v3_lead + 160;
            float v1071_data = s0[(v3_lead + 160)];
            float v1072_data = s1[14];
            float v1074_data = ir6[1];
            ir6[1] = (v1074_data + (v1071_data * v1072_data));
            int32_t v1081_a = v3_lead + 160;
            float v1088_data = s0[(v3_lead + 160)];
            float v1089_data = s1[23];
            float v1091_data = ir6[2];
            ir6[2] = (v1091_data + (v1088_data * v1089_data));
            int32_t v1098_a = v3_lead + 160;
            float v1105_data = s0[(v3_lead + 160)];
            float v1106_data = s1[32];
            float v1108_data = ir6[3];
            ir6[3] = (v1108_data + (v1105_data * v1106_data));
            int32_t v1115_a = v3_lead + 160;
            float v1122_data = s0[(v3_lead + 160)];
            float v1123_data = s1[41];
            float v1125_data = ir6[4];
            ir6[4] = (v1125_data + (v1122_data * v1123_data));
            int32_t v1132_a = v3_lead + 160;
            float v1139_data = s0[(v3_lead + 160)];
            float v1140_data = s1[50];
            float v1142_data = ir6[5];
            ir6[5] = (v1142_data + (v1139_data * v1140_data));
            int32_t v1149_a = v3_lead + 160;
            float v1156_data = s0[(v3_lead + 160)];
            float v1157_data = s1[59];
            float v1159_data = ir6[6];
            ir6[6] = (v1159_data + (v1156_data * v1157_data));
            int32_t v1166_a = v3_lead + 160;
            float v1173_data = s0[(v3_lead + 160)];
            float v1174_data = s1[68];
            float v1176_data = ir6[7];
            ir6[7] = (v1176_data + (v1173_data * v1174_data));
            int32_t v1183_a = v3_lead + 160;
            float v1190_data = s0[(v3_lead + 160)];
            float v1191_data = s1[77];
            float v1193_data = ir6[8];
            ir6[8] = (v1193_data + (v1190_data * v1191_data));
            int32_t v1203_a = v3_lead + 192;
            float v1210_data = s0[(v3_lead + 192)];
            float v1211_data = s1[6];
            float v1213_data = ir6[0];
            ir6[0] = (v1213_data + (v1210_data * v1211_data));
            int32_t v1220_a = v3_lead + 192;
            float v1227_data = s0[(v3_lead + 192)];
            float v1228_data = s1[15];
            float v1230_data = ir6[1];
            ir6[1] = (v1230_data + (v1227_data * v1228_data));
            int32_t v1237_a = v3_lead + 192;
            float v1244_data = s0[(v3_lead + 192)];
            float v1245_data = s1[24];
            float v1247_data = ir6[2];
            ir6[2] = (v1247_data + (v1244_data * v1245_data));
            int32_t v1254_a = v3_lead + 192;
            float v1261_data = s0[(v3_lead + 192)];
            float v1262_data = s1[33];
            float v1264_data = ir6[3];
            ir6[3] = (v1264_data + (v1261_data * v1262_data));
            int32_t v1271_a = v3_lead + 192;
            float v1278_data = s0[(v3_lead + 192)];
            float v1279_data = s1[42];
            float v1281_data = ir6[4];
            ir6[4] = (v1281_data + (v1278_data * v1279_data));
            int32_t v1288_a = v3_lead + 192;
            float v1295_data = s0[(v3_lead + 192)];
            float v1296_data = s1[51];
            float v1298_data = ir6[5];
            ir6[5] = (v1298_data + (v1295_data * v1296_data));
            int32_t v1305_a = v3_lead + 192;
            float v1312_data = s0[(v3_lead + 192)];
            float v1313_data = s1[60];
            float v1315_data = ir6[6];
            ir6[6] = (v1315_data + (v1312_data * v1313_data));
            int32_t v1322_a = v3_lead + 192;
            float v1329_data = s0[(v3_lead + 192)];
            float v1330_data = s1[69];
            float v1332_data = ir6[7];
            ir6[7] = (v1332_data + (v1329_data * v1330_data));
            int32_t v1339_a = v3_lead + 192;
            float v1346_data = s0[(v3_lead + 192)];
            float v1347_data = s1[78];
            float v1349_data = ir6[8];
            ir6[8] = (v1349_data + (v1346_data * v1347_data));
            int32_t v1359_a = v3_lead + 224;
            float v1366_data = s0[(v3_lead + 224)];
            float v1367_data = s1[7];
            float v1369_data = ir6[0];
            ir6[0] = (v1369_data + (v1366_data * v1367_data));
            int32_t v1376_a = v3_lead + 224;
            float v1383_data = s0[(v3_lead + 224)];
            float v1384_data = s1[16];
            float v1386_data = ir6[1];
            ir6[1] = (v1386_data + (v1383_data * v1384_data));
            int32_t v1393_a = v3_lead + 224;
            float v1400_data = s0[(v3_lead + 224)];
            float v1401_data = s1[25];
            float v1403_data = ir6[2];
            ir6[2] = (v1403_data + (v1400_data * v1401_data));
            int32_t v1410_a = v3_lead + 224;
            float v1417_data = s0[(v3_lead + 224)];
            float v1418_data = s1[34];
            float v1420_data = ir6[3];
            ir6[3] = (v1420_data + (v1417_data * v1418_data));
            int32_t v1427_a = v3_lead + 224;
            float v1434_data = s0[(v3_lead + 224)];
            float v1435_data = s1[43];
            float v1437_data = ir6[4];
            ir6[4] = (v1437_data + (v1434_data * v1435_data));
            int32_t v1444_a = v3_lead + 224;
            float v1451_data = s0[(v3_lead + 224)];
            float v1452_data = s1[52];
            float v1454_data = ir6[5];
            ir6[5] = (v1454_data + (v1451_data * v1452_data));
            int32_t v1461_a = v3_lead + 224;
            float v1468_data = s0[(v3_lead + 224)];
            float v1469_data = s1[61];
            float v1471_data = ir6[6];
            ir6[6] = (v1471_data + (v1468_data * v1469_data));
            int32_t v1478_a = v3_lead + 224;
            float v1485_data = s0[(v3_lead + 224)];
            float v1486_data = s1[70];
            float v1488_data = ir6[7];
            ir6[7] = (v1488_data + (v1485_data * v1486_data));
            int32_t v1495_a = v3_lead + 224;
            float v1502_data = s0[(v3_lead + 224)];
            float v1503_data = s1[79];
            float v1505_data = ir6[8];
            ir6[8] = (v1505_data + (v1502_data * v1503_data));
            int32_t v1515_a = v3_lead + 256;
            float v1522_data = s0[(v3_lead + 256)];
            float v1523_data = s1[8];
            float v1525_data = ir6[0];
            ir6[0] = (v1525_data + (v1522_data * v1523_data));
            int32_t v1532_a = v3_lead + 256;
            float v1539_data = s0[(v3_lead + 256)];
            float v1540_data = s1[17];
            float v1542_data = ir6[1];
            ir6[1] = (v1542_data + (v1539_data * v1540_data));
            int32_t v1549_a = v3_lead + 256;
            float v1556_data = s0[(v3_lead + 256)];
            float v1557_data = s1[26];
            float v1559_data = ir6[2];
            ir6[2] = (v1559_data + (v1556_data * v1557_data));
            int32_t v1566_a = v3_lead + 256;
            float v1573_data = s0[(v3_lead + 256)];
            float v1574_data = s1[35];
            float v1576_data = ir6[3];
            ir6[3] = (v1576_data + (v1573_data * v1574_data));
            int32_t v1583_a = v3_lead + 256;
            float v1590_data = s0[(v3_lead + 256)];
            float v1591_data = s1[44];
            float v1593_data = ir6[4];
            ir6[4] = (v1593_data + (v1590_data * v1591_data));
            int32_t v1600_a = v3_lead + 256;
            float v1607_data = s0[(v3_lead + 256)];
            float v1608_data = s1[53];
            float v1610_data = ir6[5];
            ir6[5] = (v1610_data + (v1607_data * v1608_data));
            int32_t v1617_a = v3_lead + 256;
            float v1624_data = s0[(v3_lead + 256)];
            float v1625_data = s1[62];
            float v1627_data = ir6[6];
            ir6[6] = (v1627_data + (v1624_data * v1625_data));
            int32_t v1634_a = v3_lead + 256;
            float v1641_data = s0[(v3_lead + 256)];
            float v1642_data = s1[71];
            float v1644_data = ir6[7];
            ir6[7] = (v1644_data + (v1641_data * v1642_data));
            int32_t v1651_a = v3_lead + 256;
            float v1658_data = s0[(v3_lead + 256)];
            float v1659_data = s1[80];
            float v1661_data = ir6[8];
            ir6[8] = (v1661_data + (v1658_data * v1659_data));
            #pragma unroll
            for (int32_t v1666_n0 = 0; v1666_n0 < 1; ++v1666_n0) {
              #pragma unroll
              for (int32_t v1667_n1 = 0; v1667_n1 < 9; ++v1667_n1) {
                int32_t v1668_a = v1666_n0 + v1667_n1;
                int32_t v1669_a = v1666_n0 + v1667_n1;
                float v1670_data = ir6[v1669_a];
                int32_t v1671_a = v1666_n0 + v1667_n1;
                r6[v1669_a] = v1670_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1676_i0 = 0; v1676_i0 < 1; ++v1676_i0) {
            int32_t v1685_lead = v3_lead + (v1676_i0 * 32);
            #pragma unroll
            for (int32_t v1677_i1 = 0; v1677_i1 < 9; ++v1677_i1) {
              int32_t v1678_a = v1676_i0 + v1677_i1;
              float v1680_data = r6[(v1676_i0 + v1677_i1)];
              int32_t v1687_a = v1685_lead + (v1677_i1 * 32);
              glb_m3[v1687_a] = v1680_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

