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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m0[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 32;
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 9; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 16);
              float v26_data;
              {
                v26_data = __ldcg(&glb_m1[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r2[v27_a] = v26_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v31_data = r0[0];
          float v32_data = ir1[0];
          ir1[0] = (v32_data + v31_data);
          float v34_data = r0[1];
          float v35_data = ir1[1];
          ir1[1] = (v35_data + v34_data);
          float v37_data = r0[2];
          float v38_data = ir1[2];
          ir1[2] = (v38_data + v37_data);
          float v40_data = r0[3];
          float v41_data = ir1[3];
          ir1[3] = (v41_data + v40_data);
          float v43_data = r0[4];
          float v44_data = ir1[4];
          ir1[4] = (v44_data + v43_data);
          float v46_data = r0[5];
          float v47_data = ir1[5];
          ir1[5] = (v47_data + v46_data);
          float v49_data = r0[6];
          float v50_data = ir1[6];
          ir1[6] = (v50_data + v49_data);
          float v52_data = r0[7];
          float v53_data = ir1[7];
          ir1[7] = (v53_data + v52_data);
          float v55_data = r0[8];
          float v56_data = ir1[8];
          ir1[8] = (v56_data + v55_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          int32_t v60_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v61_i0 = 0; v61_i0 < 1; ++v61_i0) {
            int32_t v70_lead = v60_lead + (v61_i0 * 32);
            #pragma unroll
            for (int32_t v62_i1 = 0; v62_i1 < 9; ++v62_i1) {
              int32_t v63_a = v61_i0 + v62_i1;
              float v65_data = r1[(v61_i0 + v62_i1)];
              int32_t v72_a = v70_lead + (v62_i1 * 32);
              s0[v72_a] = v65_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          int32_t v75_lead = threadIdx.x % 32;
          if (v75_lead < 16) {
            #pragma unroll
            for (int32_t v77_i1 = 0; v77_i1 < 9; ++v77_i1) {
              int32_t v84_a = v75_lead + (v77_i1 * 16);
              float v85_data;
              {
                v85_data = __ldcg(&glb_m2[v84_a]);
              }
              int32_t v86_a = 0 + v77_i1;
              r4[v86_a] = v85_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          {
            // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir3[9]{};
            int32_t v89_lead = threadIdx.x % 32;
            if (v89_lead < 16) {
              float v91_data = r2[0];
              float v92_data = ir3[0];
              ir3[0] = (v92_data + v91_data);
              float v94_data = r2[1];
              float v95_data = ir3[1];
              ir3[1] = (v95_data + v94_data);
              float v97_data = r2[2];
              float v98_data = ir3[2];
              ir3[2] = (v98_data + v97_data);
              float v100_data = r2[3];
              float v101_data = ir3[3];
              ir3[3] = (v101_data + v100_data);
              float v103_data = r2[4];
              float v104_data = ir3[4];
              ir3[4] = (v104_data + v103_data);
              float v106_data = r2[5];
              float v107_data = ir3[5];
              ir3[5] = (v107_data + v106_data);
              float v109_data = r2[6];
              float v110_data = ir3[6];
              ir3[6] = (v110_data + v109_data);
              float v112_data = r2[7];
              float v113_data = ir3[7];
              ir3[7] = (v113_data + v112_data);
              float v115_data = r2[8];
              float v116_data = ir3[8];
              ir3[8] = (v116_data + v115_data);
            }
            if (v89_lead < 16) {
              #pragma unroll
              for (int32_t v122_n1 = 0; v122_n1 < 9; ++v122_n1) {
                int32_t v123_a = 0 + v122_n1;
                float v125_data = ir3[v122_n1];
                int32_t v131_a = v122_n1 * 32;
                int32_t v132_a = v89_lead + v131_a;
                float v140_data = s0[(v89_lead + v131_a)];
                int32_t v142_a = 0 + v122_n1;
                r3[v142_a] = (v140_data + v125_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          int32_t v145_lead = threadIdx.x % 32;
          if (v145_lead < 16) {
            #pragma unroll
            for (int32_t v147_i1 = 0; v147_i1 < 9; ++v147_i1) {
              int32_t v148_a = 0 + v147_i1;
              float v150_data = r3[v147_i1];
              int32_t v157_a = v145_lead + (v147_i1 * 32);
              s0[v157_a] = v150_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          {
            // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir5[9]{};
            int32_t v160_lead = threadIdx.x % 32;
            if (v160_lead < 16) {
              float v162_data = r4[0];
              float v163_data = ir5[0];
              ir5[0] = (v163_data + v162_data);
              float v165_data = r4[1];
              float v166_data = ir5[1];
              ir5[1] = (v166_data + v165_data);
              float v168_data = r4[2];
              float v169_data = ir5[2];
              ir5[2] = (v169_data + v168_data);
              float v171_data = r4[3];
              float v172_data = ir5[3];
              ir5[3] = (v172_data + v171_data);
              float v174_data = r4[4];
              float v175_data = ir5[4];
              ir5[4] = (v175_data + v174_data);
              float v177_data = r4[5];
              float v178_data = ir5[5];
              ir5[5] = (v178_data + v177_data);
              float v180_data = r4[6];
              float v181_data = ir5[6];
              ir5[6] = (v181_data + v180_data);
              float v183_data = r4[7];
              float v184_data = ir5[7];
              ir5[7] = (v184_data + v183_data);
              float v186_data = r4[8];
              float v187_data = ir5[8];
              ir5[8] = (v187_data + v186_data);
            }
            if (v160_lead < 16) {
              #pragma unroll
              for (int32_t v193_n1 = 0; v193_n1 < 9; ++v193_n1) {
                int32_t v194_a = 0 + v193_n1;
                float v196_data = ir5[v193_n1];
                int32_t v202_a = v193_n1 * 32;
                int32_t v203_a = v160_lead + v202_a;
                float v211_data = s0[(v160_lead + v202_a)];
                int32_t v213_a = 0 + v193_n1;
                r5[v213_a] = (v211_data + v196_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          int32_t v216_lead = threadIdx.x % 32;
          if (v216_lead < 16) {
            #pragma unroll
            for (int32_t v218_i1 = 0; v218_i1 < 9; ++v218_i1) {
              int32_t v219_a = 0 + v218_i1;
              float v221_data = r5[v218_i1];
              int32_t v228_a = v216_lead + (v218_i1 * 32);
              s0[v228_a] = v221_data;
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
            int32_t v231_lead = threadIdx.x % 32;
            int32_t v237_a = v231_lead + 0;
            float v244_data = s0[v231_lead];
            float v245_data = s1[0];
            float v247_data = ir6[0];
            ir6[0] = (v247_data + (v244_data * v245_data));
            int32_t v254_a = v231_lead + 0;
            float v261_data = s0[v231_lead];
            float v262_data = s1[9];
            float v264_data = ir6[1];
            ir6[1] = (v264_data + (v261_data * v262_data));
            int32_t v271_a = v231_lead + 0;
            float v278_data = s0[v231_lead];
            float v279_data = s1[18];
            float v281_data = ir6[2];
            ir6[2] = (v281_data + (v278_data * v279_data));
            int32_t v288_a = v231_lead + 0;
            float v295_data = s0[v231_lead];
            float v296_data = s1[27];
            float v298_data = ir6[3];
            ir6[3] = (v298_data + (v295_data * v296_data));
            int32_t v305_a = v231_lead + 0;
            float v312_data = s0[v231_lead];
            float v313_data = s1[36];
            float v315_data = ir6[4];
            ir6[4] = (v315_data + (v312_data * v313_data));
            int32_t v322_a = v231_lead + 0;
            float v329_data = s0[v231_lead];
            float v330_data = s1[45];
            float v332_data = ir6[5];
            ir6[5] = (v332_data + (v329_data * v330_data));
            int32_t v339_a = v231_lead + 0;
            float v346_data = s0[v231_lead];
            float v347_data = s1[54];
            float v349_data = ir6[6];
            ir6[6] = (v349_data + (v346_data * v347_data));
            int32_t v356_a = v231_lead + 0;
            float v363_data = s0[v231_lead];
            float v364_data = s1[63];
            float v366_data = ir6[7];
            ir6[7] = (v366_data + (v363_data * v364_data));
            int32_t v373_a = v231_lead + 0;
            float v380_data = s0[v231_lead];
            float v381_data = s1[72];
            float v383_data = ir6[8];
            ir6[8] = (v383_data + (v380_data * v381_data));
            int32_t v393_a = v231_lead + 32;
            float v400_data = s0[(v231_lead + 32)];
            float v401_data = s1[1];
            float v403_data = ir6[0];
            ir6[0] = (v403_data + (v400_data * v401_data));
            int32_t v410_a = v231_lead + 32;
            float v417_data = s0[(v231_lead + 32)];
            float v418_data = s1[10];
            float v420_data = ir6[1];
            ir6[1] = (v420_data + (v417_data * v418_data));
            int32_t v427_a = v231_lead + 32;
            float v434_data = s0[(v231_lead + 32)];
            float v435_data = s1[19];
            float v437_data = ir6[2];
            ir6[2] = (v437_data + (v434_data * v435_data));
            int32_t v444_a = v231_lead + 32;
            float v451_data = s0[(v231_lead + 32)];
            float v452_data = s1[28];
            float v454_data = ir6[3];
            ir6[3] = (v454_data + (v451_data * v452_data));
            int32_t v461_a = v231_lead + 32;
            float v468_data = s0[(v231_lead + 32)];
            float v469_data = s1[37];
            float v471_data = ir6[4];
            ir6[4] = (v471_data + (v468_data * v469_data));
            int32_t v478_a = v231_lead + 32;
            float v485_data = s0[(v231_lead + 32)];
            float v486_data = s1[46];
            float v488_data = ir6[5];
            ir6[5] = (v488_data + (v485_data * v486_data));
            int32_t v495_a = v231_lead + 32;
            float v502_data = s0[(v231_lead + 32)];
            float v503_data = s1[55];
            float v505_data = ir6[6];
            ir6[6] = (v505_data + (v502_data * v503_data));
            int32_t v512_a = v231_lead + 32;
            float v519_data = s0[(v231_lead + 32)];
            float v520_data = s1[64];
            float v522_data = ir6[7];
            ir6[7] = (v522_data + (v519_data * v520_data));
            int32_t v529_a = v231_lead + 32;
            float v536_data = s0[(v231_lead + 32)];
            float v537_data = s1[73];
            float v539_data = ir6[8];
            ir6[8] = (v539_data + (v536_data * v537_data));
            int32_t v549_a = v231_lead + 64;
            float v556_data = s0[(v231_lead + 64)];
            float v557_data = s1[2];
            float v559_data = ir6[0];
            ir6[0] = (v559_data + (v556_data * v557_data));
            int32_t v566_a = v231_lead + 64;
            float v573_data = s0[(v231_lead + 64)];
            float v574_data = s1[11];
            float v576_data = ir6[1];
            ir6[1] = (v576_data + (v573_data * v574_data));
            int32_t v583_a = v231_lead + 64;
            float v590_data = s0[(v231_lead + 64)];
            float v591_data = s1[20];
            float v593_data = ir6[2];
            ir6[2] = (v593_data + (v590_data * v591_data));
            int32_t v600_a = v231_lead + 64;
            float v607_data = s0[(v231_lead + 64)];
            float v608_data = s1[29];
            float v610_data = ir6[3];
            ir6[3] = (v610_data + (v607_data * v608_data));
            int32_t v617_a = v231_lead + 64;
            float v624_data = s0[(v231_lead + 64)];
            float v625_data = s1[38];
            float v627_data = ir6[4];
            ir6[4] = (v627_data + (v624_data * v625_data));
            int32_t v634_a = v231_lead + 64;
            float v641_data = s0[(v231_lead + 64)];
            float v642_data = s1[47];
            float v644_data = ir6[5];
            ir6[5] = (v644_data + (v641_data * v642_data));
            int32_t v651_a = v231_lead + 64;
            float v658_data = s0[(v231_lead + 64)];
            float v659_data = s1[56];
            float v661_data = ir6[6];
            ir6[6] = (v661_data + (v658_data * v659_data));
            int32_t v668_a = v231_lead + 64;
            float v675_data = s0[(v231_lead + 64)];
            float v676_data = s1[65];
            float v678_data = ir6[7];
            ir6[7] = (v678_data + (v675_data * v676_data));
            int32_t v685_a = v231_lead + 64;
            float v692_data = s0[(v231_lead + 64)];
            float v693_data = s1[74];
            float v695_data = ir6[8];
            ir6[8] = (v695_data + (v692_data * v693_data));
            int32_t v705_a = v231_lead + 96;
            float v712_data = s0[(v231_lead + 96)];
            float v713_data = s1[3];
            float v715_data = ir6[0];
            ir6[0] = (v715_data + (v712_data * v713_data));
            int32_t v722_a = v231_lead + 96;
            float v729_data = s0[(v231_lead + 96)];
            float v730_data = s1[12];
            float v732_data = ir6[1];
            ir6[1] = (v732_data + (v729_data * v730_data));
            int32_t v739_a = v231_lead + 96;
            float v746_data = s0[(v231_lead + 96)];
            float v747_data = s1[21];
            float v749_data = ir6[2];
            ir6[2] = (v749_data + (v746_data * v747_data));
            int32_t v756_a = v231_lead + 96;
            float v763_data = s0[(v231_lead + 96)];
            float v764_data = s1[30];
            float v766_data = ir6[3];
            ir6[3] = (v766_data + (v763_data * v764_data));
            int32_t v773_a = v231_lead + 96;
            float v780_data = s0[(v231_lead + 96)];
            float v781_data = s1[39];
            float v783_data = ir6[4];
            ir6[4] = (v783_data + (v780_data * v781_data));
            int32_t v790_a = v231_lead + 96;
            float v797_data = s0[(v231_lead + 96)];
            float v798_data = s1[48];
            float v800_data = ir6[5];
            ir6[5] = (v800_data + (v797_data * v798_data));
            int32_t v807_a = v231_lead + 96;
            float v814_data = s0[(v231_lead + 96)];
            float v815_data = s1[57];
            float v817_data = ir6[6];
            ir6[6] = (v817_data + (v814_data * v815_data));
            int32_t v824_a = v231_lead + 96;
            float v831_data = s0[(v231_lead + 96)];
            float v832_data = s1[66];
            float v834_data = ir6[7];
            ir6[7] = (v834_data + (v831_data * v832_data));
            int32_t v841_a = v231_lead + 96;
            float v848_data = s0[(v231_lead + 96)];
            float v849_data = s1[75];
            float v851_data = ir6[8];
            ir6[8] = (v851_data + (v848_data * v849_data));
            int32_t v861_a = v231_lead + 128;
            float v868_data = s0[(v231_lead + 128)];
            float v869_data = s1[4];
            float v871_data = ir6[0];
            ir6[0] = (v871_data + (v868_data * v869_data));
            int32_t v878_a = v231_lead + 128;
            float v885_data = s0[(v231_lead + 128)];
            float v886_data = s1[13];
            float v888_data = ir6[1];
            ir6[1] = (v888_data + (v885_data * v886_data));
            int32_t v895_a = v231_lead + 128;
            float v902_data = s0[(v231_lead + 128)];
            float v903_data = s1[22];
            float v905_data = ir6[2];
            ir6[2] = (v905_data + (v902_data * v903_data));
            int32_t v912_a = v231_lead + 128;
            float v919_data = s0[(v231_lead + 128)];
            float v920_data = s1[31];
            float v922_data = ir6[3];
            ir6[3] = (v922_data + (v919_data * v920_data));
            int32_t v929_a = v231_lead + 128;
            float v936_data = s0[(v231_lead + 128)];
            float v937_data = s1[40];
            float v939_data = ir6[4];
            ir6[4] = (v939_data + (v936_data * v937_data));
            int32_t v946_a = v231_lead + 128;
            float v953_data = s0[(v231_lead + 128)];
            float v954_data = s1[49];
            float v956_data = ir6[5];
            ir6[5] = (v956_data + (v953_data * v954_data));
            int32_t v963_a = v231_lead + 128;
            float v970_data = s0[(v231_lead + 128)];
            float v971_data = s1[58];
            float v973_data = ir6[6];
            ir6[6] = (v973_data + (v970_data * v971_data));
            int32_t v980_a = v231_lead + 128;
            float v987_data = s0[(v231_lead + 128)];
            float v988_data = s1[67];
            float v990_data = ir6[7];
            ir6[7] = (v990_data + (v987_data * v988_data));
            int32_t v997_a = v231_lead + 128;
            float v1004_data = s0[(v231_lead + 128)];
            float v1005_data = s1[76];
            float v1007_data = ir6[8];
            ir6[8] = (v1007_data + (v1004_data * v1005_data));
            int32_t v1017_a = v231_lead + 160;
            float v1024_data = s0[(v231_lead + 160)];
            float v1025_data = s1[5];
            float v1027_data = ir6[0];
            ir6[0] = (v1027_data + (v1024_data * v1025_data));
            int32_t v1034_a = v231_lead + 160;
            float v1041_data = s0[(v231_lead + 160)];
            float v1042_data = s1[14];
            float v1044_data = ir6[1];
            ir6[1] = (v1044_data + (v1041_data * v1042_data));
            int32_t v1051_a = v231_lead + 160;
            float v1058_data = s0[(v231_lead + 160)];
            float v1059_data = s1[23];
            float v1061_data = ir6[2];
            ir6[2] = (v1061_data + (v1058_data * v1059_data));
            int32_t v1068_a = v231_lead + 160;
            float v1075_data = s0[(v231_lead + 160)];
            float v1076_data = s1[32];
            float v1078_data = ir6[3];
            ir6[3] = (v1078_data + (v1075_data * v1076_data));
            int32_t v1085_a = v231_lead + 160;
            float v1092_data = s0[(v231_lead + 160)];
            float v1093_data = s1[41];
            float v1095_data = ir6[4];
            ir6[4] = (v1095_data + (v1092_data * v1093_data));
            int32_t v1102_a = v231_lead + 160;
            float v1109_data = s0[(v231_lead + 160)];
            float v1110_data = s1[50];
            float v1112_data = ir6[5];
            ir6[5] = (v1112_data + (v1109_data * v1110_data));
            int32_t v1119_a = v231_lead + 160;
            float v1126_data = s0[(v231_lead + 160)];
            float v1127_data = s1[59];
            float v1129_data = ir6[6];
            ir6[6] = (v1129_data + (v1126_data * v1127_data));
            int32_t v1136_a = v231_lead + 160;
            float v1143_data = s0[(v231_lead + 160)];
            float v1144_data = s1[68];
            float v1146_data = ir6[7];
            ir6[7] = (v1146_data + (v1143_data * v1144_data));
            int32_t v1153_a = v231_lead + 160;
            float v1160_data = s0[(v231_lead + 160)];
            float v1161_data = s1[77];
            float v1163_data = ir6[8];
            ir6[8] = (v1163_data + (v1160_data * v1161_data));
            int32_t v1173_a = v231_lead + 192;
            float v1180_data = s0[(v231_lead + 192)];
            float v1181_data = s1[6];
            float v1183_data = ir6[0];
            ir6[0] = (v1183_data + (v1180_data * v1181_data));
            int32_t v1190_a = v231_lead + 192;
            float v1197_data = s0[(v231_lead + 192)];
            float v1198_data = s1[15];
            float v1200_data = ir6[1];
            ir6[1] = (v1200_data + (v1197_data * v1198_data));
            int32_t v1207_a = v231_lead + 192;
            float v1214_data = s0[(v231_lead + 192)];
            float v1215_data = s1[24];
            float v1217_data = ir6[2];
            ir6[2] = (v1217_data + (v1214_data * v1215_data));
            int32_t v1224_a = v231_lead + 192;
            float v1231_data = s0[(v231_lead + 192)];
            float v1232_data = s1[33];
            float v1234_data = ir6[3];
            ir6[3] = (v1234_data + (v1231_data * v1232_data));
            int32_t v1241_a = v231_lead + 192;
            float v1248_data = s0[(v231_lead + 192)];
            float v1249_data = s1[42];
            float v1251_data = ir6[4];
            ir6[4] = (v1251_data + (v1248_data * v1249_data));
            int32_t v1258_a = v231_lead + 192;
            float v1265_data = s0[(v231_lead + 192)];
            float v1266_data = s1[51];
            float v1268_data = ir6[5];
            ir6[5] = (v1268_data + (v1265_data * v1266_data));
            int32_t v1275_a = v231_lead + 192;
            float v1282_data = s0[(v231_lead + 192)];
            float v1283_data = s1[60];
            float v1285_data = ir6[6];
            ir6[6] = (v1285_data + (v1282_data * v1283_data));
            int32_t v1292_a = v231_lead + 192;
            float v1299_data = s0[(v231_lead + 192)];
            float v1300_data = s1[69];
            float v1302_data = ir6[7];
            ir6[7] = (v1302_data + (v1299_data * v1300_data));
            int32_t v1309_a = v231_lead + 192;
            float v1316_data = s0[(v231_lead + 192)];
            float v1317_data = s1[78];
            float v1319_data = ir6[8];
            ir6[8] = (v1319_data + (v1316_data * v1317_data));
            int32_t v1329_a = v231_lead + 224;
            float v1336_data = s0[(v231_lead + 224)];
            float v1337_data = s1[7];
            float v1339_data = ir6[0];
            ir6[0] = (v1339_data + (v1336_data * v1337_data));
            int32_t v1346_a = v231_lead + 224;
            float v1353_data = s0[(v231_lead + 224)];
            float v1354_data = s1[16];
            float v1356_data = ir6[1];
            ir6[1] = (v1356_data + (v1353_data * v1354_data));
            int32_t v1363_a = v231_lead + 224;
            float v1370_data = s0[(v231_lead + 224)];
            float v1371_data = s1[25];
            float v1373_data = ir6[2];
            ir6[2] = (v1373_data + (v1370_data * v1371_data));
            int32_t v1380_a = v231_lead + 224;
            float v1387_data = s0[(v231_lead + 224)];
            float v1388_data = s1[34];
            float v1390_data = ir6[3];
            ir6[3] = (v1390_data + (v1387_data * v1388_data));
            int32_t v1397_a = v231_lead + 224;
            float v1404_data = s0[(v231_lead + 224)];
            float v1405_data = s1[43];
            float v1407_data = ir6[4];
            ir6[4] = (v1407_data + (v1404_data * v1405_data));
            int32_t v1414_a = v231_lead + 224;
            float v1421_data = s0[(v231_lead + 224)];
            float v1422_data = s1[52];
            float v1424_data = ir6[5];
            ir6[5] = (v1424_data + (v1421_data * v1422_data));
            int32_t v1431_a = v231_lead + 224;
            float v1438_data = s0[(v231_lead + 224)];
            float v1439_data = s1[61];
            float v1441_data = ir6[6];
            ir6[6] = (v1441_data + (v1438_data * v1439_data));
            int32_t v1448_a = v231_lead + 224;
            float v1455_data = s0[(v231_lead + 224)];
            float v1456_data = s1[70];
            float v1458_data = ir6[7];
            ir6[7] = (v1458_data + (v1455_data * v1456_data));
            int32_t v1465_a = v231_lead + 224;
            float v1472_data = s0[(v231_lead + 224)];
            float v1473_data = s1[79];
            float v1475_data = ir6[8];
            ir6[8] = (v1475_data + (v1472_data * v1473_data));
            int32_t v1485_a = v231_lead + 256;
            float v1492_data = s0[(v231_lead + 256)];
            float v1493_data = s1[8];
            float v1495_data = ir6[0];
            ir6[0] = (v1495_data + (v1492_data * v1493_data));
            int32_t v1502_a = v231_lead + 256;
            float v1509_data = s0[(v231_lead + 256)];
            float v1510_data = s1[17];
            float v1512_data = ir6[1];
            ir6[1] = (v1512_data + (v1509_data * v1510_data));
            int32_t v1519_a = v231_lead + 256;
            float v1526_data = s0[(v231_lead + 256)];
            float v1527_data = s1[26];
            float v1529_data = ir6[2];
            ir6[2] = (v1529_data + (v1526_data * v1527_data));
            int32_t v1536_a = v231_lead + 256;
            float v1543_data = s0[(v231_lead + 256)];
            float v1544_data = s1[35];
            float v1546_data = ir6[3];
            ir6[3] = (v1546_data + (v1543_data * v1544_data));
            int32_t v1553_a = v231_lead + 256;
            float v1560_data = s0[(v231_lead + 256)];
            float v1561_data = s1[44];
            float v1563_data = ir6[4];
            ir6[4] = (v1563_data + (v1560_data * v1561_data));
            int32_t v1570_a = v231_lead + 256;
            float v1577_data = s0[(v231_lead + 256)];
            float v1578_data = s1[53];
            float v1580_data = ir6[5];
            ir6[5] = (v1580_data + (v1577_data * v1578_data));
            int32_t v1587_a = v231_lead + 256;
            float v1594_data = s0[(v231_lead + 256)];
            float v1595_data = s1[62];
            float v1597_data = ir6[6];
            ir6[6] = (v1597_data + (v1594_data * v1595_data));
            int32_t v1604_a = v231_lead + 256;
            float v1611_data = s0[(v231_lead + 256)];
            float v1612_data = s1[71];
            float v1614_data = ir6[7];
            ir6[7] = (v1614_data + (v1611_data * v1612_data));
            int32_t v1621_a = v231_lead + 256;
            float v1628_data = s0[(v231_lead + 256)];
            float v1629_data = s1[80];
            float v1631_data = ir6[8];
            ir6[8] = (v1631_data + (v1628_data * v1629_data));
            #pragma unroll
            for (int32_t v1636_n0 = 0; v1636_n0 < 1; ++v1636_n0) {
              #pragma unroll
              for (int32_t v1637_n1 = 0; v1637_n1 < 9; ++v1637_n1) {
                int32_t v1638_a = v1636_n0 + v1637_n1;
                float v1640_data = ir6[(v1636_n0 + v1637_n1)];
                int32_t v1641_a = v1636_n0 + v1637_n1;
                r6[v1641_a] = v1640_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r6);
          int32_t v1644_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v1645_i0 = 0; v1645_i0 < 1; ++v1645_i0) {
            int32_t v1654_lead = v1644_lead + (v1645_i0 * 32);
            #pragma unroll
            for (int32_t v1646_i1 = 0; v1646_i1 < 9; ++v1646_i1) {
              int32_t v1647_a = v1645_i0 + v1646_i1;
              float v1649_data = r6[(v1645_i0 + v1646_i1)];
              int32_t v1656_a = v1654_lead + (v1646_i1 * 32);
              glb_m3[v1656_a] = v1649_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

