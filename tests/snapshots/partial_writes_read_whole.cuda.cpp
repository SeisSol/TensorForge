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
                r3[v122_n1] = (v140_data + v125_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          int32_t v146_lead = threadIdx.x % 32;
          if (v146_lead < 16) {
            #pragma unroll
            for (int32_t v148_i1 = 0; v148_i1 < 9; ++v148_i1) {
              int32_t v149_a = 0 + v148_i1;
              float v151_data = r3[v148_i1];
              int32_t v158_a = v146_lead + (v148_i1 * 32);
              s0[v158_a] = v151_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          {
            // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
            // [(0, 16), (0, 9)] []
            float ir5[9]{};
            int32_t v161_lead = threadIdx.x % 32;
            if (v161_lead < 16) {
              float v163_data = r4[0];
              float v164_data = ir5[0];
              ir5[0] = (v164_data + v163_data);
              float v166_data = r4[1];
              float v167_data = ir5[1];
              ir5[1] = (v167_data + v166_data);
              float v169_data = r4[2];
              float v170_data = ir5[2];
              ir5[2] = (v170_data + v169_data);
              float v172_data = r4[3];
              float v173_data = ir5[3];
              ir5[3] = (v173_data + v172_data);
              float v175_data = r4[4];
              float v176_data = ir5[4];
              ir5[4] = (v176_data + v175_data);
              float v178_data = r4[5];
              float v179_data = ir5[5];
              ir5[5] = (v179_data + v178_data);
              float v181_data = r4[6];
              float v182_data = ir5[6];
              ir5[6] = (v182_data + v181_data);
              float v184_data = r4[7];
              float v185_data = ir5[7];
              ir5[7] = (v185_data + v184_data);
              float v187_data = r4[8];
              float v188_data = ir5[8];
              ir5[8] = (v188_data + v187_data);
            }
            if (v161_lead < 16) {
              #pragma unroll
              for (int32_t v194_n1 = 0; v194_n1 < 9; ++v194_n1) {
                int32_t v195_a = 0 + v194_n1;
                float v197_data = ir5[v194_n1];
                int32_t v203_a = v194_n1 * 32;
                int32_t v204_a = v161_lead + v203_a;
                float v212_data = s0[(v161_lead + v203_a)];
                int32_t v214_a = 0 + v194_n1;
                r5[v194_n1] = (v212_data + v197_data);
              }
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          int32_t v218_lead = threadIdx.x % 32;
          if (v218_lead < 16) {
            #pragma unroll
            for (int32_t v220_i1 = 0; v220_i1 < 9; ++v220_i1) {
              int32_t v221_a = 0 + v220_i1;
              float v223_data = r5[v220_i1];
              int32_t v230_a = v218_lead + (v220_i1 * 32);
              s0[v230_a] = v223_data;
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
            int32_t v233_lead = threadIdx.x % 32;
            int32_t v239_a = v233_lead + 0;
            float v246_data = s0[v233_lead];
            float v247_data = s1[0];
            float v249_data = ir6[0];
            ir6[0] = (v249_data + (v246_data * v247_data));
            int32_t v256_a = v233_lead + 0;
            float v263_data = s0[v233_lead];
            float v264_data = s1[9];
            float v266_data = ir6[1];
            ir6[1] = (v266_data + (v263_data * v264_data));
            int32_t v273_a = v233_lead + 0;
            float v280_data = s0[v233_lead];
            float v281_data = s1[18];
            float v283_data = ir6[2];
            ir6[2] = (v283_data + (v280_data * v281_data));
            int32_t v290_a = v233_lead + 0;
            float v297_data = s0[v233_lead];
            float v298_data = s1[27];
            float v300_data = ir6[3];
            ir6[3] = (v300_data + (v297_data * v298_data));
            int32_t v307_a = v233_lead + 0;
            float v314_data = s0[v233_lead];
            float v315_data = s1[36];
            float v317_data = ir6[4];
            ir6[4] = (v317_data + (v314_data * v315_data));
            int32_t v324_a = v233_lead + 0;
            float v331_data = s0[v233_lead];
            float v332_data = s1[45];
            float v334_data = ir6[5];
            ir6[5] = (v334_data + (v331_data * v332_data));
            int32_t v341_a = v233_lead + 0;
            float v348_data = s0[v233_lead];
            float v349_data = s1[54];
            float v351_data = ir6[6];
            ir6[6] = (v351_data + (v348_data * v349_data));
            int32_t v358_a = v233_lead + 0;
            float v365_data = s0[v233_lead];
            float v366_data = s1[63];
            float v368_data = ir6[7];
            ir6[7] = (v368_data + (v365_data * v366_data));
            int32_t v375_a = v233_lead + 0;
            float v382_data = s0[v233_lead];
            float v383_data = s1[72];
            float v385_data = ir6[8];
            ir6[8] = (v385_data + (v382_data * v383_data));
            int32_t v395_a = v233_lead + 32;
            float v402_data = s0[(v233_lead + 32)];
            float v403_data = s1[1];
            float v405_data = ir6[0];
            ir6[0] = (v405_data + (v402_data * v403_data));
            int32_t v412_a = v233_lead + 32;
            float v419_data = s0[(v233_lead + 32)];
            float v420_data = s1[10];
            float v422_data = ir6[1];
            ir6[1] = (v422_data + (v419_data * v420_data));
            int32_t v429_a = v233_lead + 32;
            float v436_data = s0[(v233_lead + 32)];
            float v437_data = s1[19];
            float v439_data = ir6[2];
            ir6[2] = (v439_data + (v436_data * v437_data));
            int32_t v446_a = v233_lead + 32;
            float v453_data = s0[(v233_lead + 32)];
            float v454_data = s1[28];
            float v456_data = ir6[3];
            ir6[3] = (v456_data + (v453_data * v454_data));
            int32_t v463_a = v233_lead + 32;
            float v470_data = s0[(v233_lead + 32)];
            float v471_data = s1[37];
            float v473_data = ir6[4];
            ir6[4] = (v473_data + (v470_data * v471_data));
            int32_t v480_a = v233_lead + 32;
            float v487_data = s0[(v233_lead + 32)];
            float v488_data = s1[46];
            float v490_data = ir6[5];
            ir6[5] = (v490_data + (v487_data * v488_data));
            int32_t v497_a = v233_lead + 32;
            float v504_data = s0[(v233_lead + 32)];
            float v505_data = s1[55];
            float v507_data = ir6[6];
            ir6[6] = (v507_data + (v504_data * v505_data));
            int32_t v514_a = v233_lead + 32;
            float v521_data = s0[(v233_lead + 32)];
            float v522_data = s1[64];
            float v524_data = ir6[7];
            ir6[7] = (v524_data + (v521_data * v522_data));
            int32_t v531_a = v233_lead + 32;
            float v538_data = s0[(v233_lead + 32)];
            float v539_data = s1[73];
            float v541_data = ir6[8];
            ir6[8] = (v541_data + (v538_data * v539_data));
            int32_t v551_a = v233_lead + 64;
            float v558_data = s0[(v233_lead + 64)];
            float v559_data = s1[2];
            float v561_data = ir6[0];
            ir6[0] = (v561_data + (v558_data * v559_data));
            int32_t v568_a = v233_lead + 64;
            float v575_data = s0[(v233_lead + 64)];
            float v576_data = s1[11];
            float v578_data = ir6[1];
            ir6[1] = (v578_data + (v575_data * v576_data));
            int32_t v585_a = v233_lead + 64;
            float v592_data = s0[(v233_lead + 64)];
            float v593_data = s1[20];
            float v595_data = ir6[2];
            ir6[2] = (v595_data + (v592_data * v593_data));
            int32_t v602_a = v233_lead + 64;
            float v609_data = s0[(v233_lead + 64)];
            float v610_data = s1[29];
            float v612_data = ir6[3];
            ir6[3] = (v612_data + (v609_data * v610_data));
            int32_t v619_a = v233_lead + 64;
            float v626_data = s0[(v233_lead + 64)];
            float v627_data = s1[38];
            float v629_data = ir6[4];
            ir6[4] = (v629_data + (v626_data * v627_data));
            int32_t v636_a = v233_lead + 64;
            float v643_data = s0[(v233_lead + 64)];
            float v644_data = s1[47];
            float v646_data = ir6[5];
            ir6[5] = (v646_data + (v643_data * v644_data));
            int32_t v653_a = v233_lead + 64;
            float v660_data = s0[(v233_lead + 64)];
            float v661_data = s1[56];
            float v663_data = ir6[6];
            ir6[6] = (v663_data + (v660_data * v661_data));
            int32_t v670_a = v233_lead + 64;
            float v677_data = s0[(v233_lead + 64)];
            float v678_data = s1[65];
            float v680_data = ir6[7];
            ir6[7] = (v680_data + (v677_data * v678_data));
            int32_t v687_a = v233_lead + 64;
            float v694_data = s0[(v233_lead + 64)];
            float v695_data = s1[74];
            float v697_data = ir6[8];
            ir6[8] = (v697_data + (v694_data * v695_data));
            int32_t v707_a = v233_lead + 96;
            float v714_data = s0[(v233_lead + 96)];
            float v715_data = s1[3];
            float v717_data = ir6[0];
            ir6[0] = (v717_data + (v714_data * v715_data));
            int32_t v724_a = v233_lead + 96;
            float v731_data = s0[(v233_lead + 96)];
            float v732_data = s1[12];
            float v734_data = ir6[1];
            ir6[1] = (v734_data + (v731_data * v732_data));
            int32_t v741_a = v233_lead + 96;
            float v748_data = s0[(v233_lead + 96)];
            float v749_data = s1[21];
            float v751_data = ir6[2];
            ir6[2] = (v751_data + (v748_data * v749_data));
            int32_t v758_a = v233_lead + 96;
            float v765_data = s0[(v233_lead + 96)];
            float v766_data = s1[30];
            float v768_data = ir6[3];
            ir6[3] = (v768_data + (v765_data * v766_data));
            int32_t v775_a = v233_lead + 96;
            float v782_data = s0[(v233_lead + 96)];
            float v783_data = s1[39];
            float v785_data = ir6[4];
            ir6[4] = (v785_data + (v782_data * v783_data));
            int32_t v792_a = v233_lead + 96;
            float v799_data = s0[(v233_lead + 96)];
            float v800_data = s1[48];
            float v802_data = ir6[5];
            ir6[5] = (v802_data + (v799_data * v800_data));
            int32_t v809_a = v233_lead + 96;
            float v816_data = s0[(v233_lead + 96)];
            float v817_data = s1[57];
            float v819_data = ir6[6];
            ir6[6] = (v819_data + (v816_data * v817_data));
            int32_t v826_a = v233_lead + 96;
            float v833_data = s0[(v233_lead + 96)];
            float v834_data = s1[66];
            float v836_data = ir6[7];
            ir6[7] = (v836_data + (v833_data * v834_data));
            int32_t v843_a = v233_lead + 96;
            float v850_data = s0[(v233_lead + 96)];
            float v851_data = s1[75];
            float v853_data = ir6[8];
            ir6[8] = (v853_data + (v850_data * v851_data));
            int32_t v863_a = v233_lead + 128;
            float v870_data = s0[(v233_lead + 128)];
            float v871_data = s1[4];
            float v873_data = ir6[0];
            ir6[0] = (v873_data + (v870_data * v871_data));
            int32_t v880_a = v233_lead + 128;
            float v887_data = s0[(v233_lead + 128)];
            float v888_data = s1[13];
            float v890_data = ir6[1];
            ir6[1] = (v890_data + (v887_data * v888_data));
            int32_t v897_a = v233_lead + 128;
            float v904_data = s0[(v233_lead + 128)];
            float v905_data = s1[22];
            float v907_data = ir6[2];
            ir6[2] = (v907_data + (v904_data * v905_data));
            int32_t v914_a = v233_lead + 128;
            float v921_data = s0[(v233_lead + 128)];
            float v922_data = s1[31];
            float v924_data = ir6[3];
            ir6[3] = (v924_data + (v921_data * v922_data));
            int32_t v931_a = v233_lead + 128;
            float v938_data = s0[(v233_lead + 128)];
            float v939_data = s1[40];
            float v941_data = ir6[4];
            ir6[4] = (v941_data + (v938_data * v939_data));
            int32_t v948_a = v233_lead + 128;
            float v955_data = s0[(v233_lead + 128)];
            float v956_data = s1[49];
            float v958_data = ir6[5];
            ir6[5] = (v958_data + (v955_data * v956_data));
            int32_t v965_a = v233_lead + 128;
            float v972_data = s0[(v233_lead + 128)];
            float v973_data = s1[58];
            float v975_data = ir6[6];
            ir6[6] = (v975_data + (v972_data * v973_data));
            int32_t v982_a = v233_lead + 128;
            float v989_data = s0[(v233_lead + 128)];
            float v990_data = s1[67];
            float v992_data = ir6[7];
            ir6[7] = (v992_data + (v989_data * v990_data));
            int32_t v999_a = v233_lead + 128;
            float v1006_data = s0[(v233_lead + 128)];
            float v1007_data = s1[76];
            float v1009_data = ir6[8];
            ir6[8] = (v1009_data + (v1006_data * v1007_data));
            int32_t v1019_a = v233_lead + 160;
            float v1026_data = s0[(v233_lead + 160)];
            float v1027_data = s1[5];
            float v1029_data = ir6[0];
            ir6[0] = (v1029_data + (v1026_data * v1027_data));
            int32_t v1036_a = v233_lead + 160;
            float v1043_data = s0[(v233_lead + 160)];
            float v1044_data = s1[14];
            float v1046_data = ir6[1];
            ir6[1] = (v1046_data + (v1043_data * v1044_data));
            int32_t v1053_a = v233_lead + 160;
            float v1060_data = s0[(v233_lead + 160)];
            float v1061_data = s1[23];
            float v1063_data = ir6[2];
            ir6[2] = (v1063_data + (v1060_data * v1061_data));
            int32_t v1070_a = v233_lead + 160;
            float v1077_data = s0[(v233_lead + 160)];
            float v1078_data = s1[32];
            float v1080_data = ir6[3];
            ir6[3] = (v1080_data + (v1077_data * v1078_data));
            int32_t v1087_a = v233_lead + 160;
            float v1094_data = s0[(v233_lead + 160)];
            float v1095_data = s1[41];
            float v1097_data = ir6[4];
            ir6[4] = (v1097_data + (v1094_data * v1095_data));
            int32_t v1104_a = v233_lead + 160;
            float v1111_data = s0[(v233_lead + 160)];
            float v1112_data = s1[50];
            float v1114_data = ir6[5];
            ir6[5] = (v1114_data + (v1111_data * v1112_data));
            int32_t v1121_a = v233_lead + 160;
            float v1128_data = s0[(v233_lead + 160)];
            float v1129_data = s1[59];
            float v1131_data = ir6[6];
            ir6[6] = (v1131_data + (v1128_data * v1129_data));
            int32_t v1138_a = v233_lead + 160;
            float v1145_data = s0[(v233_lead + 160)];
            float v1146_data = s1[68];
            float v1148_data = ir6[7];
            ir6[7] = (v1148_data + (v1145_data * v1146_data));
            int32_t v1155_a = v233_lead + 160;
            float v1162_data = s0[(v233_lead + 160)];
            float v1163_data = s1[77];
            float v1165_data = ir6[8];
            ir6[8] = (v1165_data + (v1162_data * v1163_data));
            int32_t v1175_a = v233_lead + 192;
            float v1182_data = s0[(v233_lead + 192)];
            float v1183_data = s1[6];
            float v1185_data = ir6[0];
            ir6[0] = (v1185_data + (v1182_data * v1183_data));
            int32_t v1192_a = v233_lead + 192;
            float v1199_data = s0[(v233_lead + 192)];
            float v1200_data = s1[15];
            float v1202_data = ir6[1];
            ir6[1] = (v1202_data + (v1199_data * v1200_data));
            int32_t v1209_a = v233_lead + 192;
            float v1216_data = s0[(v233_lead + 192)];
            float v1217_data = s1[24];
            float v1219_data = ir6[2];
            ir6[2] = (v1219_data + (v1216_data * v1217_data));
            int32_t v1226_a = v233_lead + 192;
            float v1233_data = s0[(v233_lead + 192)];
            float v1234_data = s1[33];
            float v1236_data = ir6[3];
            ir6[3] = (v1236_data + (v1233_data * v1234_data));
            int32_t v1243_a = v233_lead + 192;
            float v1250_data = s0[(v233_lead + 192)];
            float v1251_data = s1[42];
            float v1253_data = ir6[4];
            ir6[4] = (v1253_data + (v1250_data * v1251_data));
            int32_t v1260_a = v233_lead + 192;
            float v1267_data = s0[(v233_lead + 192)];
            float v1268_data = s1[51];
            float v1270_data = ir6[5];
            ir6[5] = (v1270_data + (v1267_data * v1268_data));
            int32_t v1277_a = v233_lead + 192;
            float v1284_data = s0[(v233_lead + 192)];
            float v1285_data = s1[60];
            float v1287_data = ir6[6];
            ir6[6] = (v1287_data + (v1284_data * v1285_data));
            int32_t v1294_a = v233_lead + 192;
            float v1301_data = s0[(v233_lead + 192)];
            float v1302_data = s1[69];
            float v1304_data = ir6[7];
            ir6[7] = (v1304_data + (v1301_data * v1302_data));
            int32_t v1311_a = v233_lead + 192;
            float v1318_data = s0[(v233_lead + 192)];
            float v1319_data = s1[78];
            float v1321_data = ir6[8];
            ir6[8] = (v1321_data + (v1318_data * v1319_data));
            int32_t v1331_a = v233_lead + 224;
            float v1338_data = s0[(v233_lead + 224)];
            float v1339_data = s1[7];
            float v1341_data = ir6[0];
            ir6[0] = (v1341_data + (v1338_data * v1339_data));
            int32_t v1348_a = v233_lead + 224;
            float v1355_data = s0[(v233_lead + 224)];
            float v1356_data = s1[16];
            float v1358_data = ir6[1];
            ir6[1] = (v1358_data + (v1355_data * v1356_data));
            int32_t v1365_a = v233_lead + 224;
            float v1372_data = s0[(v233_lead + 224)];
            float v1373_data = s1[25];
            float v1375_data = ir6[2];
            ir6[2] = (v1375_data + (v1372_data * v1373_data));
            int32_t v1382_a = v233_lead + 224;
            float v1389_data = s0[(v233_lead + 224)];
            float v1390_data = s1[34];
            float v1392_data = ir6[3];
            ir6[3] = (v1392_data + (v1389_data * v1390_data));
            int32_t v1399_a = v233_lead + 224;
            float v1406_data = s0[(v233_lead + 224)];
            float v1407_data = s1[43];
            float v1409_data = ir6[4];
            ir6[4] = (v1409_data + (v1406_data * v1407_data));
            int32_t v1416_a = v233_lead + 224;
            float v1423_data = s0[(v233_lead + 224)];
            float v1424_data = s1[52];
            float v1426_data = ir6[5];
            ir6[5] = (v1426_data + (v1423_data * v1424_data));
            int32_t v1433_a = v233_lead + 224;
            float v1440_data = s0[(v233_lead + 224)];
            float v1441_data = s1[61];
            float v1443_data = ir6[6];
            ir6[6] = (v1443_data + (v1440_data * v1441_data));
            int32_t v1450_a = v233_lead + 224;
            float v1457_data = s0[(v233_lead + 224)];
            float v1458_data = s1[70];
            float v1460_data = ir6[7];
            ir6[7] = (v1460_data + (v1457_data * v1458_data));
            int32_t v1467_a = v233_lead + 224;
            float v1474_data = s0[(v233_lead + 224)];
            float v1475_data = s1[79];
            float v1477_data = ir6[8];
            ir6[8] = (v1477_data + (v1474_data * v1475_data));
            int32_t v1487_a = v233_lead + 256;
            float v1494_data = s0[(v233_lead + 256)];
            float v1495_data = s1[8];
            float v1497_data = ir6[0];
            ir6[0] = (v1497_data + (v1494_data * v1495_data));
            int32_t v1504_a = v233_lead + 256;
            float v1511_data = s0[(v233_lead + 256)];
            float v1512_data = s1[17];
            float v1514_data = ir6[1];
            ir6[1] = (v1514_data + (v1511_data * v1512_data));
            int32_t v1521_a = v233_lead + 256;
            float v1528_data = s0[(v233_lead + 256)];
            float v1529_data = s1[26];
            float v1531_data = ir6[2];
            ir6[2] = (v1531_data + (v1528_data * v1529_data));
            int32_t v1538_a = v233_lead + 256;
            float v1545_data = s0[(v233_lead + 256)];
            float v1546_data = s1[35];
            float v1548_data = ir6[3];
            ir6[3] = (v1548_data + (v1545_data * v1546_data));
            int32_t v1555_a = v233_lead + 256;
            float v1562_data = s0[(v233_lead + 256)];
            float v1563_data = s1[44];
            float v1565_data = ir6[4];
            ir6[4] = (v1565_data + (v1562_data * v1563_data));
            int32_t v1572_a = v233_lead + 256;
            float v1579_data = s0[(v233_lead + 256)];
            float v1580_data = s1[53];
            float v1582_data = ir6[5];
            ir6[5] = (v1582_data + (v1579_data * v1580_data));
            int32_t v1589_a = v233_lead + 256;
            float v1596_data = s0[(v233_lead + 256)];
            float v1597_data = s1[62];
            float v1599_data = ir6[6];
            ir6[6] = (v1599_data + (v1596_data * v1597_data));
            int32_t v1606_a = v233_lead + 256;
            float v1613_data = s0[(v233_lead + 256)];
            float v1614_data = s1[71];
            float v1616_data = ir6[7];
            ir6[7] = (v1616_data + (v1613_data * v1614_data));
            int32_t v1623_a = v233_lead + 256;
            float v1630_data = s0[(v233_lead + 256)];
            float v1631_data = s1[80];
            float v1633_data = ir6[8];
            ir6[8] = (v1633_data + (v1630_data * v1631_data));
            #pragma unroll
            for (int32_t v1638_n0 = 0; v1638_n0 < 1; ++v1638_n0) {
              #pragma unroll
              for (int32_t v1639_n1 = 0; v1639_n1 < 9; ++v1639_n1) {
                int32_t v1640_a = v1638_n0 + v1639_n1;
                int32_t v1641_a = v1638_n0 + v1639_n1;
                float v1642_data = ir6[v1641_a];
                int32_t v1643_a = v1638_n0 + v1639_n1;
                r6[v1641_a] = v1642_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r6);
          int32_t v1647_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v1648_i0 = 0; v1648_i0 < 1; ++v1648_i0) {
            int32_t v1657_lead = v1647_lead + (v1648_i0 * 32);
            #pragma unroll
            for (int32_t v1649_i1 = 0; v1649_i1 < 9; ++v1649_i1) {
              int32_t v1650_a = v1648_i0 + v1649_i1;
              float v1652_data = r6[(v1648_i0 + v1649_i1)];
              int32_t v1659_a = v1657_lead + (v1649_i1 * 32);
              glb_m3[v1659_a] = v1652_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

