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
          int32_t v8_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
            int32_t v14_lead = v9_i0 * 32;
            int32_t v15_lead = v8_lead + v14_lead;
            int32_t v22_lead = v8_lead + v14_lead;
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 9; ++v10_i1) {
              int32_t v16_a = v10_i1 * 32;
              int32_t v17_a = v15_lead + v16_a;
              float v25_data = __ldcg(&glb_m0[(v22_lead + v16_a)]);
              int32_t v26_a = v9_i0 + v10_i1;
              r0[v26_a] = v25_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
              int32_t v38_a = v32_i1 * 16;
              int32_t v39_a = v8_lead + v38_a;
              float v47_data = __ldcg(&glb_m1[(v8_lead + v38_a)]);
              int32_t v48_a = 0 + v32_i1;
              r2[v48_a] = v47_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v53_data = r0[0];
          float v54_data = ir1[0];
          ir1[0] = (v54_data + v53_data);
          float v56_data = r0[1];
          float v57_data = ir1[1];
          ir1[1] = (v57_data + v56_data);
          float v59_data = r0[2];
          float v60_data = ir1[2];
          ir1[2] = (v60_data + v59_data);
          float v62_data = r0[3];
          float v63_data = ir1[3];
          ir1[3] = (v63_data + v62_data);
          float v65_data = r0[4];
          float v66_data = ir1[4];
          ir1[4] = (v66_data + v65_data);
          float v68_data = r0[5];
          float v69_data = ir1[5];
          ir1[5] = (v69_data + v68_data);
          float v71_data = r0[6];
          float v72_data = ir1[6];
          ir1[6] = (v72_data + v71_data);
          float v74_data = r0[7];
          float v75_data = ir1[7];
          ir1[7] = (v75_data + v74_data);
          float v77_data = r0[8];
          float v78_data = ir1[8];
          ir1[8] = (v78_data + v77_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v84_i0 = 0; v84_i0 < 1; ++v84_i0) {
            int32_t v93_lead = v8_lead + (v84_i0 * 32);
            #pragma unroll
            for (int32_t v85_i1 = 0; v85_i1 < 9; ++v85_i1) {
              int32_t v86_a = v84_i0 + v85_i1;
              float v88_data = r1[(v84_i0 + v85_i1)];
              int32_t v95_a = v93_lead + (v85_i1 * 32);
              s0[v95_a] = v88_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v101_i1 = 0; v101_i1 < 9; ++v101_i1) {
              int32_t v107_a = v101_i1 * 16;
              int32_t v108_a = v8_lead + v107_a;
              float v116_data = __ldcg(&glb_m2[(v8_lead + v107_a)]);
              int32_t v117_a = 0 + v101_i1;
              r4[v117_a] = v116_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v8_lead < 16) {
            float v124_data = r2[0];
            float v125_data = ir3[0];
            ir3[0] = (v125_data + v124_data);
            float v127_data = r2[1];
            float v128_data = ir3[1];
            ir3[1] = (v128_data + v127_data);
            float v130_data = r2[2];
            float v131_data = ir3[2];
            ir3[2] = (v131_data + v130_data);
            float v133_data = r2[3];
            float v134_data = ir3[3];
            ir3[3] = (v134_data + v133_data);
            float v136_data = r2[4];
            float v137_data = ir3[4];
            ir3[4] = (v137_data + v136_data);
            float v139_data = r2[5];
            float v140_data = ir3[5];
            ir3[5] = (v140_data + v139_data);
            float v142_data = r2[6];
            float v143_data = ir3[6];
            ir3[6] = (v143_data + v142_data);
            float v145_data = r2[7];
            float v146_data = ir3[7];
            ir3[7] = (v146_data + v145_data);
            float v148_data = r2[8];
            float v149_data = ir3[8];
            ir3[8] = (v149_data + v148_data);
          }
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v155_n1 = 0; v155_n1 < 9; ++v155_n1) {
              int32_t v156_a = 0 + v155_n1;
              float v158_data = ir3[v155_n1];
              int32_t v164_a = v155_n1 * 32;
              int32_t v165_a = v8_lead + v164_a;
              float v173_data = s0[(v8_lead + v164_a)];
              int32_t v175_a = 0 + v155_n1;
              r3[v155_n1] = (v173_data + v158_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v181_i1 = 0; v181_i1 < 9; ++v181_i1) {
              int32_t v182_a = 0 + v181_i1;
              float v184_data = r3[v181_i1];
              int32_t v191_a = v8_lead + (v181_i1 * 32);
              s0[v191_a] = v184_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v8_lead < 16) {
            float v198_data = r4[0];
            float v199_data = ir5[0];
            ir5[0] = (v199_data + v198_data);
            float v201_data = r4[1];
            float v202_data = ir5[1];
            ir5[1] = (v202_data + v201_data);
            float v204_data = r4[2];
            float v205_data = ir5[2];
            ir5[2] = (v205_data + v204_data);
            float v207_data = r4[3];
            float v208_data = ir5[3];
            ir5[3] = (v208_data + v207_data);
            float v210_data = r4[4];
            float v211_data = ir5[4];
            ir5[4] = (v211_data + v210_data);
            float v213_data = r4[5];
            float v214_data = ir5[5];
            ir5[5] = (v214_data + v213_data);
            float v216_data = r4[6];
            float v217_data = ir5[6];
            ir5[6] = (v217_data + v216_data);
            float v219_data = r4[7];
            float v220_data = ir5[7];
            ir5[7] = (v220_data + v219_data);
            float v222_data = r4[8];
            float v223_data = ir5[8];
            ir5[8] = (v223_data + v222_data);
          }
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v229_n1 = 0; v229_n1 < 9; ++v229_n1) {
              int32_t v230_a = 0 + v229_n1;
              float v232_data = ir5[v229_n1];
              int32_t v238_a = v229_n1 * 32;
              int32_t v239_a = v8_lead + v238_a;
              float v247_data = s0[(v8_lead + v238_a)];
              int32_t v249_a = 0 + v229_n1;
              r5[v229_n1] = (v247_data + v232_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v255_i1 = 0; v255_i1 < 9; ++v255_i1) {
              int32_t v256_a = 0 + v255_i1;
              float v258_data = r5[v255_i1];
              int32_t v265_a = v8_lead + (v255_i1 * 32);
              s0[v265_a] = v258_data;
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
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          int32_t v277_a = v8_lead + 0;
          float v284_data = s0[v8_lead];
          float v285_data = s1[0];
          float v287_data = ir6[0];
          ir6[0] = (v287_data + (v284_data * v285_data));
          int32_t v294_a = v8_lead + 0;
          float v301_data = s0[v8_lead];
          float v302_data = s1[9];
          float v304_data = ir6[1];
          ir6[1] = (v304_data + (v301_data * v302_data));
          int32_t v311_a = v8_lead + 0;
          float v318_data = s0[v8_lead];
          float v319_data = s1[18];
          float v321_data = ir6[2];
          ir6[2] = (v321_data + (v318_data * v319_data));
          int32_t v328_a = v8_lead + 0;
          float v335_data = s0[v8_lead];
          float v336_data = s1[27];
          float v338_data = ir6[3];
          ir6[3] = (v338_data + (v335_data * v336_data));
          int32_t v345_a = v8_lead + 0;
          float v352_data = s0[v8_lead];
          float v353_data = s1[36];
          float v355_data = ir6[4];
          ir6[4] = (v355_data + (v352_data * v353_data));
          int32_t v362_a = v8_lead + 0;
          float v369_data = s0[v8_lead];
          float v370_data = s1[45];
          float v372_data = ir6[5];
          ir6[5] = (v372_data + (v369_data * v370_data));
          int32_t v379_a = v8_lead + 0;
          float v386_data = s0[v8_lead];
          float v387_data = s1[54];
          float v389_data = ir6[6];
          ir6[6] = (v389_data + (v386_data * v387_data));
          int32_t v396_a = v8_lead + 0;
          float v403_data = s0[v8_lead];
          float v404_data = s1[63];
          float v406_data = ir6[7];
          ir6[7] = (v406_data + (v403_data * v404_data));
          int32_t v413_a = v8_lead + 0;
          float v420_data = s0[v8_lead];
          float v421_data = s1[72];
          float v423_data = ir6[8];
          ir6[8] = (v423_data + (v420_data * v421_data));
          int32_t v433_a = v8_lead + 32;
          float v440_data = s0[(v8_lead + 32)];
          float v441_data = s1[1];
          float v443_data = ir6[0];
          ir6[0] = (v443_data + (v440_data * v441_data));
          int32_t v450_a = v8_lead + 32;
          float v457_data = s0[(v8_lead + 32)];
          float v458_data = s1[10];
          float v460_data = ir6[1];
          ir6[1] = (v460_data + (v457_data * v458_data));
          int32_t v467_a = v8_lead + 32;
          float v474_data = s0[(v8_lead + 32)];
          float v475_data = s1[19];
          float v477_data = ir6[2];
          ir6[2] = (v477_data + (v474_data * v475_data));
          int32_t v484_a = v8_lead + 32;
          float v491_data = s0[(v8_lead + 32)];
          float v492_data = s1[28];
          float v494_data = ir6[3];
          ir6[3] = (v494_data + (v491_data * v492_data));
          int32_t v501_a = v8_lead + 32;
          float v508_data = s0[(v8_lead + 32)];
          float v509_data = s1[37];
          float v511_data = ir6[4];
          ir6[4] = (v511_data + (v508_data * v509_data));
          int32_t v518_a = v8_lead + 32;
          float v525_data = s0[(v8_lead + 32)];
          float v526_data = s1[46];
          float v528_data = ir6[5];
          ir6[5] = (v528_data + (v525_data * v526_data));
          int32_t v535_a = v8_lead + 32;
          float v542_data = s0[(v8_lead + 32)];
          float v543_data = s1[55];
          float v545_data = ir6[6];
          ir6[6] = (v545_data + (v542_data * v543_data));
          int32_t v552_a = v8_lead + 32;
          float v559_data = s0[(v8_lead + 32)];
          float v560_data = s1[64];
          float v562_data = ir6[7];
          ir6[7] = (v562_data + (v559_data * v560_data));
          int32_t v569_a = v8_lead + 32;
          float v576_data = s0[(v8_lead + 32)];
          float v577_data = s1[73];
          float v579_data = ir6[8];
          ir6[8] = (v579_data + (v576_data * v577_data));
          int32_t v589_a = v8_lead + 64;
          float v596_data = s0[(v8_lead + 64)];
          float v597_data = s1[2];
          float v599_data = ir6[0];
          ir6[0] = (v599_data + (v596_data * v597_data));
          int32_t v606_a = v8_lead + 64;
          float v613_data = s0[(v8_lead + 64)];
          float v614_data = s1[11];
          float v616_data = ir6[1];
          ir6[1] = (v616_data + (v613_data * v614_data));
          int32_t v623_a = v8_lead + 64;
          float v630_data = s0[(v8_lead + 64)];
          float v631_data = s1[20];
          float v633_data = ir6[2];
          ir6[2] = (v633_data + (v630_data * v631_data));
          int32_t v640_a = v8_lead + 64;
          float v647_data = s0[(v8_lead + 64)];
          float v648_data = s1[29];
          float v650_data = ir6[3];
          ir6[3] = (v650_data + (v647_data * v648_data));
          int32_t v657_a = v8_lead + 64;
          float v664_data = s0[(v8_lead + 64)];
          float v665_data = s1[38];
          float v667_data = ir6[4];
          ir6[4] = (v667_data + (v664_data * v665_data));
          int32_t v674_a = v8_lead + 64;
          float v681_data = s0[(v8_lead + 64)];
          float v682_data = s1[47];
          float v684_data = ir6[5];
          ir6[5] = (v684_data + (v681_data * v682_data));
          int32_t v691_a = v8_lead + 64;
          float v698_data = s0[(v8_lead + 64)];
          float v699_data = s1[56];
          float v701_data = ir6[6];
          ir6[6] = (v701_data + (v698_data * v699_data));
          int32_t v708_a = v8_lead + 64;
          float v715_data = s0[(v8_lead + 64)];
          float v716_data = s1[65];
          float v718_data = ir6[7];
          ir6[7] = (v718_data + (v715_data * v716_data));
          int32_t v725_a = v8_lead + 64;
          float v732_data = s0[(v8_lead + 64)];
          float v733_data = s1[74];
          float v735_data = ir6[8];
          ir6[8] = (v735_data + (v732_data * v733_data));
          int32_t v745_a = v8_lead + 96;
          float v752_data = s0[(v8_lead + 96)];
          float v753_data = s1[3];
          float v755_data = ir6[0];
          ir6[0] = (v755_data + (v752_data * v753_data));
          int32_t v762_a = v8_lead + 96;
          float v769_data = s0[(v8_lead + 96)];
          float v770_data = s1[12];
          float v772_data = ir6[1];
          ir6[1] = (v772_data + (v769_data * v770_data));
          int32_t v779_a = v8_lead + 96;
          float v786_data = s0[(v8_lead + 96)];
          float v787_data = s1[21];
          float v789_data = ir6[2];
          ir6[2] = (v789_data + (v786_data * v787_data));
          int32_t v796_a = v8_lead + 96;
          float v803_data = s0[(v8_lead + 96)];
          float v804_data = s1[30];
          float v806_data = ir6[3];
          ir6[3] = (v806_data + (v803_data * v804_data));
          int32_t v813_a = v8_lead + 96;
          float v820_data = s0[(v8_lead + 96)];
          float v821_data = s1[39];
          float v823_data = ir6[4];
          ir6[4] = (v823_data + (v820_data * v821_data));
          int32_t v830_a = v8_lead + 96;
          float v837_data = s0[(v8_lead + 96)];
          float v838_data = s1[48];
          float v840_data = ir6[5];
          ir6[5] = (v840_data + (v837_data * v838_data));
          int32_t v847_a = v8_lead + 96;
          float v854_data = s0[(v8_lead + 96)];
          float v855_data = s1[57];
          float v857_data = ir6[6];
          ir6[6] = (v857_data + (v854_data * v855_data));
          int32_t v864_a = v8_lead + 96;
          float v871_data = s0[(v8_lead + 96)];
          float v872_data = s1[66];
          float v874_data = ir6[7];
          ir6[7] = (v874_data + (v871_data * v872_data));
          int32_t v881_a = v8_lead + 96;
          float v888_data = s0[(v8_lead + 96)];
          float v889_data = s1[75];
          float v891_data = ir6[8];
          ir6[8] = (v891_data + (v888_data * v889_data));
          int32_t v901_a = v8_lead + 128;
          float v908_data = s0[(v8_lead + 128)];
          float v909_data = s1[4];
          float v911_data = ir6[0];
          ir6[0] = (v911_data + (v908_data * v909_data));
          int32_t v918_a = v8_lead + 128;
          float v925_data = s0[(v8_lead + 128)];
          float v926_data = s1[13];
          float v928_data = ir6[1];
          ir6[1] = (v928_data + (v925_data * v926_data));
          int32_t v935_a = v8_lead + 128;
          float v942_data = s0[(v8_lead + 128)];
          float v943_data = s1[22];
          float v945_data = ir6[2];
          ir6[2] = (v945_data + (v942_data * v943_data));
          int32_t v952_a = v8_lead + 128;
          float v959_data = s0[(v8_lead + 128)];
          float v960_data = s1[31];
          float v962_data = ir6[3];
          ir6[3] = (v962_data + (v959_data * v960_data));
          int32_t v969_a = v8_lead + 128;
          float v976_data = s0[(v8_lead + 128)];
          float v977_data = s1[40];
          float v979_data = ir6[4];
          ir6[4] = (v979_data + (v976_data * v977_data));
          int32_t v986_a = v8_lead + 128;
          float v993_data = s0[(v8_lead + 128)];
          float v994_data = s1[49];
          float v996_data = ir6[5];
          ir6[5] = (v996_data + (v993_data * v994_data));
          int32_t v1003_a = v8_lead + 128;
          float v1010_data = s0[(v8_lead + 128)];
          float v1011_data = s1[58];
          float v1013_data = ir6[6];
          ir6[6] = (v1013_data + (v1010_data * v1011_data));
          int32_t v1020_a = v8_lead + 128;
          float v1027_data = s0[(v8_lead + 128)];
          float v1028_data = s1[67];
          float v1030_data = ir6[7];
          ir6[7] = (v1030_data + (v1027_data * v1028_data));
          int32_t v1037_a = v8_lead + 128;
          float v1044_data = s0[(v8_lead + 128)];
          float v1045_data = s1[76];
          float v1047_data = ir6[8];
          ir6[8] = (v1047_data + (v1044_data * v1045_data));
          int32_t v1057_a = v8_lead + 160;
          float v1064_data = s0[(v8_lead + 160)];
          float v1065_data = s1[5];
          float v1067_data = ir6[0];
          ir6[0] = (v1067_data + (v1064_data * v1065_data));
          int32_t v1074_a = v8_lead + 160;
          float v1081_data = s0[(v8_lead + 160)];
          float v1082_data = s1[14];
          float v1084_data = ir6[1];
          ir6[1] = (v1084_data + (v1081_data * v1082_data));
          int32_t v1091_a = v8_lead + 160;
          float v1098_data = s0[(v8_lead + 160)];
          float v1099_data = s1[23];
          float v1101_data = ir6[2];
          ir6[2] = (v1101_data + (v1098_data * v1099_data));
          int32_t v1108_a = v8_lead + 160;
          float v1115_data = s0[(v8_lead + 160)];
          float v1116_data = s1[32];
          float v1118_data = ir6[3];
          ir6[3] = (v1118_data + (v1115_data * v1116_data));
          int32_t v1125_a = v8_lead + 160;
          float v1132_data = s0[(v8_lead + 160)];
          float v1133_data = s1[41];
          float v1135_data = ir6[4];
          ir6[4] = (v1135_data + (v1132_data * v1133_data));
          int32_t v1142_a = v8_lead + 160;
          float v1149_data = s0[(v8_lead + 160)];
          float v1150_data = s1[50];
          float v1152_data = ir6[5];
          ir6[5] = (v1152_data + (v1149_data * v1150_data));
          int32_t v1159_a = v8_lead + 160;
          float v1166_data = s0[(v8_lead + 160)];
          float v1167_data = s1[59];
          float v1169_data = ir6[6];
          ir6[6] = (v1169_data + (v1166_data * v1167_data));
          int32_t v1176_a = v8_lead + 160;
          float v1183_data = s0[(v8_lead + 160)];
          float v1184_data = s1[68];
          float v1186_data = ir6[7];
          ir6[7] = (v1186_data + (v1183_data * v1184_data));
          int32_t v1193_a = v8_lead + 160;
          float v1200_data = s0[(v8_lead + 160)];
          float v1201_data = s1[77];
          float v1203_data = ir6[8];
          ir6[8] = (v1203_data + (v1200_data * v1201_data));
          int32_t v1213_a = v8_lead + 192;
          float v1220_data = s0[(v8_lead + 192)];
          float v1221_data = s1[6];
          float v1223_data = ir6[0];
          ir6[0] = (v1223_data + (v1220_data * v1221_data));
          int32_t v1230_a = v8_lead + 192;
          float v1237_data = s0[(v8_lead + 192)];
          float v1238_data = s1[15];
          float v1240_data = ir6[1];
          ir6[1] = (v1240_data + (v1237_data * v1238_data));
          int32_t v1247_a = v8_lead + 192;
          float v1254_data = s0[(v8_lead + 192)];
          float v1255_data = s1[24];
          float v1257_data = ir6[2];
          ir6[2] = (v1257_data + (v1254_data * v1255_data));
          int32_t v1264_a = v8_lead + 192;
          float v1271_data = s0[(v8_lead + 192)];
          float v1272_data = s1[33];
          float v1274_data = ir6[3];
          ir6[3] = (v1274_data + (v1271_data * v1272_data));
          int32_t v1281_a = v8_lead + 192;
          float v1288_data = s0[(v8_lead + 192)];
          float v1289_data = s1[42];
          float v1291_data = ir6[4];
          ir6[4] = (v1291_data + (v1288_data * v1289_data));
          int32_t v1298_a = v8_lead + 192;
          float v1305_data = s0[(v8_lead + 192)];
          float v1306_data = s1[51];
          float v1308_data = ir6[5];
          ir6[5] = (v1308_data + (v1305_data * v1306_data));
          int32_t v1315_a = v8_lead + 192;
          float v1322_data = s0[(v8_lead + 192)];
          float v1323_data = s1[60];
          float v1325_data = ir6[6];
          ir6[6] = (v1325_data + (v1322_data * v1323_data));
          int32_t v1332_a = v8_lead + 192;
          float v1339_data = s0[(v8_lead + 192)];
          float v1340_data = s1[69];
          float v1342_data = ir6[7];
          ir6[7] = (v1342_data + (v1339_data * v1340_data));
          int32_t v1349_a = v8_lead + 192;
          float v1356_data = s0[(v8_lead + 192)];
          float v1357_data = s1[78];
          float v1359_data = ir6[8];
          ir6[8] = (v1359_data + (v1356_data * v1357_data));
          int32_t v1369_a = v8_lead + 224;
          float v1376_data = s0[(v8_lead + 224)];
          float v1377_data = s1[7];
          float v1379_data = ir6[0];
          ir6[0] = (v1379_data + (v1376_data * v1377_data));
          int32_t v1386_a = v8_lead + 224;
          float v1393_data = s0[(v8_lead + 224)];
          float v1394_data = s1[16];
          float v1396_data = ir6[1];
          ir6[1] = (v1396_data + (v1393_data * v1394_data));
          int32_t v1403_a = v8_lead + 224;
          float v1410_data = s0[(v8_lead + 224)];
          float v1411_data = s1[25];
          float v1413_data = ir6[2];
          ir6[2] = (v1413_data + (v1410_data * v1411_data));
          int32_t v1420_a = v8_lead + 224;
          float v1427_data = s0[(v8_lead + 224)];
          float v1428_data = s1[34];
          float v1430_data = ir6[3];
          ir6[3] = (v1430_data + (v1427_data * v1428_data));
          int32_t v1437_a = v8_lead + 224;
          float v1444_data = s0[(v8_lead + 224)];
          float v1445_data = s1[43];
          float v1447_data = ir6[4];
          ir6[4] = (v1447_data + (v1444_data * v1445_data));
          int32_t v1454_a = v8_lead + 224;
          float v1461_data = s0[(v8_lead + 224)];
          float v1462_data = s1[52];
          float v1464_data = ir6[5];
          ir6[5] = (v1464_data + (v1461_data * v1462_data));
          int32_t v1471_a = v8_lead + 224;
          float v1478_data = s0[(v8_lead + 224)];
          float v1479_data = s1[61];
          float v1481_data = ir6[6];
          ir6[6] = (v1481_data + (v1478_data * v1479_data));
          int32_t v1488_a = v8_lead + 224;
          float v1495_data = s0[(v8_lead + 224)];
          float v1496_data = s1[70];
          float v1498_data = ir6[7];
          ir6[7] = (v1498_data + (v1495_data * v1496_data));
          int32_t v1505_a = v8_lead + 224;
          float v1512_data = s0[(v8_lead + 224)];
          float v1513_data = s1[79];
          float v1515_data = ir6[8];
          ir6[8] = (v1515_data + (v1512_data * v1513_data));
          int32_t v1525_a = v8_lead + 256;
          float v1532_data = s0[(v8_lead + 256)];
          float v1533_data = s1[8];
          float v1535_data = ir6[0];
          ir6[0] = (v1535_data + (v1532_data * v1533_data));
          int32_t v1542_a = v8_lead + 256;
          float v1549_data = s0[(v8_lead + 256)];
          float v1550_data = s1[17];
          float v1552_data = ir6[1];
          ir6[1] = (v1552_data + (v1549_data * v1550_data));
          int32_t v1559_a = v8_lead + 256;
          float v1566_data = s0[(v8_lead + 256)];
          float v1567_data = s1[26];
          float v1569_data = ir6[2];
          ir6[2] = (v1569_data + (v1566_data * v1567_data));
          int32_t v1576_a = v8_lead + 256;
          float v1583_data = s0[(v8_lead + 256)];
          float v1584_data = s1[35];
          float v1586_data = ir6[3];
          ir6[3] = (v1586_data + (v1583_data * v1584_data));
          int32_t v1593_a = v8_lead + 256;
          float v1600_data = s0[(v8_lead + 256)];
          float v1601_data = s1[44];
          float v1603_data = ir6[4];
          ir6[4] = (v1603_data + (v1600_data * v1601_data));
          int32_t v1610_a = v8_lead + 256;
          float v1617_data = s0[(v8_lead + 256)];
          float v1618_data = s1[53];
          float v1620_data = ir6[5];
          ir6[5] = (v1620_data + (v1617_data * v1618_data));
          int32_t v1627_a = v8_lead + 256;
          float v1634_data = s0[(v8_lead + 256)];
          float v1635_data = s1[62];
          float v1637_data = ir6[6];
          ir6[6] = (v1637_data + (v1634_data * v1635_data));
          int32_t v1644_a = v8_lead + 256;
          float v1651_data = s0[(v8_lead + 256)];
          float v1652_data = s1[71];
          float v1654_data = ir6[7];
          ir6[7] = (v1654_data + (v1651_data * v1652_data));
          int32_t v1661_a = v8_lead + 256;
          float v1668_data = s0[(v8_lead + 256)];
          float v1669_data = s1[80];
          float v1671_data = ir6[8];
          ir6[8] = (v1671_data + (v1668_data * v1669_data));
          #pragma unroll
          for (int32_t v1676_n0 = 0; v1676_n0 < 1; ++v1676_n0) {
            #pragma unroll
            for (int32_t v1677_n1 = 0; v1677_n1 < 9; ++v1677_n1) {
              int32_t v1678_a = v1676_n0 + v1677_n1;
              int32_t v1679_a = v1676_n0 + v1677_n1;
              float v1680_data = ir6[v1679_a];
              int32_t v1681_a = v1676_n0 + v1677_n1;
              r6[v1679_a] = v1680_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1686_i0 = 0; v1686_i0 < 1; ++v1686_i0) {
            int32_t v1695_lead = v8_lead + (v1686_i0 * 32);
            #pragma unroll
            for (int32_t v1687_i1 = 0; v1687_i1 < 9; ++v1687_i1) {
              int32_t v1688_a = v1686_i0 + v1687_i1;
              float v1690_data = r6[(v1686_i0 + v1687_i1)];
              int32_t v1697_a = v1695_lead + (v1687_i1 * 32);
              glb_m3[v1697_a] = v1690_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

