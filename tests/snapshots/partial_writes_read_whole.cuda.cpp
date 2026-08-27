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
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v3_lead < 16) {
            float v118_data = r2[0];
            float v119_data = ir3[0];
            ir3[0] = (v119_data + v118_data);
            float v121_data = r2[1];
            float v122_data = ir3[1];
            ir3[1] = (v122_data + v121_data);
            float v124_data = r2[2];
            float v125_data = ir3[2];
            ir3[2] = (v125_data + v124_data);
            float v127_data = r2[3];
            float v128_data = ir3[3];
            ir3[3] = (v128_data + v127_data);
            float v130_data = r2[4];
            float v131_data = ir3[4];
            ir3[4] = (v131_data + v130_data);
            float v133_data = r2[5];
            float v134_data = ir3[5];
            ir3[5] = (v134_data + v133_data);
            float v136_data = r2[6];
            float v137_data = ir3[6];
            ir3[6] = (v137_data + v136_data);
            float v139_data = r2[7];
            float v140_data = ir3[7];
            ir3[7] = (v140_data + v139_data);
            float v142_data = r2[8];
            float v143_data = ir3[8];
            ir3[8] = (v143_data + v142_data);
          }
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v149_n1 = 0; v149_n1 < 9; ++v149_n1) {
              int32_t v150_a = 0 + v149_n1;
              float v152_data = ir3[v149_n1];
              int32_t v158_a = v149_n1 * 32;
              int32_t v159_a = v3_lead + v158_a;
              float v167_data = s0[(v3_lead + v158_a)];
              int32_t v169_a = 0 + v149_n1;
              r3[v149_n1] = (v167_data + v152_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v175_i1 = 0; v175_i1 < 9; ++v175_i1) {
              int32_t v176_a = 0 + v175_i1;
              float v178_data = r3[v175_i1];
              int32_t v185_a = v3_lead + (v175_i1 * 32);
              s0[v185_a] = v178_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v3_lead < 16) {
            float v192_data = r4[0];
            float v193_data = ir5[0];
            ir5[0] = (v193_data + v192_data);
            float v195_data = r4[1];
            float v196_data = ir5[1];
            ir5[1] = (v196_data + v195_data);
            float v198_data = r4[2];
            float v199_data = ir5[2];
            ir5[2] = (v199_data + v198_data);
            float v201_data = r4[3];
            float v202_data = ir5[3];
            ir5[3] = (v202_data + v201_data);
            float v204_data = r4[4];
            float v205_data = ir5[4];
            ir5[4] = (v205_data + v204_data);
            float v207_data = r4[5];
            float v208_data = ir5[5];
            ir5[5] = (v208_data + v207_data);
            float v210_data = r4[6];
            float v211_data = ir5[6];
            ir5[6] = (v211_data + v210_data);
            float v213_data = r4[7];
            float v214_data = ir5[7];
            ir5[7] = (v214_data + v213_data);
            float v216_data = r4[8];
            float v217_data = ir5[8];
            ir5[8] = (v217_data + v216_data);
          }
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v223_n1 = 0; v223_n1 < 9; ++v223_n1) {
              int32_t v224_a = 0 + v223_n1;
              float v226_data = ir5[v223_n1];
              int32_t v232_a = v223_n1 * 32;
              int32_t v233_a = v3_lead + v232_a;
              float v241_data = s0[(v3_lead + v232_a)];
              int32_t v243_a = 0 + v223_n1;
              r5[v223_n1] = (v241_data + v226_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v249_i1 = 0; v249_i1 < 9; ++v249_i1) {
              int32_t v250_a = 0 + v249_i1;
              float v252_data = r5[v249_i1];
              int32_t v259_a = v3_lead + (v249_i1 * 32);
              s0[v259_a] = v252_data;
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
          int32_t v270_a = v3_lead + 0;
          float v277_data = s0[v3_lead];
          float v278_data = s1[0];
          float v280_data = ir6[0];
          ir6[0] = (v280_data + (v277_data * v278_data));
          int32_t v287_a = v3_lead + 0;
          float v294_data = s0[v3_lead];
          float v295_data = s1[9];
          float v297_data = ir6[1];
          ir6[1] = (v297_data + (v294_data * v295_data));
          int32_t v304_a = v3_lead + 0;
          float v311_data = s0[v3_lead];
          float v312_data = s1[18];
          float v314_data = ir6[2];
          ir6[2] = (v314_data + (v311_data * v312_data));
          int32_t v321_a = v3_lead + 0;
          float v328_data = s0[v3_lead];
          float v329_data = s1[27];
          float v331_data = ir6[3];
          ir6[3] = (v331_data + (v328_data * v329_data));
          int32_t v338_a = v3_lead + 0;
          float v345_data = s0[v3_lead];
          float v346_data = s1[36];
          float v348_data = ir6[4];
          ir6[4] = (v348_data + (v345_data * v346_data));
          int32_t v355_a = v3_lead + 0;
          float v362_data = s0[v3_lead];
          float v363_data = s1[45];
          float v365_data = ir6[5];
          ir6[5] = (v365_data + (v362_data * v363_data));
          int32_t v372_a = v3_lead + 0;
          float v379_data = s0[v3_lead];
          float v380_data = s1[54];
          float v382_data = ir6[6];
          ir6[6] = (v382_data + (v379_data * v380_data));
          int32_t v389_a = v3_lead + 0;
          float v396_data = s0[v3_lead];
          float v397_data = s1[63];
          float v399_data = ir6[7];
          ir6[7] = (v399_data + (v396_data * v397_data));
          int32_t v406_a = v3_lead + 0;
          float v413_data = s0[v3_lead];
          float v414_data = s1[72];
          float v416_data = ir6[8];
          ir6[8] = (v416_data + (v413_data * v414_data));
          int32_t v426_a = v3_lead + 32;
          float v433_data = s0[(v3_lead + 32)];
          float v434_data = s1[1];
          float v436_data = ir6[0];
          ir6[0] = (v436_data + (v433_data * v434_data));
          int32_t v443_a = v3_lead + 32;
          float v450_data = s0[(v3_lead + 32)];
          float v451_data = s1[10];
          float v453_data = ir6[1];
          ir6[1] = (v453_data + (v450_data * v451_data));
          int32_t v460_a = v3_lead + 32;
          float v467_data = s0[(v3_lead + 32)];
          float v468_data = s1[19];
          float v470_data = ir6[2];
          ir6[2] = (v470_data + (v467_data * v468_data));
          int32_t v477_a = v3_lead + 32;
          float v484_data = s0[(v3_lead + 32)];
          float v485_data = s1[28];
          float v487_data = ir6[3];
          ir6[3] = (v487_data + (v484_data * v485_data));
          int32_t v494_a = v3_lead + 32;
          float v501_data = s0[(v3_lead + 32)];
          float v502_data = s1[37];
          float v504_data = ir6[4];
          ir6[4] = (v504_data + (v501_data * v502_data));
          int32_t v511_a = v3_lead + 32;
          float v518_data = s0[(v3_lead + 32)];
          float v519_data = s1[46];
          float v521_data = ir6[5];
          ir6[5] = (v521_data + (v518_data * v519_data));
          int32_t v528_a = v3_lead + 32;
          float v535_data = s0[(v3_lead + 32)];
          float v536_data = s1[55];
          float v538_data = ir6[6];
          ir6[6] = (v538_data + (v535_data * v536_data));
          int32_t v545_a = v3_lead + 32;
          float v552_data = s0[(v3_lead + 32)];
          float v553_data = s1[64];
          float v555_data = ir6[7];
          ir6[7] = (v555_data + (v552_data * v553_data));
          int32_t v562_a = v3_lead + 32;
          float v569_data = s0[(v3_lead + 32)];
          float v570_data = s1[73];
          float v572_data = ir6[8];
          ir6[8] = (v572_data + (v569_data * v570_data));
          int32_t v582_a = v3_lead + 64;
          float v589_data = s0[(v3_lead + 64)];
          float v590_data = s1[2];
          float v592_data = ir6[0];
          ir6[0] = (v592_data + (v589_data * v590_data));
          int32_t v599_a = v3_lead + 64;
          float v606_data = s0[(v3_lead + 64)];
          float v607_data = s1[11];
          float v609_data = ir6[1];
          ir6[1] = (v609_data + (v606_data * v607_data));
          int32_t v616_a = v3_lead + 64;
          float v623_data = s0[(v3_lead + 64)];
          float v624_data = s1[20];
          float v626_data = ir6[2];
          ir6[2] = (v626_data + (v623_data * v624_data));
          int32_t v633_a = v3_lead + 64;
          float v640_data = s0[(v3_lead + 64)];
          float v641_data = s1[29];
          float v643_data = ir6[3];
          ir6[3] = (v643_data + (v640_data * v641_data));
          int32_t v650_a = v3_lead + 64;
          float v657_data = s0[(v3_lead + 64)];
          float v658_data = s1[38];
          float v660_data = ir6[4];
          ir6[4] = (v660_data + (v657_data * v658_data));
          int32_t v667_a = v3_lead + 64;
          float v674_data = s0[(v3_lead + 64)];
          float v675_data = s1[47];
          float v677_data = ir6[5];
          ir6[5] = (v677_data + (v674_data * v675_data));
          int32_t v684_a = v3_lead + 64;
          float v691_data = s0[(v3_lead + 64)];
          float v692_data = s1[56];
          float v694_data = ir6[6];
          ir6[6] = (v694_data + (v691_data * v692_data));
          int32_t v701_a = v3_lead + 64;
          float v708_data = s0[(v3_lead + 64)];
          float v709_data = s1[65];
          float v711_data = ir6[7];
          ir6[7] = (v711_data + (v708_data * v709_data));
          int32_t v718_a = v3_lead + 64;
          float v725_data = s0[(v3_lead + 64)];
          float v726_data = s1[74];
          float v728_data = ir6[8];
          ir6[8] = (v728_data + (v725_data * v726_data));
          int32_t v738_a = v3_lead + 96;
          float v745_data = s0[(v3_lead + 96)];
          float v746_data = s1[3];
          float v748_data = ir6[0];
          ir6[0] = (v748_data + (v745_data * v746_data));
          int32_t v755_a = v3_lead + 96;
          float v762_data = s0[(v3_lead + 96)];
          float v763_data = s1[12];
          float v765_data = ir6[1];
          ir6[1] = (v765_data + (v762_data * v763_data));
          int32_t v772_a = v3_lead + 96;
          float v779_data = s0[(v3_lead + 96)];
          float v780_data = s1[21];
          float v782_data = ir6[2];
          ir6[2] = (v782_data + (v779_data * v780_data));
          int32_t v789_a = v3_lead + 96;
          float v796_data = s0[(v3_lead + 96)];
          float v797_data = s1[30];
          float v799_data = ir6[3];
          ir6[3] = (v799_data + (v796_data * v797_data));
          int32_t v806_a = v3_lead + 96;
          float v813_data = s0[(v3_lead + 96)];
          float v814_data = s1[39];
          float v816_data = ir6[4];
          ir6[4] = (v816_data + (v813_data * v814_data));
          int32_t v823_a = v3_lead + 96;
          float v830_data = s0[(v3_lead + 96)];
          float v831_data = s1[48];
          float v833_data = ir6[5];
          ir6[5] = (v833_data + (v830_data * v831_data));
          int32_t v840_a = v3_lead + 96;
          float v847_data = s0[(v3_lead + 96)];
          float v848_data = s1[57];
          float v850_data = ir6[6];
          ir6[6] = (v850_data + (v847_data * v848_data));
          int32_t v857_a = v3_lead + 96;
          float v864_data = s0[(v3_lead + 96)];
          float v865_data = s1[66];
          float v867_data = ir6[7];
          ir6[7] = (v867_data + (v864_data * v865_data));
          int32_t v874_a = v3_lead + 96;
          float v881_data = s0[(v3_lead + 96)];
          float v882_data = s1[75];
          float v884_data = ir6[8];
          ir6[8] = (v884_data + (v881_data * v882_data));
          int32_t v894_a = v3_lead + 128;
          float v901_data = s0[(v3_lead + 128)];
          float v902_data = s1[4];
          float v904_data = ir6[0];
          ir6[0] = (v904_data + (v901_data * v902_data));
          int32_t v911_a = v3_lead + 128;
          float v918_data = s0[(v3_lead + 128)];
          float v919_data = s1[13];
          float v921_data = ir6[1];
          ir6[1] = (v921_data + (v918_data * v919_data));
          int32_t v928_a = v3_lead + 128;
          float v935_data = s0[(v3_lead + 128)];
          float v936_data = s1[22];
          float v938_data = ir6[2];
          ir6[2] = (v938_data + (v935_data * v936_data));
          int32_t v945_a = v3_lead + 128;
          float v952_data = s0[(v3_lead + 128)];
          float v953_data = s1[31];
          float v955_data = ir6[3];
          ir6[3] = (v955_data + (v952_data * v953_data));
          int32_t v962_a = v3_lead + 128;
          float v969_data = s0[(v3_lead + 128)];
          float v970_data = s1[40];
          float v972_data = ir6[4];
          ir6[4] = (v972_data + (v969_data * v970_data));
          int32_t v979_a = v3_lead + 128;
          float v986_data = s0[(v3_lead + 128)];
          float v987_data = s1[49];
          float v989_data = ir6[5];
          ir6[5] = (v989_data + (v986_data * v987_data));
          int32_t v996_a = v3_lead + 128;
          float v1003_data = s0[(v3_lead + 128)];
          float v1004_data = s1[58];
          float v1006_data = ir6[6];
          ir6[6] = (v1006_data + (v1003_data * v1004_data));
          int32_t v1013_a = v3_lead + 128;
          float v1020_data = s0[(v3_lead + 128)];
          float v1021_data = s1[67];
          float v1023_data = ir6[7];
          ir6[7] = (v1023_data + (v1020_data * v1021_data));
          int32_t v1030_a = v3_lead + 128;
          float v1037_data = s0[(v3_lead + 128)];
          float v1038_data = s1[76];
          float v1040_data = ir6[8];
          ir6[8] = (v1040_data + (v1037_data * v1038_data));
          int32_t v1050_a = v3_lead + 160;
          float v1057_data = s0[(v3_lead + 160)];
          float v1058_data = s1[5];
          float v1060_data = ir6[0];
          ir6[0] = (v1060_data + (v1057_data * v1058_data));
          int32_t v1067_a = v3_lead + 160;
          float v1074_data = s0[(v3_lead + 160)];
          float v1075_data = s1[14];
          float v1077_data = ir6[1];
          ir6[1] = (v1077_data + (v1074_data * v1075_data));
          int32_t v1084_a = v3_lead + 160;
          float v1091_data = s0[(v3_lead + 160)];
          float v1092_data = s1[23];
          float v1094_data = ir6[2];
          ir6[2] = (v1094_data + (v1091_data * v1092_data));
          int32_t v1101_a = v3_lead + 160;
          float v1108_data = s0[(v3_lead + 160)];
          float v1109_data = s1[32];
          float v1111_data = ir6[3];
          ir6[3] = (v1111_data + (v1108_data * v1109_data));
          int32_t v1118_a = v3_lead + 160;
          float v1125_data = s0[(v3_lead + 160)];
          float v1126_data = s1[41];
          float v1128_data = ir6[4];
          ir6[4] = (v1128_data + (v1125_data * v1126_data));
          int32_t v1135_a = v3_lead + 160;
          float v1142_data = s0[(v3_lead + 160)];
          float v1143_data = s1[50];
          float v1145_data = ir6[5];
          ir6[5] = (v1145_data + (v1142_data * v1143_data));
          int32_t v1152_a = v3_lead + 160;
          float v1159_data = s0[(v3_lead + 160)];
          float v1160_data = s1[59];
          float v1162_data = ir6[6];
          ir6[6] = (v1162_data + (v1159_data * v1160_data));
          int32_t v1169_a = v3_lead + 160;
          float v1176_data = s0[(v3_lead + 160)];
          float v1177_data = s1[68];
          float v1179_data = ir6[7];
          ir6[7] = (v1179_data + (v1176_data * v1177_data));
          int32_t v1186_a = v3_lead + 160;
          float v1193_data = s0[(v3_lead + 160)];
          float v1194_data = s1[77];
          float v1196_data = ir6[8];
          ir6[8] = (v1196_data + (v1193_data * v1194_data));
          int32_t v1206_a = v3_lead + 192;
          float v1213_data = s0[(v3_lead + 192)];
          float v1214_data = s1[6];
          float v1216_data = ir6[0];
          ir6[0] = (v1216_data + (v1213_data * v1214_data));
          int32_t v1223_a = v3_lead + 192;
          float v1230_data = s0[(v3_lead + 192)];
          float v1231_data = s1[15];
          float v1233_data = ir6[1];
          ir6[1] = (v1233_data + (v1230_data * v1231_data));
          int32_t v1240_a = v3_lead + 192;
          float v1247_data = s0[(v3_lead + 192)];
          float v1248_data = s1[24];
          float v1250_data = ir6[2];
          ir6[2] = (v1250_data + (v1247_data * v1248_data));
          int32_t v1257_a = v3_lead + 192;
          float v1264_data = s0[(v3_lead + 192)];
          float v1265_data = s1[33];
          float v1267_data = ir6[3];
          ir6[3] = (v1267_data + (v1264_data * v1265_data));
          int32_t v1274_a = v3_lead + 192;
          float v1281_data = s0[(v3_lead + 192)];
          float v1282_data = s1[42];
          float v1284_data = ir6[4];
          ir6[4] = (v1284_data + (v1281_data * v1282_data));
          int32_t v1291_a = v3_lead + 192;
          float v1298_data = s0[(v3_lead + 192)];
          float v1299_data = s1[51];
          float v1301_data = ir6[5];
          ir6[5] = (v1301_data + (v1298_data * v1299_data));
          int32_t v1308_a = v3_lead + 192;
          float v1315_data = s0[(v3_lead + 192)];
          float v1316_data = s1[60];
          float v1318_data = ir6[6];
          ir6[6] = (v1318_data + (v1315_data * v1316_data));
          int32_t v1325_a = v3_lead + 192;
          float v1332_data = s0[(v3_lead + 192)];
          float v1333_data = s1[69];
          float v1335_data = ir6[7];
          ir6[7] = (v1335_data + (v1332_data * v1333_data));
          int32_t v1342_a = v3_lead + 192;
          float v1349_data = s0[(v3_lead + 192)];
          float v1350_data = s1[78];
          float v1352_data = ir6[8];
          ir6[8] = (v1352_data + (v1349_data * v1350_data));
          int32_t v1362_a = v3_lead + 224;
          float v1369_data = s0[(v3_lead + 224)];
          float v1370_data = s1[7];
          float v1372_data = ir6[0];
          ir6[0] = (v1372_data + (v1369_data * v1370_data));
          int32_t v1379_a = v3_lead + 224;
          float v1386_data = s0[(v3_lead + 224)];
          float v1387_data = s1[16];
          float v1389_data = ir6[1];
          ir6[1] = (v1389_data + (v1386_data * v1387_data));
          int32_t v1396_a = v3_lead + 224;
          float v1403_data = s0[(v3_lead + 224)];
          float v1404_data = s1[25];
          float v1406_data = ir6[2];
          ir6[2] = (v1406_data + (v1403_data * v1404_data));
          int32_t v1413_a = v3_lead + 224;
          float v1420_data = s0[(v3_lead + 224)];
          float v1421_data = s1[34];
          float v1423_data = ir6[3];
          ir6[3] = (v1423_data + (v1420_data * v1421_data));
          int32_t v1430_a = v3_lead + 224;
          float v1437_data = s0[(v3_lead + 224)];
          float v1438_data = s1[43];
          float v1440_data = ir6[4];
          ir6[4] = (v1440_data + (v1437_data * v1438_data));
          int32_t v1447_a = v3_lead + 224;
          float v1454_data = s0[(v3_lead + 224)];
          float v1455_data = s1[52];
          float v1457_data = ir6[5];
          ir6[5] = (v1457_data + (v1454_data * v1455_data));
          int32_t v1464_a = v3_lead + 224;
          float v1471_data = s0[(v3_lead + 224)];
          float v1472_data = s1[61];
          float v1474_data = ir6[6];
          ir6[6] = (v1474_data + (v1471_data * v1472_data));
          int32_t v1481_a = v3_lead + 224;
          float v1488_data = s0[(v3_lead + 224)];
          float v1489_data = s1[70];
          float v1491_data = ir6[7];
          ir6[7] = (v1491_data + (v1488_data * v1489_data));
          int32_t v1498_a = v3_lead + 224;
          float v1505_data = s0[(v3_lead + 224)];
          float v1506_data = s1[79];
          float v1508_data = ir6[8];
          ir6[8] = (v1508_data + (v1505_data * v1506_data));
          int32_t v1518_a = v3_lead + 256;
          float v1525_data = s0[(v3_lead + 256)];
          float v1526_data = s1[8];
          float v1528_data = ir6[0];
          ir6[0] = (v1528_data + (v1525_data * v1526_data));
          int32_t v1535_a = v3_lead + 256;
          float v1542_data = s0[(v3_lead + 256)];
          float v1543_data = s1[17];
          float v1545_data = ir6[1];
          ir6[1] = (v1545_data + (v1542_data * v1543_data));
          int32_t v1552_a = v3_lead + 256;
          float v1559_data = s0[(v3_lead + 256)];
          float v1560_data = s1[26];
          float v1562_data = ir6[2];
          ir6[2] = (v1562_data + (v1559_data * v1560_data));
          int32_t v1569_a = v3_lead + 256;
          float v1576_data = s0[(v3_lead + 256)];
          float v1577_data = s1[35];
          float v1579_data = ir6[3];
          ir6[3] = (v1579_data + (v1576_data * v1577_data));
          int32_t v1586_a = v3_lead + 256;
          float v1593_data = s0[(v3_lead + 256)];
          float v1594_data = s1[44];
          float v1596_data = ir6[4];
          ir6[4] = (v1596_data + (v1593_data * v1594_data));
          int32_t v1603_a = v3_lead + 256;
          float v1610_data = s0[(v3_lead + 256)];
          float v1611_data = s1[53];
          float v1613_data = ir6[5];
          ir6[5] = (v1613_data + (v1610_data * v1611_data));
          int32_t v1620_a = v3_lead + 256;
          float v1627_data = s0[(v3_lead + 256)];
          float v1628_data = s1[62];
          float v1630_data = ir6[6];
          ir6[6] = (v1630_data + (v1627_data * v1628_data));
          int32_t v1637_a = v3_lead + 256;
          float v1644_data = s0[(v3_lead + 256)];
          float v1645_data = s1[71];
          float v1647_data = ir6[7];
          ir6[7] = (v1647_data + (v1644_data * v1645_data));
          int32_t v1654_a = v3_lead + 256;
          float v1661_data = s0[(v3_lead + 256)];
          float v1662_data = s1[80];
          float v1664_data = ir6[8];
          ir6[8] = (v1664_data + (v1661_data * v1662_data));
          #pragma unroll
          for (int32_t v1669_n0 = 0; v1669_n0 < 1; ++v1669_n0) {
            #pragma unroll
            for (int32_t v1670_n1 = 0; v1670_n1 < 9; ++v1670_n1) {
              int32_t v1671_a = v1669_n0 + v1670_n1;
              int32_t v1672_a = v1669_n0 + v1670_n1;
              float v1673_data = ir6[v1672_a];
              int32_t v1674_a = v1669_n0 + v1670_n1;
              r6[v1672_a] = v1673_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1679_i0 = 0; v1679_i0 < 1; ++v1679_i0) {
            int32_t v1688_lead = v3_lead + (v1679_i0 * 32);
            #pragma unroll
            for (int32_t v1680_i1 = 0; v1680_i1 < 9; ++v1680_i1) {
              int32_t v1681_a = v1679_i0 + v1680_i1;
              float v1683_data = r6[(v1679_i0 + v1680_i1)];
              int32_t v1690_a = v1688_lead + (v1680_i1 * 32);
              glb_m3[v1690_a] = v1683_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

