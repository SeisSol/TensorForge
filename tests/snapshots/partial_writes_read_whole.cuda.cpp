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
          float v53_data = r0[0];
          float v54_data = r1[0];
          r1[0] = (v54_data + v53_data);
          float v56_data = r0[1];
          float v57_data = r1[1];
          r1[1] = (v57_data + v56_data);
          float v59_data = r0[2];
          float v60_data = r1[2];
          r1[2] = (v60_data + v59_data);
          float v62_data = r0[3];
          float v63_data = r1[3];
          r1[3] = (v63_data + v62_data);
          float v65_data = r0[4];
          float v66_data = r1[4];
          r1[4] = (v66_data + v65_data);
          float v68_data = r0[5];
          float v69_data = r1[5];
          r1[5] = (v69_data + v68_data);
          float v71_data = r0[6];
          float v72_data = r1[6];
          r1[6] = (v72_data + v71_data);
          float v74_data = r0[7];
          float v75_data = r1[7];
          r1[7] = (v75_data + v74_data);
          float v77_data = r0[8];
          float v78_data = r1[8];
          r1[8] = (v78_data + v77_data);
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
              r3[v155_n1] = (v173_data + v158_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v180_i1 = 0; v180_i1 < 9; ++v180_i1) {
              int32_t v181_a = 0 + v180_i1;
              float v183_data = r3[v180_i1];
              int32_t v190_a = v8_lead + (v180_i1 * 32);
              s0[v190_a] = v183_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v8_lead < 16) {
            float v197_data = r4[0];
            float v198_data = ir5[0];
            ir5[0] = (v198_data + v197_data);
            float v200_data = r4[1];
            float v201_data = ir5[1];
            ir5[1] = (v201_data + v200_data);
            float v203_data = r4[2];
            float v204_data = ir5[2];
            ir5[2] = (v204_data + v203_data);
            float v206_data = r4[3];
            float v207_data = ir5[3];
            ir5[3] = (v207_data + v206_data);
            float v209_data = r4[4];
            float v210_data = ir5[4];
            ir5[4] = (v210_data + v209_data);
            float v212_data = r4[5];
            float v213_data = ir5[5];
            ir5[5] = (v213_data + v212_data);
            float v215_data = r4[6];
            float v216_data = ir5[6];
            ir5[6] = (v216_data + v215_data);
            float v218_data = r4[7];
            float v219_data = ir5[7];
            ir5[7] = (v219_data + v218_data);
            float v221_data = r4[8];
            float v222_data = ir5[8];
            ir5[8] = (v222_data + v221_data);
          }
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v228_n1 = 0; v228_n1 < 9; ++v228_n1) {
              int32_t v229_a = 0 + v228_n1;
              float v231_data = ir5[v228_n1];
              int32_t v237_a = v228_n1 * 32;
              int32_t v238_a = v8_lead + v237_a;
              float v246_data = s0[(v8_lead + v237_a)];
              r5[v228_n1] = (v246_data + v231_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v253_i1 = 0; v253_i1 < 9; ++v253_i1) {
              int32_t v254_a = 0 + v253_i1;
              float v256_data = r5[v253_i1];
              int32_t v263_a = v8_lead + (v253_i1 * 32);
              s0[v263_a] = v256_data;
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
          int32_t v278_a = v8_lead + 0;
          float v285_data = s0[v8_lead];
          float v286_data = s1[0];
          float v288_data = ir6[0];
          ir6[0] = (v288_data + (v285_data * v286_data));
          int32_t v295_a = v8_lead + 0;
          float v302_data = s0[v8_lead];
          float v303_data = s1[9];
          float v305_data = ir6[1];
          ir6[1] = (v305_data + (v302_data * v303_data));
          int32_t v312_a = v8_lead + 0;
          float v319_data = s0[v8_lead];
          float v320_data = s1[18];
          float v322_data = ir6[2];
          ir6[2] = (v322_data + (v319_data * v320_data));
          int32_t v329_a = v8_lead + 0;
          float v336_data = s0[v8_lead];
          float v337_data = s1[27];
          float v339_data = ir6[3];
          ir6[3] = (v339_data + (v336_data * v337_data));
          int32_t v346_a = v8_lead + 0;
          float v353_data = s0[v8_lead];
          float v354_data = s1[36];
          float v356_data = ir6[4];
          ir6[4] = (v356_data + (v353_data * v354_data));
          int32_t v363_a = v8_lead + 0;
          float v370_data = s0[v8_lead];
          float v371_data = s1[45];
          float v373_data = ir6[5];
          ir6[5] = (v373_data + (v370_data * v371_data));
          int32_t v380_a = v8_lead + 0;
          float v387_data = s0[v8_lead];
          float v388_data = s1[54];
          float v390_data = ir6[6];
          ir6[6] = (v390_data + (v387_data * v388_data));
          int32_t v397_a = v8_lead + 0;
          float v404_data = s0[v8_lead];
          float v405_data = s1[63];
          float v407_data = ir6[7];
          ir6[7] = (v407_data + (v404_data * v405_data));
          int32_t v414_a = v8_lead + 0;
          float v421_data = s0[v8_lead];
          float v422_data = s1[72];
          float v424_data = ir6[8];
          ir6[8] = (v424_data + (v421_data * v422_data));
          int32_t v434_a = v8_lead + 32;
          float v441_data = s0[(v8_lead + 32)];
          float v442_data = s1[1];
          float v444_data = ir6[0];
          ir6[0] = (v444_data + (v441_data * v442_data));
          int32_t v451_a = v8_lead + 32;
          float v458_data = s0[(v8_lead + 32)];
          float v459_data = s1[10];
          float v461_data = ir6[1];
          ir6[1] = (v461_data + (v458_data * v459_data));
          int32_t v468_a = v8_lead + 32;
          float v475_data = s0[(v8_lead + 32)];
          float v476_data = s1[19];
          float v478_data = ir6[2];
          ir6[2] = (v478_data + (v475_data * v476_data));
          int32_t v485_a = v8_lead + 32;
          float v492_data = s0[(v8_lead + 32)];
          float v493_data = s1[28];
          float v495_data = ir6[3];
          ir6[3] = (v495_data + (v492_data * v493_data));
          int32_t v502_a = v8_lead + 32;
          float v509_data = s0[(v8_lead + 32)];
          float v510_data = s1[37];
          float v512_data = ir6[4];
          ir6[4] = (v512_data + (v509_data * v510_data));
          int32_t v519_a = v8_lead + 32;
          float v526_data = s0[(v8_lead + 32)];
          float v527_data = s1[46];
          float v529_data = ir6[5];
          ir6[5] = (v529_data + (v526_data * v527_data));
          int32_t v536_a = v8_lead + 32;
          float v543_data = s0[(v8_lead + 32)];
          float v544_data = s1[55];
          float v546_data = ir6[6];
          ir6[6] = (v546_data + (v543_data * v544_data));
          int32_t v553_a = v8_lead + 32;
          float v560_data = s0[(v8_lead + 32)];
          float v561_data = s1[64];
          float v563_data = ir6[7];
          ir6[7] = (v563_data + (v560_data * v561_data));
          int32_t v570_a = v8_lead + 32;
          float v577_data = s0[(v8_lead + 32)];
          float v578_data = s1[73];
          float v580_data = ir6[8];
          ir6[8] = (v580_data + (v577_data * v578_data));
          int32_t v590_a = v8_lead + 64;
          float v597_data = s0[(v8_lead + 64)];
          float v598_data = s1[2];
          float v600_data = ir6[0];
          ir6[0] = (v600_data + (v597_data * v598_data));
          int32_t v607_a = v8_lead + 64;
          float v614_data = s0[(v8_lead + 64)];
          float v615_data = s1[11];
          float v617_data = ir6[1];
          ir6[1] = (v617_data + (v614_data * v615_data));
          int32_t v624_a = v8_lead + 64;
          float v631_data = s0[(v8_lead + 64)];
          float v632_data = s1[20];
          float v634_data = ir6[2];
          ir6[2] = (v634_data + (v631_data * v632_data));
          int32_t v641_a = v8_lead + 64;
          float v648_data = s0[(v8_lead + 64)];
          float v649_data = s1[29];
          float v651_data = ir6[3];
          ir6[3] = (v651_data + (v648_data * v649_data));
          int32_t v658_a = v8_lead + 64;
          float v665_data = s0[(v8_lead + 64)];
          float v666_data = s1[38];
          float v668_data = ir6[4];
          ir6[4] = (v668_data + (v665_data * v666_data));
          int32_t v675_a = v8_lead + 64;
          float v682_data = s0[(v8_lead + 64)];
          float v683_data = s1[47];
          float v685_data = ir6[5];
          ir6[5] = (v685_data + (v682_data * v683_data));
          int32_t v692_a = v8_lead + 64;
          float v699_data = s0[(v8_lead + 64)];
          float v700_data = s1[56];
          float v702_data = ir6[6];
          ir6[6] = (v702_data + (v699_data * v700_data));
          int32_t v709_a = v8_lead + 64;
          float v716_data = s0[(v8_lead + 64)];
          float v717_data = s1[65];
          float v719_data = ir6[7];
          ir6[7] = (v719_data + (v716_data * v717_data));
          int32_t v726_a = v8_lead + 64;
          float v733_data = s0[(v8_lead + 64)];
          float v734_data = s1[74];
          float v736_data = ir6[8];
          ir6[8] = (v736_data + (v733_data * v734_data));
          int32_t v746_a = v8_lead + 96;
          float v753_data = s0[(v8_lead + 96)];
          float v754_data = s1[3];
          float v756_data = ir6[0];
          ir6[0] = (v756_data + (v753_data * v754_data));
          int32_t v763_a = v8_lead + 96;
          float v770_data = s0[(v8_lead + 96)];
          float v771_data = s1[12];
          float v773_data = ir6[1];
          ir6[1] = (v773_data + (v770_data * v771_data));
          int32_t v780_a = v8_lead + 96;
          float v787_data = s0[(v8_lead + 96)];
          float v788_data = s1[21];
          float v790_data = ir6[2];
          ir6[2] = (v790_data + (v787_data * v788_data));
          int32_t v797_a = v8_lead + 96;
          float v804_data = s0[(v8_lead + 96)];
          float v805_data = s1[30];
          float v807_data = ir6[3];
          ir6[3] = (v807_data + (v804_data * v805_data));
          int32_t v814_a = v8_lead + 96;
          float v821_data = s0[(v8_lead + 96)];
          float v822_data = s1[39];
          float v824_data = ir6[4];
          ir6[4] = (v824_data + (v821_data * v822_data));
          int32_t v831_a = v8_lead + 96;
          float v838_data = s0[(v8_lead + 96)];
          float v839_data = s1[48];
          float v841_data = ir6[5];
          ir6[5] = (v841_data + (v838_data * v839_data));
          int32_t v848_a = v8_lead + 96;
          float v855_data = s0[(v8_lead + 96)];
          float v856_data = s1[57];
          float v858_data = ir6[6];
          ir6[6] = (v858_data + (v855_data * v856_data));
          int32_t v865_a = v8_lead + 96;
          float v872_data = s0[(v8_lead + 96)];
          float v873_data = s1[66];
          float v875_data = ir6[7];
          ir6[7] = (v875_data + (v872_data * v873_data));
          int32_t v882_a = v8_lead + 96;
          float v889_data = s0[(v8_lead + 96)];
          float v890_data = s1[75];
          float v892_data = ir6[8];
          ir6[8] = (v892_data + (v889_data * v890_data));
          int32_t v902_a = v8_lead + 128;
          float v909_data = s0[(v8_lead + 128)];
          float v910_data = s1[4];
          float v912_data = ir6[0];
          ir6[0] = (v912_data + (v909_data * v910_data));
          int32_t v919_a = v8_lead + 128;
          float v926_data = s0[(v8_lead + 128)];
          float v927_data = s1[13];
          float v929_data = ir6[1];
          ir6[1] = (v929_data + (v926_data * v927_data));
          int32_t v936_a = v8_lead + 128;
          float v943_data = s0[(v8_lead + 128)];
          float v944_data = s1[22];
          float v946_data = ir6[2];
          ir6[2] = (v946_data + (v943_data * v944_data));
          int32_t v953_a = v8_lead + 128;
          float v960_data = s0[(v8_lead + 128)];
          float v961_data = s1[31];
          float v963_data = ir6[3];
          ir6[3] = (v963_data + (v960_data * v961_data));
          int32_t v970_a = v8_lead + 128;
          float v977_data = s0[(v8_lead + 128)];
          float v978_data = s1[40];
          float v980_data = ir6[4];
          ir6[4] = (v980_data + (v977_data * v978_data));
          int32_t v987_a = v8_lead + 128;
          float v994_data = s0[(v8_lead + 128)];
          float v995_data = s1[49];
          float v997_data = ir6[5];
          ir6[5] = (v997_data + (v994_data * v995_data));
          int32_t v1004_a = v8_lead + 128;
          float v1011_data = s0[(v8_lead + 128)];
          float v1012_data = s1[58];
          float v1014_data = ir6[6];
          ir6[6] = (v1014_data + (v1011_data * v1012_data));
          int32_t v1021_a = v8_lead + 128;
          float v1028_data = s0[(v8_lead + 128)];
          float v1029_data = s1[67];
          float v1031_data = ir6[7];
          ir6[7] = (v1031_data + (v1028_data * v1029_data));
          int32_t v1038_a = v8_lead + 128;
          float v1045_data = s0[(v8_lead + 128)];
          float v1046_data = s1[76];
          float v1048_data = ir6[8];
          ir6[8] = (v1048_data + (v1045_data * v1046_data));
          int32_t v1058_a = v8_lead + 160;
          float v1065_data = s0[(v8_lead + 160)];
          float v1066_data = s1[5];
          float v1068_data = ir6[0];
          ir6[0] = (v1068_data + (v1065_data * v1066_data));
          int32_t v1075_a = v8_lead + 160;
          float v1082_data = s0[(v8_lead + 160)];
          float v1083_data = s1[14];
          float v1085_data = ir6[1];
          ir6[1] = (v1085_data + (v1082_data * v1083_data));
          int32_t v1092_a = v8_lead + 160;
          float v1099_data = s0[(v8_lead + 160)];
          float v1100_data = s1[23];
          float v1102_data = ir6[2];
          ir6[2] = (v1102_data + (v1099_data * v1100_data));
          int32_t v1109_a = v8_lead + 160;
          float v1116_data = s0[(v8_lead + 160)];
          float v1117_data = s1[32];
          float v1119_data = ir6[3];
          ir6[3] = (v1119_data + (v1116_data * v1117_data));
          int32_t v1126_a = v8_lead + 160;
          float v1133_data = s0[(v8_lead + 160)];
          float v1134_data = s1[41];
          float v1136_data = ir6[4];
          ir6[4] = (v1136_data + (v1133_data * v1134_data));
          int32_t v1143_a = v8_lead + 160;
          float v1150_data = s0[(v8_lead + 160)];
          float v1151_data = s1[50];
          float v1153_data = ir6[5];
          ir6[5] = (v1153_data + (v1150_data * v1151_data));
          int32_t v1160_a = v8_lead + 160;
          float v1167_data = s0[(v8_lead + 160)];
          float v1168_data = s1[59];
          float v1170_data = ir6[6];
          ir6[6] = (v1170_data + (v1167_data * v1168_data));
          int32_t v1177_a = v8_lead + 160;
          float v1184_data = s0[(v8_lead + 160)];
          float v1185_data = s1[68];
          float v1187_data = ir6[7];
          ir6[7] = (v1187_data + (v1184_data * v1185_data));
          int32_t v1194_a = v8_lead + 160;
          float v1201_data = s0[(v8_lead + 160)];
          float v1202_data = s1[77];
          float v1204_data = ir6[8];
          ir6[8] = (v1204_data + (v1201_data * v1202_data));
          int32_t v1214_a = v8_lead + 192;
          float v1221_data = s0[(v8_lead + 192)];
          float v1222_data = s1[6];
          float v1224_data = ir6[0];
          ir6[0] = (v1224_data + (v1221_data * v1222_data));
          int32_t v1231_a = v8_lead + 192;
          float v1238_data = s0[(v8_lead + 192)];
          float v1239_data = s1[15];
          float v1241_data = ir6[1];
          ir6[1] = (v1241_data + (v1238_data * v1239_data));
          int32_t v1248_a = v8_lead + 192;
          float v1255_data = s0[(v8_lead + 192)];
          float v1256_data = s1[24];
          float v1258_data = ir6[2];
          ir6[2] = (v1258_data + (v1255_data * v1256_data));
          int32_t v1265_a = v8_lead + 192;
          float v1272_data = s0[(v8_lead + 192)];
          float v1273_data = s1[33];
          float v1275_data = ir6[3];
          ir6[3] = (v1275_data + (v1272_data * v1273_data));
          int32_t v1282_a = v8_lead + 192;
          float v1289_data = s0[(v8_lead + 192)];
          float v1290_data = s1[42];
          float v1292_data = ir6[4];
          ir6[4] = (v1292_data + (v1289_data * v1290_data));
          int32_t v1299_a = v8_lead + 192;
          float v1306_data = s0[(v8_lead + 192)];
          float v1307_data = s1[51];
          float v1309_data = ir6[5];
          ir6[5] = (v1309_data + (v1306_data * v1307_data));
          int32_t v1316_a = v8_lead + 192;
          float v1323_data = s0[(v8_lead + 192)];
          float v1324_data = s1[60];
          float v1326_data = ir6[6];
          ir6[6] = (v1326_data + (v1323_data * v1324_data));
          int32_t v1333_a = v8_lead + 192;
          float v1340_data = s0[(v8_lead + 192)];
          float v1341_data = s1[69];
          float v1343_data = ir6[7];
          ir6[7] = (v1343_data + (v1340_data * v1341_data));
          int32_t v1350_a = v8_lead + 192;
          float v1357_data = s0[(v8_lead + 192)];
          float v1358_data = s1[78];
          float v1360_data = ir6[8];
          ir6[8] = (v1360_data + (v1357_data * v1358_data));
          int32_t v1370_a = v8_lead + 224;
          float v1377_data = s0[(v8_lead + 224)];
          float v1378_data = s1[7];
          float v1380_data = ir6[0];
          ir6[0] = (v1380_data + (v1377_data * v1378_data));
          int32_t v1387_a = v8_lead + 224;
          float v1394_data = s0[(v8_lead + 224)];
          float v1395_data = s1[16];
          float v1397_data = ir6[1];
          ir6[1] = (v1397_data + (v1394_data * v1395_data));
          int32_t v1404_a = v8_lead + 224;
          float v1411_data = s0[(v8_lead + 224)];
          float v1412_data = s1[25];
          float v1414_data = ir6[2];
          ir6[2] = (v1414_data + (v1411_data * v1412_data));
          int32_t v1421_a = v8_lead + 224;
          float v1428_data = s0[(v8_lead + 224)];
          float v1429_data = s1[34];
          float v1431_data = ir6[3];
          ir6[3] = (v1431_data + (v1428_data * v1429_data));
          int32_t v1438_a = v8_lead + 224;
          float v1445_data = s0[(v8_lead + 224)];
          float v1446_data = s1[43];
          float v1448_data = ir6[4];
          ir6[4] = (v1448_data + (v1445_data * v1446_data));
          int32_t v1455_a = v8_lead + 224;
          float v1462_data = s0[(v8_lead + 224)];
          float v1463_data = s1[52];
          float v1465_data = ir6[5];
          ir6[5] = (v1465_data + (v1462_data * v1463_data));
          int32_t v1472_a = v8_lead + 224;
          float v1479_data = s0[(v8_lead + 224)];
          float v1480_data = s1[61];
          float v1482_data = ir6[6];
          ir6[6] = (v1482_data + (v1479_data * v1480_data));
          int32_t v1489_a = v8_lead + 224;
          float v1496_data = s0[(v8_lead + 224)];
          float v1497_data = s1[70];
          float v1499_data = ir6[7];
          ir6[7] = (v1499_data + (v1496_data * v1497_data));
          int32_t v1506_a = v8_lead + 224;
          float v1513_data = s0[(v8_lead + 224)];
          float v1514_data = s1[79];
          float v1516_data = ir6[8];
          ir6[8] = (v1516_data + (v1513_data * v1514_data));
          int32_t v1526_a = v8_lead + 256;
          float v1533_data = s0[(v8_lead + 256)];
          float v1534_data = s1[8];
          float v1536_data = ir6[0];
          ir6[0] = (v1536_data + (v1533_data * v1534_data));
          int32_t v1543_a = v8_lead + 256;
          float v1550_data = s0[(v8_lead + 256)];
          float v1551_data = s1[17];
          float v1553_data = ir6[1];
          ir6[1] = (v1553_data + (v1550_data * v1551_data));
          int32_t v1560_a = v8_lead + 256;
          float v1567_data = s0[(v8_lead + 256)];
          float v1568_data = s1[26];
          float v1570_data = ir6[2];
          ir6[2] = (v1570_data + (v1567_data * v1568_data));
          int32_t v1577_a = v8_lead + 256;
          float v1584_data = s0[(v8_lead + 256)];
          float v1585_data = s1[35];
          float v1587_data = ir6[3];
          ir6[3] = (v1587_data + (v1584_data * v1585_data));
          int32_t v1594_a = v8_lead + 256;
          float v1601_data = s0[(v8_lead + 256)];
          float v1602_data = s1[44];
          float v1604_data = ir6[4];
          ir6[4] = (v1604_data + (v1601_data * v1602_data));
          int32_t v1611_a = v8_lead + 256;
          float v1618_data = s0[(v8_lead + 256)];
          float v1619_data = s1[53];
          float v1621_data = ir6[5];
          ir6[5] = (v1621_data + (v1618_data * v1619_data));
          int32_t v1628_a = v8_lead + 256;
          float v1635_data = s0[(v8_lead + 256)];
          float v1636_data = s1[62];
          float v1638_data = ir6[6];
          ir6[6] = (v1638_data + (v1635_data * v1636_data));
          int32_t v1645_a = v8_lead + 256;
          float v1652_data = s0[(v8_lead + 256)];
          float v1653_data = s1[71];
          float v1655_data = ir6[7];
          ir6[7] = (v1655_data + (v1652_data * v1653_data));
          int32_t v1662_a = v8_lead + 256;
          float v1669_data = s0[(v8_lead + 256)];
          float v1670_data = s1[80];
          float v1672_data = ir6[8];
          ir6[8] = (v1672_data + (v1669_data * v1670_data));
          #pragma unroll
          for (int32_t v1677_n0 = 0; v1677_n0 < 1; ++v1677_n0) {
            #pragma unroll
            for (int32_t v1678_n1 = 0; v1678_n1 < 9; ++v1678_n1) {
              int32_t v1679_a = v1677_n0 + v1678_n1;
              int32_t v1680_a = v1677_n0 + v1678_n1;
              float v1681_data = ir6[v1680_a];
              r6[v1680_a] = v1681_data;
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
              glb_m3[(v1695_lead + (v1687_i1 * 32))] = v1690_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

