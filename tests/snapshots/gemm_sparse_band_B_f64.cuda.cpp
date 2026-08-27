// === base name ===
kernel_417e1ddcc4

// === header ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_417e1ddcc4, block.x * block.y * block.z, 1024 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_417e1ddcc4, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_417e1ddcc4<<<grid,block,1024 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[48];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 16;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v14_a = v8_i1 * 16;
              int32_t v15_a = v13_lead + v14_a;
              double v23_data = __ldcg(&glb_m1[(v20_lead + v14_a)]);
              int32_t v24_a = v7_i0 + v8_i1;
              r0[v24_a] = v23_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<8>(8), pipeline);
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], cuda::aligned_size_t<8>(8), pipeline);
          if (threadIdx.x < 14) {
            cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<8>(8), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          double r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v31_data = r0[0];
          double v32_data = s0[0];
          double v34_data = ir1[0];
          ir1[0] = (v34_data + (v31_data * v32_data));
          double v37_data = s0[2];
          double v39_data = ir1[1];
          ir1[1] = (v39_data + (v31_data * v37_data));
          double v58_data = r0[1];
          double v59_data = s0[1];
          double v61_data = ir1[0];
          ir1[0] = (v61_data + (v58_data * v59_data));
          double v64_data = s0[3];
          double v66_data = ir1[1];
          ir1[1] = (v66_data + (v58_data * v64_data));
          double v69_data = s0[5];
          double v71_data = ir1[2];
          ir1[2] = (v71_data + (v58_data * v69_data));
          double v89_data = r0[2];
          double v91_data = s0[4];
          double v93_data = ir1[1];
          ir1[1] = (v93_data + (v89_data * v91_data));
          double v96_data = s0[6];
          double v98_data = ir1[2];
          ir1[2] = (v98_data + (v89_data * v96_data));
          double v101_data = s0[8];
          double v103_data = ir1[3];
          ir1[3] = (v103_data + (v89_data * v101_data));
          double v120_data = r0[3];
          double v123_data = s0[7];
          double v125_data = ir1[2];
          ir1[2] = (v125_data + (v120_data * v123_data));
          double v128_data = s0[9];
          double v130_data = ir1[3];
          ir1[3] = (v130_data + (v120_data * v128_data));
          double v133_data = s0[11];
          double v135_data = ir1[4];
          ir1[4] = (v135_data + (v120_data * v133_data));
          double v151_data = r0[4];
          double v155_data = s0[10];
          double v157_data = ir1[3];
          ir1[3] = (v157_data + (v151_data * v155_data));
          double v160_data = s0[12];
          double v162_data = ir1[4];
          ir1[4] = (v162_data + (v151_data * v160_data));
          double v165_data = s0[14];
          double v167_data = ir1[5];
          ir1[5] = (v167_data + (v151_data * v165_data));
          double v182_data = r0[5];
          double v187_data = s0[13];
          double v189_data = ir1[4];
          ir1[4] = (v189_data + (v182_data * v187_data));
          double v192_data = s0[15];
          double v194_data = ir1[5];
          ir1[5] = (v194_data + (v182_data * v192_data));
          double v197_data = s0[17];
          double v199_data = ir1[6];
          ir1[6] = (v199_data + (v182_data * v197_data));
          double v213_data = r0[6];
          double v219_data = s0[16];
          double v221_data = ir1[5];
          ir1[5] = (v221_data + (v213_data * v219_data));
          double v224_data = s0[18];
          double v226_data = ir1[6];
          ir1[6] = (v226_data + (v213_data * v224_data));
          double v229_data = s0[20];
          double v231_data = ir1[7];
          ir1[7] = (v231_data + (v213_data * v229_data));
          double v244_data = r0[7];
          double v251_data = s0[19];
          double v253_data = ir1[6];
          ir1[6] = (v253_data + (v244_data * v251_data));
          double v256_data = s0[21];
          double v258_data = ir1[7];
          ir1[7] = (v258_data + (v244_data * v256_data));
          double v261_data = s0[23];
          double v263_data = ir1[8];
          ir1[8] = (v263_data + (v244_data * v261_data));
          double v275_data = r0[8];
          double v283_data = s0[22];
          double v285_data = ir1[7];
          ir1[7] = (v285_data + (v275_data * v283_data));
          double v288_data = s0[24];
          double v290_data = ir1[8];
          ir1[8] = (v290_data + (v275_data * v288_data));
          double v293_data = s0[26];
          double v295_data = ir1[9];
          ir1[9] = (v295_data + (v275_data * v293_data));
          double v306_data = r0[9];
          double v315_data = s0[25];
          double v317_data = ir1[8];
          ir1[8] = (v317_data + (v306_data * v315_data));
          double v320_data = s0[27];
          double v322_data = ir1[9];
          ir1[9] = (v322_data + (v306_data * v320_data));
          double v325_data = s0[29];
          double v327_data = ir1[10];
          ir1[10] = (v327_data + (v306_data * v325_data));
          double v337_data = r0[10];
          double v347_data = s0[28];
          double v349_data = ir1[9];
          ir1[9] = (v349_data + (v337_data * v347_data));
          double v352_data = s0[30];
          double v354_data = ir1[10];
          ir1[10] = (v354_data + (v337_data * v352_data));
          double v357_data = s0[32];
          double v359_data = ir1[11];
          ir1[11] = (v359_data + (v337_data * v357_data));
          double v368_data = r0[11];
          double v379_data = s0[31];
          double v381_data = ir1[10];
          ir1[10] = (v381_data + (v368_data * v379_data));
          double v384_data = s0[33];
          double v386_data = ir1[11];
          ir1[11] = (v386_data + (v368_data * v384_data));
          double v389_data = s0[35];
          double v391_data = ir1[12];
          ir1[12] = (v391_data + (v368_data * v389_data));
          double v399_data = r0[12];
          double v411_data = s0[34];
          double v413_data = ir1[11];
          ir1[11] = (v413_data + (v399_data * v411_data));
          double v416_data = s0[36];
          double v418_data = ir1[12];
          ir1[12] = (v418_data + (v399_data * v416_data));
          double v421_data = s0[38];
          double v423_data = ir1[13];
          ir1[13] = (v423_data + (v399_data * v421_data));
          double v430_data = r0[13];
          double v443_data = s0[37];
          double v445_data = ir1[12];
          ir1[12] = (v445_data + (v430_data * v443_data));
          double v448_data = s0[39];
          double v450_data = ir1[13];
          ir1[13] = (v450_data + (v430_data * v448_data));
          double v453_data = s0[41];
          double v455_data = ir1[14];
          ir1[14] = (v455_data + (v430_data * v453_data));
          double v461_data = r0[14];
          double v475_data = s0[40];
          double v477_data = ir1[13];
          ir1[13] = (v477_data + (v461_data * v475_data));
          double v480_data = s0[42];
          double v482_data = ir1[14];
          ir1[14] = (v482_data + (v461_data * v480_data));
          double v485_data = s0[44];
          double v487_data = ir1[15];
          ir1[15] = (v487_data + (v461_data * v485_data));
          double v492_data = r0[15];
          double v507_data = s0[43];
          double v509_data = ir1[14];
          ir1[14] = (v509_data + (v492_data * v507_data));
          double v512_data = s0[45];
          double v514_data = ir1[15];
          ir1[15] = (v514_data + (v492_data * v512_data));
          #pragma unroll
          for (int32_t v519_n0 = 0; v519_n0 < 1; ++v519_n0) {
            #pragma unroll
            for (int32_t v520_n1 = 0; v520_n1 < 16; ++v520_n1) {
              int32_t v521_a = v519_n0 + v520_n1;
              int32_t v522_a = v519_n0 + v520_n1;
              double v523_data = ir1[v522_a];
              int32_t v524_a = v519_n0 + v520_n1;
              r1[v522_a] = v523_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v529_i0 = 0; v529_i0 < 1; ++v529_i0) {
            int32_t v538_lead = v6_lead + (v529_i0 * 16);
            #pragma unroll
            for (int32_t v530_i1 = 0; v530_i1 < 16; ++v530_i1) {
              int32_t v531_a = v529_i0 + v530_i1;
              double v533_data = r1[(v529_i0 + v530_i1)];
              int32_t v540_a = v538_lead + (v530_i1 * 16);
              glb_m0[v540_a] = v533_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

