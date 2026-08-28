// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_30948bd44e, block.x * block.y * block.z, 1280 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_30948bd44e, cudaFuncAttributeMaxDynamicSharedMemorySize, 1280 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_30948bd44e<<<grid,block,1280 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 16);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 16))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_commit();
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v31_data = r0[0];
          float v32_data = s0[0];
          float v34_data = ir1[0];
          ir1[0] = (v34_data + (v31_data * v32_data));
          float v37_data = s0[2];
          float v39_data = ir1[1];
          ir1[1] = (v39_data + (v31_data * v37_data));
          float v58_data = r0[1];
          float v59_data = s0[1];
          float v61_data = ir1[0];
          ir1[0] = (v61_data + (v58_data * v59_data));
          float v64_data = s0[3];
          float v66_data = ir1[1];
          ir1[1] = (v66_data + (v58_data * v64_data));
          float v69_data = s0[5];
          float v71_data = ir1[2];
          ir1[2] = (v71_data + (v58_data * v69_data));
          float v89_data = r0[2];
          float v91_data = s0[4];
          float v93_data = ir1[1];
          ir1[1] = (v93_data + (v89_data * v91_data));
          float v96_data = s0[6];
          float v98_data = ir1[2];
          ir1[2] = (v98_data + (v89_data * v96_data));
          float v101_data = s0[8];
          float v103_data = ir1[3];
          ir1[3] = (v103_data + (v89_data * v101_data));
          float v120_data = r0[3];
          float v123_data = s0[7];
          float v125_data = ir1[2];
          ir1[2] = (v125_data + (v120_data * v123_data));
          float v128_data = s0[9];
          float v130_data = ir1[3];
          ir1[3] = (v130_data + (v120_data * v128_data));
          float v133_data = s0[11];
          float v135_data = ir1[4];
          ir1[4] = (v135_data + (v120_data * v133_data));
          float v151_data = r0[4];
          float v155_data = s0[10];
          float v157_data = ir1[3];
          ir1[3] = (v157_data + (v151_data * v155_data));
          float v160_data = s0[12];
          float v162_data = ir1[4];
          ir1[4] = (v162_data + (v151_data * v160_data));
          float v165_data = s0[14];
          float v167_data = ir1[5];
          ir1[5] = (v167_data + (v151_data * v165_data));
          float v182_data = r0[5];
          float v187_data = s0[13];
          float v189_data = ir1[4];
          ir1[4] = (v189_data + (v182_data * v187_data));
          float v192_data = s0[15];
          float v194_data = ir1[5];
          ir1[5] = (v194_data + (v182_data * v192_data));
          float v197_data = s0[17];
          float v199_data = ir1[6];
          ir1[6] = (v199_data + (v182_data * v197_data));
          float v213_data = r0[6];
          float v219_data = s0[16];
          float v221_data = ir1[5];
          ir1[5] = (v221_data + (v213_data * v219_data));
          float v224_data = s0[18];
          float v226_data = ir1[6];
          ir1[6] = (v226_data + (v213_data * v224_data));
          float v229_data = s0[20];
          float v231_data = ir1[7];
          ir1[7] = (v231_data + (v213_data * v229_data));
          float v244_data = r0[7];
          float v251_data = s0[19];
          float v253_data = ir1[6];
          ir1[6] = (v253_data + (v244_data * v251_data));
          float v256_data = s0[21];
          float v258_data = ir1[7];
          ir1[7] = (v258_data + (v244_data * v256_data));
          float v261_data = s0[23];
          float v263_data = ir1[8];
          ir1[8] = (v263_data + (v244_data * v261_data));
          float v275_data = r0[8];
          float v283_data = s0[22];
          float v285_data = ir1[7];
          ir1[7] = (v285_data + (v275_data * v283_data));
          float v288_data = s0[24];
          float v290_data = ir1[8];
          ir1[8] = (v290_data + (v275_data * v288_data));
          float v293_data = s0[26];
          float v295_data = ir1[9];
          ir1[9] = (v295_data + (v275_data * v293_data));
          float v306_data = r0[9];
          float v315_data = s0[25];
          float v317_data = ir1[8];
          ir1[8] = (v317_data + (v306_data * v315_data));
          float v320_data = s0[27];
          float v322_data = ir1[9];
          ir1[9] = (v322_data + (v306_data * v320_data));
          float v325_data = s0[29];
          float v327_data = ir1[10];
          ir1[10] = (v327_data + (v306_data * v325_data));
          float v337_data = r0[10];
          float v347_data = s0[28];
          float v349_data = ir1[9];
          ir1[9] = (v349_data + (v337_data * v347_data));
          float v352_data = s0[30];
          float v354_data = ir1[10];
          ir1[10] = (v354_data + (v337_data * v352_data));
          float v357_data = s0[32];
          float v359_data = ir1[11];
          ir1[11] = (v359_data + (v337_data * v357_data));
          float v368_data = r0[11];
          float v379_data = s0[31];
          float v381_data = ir1[10];
          ir1[10] = (v381_data + (v368_data * v379_data));
          float v384_data = s0[33];
          float v386_data = ir1[11];
          ir1[11] = (v386_data + (v368_data * v384_data));
          float v389_data = s0[35];
          float v391_data = ir1[12];
          ir1[12] = (v391_data + (v368_data * v389_data));
          float v399_data = r0[12];
          float v411_data = s0[34];
          float v413_data = ir1[11];
          ir1[11] = (v413_data + (v399_data * v411_data));
          float v416_data = s0[36];
          float v418_data = ir1[12];
          ir1[12] = (v418_data + (v399_data * v416_data));
          float v421_data = s0[38];
          float v423_data = ir1[13];
          ir1[13] = (v423_data + (v399_data * v421_data));
          float v430_data = r0[13];
          float v443_data = s0[37];
          float v445_data = ir1[12];
          ir1[12] = (v445_data + (v430_data * v443_data));
          float v448_data = s0[39];
          float v450_data = ir1[13];
          ir1[13] = (v450_data + (v430_data * v448_data));
          float v453_data = s0[41];
          float v455_data = ir1[14];
          ir1[14] = (v455_data + (v430_data * v453_data));
          float v461_data = r0[14];
          float v475_data = s0[40];
          float v477_data = ir1[13];
          ir1[13] = (v477_data + (v461_data * v475_data));
          float v480_data = s0[42];
          float v482_data = ir1[14];
          ir1[14] = (v482_data + (v461_data * v480_data));
          float v485_data = s0[44];
          float v487_data = ir1[15];
          ir1[15] = (v487_data + (v461_data * v485_data));
          float v492_data = r0[15];
          float v507_data = s0[43];
          float v509_data = ir1[14];
          ir1[14] = (v509_data + (v492_data * v507_data));
          float v512_data = s0[45];
          float v514_data = ir1[15];
          ir1[15] = (v514_data + (v492_data * v512_data));
          #pragma unroll
          for (int32_t v519_n0 = 0; v519_n0 < 1; ++v519_n0) {
            #pragma unroll
            for (int32_t v520_n1 = 0; v520_n1 < 16; ++v520_n1) {
              int32_t v521_a = v519_n0 + v520_n1;
              float v522_data = ir1[v521_a];
              r1[v521_a] = v522_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v527_i0 = 0; v527_i0 < 1; ++v527_i0) {
            int32_t v535_lead = v10_lead + (v527_i0 * 16);
            #pragma unroll
            for (int32_t v528_i1 = 0; v528_i1 < 16; ++v528_i1) {
              float v530_data = r1[(v527_i0 + v528_i1)];
              glb_m0[(v535_lead + (v528_i1 * 16))] = v530_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

