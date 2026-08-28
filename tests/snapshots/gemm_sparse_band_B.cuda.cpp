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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
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
              float v23_data = __ldcg(&glb_m1[(v20_lead + v14_a)]);
              int32_t v24_a = v7_i0 + v8_i1;
              r0[v24_a] = v23_data;
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
          float v34_data = r0[0];
          float v35_data = s0[0];
          float v37_data = ir1[0];
          ir1[0] = (v37_data + (v34_data * v35_data));
          float v40_data = s0[2];
          float v42_data = ir1[1];
          ir1[1] = (v42_data + (v34_data * v40_data));
          float v61_data = r0[1];
          float v62_data = s0[1];
          float v64_data = ir1[0];
          ir1[0] = (v64_data + (v61_data * v62_data));
          float v67_data = s0[3];
          float v69_data = ir1[1];
          ir1[1] = (v69_data + (v61_data * v67_data));
          float v72_data = s0[5];
          float v74_data = ir1[2];
          ir1[2] = (v74_data + (v61_data * v72_data));
          float v92_data = r0[2];
          float v94_data = s0[4];
          float v96_data = ir1[1];
          ir1[1] = (v96_data + (v92_data * v94_data));
          float v99_data = s0[6];
          float v101_data = ir1[2];
          ir1[2] = (v101_data + (v92_data * v99_data));
          float v104_data = s0[8];
          float v106_data = ir1[3];
          ir1[3] = (v106_data + (v92_data * v104_data));
          float v123_data = r0[3];
          float v126_data = s0[7];
          float v128_data = ir1[2];
          ir1[2] = (v128_data + (v123_data * v126_data));
          float v131_data = s0[9];
          float v133_data = ir1[3];
          ir1[3] = (v133_data + (v123_data * v131_data));
          float v136_data = s0[11];
          float v138_data = ir1[4];
          ir1[4] = (v138_data + (v123_data * v136_data));
          float v154_data = r0[4];
          float v158_data = s0[10];
          float v160_data = ir1[3];
          ir1[3] = (v160_data + (v154_data * v158_data));
          float v163_data = s0[12];
          float v165_data = ir1[4];
          ir1[4] = (v165_data + (v154_data * v163_data));
          float v168_data = s0[14];
          float v170_data = ir1[5];
          ir1[5] = (v170_data + (v154_data * v168_data));
          float v185_data = r0[5];
          float v190_data = s0[13];
          float v192_data = ir1[4];
          ir1[4] = (v192_data + (v185_data * v190_data));
          float v195_data = s0[15];
          float v197_data = ir1[5];
          ir1[5] = (v197_data + (v185_data * v195_data));
          float v200_data = s0[17];
          float v202_data = ir1[6];
          ir1[6] = (v202_data + (v185_data * v200_data));
          float v216_data = r0[6];
          float v222_data = s0[16];
          float v224_data = ir1[5];
          ir1[5] = (v224_data + (v216_data * v222_data));
          float v227_data = s0[18];
          float v229_data = ir1[6];
          ir1[6] = (v229_data + (v216_data * v227_data));
          float v232_data = s0[20];
          float v234_data = ir1[7];
          ir1[7] = (v234_data + (v216_data * v232_data));
          float v247_data = r0[7];
          float v254_data = s0[19];
          float v256_data = ir1[6];
          ir1[6] = (v256_data + (v247_data * v254_data));
          float v259_data = s0[21];
          float v261_data = ir1[7];
          ir1[7] = (v261_data + (v247_data * v259_data));
          float v264_data = s0[23];
          float v266_data = ir1[8];
          ir1[8] = (v266_data + (v247_data * v264_data));
          float v278_data = r0[8];
          float v286_data = s0[22];
          float v288_data = ir1[7];
          ir1[7] = (v288_data + (v278_data * v286_data));
          float v291_data = s0[24];
          float v293_data = ir1[8];
          ir1[8] = (v293_data + (v278_data * v291_data));
          float v296_data = s0[26];
          float v298_data = ir1[9];
          ir1[9] = (v298_data + (v278_data * v296_data));
          float v309_data = r0[9];
          float v318_data = s0[25];
          float v320_data = ir1[8];
          ir1[8] = (v320_data + (v309_data * v318_data));
          float v323_data = s0[27];
          float v325_data = ir1[9];
          ir1[9] = (v325_data + (v309_data * v323_data));
          float v328_data = s0[29];
          float v330_data = ir1[10];
          ir1[10] = (v330_data + (v309_data * v328_data));
          float v340_data = r0[10];
          float v350_data = s0[28];
          float v352_data = ir1[9];
          ir1[9] = (v352_data + (v340_data * v350_data));
          float v355_data = s0[30];
          float v357_data = ir1[10];
          ir1[10] = (v357_data + (v340_data * v355_data));
          float v360_data = s0[32];
          float v362_data = ir1[11];
          ir1[11] = (v362_data + (v340_data * v360_data));
          float v371_data = r0[11];
          float v382_data = s0[31];
          float v384_data = ir1[10];
          ir1[10] = (v384_data + (v371_data * v382_data));
          float v387_data = s0[33];
          float v389_data = ir1[11];
          ir1[11] = (v389_data + (v371_data * v387_data));
          float v392_data = s0[35];
          float v394_data = ir1[12];
          ir1[12] = (v394_data + (v371_data * v392_data));
          float v402_data = r0[12];
          float v414_data = s0[34];
          float v416_data = ir1[11];
          ir1[11] = (v416_data + (v402_data * v414_data));
          float v419_data = s0[36];
          float v421_data = ir1[12];
          ir1[12] = (v421_data + (v402_data * v419_data));
          float v424_data = s0[38];
          float v426_data = ir1[13];
          ir1[13] = (v426_data + (v402_data * v424_data));
          float v433_data = r0[13];
          float v446_data = s0[37];
          float v448_data = ir1[12];
          ir1[12] = (v448_data + (v433_data * v446_data));
          float v451_data = s0[39];
          float v453_data = ir1[13];
          ir1[13] = (v453_data + (v433_data * v451_data));
          float v456_data = s0[41];
          float v458_data = ir1[14];
          ir1[14] = (v458_data + (v433_data * v456_data));
          float v464_data = r0[14];
          float v478_data = s0[40];
          float v480_data = ir1[13];
          ir1[13] = (v480_data + (v464_data * v478_data));
          float v483_data = s0[42];
          float v485_data = ir1[14];
          ir1[14] = (v485_data + (v464_data * v483_data));
          float v488_data = s0[44];
          float v490_data = ir1[15];
          ir1[15] = (v490_data + (v464_data * v488_data));
          float v495_data = r0[15];
          float v510_data = s0[43];
          float v512_data = ir1[14];
          ir1[14] = (v512_data + (v495_data * v510_data));
          float v515_data = s0[45];
          float v517_data = ir1[15];
          ir1[15] = (v517_data + (v495_data * v515_data));
          #pragma unroll
          for (int32_t v522_n0 = 0; v522_n0 < 1; ++v522_n0) {
            #pragma unroll
            for (int32_t v523_n1 = 0; v523_n1 < 16; ++v523_n1) {
              int32_t v524_a = v522_n0 + v523_n1;
              int32_t v525_a = v522_n0 + v523_n1;
              float v526_data = ir1[v525_a];
              r1[v525_a] = v526_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v531_i0 = 0; v531_i0 < 1; ++v531_i0) {
            int32_t v540_lead = v6_lead + (v531_i0 * 16);
            #pragma unroll
            for (int32_t v532_i1 = 0; v532_i1 < 16; ++v532_i1) {
              int32_t v533_a = v531_i0 + v532_i1;
              float v535_data = r1[(v531_i0 + v532_i1)];
              glb_m0[(v540_lead + (v532_i1 * 16))] = v535_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

