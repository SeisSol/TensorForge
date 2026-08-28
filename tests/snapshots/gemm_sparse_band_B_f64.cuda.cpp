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
          alignas(16) double r0[16]{};
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
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 8);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 8);
          __pipeline_commit();
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 8);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          alignas(16) double r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v34_data = r0[0];
          double v35_data = s0[0];
          double v37_data = ir1[0];
          ir1[0] = (v37_data + (v34_data * v35_data));
          double v40_data = s0[2];
          double v42_data = ir1[1];
          ir1[1] = (v42_data + (v34_data * v40_data));
          double v61_data = r0[1];
          double v62_data = s0[1];
          double v64_data = ir1[0];
          ir1[0] = (v64_data + (v61_data * v62_data));
          double v67_data = s0[3];
          double v69_data = ir1[1];
          ir1[1] = (v69_data + (v61_data * v67_data));
          double v72_data = s0[5];
          double v74_data = ir1[2];
          ir1[2] = (v74_data + (v61_data * v72_data));
          double v92_data = r0[2];
          double v94_data = s0[4];
          double v96_data = ir1[1];
          ir1[1] = (v96_data + (v92_data * v94_data));
          double v99_data = s0[6];
          double v101_data = ir1[2];
          ir1[2] = (v101_data + (v92_data * v99_data));
          double v104_data = s0[8];
          double v106_data = ir1[3];
          ir1[3] = (v106_data + (v92_data * v104_data));
          double v123_data = r0[3];
          double v126_data = s0[7];
          double v128_data = ir1[2];
          ir1[2] = (v128_data + (v123_data * v126_data));
          double v131_data = s0[9];
          double v133_data = ir1[3];
          ir1[3] = (v133_data + (v123_data * v131_data));
          double v136_data = s0[11];
          double v138_data = ir1[4];
          ir1[4] = (v138_data + (v123_data * v136_data));
          double v154_data = r0[4];
          double v158_data = s0[10];
          double v160_data = ir1[3];
          ir1[3] = (v160_data + (v154_data * v158_data));
          double v163_data = s0[12];
          double v165_data = ir1[4];
          ir1[4] = (v165_data + (v154_data * v163_data));
          double v168_data = s0[14];
          double v170_data = ir1[5];
          ir1[5] = (v170_data + (v154_data * v168_data));
          double v185_data = r0[5];
          double v190_data = s0[13];
          double v192_data = ir1[4];
          ir1[4] = (v192_data + (v185_data * v190_data));
          double v195_data = s0[15];
          double v197_data = ir1[5];
          ir1[5] = (v197_data + (v185_data * v195_data));
          double v200_data = s0[17];
          double v202_data = ir1[6];
          ir1[6] = (v202_data + (v185_data * v200_data));
          double v216_data = r0[6];
          double v222_data = s0[16];
          double v224_data = ir1[5];
          ir1[5] = (v224_data + (v216_data * v222_data));
          double v227_data = s0[18];
          double v229_data = ir1[6];
          ir1[6] = (v229_data + (v216_data * v227_data));
          double v232_data = s0[20];
          double v234_data = ir1[7];
          ir1[7] = (v234_data + (v216_data * v232_data));
          double v247_data = r0[7];
          double v254_data = s0[19];
          double v256_data = ir1[6];
          ir1[6] = (v256_data + (v247_data * v254_data));
          double v259_data = s0[21];
          double v261_data = ir1[7];
          ir1[7] = (v261_data + (v247_data * v259_data));
          double v264_data = s0[23];
          double v266_data = ir1[8];
          ir1[8] = (v266_data + (v247_data * v264_data));
          double v278_data = r0[8];
          double v286_data = s0[22];
          double v288_data = ir1[7];
          ir1[7] = (v288_data + (v278_data * v286_data));
          double v291_data = s0[24];
          double v293_data = ir1[8];
          ir1[8] = (v293_data + (v278_data * v291_data));
          double v296_data = s0[26];
          double v298_data = ir1[9];
          ir1[9] = (v298_data + (v278_data * v296_data));
          double v309_data = r0[9];
          double v318_data = s0[25];
          double v320_data = ir1[8];
          ir1[8] = (v320_data + (v309_data * v318_data));
          double v323_data = s0[27];
          double v325_data = ir1[9];
          ir1[9] = (v325_data + (v309_data * v323_data));
          double v328_data = s0[29];
          double v330_data = ir1[10];
          ir1[10] = (v330_data + (v309_data * v328_data));
          double v340_data = r0[10];
          double v350_data = s0[28];
          double v352_data = ir1[9];
          ir1[9] = (v352_data + (v340_data * v350_data));
          double v355_data = s0[30];
          double v357_data = ir1[10];
          ir1[10] = (v357_data + (v340_data * v355_data));
          double v360_data = s0[32];
          double v362_data = ir1[11];
          ir1[11] = (v362_data + (v340_data * v360_data));
          double v371_data = r0[11];
          double v382_data = s0[31];
          double v384_data = ir1[10];
          ir1[10] = (v384_data + (v371_data * v382_data));
          double v387_data = s0[33];
          double v389_data = ir1[11];
          ir1[11] = (v389_data + (v371_data * v387_data));
          double v392_data = s0[35];
          double v394_data = ir1[12];
          ir1[12] = (v394_data + (v371_data * v392_data));
          double v402_data = r0[12];
          double v414_data = s0[34];
          double v416_data = ir1[11];
          ir1[11] = (v416_data + (v402_data * v414_data));
          double v419_data = s0[36];
          double v421_data = ir1[12];
          ir1[12] = (v421_data + (v402_data * v419_data));
          double v424_data = s0[38];
          double v426_data = ir1[13];
          ir1[13] = (v426_data + (v402_data * v424_data));
          double v433_data = r0[13];
          double v446_data = s0[37];
          double v448_data = ir1[12];
          ir1[12] = (v448_data + (v433_data * v446_data));
          double v451_data = s0[39];
          double v453_data = ir1[13];
          ir1[13] = (v453_data + (v433_data * v451_data));
          double v456_data = s0[41];
          double v458_data = ir1[14];
          ir1[14] = (v458_data + (v433_data * v456_data));
          double v464_data = r0[14];
          double v478_data = s0[40];
          double v480_data = ir1[13];
          ir1[13] = (v480_data + (v464_data * v478_data));
          double v483_data = s0[42];
          double v485_data = ir1[14];
          ir1[14] = (v485_data + (v464_data * v483_data));
          double v488_data = s0[44];
          double v490_data = ir1[15];
          ir1[15] = (v490_data + (v464_data * v488_data));
          double v495_data = r0[15];
          double v510_data = s0[43];
          double v512_data = ir1[14];
          ir1[14] = (v512_data + (v495_data * v510_data));
          double v515_data = s0[45];
          double v517_data = ir1[15];
          ir1[15] = (v517_data + (v495_data * v515_data));
          #pragma unroll
          for (int32_t v522_n0 = 0; v522_n0 < 1; ++v522_n0) {
            #pragma unroll
            for (int32_t v523_n1 = 0; v523_n1 < 16; ++v523_n1) {
              int32_t v524_a = v522_n0 + v523_n1;
              int32_t v525_a = v522_n0 + v523_n1;
              double v526_data = ir1[v525_a];
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
              double v535_data = r1[(v531_i0 + v532_i1)];
              glb_m0[(v540_lead + (v532_i1 * 16))] = v535_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

