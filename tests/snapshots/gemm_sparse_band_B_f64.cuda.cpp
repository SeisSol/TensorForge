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
          int32_t v3_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 16;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 16;
              int32_t v12_a = v10_lead + v11_a;
              double v20_data = __ldcg(&glb_m1[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
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
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            double ir1[16]{};
            double v26_data = r0[0];
            double v27_data = s0[0];
            double v29_data = ir1[0];
            ir1[0] = (v29_data + (v26_data * v27_data));
            double v32_data = s0[2];
            double v34_data = ir1[1];
            ir1[1] = (v34_data + (v26_data * v32_data));
            double v53_data = r0[1];
            double v54_data = s0[1];
            double v56_data = ir1[0];
            ir1[0] = (v56_data + (v53_data * v54_data));
            double v59_data = s0[3];
            double v61_data = ir1[1];
            ir1[1] = (v61_data + (v53_data * v59_data));
            double v64_data = s0[5];
            double v66_data = ir1[2];
            ir1[2] = (v66_data + (v53_data * v64_data));
            double v84_data = r0[2];
            double v86_data = s0[4];
            double v88_data = ir1[1];
            ir1[1] = (v88_data + (v84_data * v86_data));
            double v91_data = s0[6];
            double v93_data = ir1[2];
            ir1[2] = (v93_data + (v84_data * v91_data));
            double v96_data = s0[8];
            double v98_data = ir1[3];
            ir1[3] = (v98_data + (v84_data * v96_data));
            double v115_data = r0[3];
            double v118_data = s0[7];
            double v120_data = ir1[2];
            ir1[2] = (v120_data + (v115_data * v118_data));
            double v123_data = s0[9];
            double v125_data = ir1[3];
            ir1[3] = (v125_data + (v115_data * v123_data));
            double v128_data = s0[11];
            double v130_data = ir1[4];
            ir1[4] = (v130_data + (v115_data * v128_data));
            double v146_data = r0[4];
            double v150_data = s0[10];
            double v152_data = ir1[3];
            ir1[3] = (v152_data + (v146_data * v150_data));
            double v155_data = s0[12];
            double v157_data = ir1[4];
            ir1[4] = (v157_data + (v146_data * v155_data));
            double v160_data = s0[14];
            double v162_data = ir1[5];
            ir1[5] = (v162_data + (v146_data * v160_data));
            double v177_data = r0[5];
            double v182_data = s0[13];
            double v184_data = ir1[4];
            ir1[4] = (v184_data + (v177_data * v182_data));
            double v187_data = s0[15];
            double v189_data = ir1[5];
            ir1[5] = (v189_data + (v177_data * v187_data));
            double v192_data = s0[17];
            double v194_data = ir1[6];
            ir1[6] = (v194_data + (v177_data * v192_data));
            double v208_data = r0[6];
            double v214_data = s0[16];
            double v216_data = ir1[5];
            ir1[5] = (v216_data + (v208_data * v214_data));
            double v219_data = s0[18];
            double v221_data = ir1[6];
            ir1[6] = (v221_data + (v208_data * v219_data));
            double v224_data = s0[20];
            double v226_data = ir1[7];
            ir1[7] = (v226_data + (v208_data * v224_data));
            double v239_data = r0[7];
            double v246_data = s0[19];
            double v248_data = ir1[6];
            ir1[6] = (v248_data + (v239_data * v246_data));
            double v251_data = s0[21];
            double v253_data = ir1[7];
            ir1[7] = (v253_data + (v239_data * v251_data));
            double v256_data = s0[23];
            double v258_data = ir1[8];
            ir1[8] = (v258_data + (v239_data * v256_data));
            double v270_data = r0[8];
            double v278_data = s0[22];
            double v280_data = ir1[7];
            ir1[7] = (v280_data + (v270_data * v278_data));
            double v283_data = s0[24];
            double v285_data = ir1[8];
            ir1[8] = (v285_data + (v270_data * v283_data));
            double v288_data = s0[26];
            double v290_data = ir1[9];
            ir1[9] = (v290_data + (v270_data * v288_data));
            double v301_data = r0[9];
            double v310_data = s0[25];
            double v312_data = ir1[8];
            ir1[8] = (v312_data + (v301_data * v310_data));
            double v315_data = s0[27];
            double v317_data = ir1[9];
            ir1[9] = (v317_data + (v301_data * v315_data));
            double v320_data = s0[29];
            double v322_data = ir1[10];
            ir1[10] = (v322_data + (v301_data * v320_data));
            double v332_data = r0[10];
            double v342_data = s0[28];
            double v344_data = ir1[9];
            ir1[9] = (v344_data + (v332_data * v342_data));
            double v347_data = s0[30];
            double v349_data = ir1[10];
            ir1[10] = (v349_data + (v332_data * v347_data));
            double v352_data = s0[32];
            double v354_data = ir1[11];
            ir1[11] = (v354_data + (v332_data * v352_data));
            double v363_data = r0[11];
            double v374_data = s0[31];
            double v376_data = ir1[10];
            ir1[10] = (v376_data + (v363_data * v374_data));
            double v379_data = s0[33];
            double v381_data = ir1[11];
            ir1[11] = (v381_data + (v363_data * v379_data));
            double v384_data = s0[35];
            double v386_data = ir1[12];
            ir1[12] = (v386_data + (v363_data * v384_data));
            double v394_data = r0[12];
            double v406_data = s0[34];
            double v408_data = ir1[11];
            ir1[11] = (v408_data + (v394_data * v406_data));
            double v411_data = s0[36];
            double v413_data = ir1[12];
            ir1[12] = (v413_data + (v394_data * v411_data));
            double v416_data = s0[38];
            double v418_data = ir1[13];
            ir1[13] = (v418_data + (v394_data * v416_data));
            double v425_data = r0[13];
            double v438_data = s0[37];
            double v440_data = ir1[12];
            ir1[12] = (v440_data + (v425_data * v438_data));
            double v443_data = s0[39];
            double v445_data = ir1[13];
            ir1[13] = (v445_data + (v425_data * v443_data));
            double v448_data = s0[41];
            double v450_data = ir1[14];
            ir1[14] = (v450_data + (v425_data * v448_data));
            double v456_data = r0[14];
            double v470_data = s0[40];
            double v472_data = ir1[13];
            ir1[13] = (v472_data + (v456_data * v470_data));
            double v475_data = s0[42];
            double v477_data = ir1[14];
            ir1[14] = (v477_data + (v456_data * v475_data));
            double v480_data = s0[44];
            double v482_data = ir1[15];
            ir1[15] = (v482_data + (v456_data * v480_data));
            double v487_data = r0[15];
            double v502_data = s0[43];
            double v504_data = ir1[14];
            ir1[14] = (v504_data + (v487_data * v502_data));
            double v507_data = s0[45];
            double v509_data = ir1[15];
            ir1[15] = (v509_data + (v487_data * v507_data));
            #pragma unroll
            for (int32_t v514_n0 = 0; v514_n0 < 1; ++v514_n0) {
              #pragma unroll
              for (int32_t v515_n1 = 0; v515_n1 < 16; ++v515_n1) {
                int32_t v516_a = v514_n0 + v515_n1;
                int32_t v517_a = v514_n0 + v515_n1;
                double v518_data = ir1[v517_a];
                int32_t v519_a = v514_n0 + v515_n1;
                r1[v517_a] = v518_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v524_i0 = 0; v524_i0 < 1; ++v524_i0) {
            int32_t v533_lead = v3_lead + (v524_i0 * 16);
            #pragma unroll
            for (int32_t v525_i1 = 0; v525_i1 < 16; ++v525_i1) {
              int32_t v526_a = v524_i0 + v525_i1;
              double v528_data = r1[(v524_i0 + v525_i1)];
              int32_t v535_a = v533_lead + (v525_i1 * 16);
              glb_m0[v535_a] = v528_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

