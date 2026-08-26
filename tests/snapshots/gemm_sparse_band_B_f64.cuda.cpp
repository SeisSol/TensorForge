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
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 16;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 16;
              int32_t v11_a = v9_lead + v10_a;
              double v19_data = __ldcg(&glb_m1[(v16_lead + v10_a)]);
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
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
            double v24_data = r0[0];
            double v25_data = s0[0];
            double v27_data = ir1[0];
            ir1[0] = (v27_data + (v24_data * v25_data));
            double v30_data = s0[2];
            double v32_data = ir1[1];
            ir1[1] = (v32_data + (v24_data * v30_data));
            double v51_data = r0[1];
            double v52_data = s0[1];
            double v54_data = ir1[0];
            ir1[0] = (v54_data + (v51_data * v52_data));
            double v57_data = s0[3];
            double v59_data = ir1[1];
            ir1[1] = (v59_data + (v51_data * v57_data));
            double v62_data = s0[5];
            double v64_data = ir1[2];
            ir1[2] = (v64_data + (v51_data * v62_data));
            double v82_data = r0[2];
            double v84_data = s0[4];
            double v86_data = ir1[1];
            ir1[1] = (v86_data + (v82_data * v84_data));
            double v89_data = s0[6];
            double v91_data = ir1[2];
            ir1[2] = (v91_data + (v82_data * v89_data));
            double v94_data = s0[8];
            double v96_data = ir1[3];
            ir1[3] = (v96_data + (v82_data * v94_data));
            double v113_data = r0[3];
            double v116_data = s0[7];
            double v118_data = ir1[2];
            ir1[2] = (v118_data + (v113_data * v116_data));
            double v121_data = s0[9];
            double v123_data = ir1[3];
            ir1[3] = (v123_data + (v113_data * v121_data));
            double v126_data = s0[11];
            double v128_data = ir1[4];
            ir1[4] = (v128_data + (v113_data * v126_data));
            double v144_data = r0[4];
            double v148_data = s0[10];
            double v150_data = ir1[3];
            ir1[3] = (v150_data + (v144_data * v148_data));
            double v153_data = s0[12];
            double v155_data = ir1[4];
            ir1[4] = (v155_data + (v144_data * v153_data));
            double v158_data = s0[14];
            double v160_data = ir1[5];
            ir1[5] = (v160_data + (v144_data * v158_data));
            double v175_data = r0[5];
            double v180_data = s0[13];
            double v182_data = ir1[4];
            ir1[4] = (v182_data + (v175_data * v180_data));
            double v185_data = s0[15];
            double v187_data = ir1[5];
            ir1[5] = (v187_data + (v175_data * v185_data));
            double v190_data = s0[17];
            double v192_data = ir1[6];
            ir1[6] = (v192_data + (v175_data * v190_data));
            double v206_data = r0[6];
            double v212_data = s0[16];
            double v214_data = ir1[5];
            ir1[5] = (v214_data + (v206_data * v212_data));
            double v217_data = s0[18];
            double v219_data = ir1[6];
            ir1[6] = (v219_data + (v206_data * v217_data));
            double v222_data = s0[20];
            double v224_data = ir1[7];
            ir1[7] = (v224_data + (v206_data * v222_data));
            double v237_data = r0[7];
            double v244_data = s0[19];
            double v246_data = ir1[6];
            ir1[6] = (v246_data + (v237_data * v244_data));
            double v249_data = s0[21];
            double v251_data = ir1[7];
            ir1[7] = (v251_data + (v237_data * v249_data));
            double v254_data = s0[23];
            double v256_data = ir1[8];
            ir1[8] = (v256_data + (v237_data * v254_data));
            double v268_data = r0[8];
            double v276_data = s0[22];
            double v278_data = ir1[7];
            ir1[7] = (v278_data + (v268_data * v276_data));
            double v281_data = s0[24];
            double v283_data = ir1[8];
            ir1[8] = (v283_data + (v268_data * v281_data));
            double v286_data = s0[26];
            double v288_data = ir1[9];
            ir1[9] = (v288_data + (v268_data * v286_data));
            double v299_data = r0[9];
            double v308_data = s0[25];
            double v310_data = ir1[8];
            ir1[8] = (v310_data + (v299_data * v308_data));
            double v313_data = s0[27];
            double v315_data = ir1[9];
            ir1[9] = (v315_data + (v299_data * v313_data));
            double v318_data = s0[29];
            double v320_data = ir1[10];
            ir1[10] = (v320_data + (v299_data * v318_data));
            double v330_data = r0[10];
            double v340_data = s0[28];
            double v342_data = ir1[9];
            ir1[9] = (v342_data + (v330_data * v340_data));
            double v345_data = s0[30];
            double v347_data = ir1[10];
            ir1[10] = (v347_data + (v330_data * v345_data));
            double v350_data = s0[32];
            double v352_data = ir1[11];
            ir1[11] = (v352_data + (v330_data * v350_data));
            double v361_data = r0[11];
            double v372_data = s0[31];
            double v374_data = ir1[10];
            ir1[10] = (v374_data + (v361_data * v372_data));
            double v377_data = s0[33];
            double v379_data = ir1[11];
            ir1[11] = (v379_data + (v361_data * v377_data));
            double v382_data = s0[35];
            double v384_data = ir1[12];
            ir1[12] = (v384_data + (v361_data * v382_data));
            double v392_data = r0[12];
            double v404_data = s0[34];
            double v406_data = ir1[11];
            ir1[11] = (v406_data + (v392_data * v404_data));
            double v409_data = s0[36];
            double v411_data = ir1[12];
            ir1[12] = (v411_data + (v392_data * v409_data));
            double v414_data = s0[38];
            double v416_data = ir1[13];
            ir1[13] = (v416_data + (v392_data * v414_data));
            double v423_data = r0[13];
            double v436_data = s0[37];
            double v438_data = ir1[12];
            ir1[12] = (v438_data + (v423_data * v436_data));
            double v441_data = s0[39];
            double v443_data = ir1[13];
            ir1[13] = (v443_data + (v423_data * v441_data));
            double v446_data = s0[41];
            double v448_data = ir1[14];
            ir1[14] = (v448_data + (v423_data * v446_data));
            double v454_data = r0[14];
            double v468_data = s0[40];
            double v470_data = ir1[13];
            ir1[13] = (v470_data + (v454_data * v468_data));
            double v473_data = s0[42];
            double v475_data = ir1[14];
            ir1[14] = (v475_data + (v454_data * v473_data));
            double v478_data = s0[44];
            double v480_data = ir1[15];
            ir1[15] = (v480_data + (v454_data * v478_data));
            double v485_data = r0[15];
            double v500_data = s0[43];
            double v502_data = ir1[14];
            ir1[14] = (v502_data + (v485_data * v500_data));
            double v505_data = s0[45];
            double v507_data = ir1[15];
            ir1[15] = (v507_data + (v485_data * v505_data));
            #pragma unroll
            for (int32_t v512_n0 = 0; v512_n0 < 1; ++v512_n0) {
              #pragma unroll
              for (int32_t v513_n1 = 0; v513_n1 < 16; ++v513_n1) {
                int32_t v514_a = v512_n0 + v513_n1;
                int32_t v515_a = v512_n0 + v513_n1;
                double v516_data = ir1[v515_a];
                int32_t v517_a = v512_n0 + v513_n1;
                r1[v515_a] = v516_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v521_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v522_i0 = 0; v522_i0 < 1; ++v522_i0) {
            int32_t v531_lead = v521_lead + (v522_i0 * 16);
            #pragma unroll
            for (int32_t v523_i1 = 0; v523_i1 < 16; ++v523_i1) {
              int32_t v524_a = v522_i0 + v523_i1;
              double v526_data = r1[(v522_i0 + v523_i1)];
              int32_t v533_a = v531_lead + (v523_i1 * 16);
              glb_m0[v533_a] = v526_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

