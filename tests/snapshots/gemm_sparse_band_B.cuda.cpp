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
              float v20_data = __ldcg(&glb_m1[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<4>(4), pipeline);
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], cuda::aligned_size_t<4>(4), pipeline);
          if (threadIdx.x < 14) {
            cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<4>(4), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[16]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir1[16]{};
            float v26_data = r0[0];
            float v27_data = s0[0];
            float v29_data = ir1[0];
            ir1[0] = (v29_data + (v26_data * v27_data));
            float v32_data = s0[2];
            float v34_data = ir1[1];
            ir1[1] = (v34_data + (v26_data * v32_data));
            float v53_data = r0[1];
            float v54_data = s0[1];
            float v56_data = ir1[0];
            ir1[0] = (v56_data + (v53_data * v54_data));
            float v59_data = s0[3];
            float v61_data = ir1[1];
            ir1[1] = (v61_data + (v53_data * v59_data));
            float v64_data = s0[5];
            float v66_data = ir1[2];
            ir1[2] = (v66_data + (v53_data * v64_data));
            float v84_data = r0[2];
            float v86_data = s0[4];
            float v88_data = ir1[1];
            ir1[1] = (v88_data + (v84_data * v86_data));
            float v91_data = s0[6];
            float v93_data = ir1[2];
            ir1[2] = (v93_data + (v84_data * v91_data));
            float v96_data = s0[8];
            float v98_data = ir1[3];
            ir1[3] = (v98_data + (v84_data * v96_data));
            float v115_data = r0[3];
            float v118_data = s0[7];
            float v120_data = ir1[2];
            ir1[2] = (v120_data + (v115_data * v118_data));
            float v123_data = s0[9];
            float v125_data = ir1[3];
            ir1[3] = (v125_data + (v115_data * v123_data));
            float v128_data = s0[11];
            float v130_data = ir1[4];
            ir1[4] = (v130_data + (v115_data * v128_data));
            float v146_data = r0[4];
            float v150_data = s0[10];
            float v152_data = ir1[3];
            ir1[3] = (v152_data + (v146_data * v150_data));
            float v155_data = s0[12];
            float v157_data = ir1[4];
            ir1[4] = (v157_data + (v146_data * v155_data));
            float v160_data = s0[14];
            float v162_data = ir1[5];
            ir1[5] = (v162_data + (v146_data * v160_data));
            float v177_data = r0[5];
            float v182_data = s0[13];
            float v184_data = ir1[4];
            ir1[4] = (v184_data + (v177_data * v182_data));
            float v187_data = s0[15];
            float v189_data = ir1[5];
            ir1[5] = (v189_data + (v177_data * v187_data));
            float v192_data = s0[17];
            float v194_data = ir1[6];
            ir1[6] = (v194_data + (v177_data * v192_data));
            float v208_data = r0[6];
            float v214_data = s0[16];
            float v216_data = ir1[5];
            ir1[5] = (v216_data + (v208_data * v214_data));
            float v219_data = s0[18];
            float v221_data = ir1[6];
            ir1[6] = (v221_data + (v208_data * v219_data));
            float v224_data = s0[20];
            float v226_data = ir1[7];
            ir1[7] = (v226_data + (v208_data * v224_data));
            float v239_data = r0[7];
            float v246_data = s0[19];
            float v248_data = ir1[6];
            ir1[6] = (v248_data + (v239_data * v246_data));
            float v251_data = s0[21];
            float v253_data = ir1[7];
            ir1[7] = (v253_data + (v239_data * v251_data));
            float v256_data = s0[23];
            float v258_data = ir1[8];
            ir1[8] = (v258_data + (v239_data * v256_data));
            float v270_data = r0[8];
            float v278_data = s0[22];
            float v280_data = ir1[7];
            ir1[7] = (v280_data + (v270_data * v278_data));
            float v283_data = s0[24];
            float v285_data = ir1[8];
            ir1[8] = (v285_data + (v270_data * v283_data));
            float v288_data = s0[26];
            float v290_data = ir1[9];
            ir1[9] = (v290_data + (v270_data * v288_data));
            float v301_data = r0[9];
            float v310_data = s0[25];
            float v312_data = ir1[8];
            ir1[8] = (v312_data + (v301_data * v310_data));
            float v315_data = s0[27];
            float v317_data = ir1[9];
            ir1[9] = (v317_data + (v301_data * v315_data));
            float v320_data = s0[29];
            float v322_data = ir1[10];
            ir1[10] = (v322_data + (v301_data * v320_data));
            float v332_data = r0[10];
            float v342_data = s0[28];
            float v344_data = ir1[9];
            ir1[9] = (v344_data + (v332_data * v342_data));
            float v347_data = s0[30];
            float v349_data = ir1[10];
            ir1[10] = (v349_data + (v332_data * v347_data));
            float v352_data = s0[32];
            float v354_data = ir1[11];
            ir1[11] = (v354_data + (v332_data * v352_data));
            float v363_data = r0[11];
            float v374_data = s0[31];
            float v376_data = ir1[10];
            ir1[10] = (v376_data + (v363_data * v374_data));
            float v379_data = s0[33];
            float v381_data = ir1[11];
            ir1[11] = (v381_data + (v363_data * v379_data));
            float v384_data = s0[35];
            float v386_data = ir1[12];
            ir1[12] = (v386_data + (v363_data * v384_data));
            float v394_data = r0[12];
            float v406_data = s0[34];
            float v408_data = ir1[11];
            ir1[11] = (v408_data + (v394_data * v406_data));
            float v411_data = s0[36];
            float v413_data = ir1[12];
            ir1[12] = (v413_data + (v394_data * v411_data));
            float v416_data = s0[38];
            float v418_data = ir1[13];
            ir1[13] = (v418_data + (v394_data * v416_data));
            float v425_data = r0[13];
            float v438_data = s0[37];
            float v440_data = ir1[12];
            ir1[12] = (v440_data + (v425_data * v438_data));
            float v443_data = s0[39];
            float v445_data = ir1[13];
            ir1[13] = (v445_data + (v425_data * v443_data));
            float v448_data = s0[41];
            float v450_data = ir1[14];
            ir1[14] = (v450_data + (v425_data * v448_data));
            float v456_data = r0[14];
            float v470_data = s0[40];
            float v472_data = ir1[13];
            ir1[13] = (v472_data + (v456_data * v470_data));
            float v475_data = s0[42];
            float v477_data = ir1[14];
            ir1[14] = (v477_data + (v456_data * v475_data));
            float v480_data = s0[44];
            float v482_data = ir1[15];
            ir1[15] = (v482_data + (v456_data * v480_data));
            float v487_data = r0[15];
            float v502_data = s0[43];
            float v504_data = ir1[14];
            ir1[14] = (v504_data + (v487_data * v502_data));
            float v507_data = s0[45];
            float v509_data = ir1[15];
            ir1[15] = (v509_data + (v487_data * v507_data));
            #pragma unroll
            for (int32_t v514_n0 = 0; v514_n0 < 1; ++v514_n0) {
              #pragma unroll
              for (int32_t v515_n1 = 0; v515_n1 < 16; ++v515_n1) {
                int32_t v516_a = v514_n0 + v515_n1;
                int32_t v517_a = v514_n0 + v515_n1;
                float v518_data = ir1[v517_a];
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
              float v528_data = r1[(v524_i0 + v525_i1)];
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

