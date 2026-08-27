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
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v28_data = r0[0];
          double v29_data = s0[0];
          double v31_data = ir1[0];
          ir1[0] = (v31_data + (v28_data * v29_data));
          double v34_data = s0[2];
          double v36_data = ir1[1];
          ir1[1] = (v36_data + (v28_data * v34_data));
          double v55_data = r0[1];
          double v56_data = s0[1];
          double v58_data = ir1[0];
          ir1[0] = (v58_data + (v55_data * v56_data));
          double v61_data = s0[3];
          double v63_data = ir1[1];
          ir1[1] = (v63_data + (v55_data * v61_data));
          double v66_data = s0[5];
          double v68_data = ir1[2];
          ir1[2] = (v68_data + (v55_data * v66_data));
          double v86_data = r0[2];
          double v88_data = s0[4];
          double v90_data = ir1[1];
          ir1[1] = (v90_data + (v86_data * v88_data));
          double v93_data = s0[6];
          double v95_data = ir1[2];
          ir1[2] = (v95_data + (v86_data * v93_data));
          double v98_data = s0[8];
          double v100_data = ir1[3];
          ir1[3] = (v100_data + (v86_data * v98_data));
          double v117_data = r0[3];
          double v120_data = s0[7];
          double v122_data = ir1[2];
          ir1[2] = (v122_data + (v117_data * v120_data));
          double v125_data = s0[9];
          double v127_data = ir1[3];
          ir1[3] = (v127_data + (v117_data * v125_data));
          double v130_data = s0[11];
          double v132_data = ir1[4];
          ir1[4] = (v132_data + (v117_data * v130_data));
          double v148_data = r0[4];
          double v152_data = s0[10];
          double v154_data = ir1[3];
          ir1[3] = (v154_data + (v148_data * v152_data));
          double v157_data = s0[12];
          double v159_data = ir1[4];
          ir1[4] = (v159_data + (v148_data * v157_data));
          double v162_data = s0[14];
          double v164_data = ir1[5];
          ir1[5] = (v164_data + (v148_data * v162_data));
          double v179_data = r0[5];
          double v184_data = s0[13];
          double v186_data = ir1[4];
          ir1[4] = (v186_data + (v179_data * v184_data));
          double v189_data = s0[15];
          double v191_data = ir1[5];
          ir1[5] = (v191_data + (v179_data * v189_data));
          double v194_data = s0[17];
          double v196_data = ir1[6];
          ir1[6] = (v196_data + (v179_data * v194_data));
          double v210_data = r0[6];
          double v216_data = s0[16];
          double v218_data = ir1[5];
          ir1[5] = (v218_data + (v210_data * v216_data));
          double v221_data = s0[18];
          double v223_data = ir1[6];
          ir1[6] = (v223_data + (v210_data * v221_data));
          double v226_data = s0[20];
          double v228_data = ir1[7];
          ir1[7] = (v228_data + (v210_data * v226_data));
          double v241_data = r0[7];
          double v248_data = s0[19];
          double v250_data = ir1[6];
          ir1[6] = (v250_data + (v241_data * v248_data));
          double v253_data = s0[21];
          double v255_data = ir1[7];
          ir1[7] = (v255_data + (v241_data * v253_data));
          double v258_data = s0[23];
          double v260_data = ir1[8];
          ir1[8] = (v260_data + (v241_data * v258_data));
          double v272_data = r0[8];
          double v280_data = s0[22];
          double v282_data = ir1[7];
          ir1[7] = (v282_data + (v272_data * v280_data));
          double v285_data = s0[24];
          double v287_data = ir1[8];
          ir1[8] = (v287_data + (v272_data * v285_data));
          double v290_data = s0[26];
          double v292_data = ir1[9];
          ir1[9] = (v292_data + (v272_data * v290_data));
          double v303_data = r0[9];
          double v312_data = s0[25];
          double v314_data = ir1[8];
          ir1[8] = (v314_data + (v303_data * v312_data));
          double v317_data = s0[27];
          double v319_data = ir1[9];
          ir1[9] = (v319_data + (v303_data * v317_data));
          double v322_data = s0[29];
          double v324_data = ir1[10];
          ir1[10] = (v324_data + (v303_data * v322_data));
          double v334_data = r0[10];
          double v344_data = s0[28];
          double v346_data = ir1[9];
          ir1[9] = (v346_data + (v334_data * v344_data));
          double v349_data = s0[30];
          double v351_data = ir1[10];
          ir1[10] = (v351_data + (v334_data * v349_data));
          double v354_data = s0[32];
          double v356_data = ir1[11];
          ir1[11] = (v356_data + (v334_data * v354_data));
          double v365_data = r0[11];
          double v376_data = s0[31];
          double v378_data = ir1[10];
          ir1[10] = (v378_data + (v365_data * v376_data));
          double v381_data = s0[33];
          double v383_data = ir1[11];
          ir1[11] = (v383_data + (v365_data * v381_data));
          double v386_data = s0[35];
          double v388_data = ir1[12];
          ir1[12] = (v388_data + (v365_data * v386_data));
          double v396_data = r0[12];
          double v408_data = s0[34];
          double v410_data = ir1[11];
          ir1[11] = (v410_data + (v396_data * v408_data));
          double v413_data = s0[36];
          double v415_data = ir1[12];
          ir1[12] = (v415_data + (v396_data * v413_data));
          double v418_data = s0[38];
          double v420_data = ir1[13];
          ir1[13] = (v420_data + (v396_data * v418_data));
          double v427_data = r0[13];
          double v440_data = s0[37];
          double v442_data = ir1[12];
          ir1[12] = (v442_data + (v427_data * v440_data));
          double v445_data = s0[39];
          double v447_data = ir1[13];
          ir1[13] = (v447_data + (v427_data * v445_data));
          double v450_data = s0[41];
          double v452_data = ir1[14];
          ir1[14] = (v452_data + (v427_data * v450_data));
          double v458_data = r0[14];
          double v472_data = s0[40];
          double v474_data = ir1[13];
          ir1[13] = (v474_data + (v458_data * v472_data));
          double v477_data = s0[42];
          double v479_data = ir1[14];
          ir1[14] = (v479_data + (v458_data * v477_data));
          double v482_data = s0[44];
          double v484_data = ir1[15];
          ir1[15] = (v484_data + (v458_data * v482_data));
          double v489_data = r0[15];
          double v504_data = s0[43];
          double v506_data = ir1[14];
          ir1[14] = (v506_data + (v489_data * v504_data));
          double v509_data = s0[45];
          double v511_data = ir1[15];
          ir1[15] = (v511_data + (v489_data * v509_data));
          #pragma unroll
          for (int32_t v516_n0 = 0; v516_n0 < 1; ++v516_n0) {
            #pragma unroll
            for (int32_t v517_n1 = 0; v517_n1 < 16; ++v517_n1) {
              int32_t v518_a = v516_n0 + v517_n1;
              int32_t v519_a = v516_n0 + v517_n1;
              double v520_data = ir1[v519_a];
              int32_t v521_a = v516_n0 + v517_n1;
              r1[v519_a] = v520_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v526_i0 = 0; v526_i0 < 1; ++v526_i0) {
            int32_t v535_lead = v3_lead + (v526_i0 * 16);
            #pragma unroll
            for (int32_t v527_i1 = 0; v527_i1 < 16; ++v527_i1) {
              int32_t v528_a = v526_i0 + v527_i1;
              double v530_data = r1[(v526_i0 + v527_i1)];
              int32_t v537_a = v535_lead + (v527_i1 * 16);
              glb_m0[v537_a] = v530_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

