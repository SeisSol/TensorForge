// === base name ===
kernel_058319e67db3fd64

// === header ===
void launcher_kernel_058319e67db3fd64(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_058319e67db3fd64(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_058319e67db3fd64, block.x * block.y * block.z, 640 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_058319e67db3fd64, cudaFuncAttributeMaxDynamicSharedMemorySize, 640 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_058319e67db3fd64<<<grid,block,640 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_058319e67db3fd64(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 46 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 16);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              float v28_data = __ldcg(&glb_m1[(v25_lead + (v20_i1 * 16))]);
              r0[(v19_i0 + v20_i1)] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v38_data = r0[0];
          float v39_data = s0[0];
          float v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          float v44_data = s0[2];
          float v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          float v65_data = r0[1];
          float v66_data = s0[1];
          float v68_data = ir1[0];
          ir1[0] = (v68_data + (v65_data * v66_data));
          float v71_data = s0[3];
          float v73_data = ir1[1];
          ir1[1] = (v73_data + (v65_data * v71_data));
          float v76_data = s0[5];
          float v78_data = ir1[2];
          ir1[2] = (v78_data + (v65_data * v76_data));
          float v96_data = r0[2];
          float v98_data = s0[4];
          float v100_data = ir1[1];
          ir1[1] = (v100_data + (v96_data * v98_data));
          float v103_data = s0[6];
          float v105_data = ir1[2];
          ir1[2] = (v105_data + (v96_data * v103_data));
          float v108_data = s0[8];
          float v110_data = ir1[3];
          ir1[3] = (v110_data + (v96_data * v108_data));
          float v127_data = r0[3];
          float v130_data = s0[7];
          float v132_data = ir1[2];
          ir1[2] = (v132_data + (v127_data * v130_data));
          float v135_data = s0[9];
          float v137_data = ir1[3];
          ir1[3] = (v137_data + (v127_data * v135_data));
          float v140_data = s0[11];
          float v142_data = ir1[4];
          ir1[4] = (v142_data + (v127_data * v140_data));
          float v158_data = r0[4];
          float v162_data = s0[10];
          float v164_data = ir1[3];
          ir1[3] = (v164_data + (v158_data * v162_data));
          float v167_data = s0[12];
          float v169_data = ir1[4];
          ir1[4] = (v169_data + (v158_data * v167_data));
          float v172_data = s0[14];
          float v174_data = ir1[5];
          ir1[5] = (v174_data + (v158_data * v172_data));
          float v189_data = r0[5];
          float v194_data = s0[13];
          float v196_data = ir1[4];
          ir1[4] = (v196_data + (v189_data * v194_data));
          float v199_data = s0[15];
          float v201_data = ir1[5];
          ir1[5] = (v201_data + (v189_data * v199_data));
          float v204_data = s0[17];
          float v206_data = ir1[6];
          ir1[6] = (v206_data + (v189_data * v204_data));
          float v220_data = r0[6];
          float v226_data = s0[16];
          float v228_data = ir1[5];
          ir1[5] = (v228_data + (v220_data * v226_data));
          float v231_data = s0[18];
          float v233_data = ir1[6];
          ir1[6] = (v233_data + (v220_data * v231_data));
          float v236_data = s0[20];
          float v238_data = ir1[7];
          ir1[7] = (v238_data + (v220_data * v236_data));
          float v251_data = r0[7];
          float v258_data = s0[19];
          float v260_data = ir1[6];
          ir1[6] = (v260_data + (v251_data * v258_data));
          float v263_data = s0[21];
          float v265_data = ir1[7];
          ir1[7] = (v265_data + (v251_data * v263_data));
          float v268_data = s0[23];
          float v270_data = ir1[8];
          ir1[8] = (v270_data + (v251_data * v268_data));
          float v282_data = r0[8];
          float v290_data = s0[22];
          float v292_data = ir1[7];
          ir1[7] = (v292_data + (v282_data * v290_data));
          float v295_data = s0[24];
          float v297_data = ir1[8];
          ir1[8] = (v297_data + (v282_data * v295_data));
          float v300_data = s0[26];
          float v302_data = ir1[9];
          ir1[9] = (v302_data + (v282_data * v300_data));
          float v313_data = r0[9];
          float v322_data = s0[25];
          float v324_data = ir1[8];
          ir1[8] = (v324_data + (v313_data * v322_data));
          float v327_data = s0[27];
          float v329_data = ir1[9];
          ir1[9] = (v329_data + (v313_data * v327_data));
          float v332_data = s0[29];
          float v334_data = ir1[10];
          ir1[10] = (v334_data + (v313_data * v332_data));
          float v344_data = r0[10];
          float v354_data = s0[28];
          float v356_data = ir1[9];
          ir1[9] = (v356_data + (v344_data * v354_data));
          float v359_data = s0[30];
          float v361_data = ir1[10];
          ir1[10] = (v361_data + (v344_data * v359_data));
          float v364_data = s0[32];
          float v366_data = ir1[11];
          ir1[11] = (v366_data + (v344_data * v364_data));
          float v375_data = r0[11];
          float v386_data = s0[31];
          float v388_data = ir1[10];
          ir1[10] = (v388_data + (v375_data * v386_data));
          float v391_data = s0[33];
          float v393_data = ir1[11];
          ir1[11] = (v393_data + (v375_data * v391_data));
          float v396_data = s0[35];
          float v398_data = ir1[12];
          ir1[12] = (v398_data + (v375_data * v396_data));
          float v406_data = r0[12];
          float v418_data = s0[34];
          float v420_data = ir1[11];
          ir1[11] = (v420_data + (v406_data * v418_data));
          float v423_data = s0[36];
          float v425_data = ir1[12];
          ir1[12] = (v425_data + (v406_data * v423_data));
          float v428_data = s0[38];
          float v430_data = ir1[13];
          ir1[13] = (v430_data + (v406_data * v428_data));
          float v437_data = r0[13];
          float v450_data = s0[37];
          float v452_data = ir1[12];
          ir1[12] = (v452_data + (v437_data * v450_data));
          float v455_data = s0[39];
          float v457_data = ir1[13];
          ir1[13] = (v457_data + (v437_data * v455_data));
          float v460_data = s0[41];
          float v462_data = ir1[14];
          ir1[14] = (v462_data + (v437_data * v460_data));
          float v468_data = r0[14];
          float v482_data = s0[40];
          float v484_data = ir1[13];
          ir1[13] = (v484_data + (v468_data * v482_data));
          float v487_data = s0[42];
          float v489_data = ir1[14];
          ir1[14] = (v489_data + (v468_data * v487_data));
          float v492_data = s0[44];
          float v494_data = ir1[15];
          ir1[15] = (v494_data + (v468_data * v492_data));
          float v499_data = r0[15];
          float v514_data = s0[43];
          float v516_data = ir1[14];
          ir1[14] = (v516_data + (v499_data * v514_data));
          float v519_data = s0[45];
          float v521_data = ir1[15];
          ir1[15] = (v521_data + (v499_data * v519_data));
          #pragma unroll
          for (int32_t v526_n0 = 0; v526_n0 < 1; ++v526_n0) {
            #pragma unroll
            for (int32_t v527_n1 = 0; v527_n1 < 16; ++v527_n1) {
              int32_t v528_a = v526_n0 + v527_n1;
              float v529_data = ir1[v528_a];
              r1[v528_a] = v529_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v534_i0 = 0; v534_i0 < 1; ++v534_i0) {
            int32_t v542_lead = v18_lead + (v534_i0 * 16);
            #pragma unroll
            for (int32_t v535_i1 = 0; v535_i1 < 16; ++v535_i1) {
              float v537_data = r1[(v534_i0 + v535_i1)];
              glb_m0[(v542_lead + (v535_i1 * 16))] = v537_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

