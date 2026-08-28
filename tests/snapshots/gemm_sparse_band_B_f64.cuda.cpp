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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 16;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              int32_t v18_a = v12_i1 * 16;
              int32_t v19_a = v17_lead + v18_a;
              double v27_data = __ldcg(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + v12_i1)] = v27_data;
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
          double r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v38_data = r0[0];
          double v39_data = s0[0];
          double v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          double v44_data = s0[2];
          double v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          double v65_data = r0[1];
          double v66_data = s0[1];
          double v68_data = ir1[0];
          ir1[0] = (v68_data + (v65_data * v66_data));
          double v71_data = s0[3];
          double v73_data = ir1[1];
          ir1[1] = (v73_data + (v65_data * v71_data));
          double v76_data = s0[5];
          double v78_data = ir1[2];
          ir1[2] = (v78_data + (v65_data * v76_data));
          double v96_data = r0[2];
          double v98_data = s0[4];
          double v100_data = ir1[1];
          ir1[1] = (v100_data + (v96_data * v98_data));
          double v103_data = s0[6];
          double v105_data = ir1[2];
          ir1[2] = (v105_data + (v96_data * v103_data));
          double v108_data = s0[8];
          double v110_data = ir1[3];
          ir1[3] = (v110_data + (v96_data * v108_data));
          double v127_data = r0[3];
          double v130_data = s0[7];
          double v132_data = ir1[2];
          ir1[2] = (v132_data + (v127_data * v130_data));
          double v135_data = s0[9];
          double v137_data = ir1[3];
          ir1[3] = (v137_data + (v127_data * v135_data));
          double v140_data = s0[11];
          double v142_data = ir1[4];
          ir1[4] = (v142_data + (v127_data * v140_data));
          double v158_data = r0[4];
          double v162_data = s0[10];
          double v164_data = ir1[3];
          ir1[3] = (v164_data + (v158_data * v162_data));
          double v167_data = s0[12];
          double v169_data = ir1[4];
          ir1[4] = (v169_data + (v158_data * v167_data));
          double v172_data = s0[14];
          double v174_data = ir1[5];
          ir1[5] = (v174_data + (v158_data * v172_data));
          double v189_data = r0[5];
          double v194_data = s0[13];
          double v196_data = ir1[4];
          ir1[4] = (v196_data + (v189_data * v194_data));
          double v199_data = s0[15];
          double v201_data = ir1[5];
          ir1[5] = (v201_data + (v189_data * v199_data));
          double v204_data = s0[17];
          double v206_data = ir1[6];
          ir1[6] = (v206_data + (v189_data * v204_data));
          double v220_data = r0[6];
          double v226_data = s0[16];
          double v228_data = ir1[5];
          ir1[5] = (v228_data + (v220_data * v226_data));
          double v231_data = s0[18];
          double v233_data = ir1[6];
          ir1[6] = (v233_data + (v220_data * v231_data));
          double v236_data = s0[20];
          double v238_data = ir1[7];
          ir1[7] = (v238_data + (v220_data * v236_data));
          double v251_data = r0[7];
          double v258_data = s0[19];
          double v260_data = ir1[6];
          ir1[6] = (v260_data + (v251_data * v258_data));
          double v263_data = s0[21];
          double v265_data = ir1[7];
          ir1[7] = (v265_data + (v251_data * v263_data));
          double v268_data = s0[23];
          double v270_data = ir1[8];
          ir1[8] = (v270_data + (v251_data * v268_data));
          double v282_data = r0[8];
          double v290_data = s0[22];
          double v292_data = ir1[7];
          ir1[7] = (v292_data + (v282_data * v290_data));
          double v295_data = s0[24];
          double v297_data = ir1[8];
          ir1[8] = (v297_data + (v282_data * v295_data));
          double v300_data = s0[26];
          double v302_data = ir1[9];
          ir1[9] = (v302_data + (v282_data * v300_data));
          double v313_data = r0[9];
          double v322_data = s0[25];
          double v324_data = ir1[8];
          ir1[8] = (v324_data + (v313_data * v322_data));
          double v327_data = s0[27];
          double v329_data = ir1[9];
          ir1[9] = (v329_data + (v313_data * v327_data));
          double v332_data = s0[29];
          double v334_data = ir1[10];
          ir1[10] = (v334_data + (v313_data * v332_data));
          double v344_data = r0[10];
          double v354_data = s0[28];
          double v356_data = ir1[9];
          ir1[9] = (v356_data + (v344_data * v354_data));
          double v359_data = s0[30];
          double v361_data = ir1[10];
          ir1[10] = (v361_data + (v344_data * v359_data));
          double v364_data = s0[32];
          double v366_data = ir1[11];
          ir1[11] = (v366_data + (v344_data * v364_data));
          double v375_data = r0[11];
          double v386_data = s0[31];
          double v388_data = ir1[10];
          ir1[10] = (v388_data + (v375_data * v386_data));
          double v391_data = s0[33];
          double v393_data = ir1[11];
          ir1[11] = (v393_data + (v375_data * v391_data));
          double v396_data = s0[35];
          double v398_data = ir1[12];
          ir1[12] = (v398_data + (v375_data * v396_data));
          double v406_data = r0[12];
          double v418_data = s0[34];
          double v420_data = ir1[11];
          ir1[11] = (v420_data + (v406_data * v418_data));
          double v423_data = s0[36];
          double v425_data = ir1[12];
          ir1[12] = (v425_data + (v406_data * v423_data));
          double v428_data = s0[38];
          double v430_data = ir1[13];
          ir1[13] = (v430_data + (v406_data * v428_data));
          double v437_data = r0[13];
          double v450_data = s0[37];
          double v452_data = ir1[12];
          ir1[12] = (v452_data + (v437_data * v450_data));
          double v455_data = s0[39];
          double v457_data = ir1[13];
          ir1[13] = (v457_data + (v437_data * v455_data));
          double v460_data = s0[41];
          double v462_data = ir1[14];
          ir1[14] = (v462_data + (v437_data * v460_data));
          double v468_data = r0[14];
          double v482_data = s0[40];
          double v484_data = ir1[13];
          ir1[13] = (v484_data + (v468_data * v482_data));
          double v487_data = s0[42];
          double v489_data = ir1[14];
          ir1[14] = (v489_data + (v468_data * v487_data));
          double v492_data = s0[44];
          double v494_data = ir1[15];
          ir1[15] = (v494_data + (v468_data * v492_data));
          double v499_data = r0[15];
          double v514_data = s0[43];
          double v516_data = ir1[14];
          ir1[14] = (v516_data + (v499_data * v514_data));
          double v519_data = s0[45];
          double v521_data = ir1[15];
          ir1[15] = (v521_data + (v499_data * v519_data));
          #pragma unroll
          for (int32_t v526_n0 = 0; v526_n0 < 1; ++v526_n0) {
            #pragma unroll
            for (int32_t v527_n1 = 0; v527_n1 < 16; ++v527_n1) {
              int32_t v528_a = v526_n0 + v527_n1;
              int32_t v529_a = v526_n0 + v527_n1;
              double v530_data = ir1[v529_a];
              r1[v529_a] = v530_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v535_i0 = 0; v535_i0 < 1; ++v535_i0) {
            int32_t v544_lead = v10_lead + (v535_i0 * 16);
            #pragma unroll
            for (int32_t v536_i1 = 0; v536_i1 < 16; ++v536_i1) {
              int32_t v537_a = v535_i0 + v536_i1;
              double v539_data = r1[(v535_i0 + v536_i1)];
              glb_m0[(v544_lead + (v536_i1 * 16))] = v539_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

