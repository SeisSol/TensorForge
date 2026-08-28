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
          int32_t v7_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 16;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 16;
              int32_t v16_a = v14_lead + v15_a;
              double v24_data = __ldcg(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + v9_i1)] = v24_data;
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
          double v35_data = r0[0];
          double v36_data = s0[0];
          double v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          double v41_data = s0[2];
          double v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          double v62_data = r0[1];
          double v63_data = s0[1];
          double v65_data = ir1[0];
          ir1[0] = (v65_data + (v62_data * v63_data));
          double v68_data = s0[3];
          double v70_data = ir1[1];
          ir1[1] = (v70_data + (v62_data * v68_data));
          double v73_data = s0[5];
          double v75_data = ir1[2];
          ir1[2] = (v75_data + (v62_data * v73_data));
          double v93_data = r0[2];
          double v95_data = s0[4];
          double v97_data = ir1[1];
          ir1[1] = (v97_data + (v93_data * v95_data));
          double v100_data = s0[6];
          double v102_data = ir1[2];
          ir1[2] = (v102_data + (v93_data * v100_data));
          double v105_data = s0[8];
          double v107_data = ir1[3];
          ir1[3] = (v107_data + (v93_data * v105_data));
          double v124_data = r0[3];
          double v127_data = s0[7];
          double v129_data = ir1[2];
          ir1[2] = (v129_data + (v124_data * v127_data));
          double v132_data = s0[9];
          double v134_data = ir1[3];
          ir1[3] = (v134_data + (v124_data * v132_data));
          double v137_data = s0[11];
          double v139_data = ir1[4];
          ir1[4] = (v139_data + (v124_data * v137_data));
          double v155_data = r0[4];
          double v159_data = s0[10];
          double v161_data = ir1[3];
          ir1[3] = (v161_data + (v155_data * v159_data));
          double v164_data = s0[12];
          double v166_data = ir1[4];
          ir1[4] = (v166_data + (v155_data * v164_data));
          double v169_data = s0[14];
          double v171_data = ir1[5];
          ir1[5] = (v171_data + (v155_data * v169_data));
          double v186_data = r0[5];
          double v191_data = s0[13];
          double v193_data = ir1[4];
          ir1[4] = (v193_data + (v186_data * v191_data));
          double v196_data = s0[15];
          double v198_data = ir1[5];
          ir1[5] = (v198_data + (v186_data * v196_data));
          double v201_data = s0[17];
          double v203_data = ir1[6];
          ir1[6] = (v203_data + (v186_data * v201_data));
          double v217_data = r0[6];
          double v223_data = s0[16];
          double v225_data = ir1[5];
          ir1[5] = (v225_data + (v217_data * v223_data));
          double v228_data = s0[18];
          double v230_data = ir1[6];
          ir1[6] = (v230_data + (v217_data * v228_data));
          double v233_data = s0[20];
          double v235_data = ir1[7];
          ir1[7] = (v235_data + (v217_data * v233_data));
          double v248_data = r0[7];
          double v255_data = s0[19];
          double v257_data = ir1[6];
          ir1[6] = (v257_data + (v248_data * v255_data));
          double v260_data = s0[21];
          double v262_data = ir1[7];
          ir1[7] = (v262_data + (v248_data * v260_data));
          double v265_data = s0[23];
          double v267_data = ir1[8];
          ir1[8] = (v267_data + (v248_data * v265_data));
          double v279_data = r0[8];
          double v287_data = s0[22];
          double v289_data = ir1[7];
          ir1[7] = (v289_data + (v279_data * v287_data));
          double v292_data = s0[24];
          double v294_data = ir1[8];
          ir1[8] = (v294_data + (v279_data * v292_data));
          double v297_data = s0[26];
          double v299_data = ir1[9];
          ir1[9] = (v299_data + (v279_data * v297_data));
          double v310_data = r0[9];
          double v319_data = s0[25];
          double v321_data = ir1[8];
          ir1[8] = (v321_data + (v310_data * v319_data));
          double v324_data = s0[27];
          double v326_data = ir1[9];
          ir1[9] = (v326_data + (v310_data * v324_data));
          double v329_data = s0[29];
          double v331_data = ir1[10];
          ir1[10] = (v331_data + (v310_data * v329_data));
          double v341_data = r0[10];
          double v351_data = s0[28];
          double v353_data = ir1[9];
          ir1[9] = (v353_data + (v341_data * v351_data));
          double v356_data = s0[30];
          double v358_data = ir1[10];
          ir1[10] = (v358_data + (v341_data * v356_data));
          double v361_data = s0[32];
          double v363_data = ir1[11];
          ir1[11] = (v363_data + (v341_data * v361_data));
          double v372_data = r0[11];
          double v383_data = s0[31];
          double v385_data = ir1[10];
          ir1[10] = (v385_data + (v372_data * v383_data));
          double v388_data = s0[33];
          double v390_data = ir1[11];
          ir1[11] = (v390_data + (v372_data * v388_data));
          double v393_data = s0[35];
          double v395_data = ir1[12];
          ir1[12] = (v395_data + (v372_data * v393_data));
          double v403_data = r0[12];
          double v415_data = s0[34];
          double v417_data = ir1[11];
          ir1[11] = (v417_data + (v403_data * v415_data));
          double v420_data = s0[36];
          double v422_data = ir1[12];
          ir1[12] = (v422_data + (v403_data * v420_data));
          double v425_data = s0[38];
          double v427_data = ir1[13];
          ir1[13] = (v427_data + (v403_data * v425_data));
          double v434_data = r0[13];
          double v447_data = s0[37];
          double v449_data = ir1[12];
          ir1[12] = (v449_data + (v434_data * v447_data));
          double v452_data = s0[39];
          double v454_data = ir1[13];
          ir1[13] = (v454_data + (v434_data * v452_data));
          double v457_data = s0[41];
          double v459_data = ir1[14];
          ir1[14] = (v459_data + (v434_data * v457_data));
          double v465_data = r0[14];
          double v479_data = s0[40];
          double v481_data = ir1[13];
          ir1[13] = (v481_data + (v465_data * v479_data));
          double v484_data = s0[42];
          double v486_data = ir1[14];
          ir1[14] = (v486_data + (v465_data * v484_data));
          double v489_data = s0[44];
          double v491_data = ir1[15];
          ir1[15] = (v491_data + (v465_data * v489_data));
          double v496_data = r0[15];
          double v511_data = s0[43];
          double v513_data = ir1[14];
          ir1[14] = (v513_data + (v496_data * v511_data));
          double v516_data = s0[45];
          double v518_data = ir1[15];
          ir1[15] = (v518_data + (v496_data * v516_data));
          #pragma unroll
          for (int32_t v523_n0 = 0; v523_n0 < 1; ++v523_n0) {
            #pragma unroll
            for (int32_t v524_n1 = 0; v524_n1 < 16; ++v524_n1) {
              int32_t v525_a = v523_n0 + v524_n1;
              int32_t v526_a = v523_n0 + v524_n1;
              double v527_data = ir1[v526_a];
              r1[v526_a] = v527_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v532_i0 = 0; v532_i0 < 1; ++v532_i0) {
            int32_t v541_lead = v7_lead + (v532_i0 * 16);
            #pragma unroll
            for (int32_t v533_i1 = 0; v533_i1 < 16; ++v533_i1) {
              int32_t v534_a = v532_i0 + v533_i1;
              double v536_data = r1[(v532_i0 + v533_i1)];
              glb_m0[(v541_lead + (v533_i1 * 16))] = v536_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

