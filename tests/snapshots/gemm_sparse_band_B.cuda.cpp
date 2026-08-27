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
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v27_data = r0[0];
          float v28_data = s0[0];
          float v30_data = ir1[0];
          ir1[0] = (v30_data + (v27_data * v28_data));
          float v33_data = s0[2];
          float v35_data = ir1[1];
          ir1[1] = (v35_data + (v27_data * v33_data));
          float v54_data = r0[1];
          float v55_data = s0[1];
          float v57_data = ir1[0];
          ir1[0] = (v57_data + (v54_data * v55_data));
          float v60_data = s0[3];
          float v62_data = ir1[1];
          ir1[1] = (v62_data + (v54_data * v60_data));
          float v65_data = s0[5];
          float v67_data = ir1[2];
          ir1[2] = (v67_data + (v54_data * v65_data));
          float v85_data = r0[2];
          float v87_data = s0[4];
          float v89_data = ir1[1];
          ir1[1] = (v89_data + (v85_data * v87_data));
          float v92_data = s0[6];
          float v94_data = ir1[2];
          ir1[2] = (v94_data + (v85_data * v92_data));
          float v97_data = s0[8];
          float v99_data = ir1[3];
          ir1[3] = (v99_data + (v85_data * v97_data));
          float v116_data = r0[3];
          float v119_data = s0[7];
          float v121_data = ir1[2];
          ir1[2] = (v121_data + (v116_data * v119_data));
          float v124_data = s0[9];
          float v126_data = ir1[3];
          ir1[3] = (v126_data + (v116_data * v124_data));
          float v129_data = s0[11];
          float v131_data = ir1[4];
          ir1[4] = (v131_data + (v116_data * v129_data));
          float v147_data = r0[4];
          float v151_data = s0[10];
          float v153_data = ir1[3];
          ir1[3] = (v153_data + (v147_data * v151_data));
          float v156_data = s0[12];
          float v158_data = ir1[4];
          ir1[4] = (v158_data + (v147_data * v156_data));
          float v161_data = s0[14];
          float v163_data = ir1[5];
          ir1[5] = (v163_data + (v147_data * v161_data));
          float v178_data = r0[5];
          float v183_data = s0[13];
          float v185_data = ir1[4];
          ir1[4] = (v185_data + (v178_data * v183_data));
          float v188_data = s0[15];
          float v190_data = ir1[5];
          ir1[5] = (v190_data + (v178_data * v188_data));
          float v193_data = s0[17];
          float v195_data = ir1[6];
          ir1[6] = (v195_data + (v178_data * v193_data));
          float v209_data = r0[6];
          float v215_data = s0[16];
          float v217_data = ir1[5];
          ir1[5] = (v217_data + (v209_data * v215_data));
          float v220_data = s0[18];
          float v222_data = ir1[6];
          ir1[6] = (v222_data + (v209_data * v220_data));
          float v225_data = s0[20];
          float v227_data = ir1[7];
          ir1[7] = (v227_data + (v209_data * v225_data));
          float v240_data = r0[7];
          float v247_data = s0[19];
          float v249_data = ir1[6];
          ir1[6] = (v249_data + (v240_data * v247_data));
          float v252_data = s0[21];
          float v254_data = ir1[7];
          ir1[7] = (v254_data + (v240_data * v252_data));
          float v257_data = s0[23];
          float v259_data = ir1[8];
          ir1[8] = (v259_data + (v240_data * v257_data));
          float v271_data = r0[8];
          float v279_data = s0[22];
          float v281_data = ir1[7];
          ir1[7] = (v281_data + (v271_data * v279_data));
          float v284_data = s0[24];
          float v286_data = ir1[8];
          ir1[8] = (v286_data + (v271_data * v284_data));
          float v289_data = s0[26];
          float v291_data = ir1[9];
          ir1[9] = (v291_data + (v271_data * v289_data));
          float v302_data = r0[9];
          float v311_data = s0[25];
          float v313_data = ir1[8];
          ir1[8] = (v313_data + (v302_data * v311_data));
          float v316_data = s0[27];
          float v318_data = ir1[9];
          ir1[9] = (v318_data + (v302_data * v316_data));
          float v321_data = s0[29];
          float v323_data = ir1[10];
          ir1[10] = (v323_data + (v302_data * v321_data));
          float v333_data = r0[10];
          float v343_data = s0[28];
          float v345_data = ir1[9];
          ir1[9] = (v345_data + (v333_data * v343_data));
          float v348_data = s0[30];
          float v350_data = ir1[10];
          ir1[10] = (v350_data + (v333_data * v348_data));
          float v353_data = s0[32];
          float v355_data = ir1[11];
          ir1[11] = (v355_data + (v333_data * v353_data));
          float v364_data = r0[11];
          float v375_data = s0[31];
          float v377_data = ir1[10];
          ir1[10] = (v377_data + (v364_data * v375_data));
          float v380_data = s0[33];
          float v382_data = ir1[11];
          ir1[11] = (v382_data + (v364_data * v380_data));
          float v385_data = s0[35];
          float v387_data = ir1[12];
          ir1[12] = (v387_data + (v364_data * v385_data));
          float v395_data = r0[12];
          float v407_data = s0[34];
          float v409_data = ir1[11];
          ir1[11] = (v409_data + (v395_data * v407_data));
          float v412_data = s0[36];
          float v414_data = ir1[12];
          ir1[12] = (v414_data + (v395_data * v412_data));
          float v417_data = s0[38];
          float v419_data = ir1[13];
          ir1[13] = (v419_data + (v395_data * v417_data));
          float v426_data = r0[13];
          float v439_data = s0[37];
          float v441_data = ir1[12];
          ir1[12] = (v441_data + (v426_data * v439_data));
          float v444_data = s0[39];
          float v446_data = ir1[13];
          ir1[13] = (v446_data + (v426_data * v444_data));
          float v449_data = s0[41];
          float v451_data = ir1[14];
          ir1[14] = (v451_data + (v426_data * v449_data));
          float v457_data = r0[14];
          float v471_data = s0[40];
          float v473_data = ir1[13];
          ir1[13] = (v473_data + (v457_data * v471_data));
          float v476_data = s0[42];
          float v478_data = ir1[14];
          ir1[14] = (v478_data + (v457_data * v476_data));
          float v481_data = s0[44];
          float v483_data = ir1[15];
          ir1[15] = (v483_data + (v457_data * v481_data));
          float v488_data = r0[15];
          float v503_data = s0[43];
          float v505_data = ir1[14];
          ir1[14] = (v505_data + (v488_data * v503_data));
          float v508_data = s0[45];
          float v510_data = ir1[15];
          ir1[15] = (v510_data + (v488_data * v508_data));
          #pragma unroll
          for (int32_t v515_n0 = 0; v515_n0 < 1; ++v515_n0) {
            #pragma unroll
            for (int32_t v516_n1 = 0; v516_n1 < 16; ++v516_n1) {
              int32_t v517_a = v515_n0 + v516_n1;
              int32_t v518_a = v515_n0 + v516_n1;
              float v519_data = ir1[v518_a];
              int32_t v520_a = v515_n0 + v516_n1;
              r1[v518_a] = v519_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v525_i0 = 0; v525_i0 < 1; ++v525_i0) {
            int32_t v534_lead = v3_lead + (v525_i0 * 16);
            #pragma unroll
            for (int32_t v526_i1 = 0; v526_i1 < 16; ++v526_i1) {
              int32_t v527_a = v525_i0 + v526_i1;
              float v529_data = r1[(v525_i0 + v526_i1)];
              int32_t v536_a = v534_lead + (v526_i1 * 16);
              glb_m0[v536_a] = v529_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

