// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_69f2bb9311, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_69f2bb9311, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_69f2bb9311<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 35×4(35×4) {0..35}×{0..4} strided
    // m1 35×8(35×8) {0..35}×{0..8} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 8; ++v9_i1) {
              int32_t v15_a = v9_i1 * 35;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __ldcg(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + (v9_i1 * 2))] = v24_data;
            }
          }
          if (v7_lead < 3) {
            int32_t v33_lead = v7_lead + 32_i32;
            int32_t v40_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              int32_t v34_a = v28_i1 * 35;
              int32_t v35_a = v33_lead + v34_a;
              float v43_data = __ldcg(&glb_m1[(v40_lead + v34_a)]);
              r0[(1 + (v28_i1 * 2))] = v43_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float ir1[8]{};
          float v53_data = r0[0];
          float v54_data = s0[0];
          float v56_data = ir1[0];
          ir1[0] = (v56_data + (v53_data * v54_data));
          float v59_data = s0[8];
          float v61_data = ir1[2];
          ir1[2] = (v61_data + (v53_data * v59_data));
          float v64_data = s0[16];
          float v66_data = ir1[4];
          ir1[4] = (v66_data + (v53_data * v64_data));
          float v69_data = s0[24];
          float v71_data = ir1[6];
          ir1[6] = (v71_data + (v53_data * v69_data));
          if (v7_lead < 3) {
            float v74_data = r0[1];
            float v77_data = ir1[1];
            ir1[1] = (v77_data + (v74_data * v54_data));
            float v82_data = ir1[3];
            ir1[3] = (v82_data + (v74_data * v59_data));
            float v87_data = ir1[5];
            ir1[5] = (v87_data + (v74_data * v64_data));
            float v92_data = ir1[7];
            ir1[7] = (v92_data + (v74_data * v69_data));
          }
          float v97_data = r0[2];
          float v98_data = s0[1];
          float v100_data = ir1[0];
          ir1[0] = (v100_data + (v97_data * v98_data));
          float v103_data = s0[9];
          float v105_data = ir1[2];
          ir1[2] = (v105_data + (v97_data * v103_data));
          float v108_data = s0[17];
          float v110_data = ir1[4];
          ir1[4] = (v110_data + (v97_data * v108_data));
          float v113_data = s0[25];
          float v115_data = ir1[6];
          ir1[6] = (v115_data + (v97_data * v113_data));
          if (v7_lead < 3) {
            float v118_data = r0[3];
            float v121_data = ir1[1];
            ir1[1] = (v121_data + (v118_data * v98_data));
            float v126_data = ir1[3];
            ir1[3] = (v126_data + (v118_data * v103_data));
            float v131_data = ir1[5];
            ir1[5] = (v131_data + (v118_data * v108_data));
            float v136_data = ir1[7];
            ir1[7] = (v136_data + (v118_data * v113_data));
          }
          float v141_data = r0[4];
          float v142_data = s0[2];
          float v144_data = ir1[0];
          ir1[0] = (v144_data + (v141_data * v142_data));
          float v147_data = s0[10];
          float v149_data = ir1[2];
          ir1[2] = (v149_data + (v141_data * v147_data));
          float v152_data = s0[18];
          float v154_data = ir1[4];
          ir1[4] = (v154_data + (v141_data * v152_data));
          float v157_data = s0[26];
          float v159_data = ir1[6];
          ir1[6] = (v159_data + (v141_data * v157_data));
          if (v7_lead < 3) {
            float v162_data = r0[5];
            float v165_data = ir1[1];
            ir1[1] = (v165_data + (v162_data * v142_data));
            float v170_data = ir1[3];
            ir1[3] = (v170_data + (v162_data * v147_data));
            float v175_data = ir1[5];
            ir1[5] = (v175_data + (v162_data * v152_data));
            float v180_data = ir1[7];
            ir1[7] = (v180_data + (v162_data * v157_data));
          }
          float v185_data = r0[6];
          float v186_data = s0[3];
          float v188_data = ir1[0];
          ir1[0] = (v188_data + (v185_data * v186_data));
          float v191_data = s0[11];
          float v193_data = ir1[2];
          ir1[2] = (v193_data + (v185_data * v191_data));
          float v196_data = s0[19];
          float v198_data = ir1[4];
          ir1[4] = (v198_data + (v185_data * v196_data));
          float v201_data = s0[27];
          float v203_data = ir1[6];
          ir1[6] = (v203_data + (v185_data * v201_data));
          if (v7_lead < 3) {
            float v206_data = r0[7];
            float v209_data = ir1[1];
            ir1[1] = (v209_data + (v206_data * v186_data));
            float v214_data = ir1[3];
            ir1[3] = (v214_data + (v206_data * v191_data));
            float v219_data = ir1[5];
            ir1[5] = (v219_data + (v206_data * v196_data));
            float v224_data = ir1[7];
            ir1[7] = (v224_data + (v206_data * v201_data));
          }
          float v229_data = r0[8];
          float v230_data = s0[4];
          float v232_data = ir1[0];
          ir1[0] = (v232_data + (v229_data * v230_data));
          float v235_data = s0[12];
          float v237_data = ir1[2];
          ir1[2] = (v237_data + (v229_data * v235_data));
          float v240_data = s0[20];
          float v242_data = ir1[4];
          ir1[4] = (v242_data + (v229_data * v240_data));
          float v245_data = s0[28];
          float v247_data = ir1[6];
          ir1[6] = (v247_data + (v229_data * v245_data));
          if (v7_lead < 3) {
            float v250_data = r0[9];
            float v253_data = ir1[1];
            ir1[1] = (v253_data + (v250_data * v230_data));
            float v258_data = ir1[3];
            ir1[3] = (v258_data + (v250_data * v235_data));
            float v263_data = ir1[5];
            ir1[5] = (v263_data + (v250_data * v240_data));
            float v268_data = ir1[7];
            ir1[7] = (v268_data + (v250_data * v245_data));
          }
          float v273_data = r0[10];
          float v274_data = s0[5];
          float v276_data = ir1[0];
          ir1[0] = (v276_data + (v273_data * v274_data));
          float v279_data = s0[13];
          float v281_data = ir1[2];
          ir1[2] = (v281_data + (v273_data * v279_data));
          float v284_data = s0[21];
          float v286_data = ir1[4];
          ir1[4] = (v286_data + (v273_data * v284_data));
          float v289_data = s0[29];
          float v291_data = ir1[6];
          ir1[6] = (v291_data + (v273_data * v289_data));
          if (v7_lead < 3) {
            float v294_data = r0[11];
            float v297_data = ir1[1];
            ir1[1] = (v297_data + (v294_data * v274_data));
            float v302_data = ir1[3];
            ir1[3] = (v302_data + (v294_data * v279_data));
            float v307_data = ir1[5];
            ir1[5] = (v307_data + (v294_data * v284_data));
            float v312_data = ir1[7];
            ir1[7] = (v312_data + (v294_data * v289_data));
          }
          float v317_data = r0[12];
          float v318_data = s0[6];
          float v320_data = ir1[0];
          ir1[0] = (v320_data + (v317_data * v318_data));
          float v323_data = s0[14];
          float v325_data = ir1[2];
          ir1[2] = (v325_data + (v317_data * v323_data));
          float v328_data = s0[22];
          float v330_data = ir1[4];
          ir1[4] = (v330_data + (v317_data * v328_data));
          float v333_data = s0[30];
          float v335_data = ir1[6];
          ir1[6] = (v335_data + (v317_data * v333_data));
          if (v7_lead < 3) {
            float v338_data = r0[13];
            float v341_data = ir1[1];
            ir1[1] = (v341_data + (v338_data * v318_data));
            float v346_data = ir1[3];
            ir1[3] = (v346_data + (v338_data * v323_data));
            float v351_data = ir1[5];
            ir1[5] = (v351_data + (v338_data * v328_data));
            float v356_data = ir1[7];
            ir1[7] = (v356_data + (v338_data * v333_data));
          }
          float v361_data = r0[14];
          float v362_data = s0[7];
          float v364_data = ir1[0];
          ir1[0] = (v364_data + (v361_data * v362_data));
          float v367_data = s0[15];
          float v369_data = ir1[2];
          ir1[2] = (v369_data + (v361_data * v367_data));
          float v372_data = s0[23];
          float v374_data = ir1[4];
          ir1[4] = (v374_data + (v361_data * v372_data));
          float v377_data = s0[31];
          float v379_data = ir1[6];
          ir1[6] = (v379_data + (v361_data * v377_data));
          if (v7_lead < 3) {
            float v382_data = r0[15];
            float v385_data = ir1[1];
            ir1[1] = (v385_data + (v382_data * v362_data));
            float v390_data = ir1[3];
            ir1[3] = (v390_data + (v382_data * v367_data));
            float v395_data = ir1[5];
            ir1[5] = (v395_data + (v382_data * v372_data));
            float v400_data = ir1[7];
            ir1[7] = (v400_data + (v382_data * v377_data));
          }
          #pragma unroll
          for (int32_t v405_n0 = 0; v405_n0 < 1; ++v405_n0) {
            #pragma unroll
            for (int32_t v406_n1 = 0; v406_n1 < 4; ++v406_n1) {
              int32_t v407_a = v406_n1 * 2;
              int32_t v408_a = v405_n0 + v407_a;
              int32_t v410_a = v405_n0 + v407_a;
              float v411_data = ir1[v410_a];
              r1[v410_a] = v411_data;
            }
          }
          if (v7_lead < 3) {
            #pragma unroll
            for (int32_t v415_n1 = 0; v415_n1 < 4; ++v415_n1) {
              int32_t v416_a = v415_n1 * 2;
              int32_t v417_a = 1 + v416_a;
              int32_t v419_a = 1 + v416_a;
              float v420_data = ir1[v419_a];
              r1[v419_a] = v420_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v426_i0 = 0; v426_i0 < 1; ++v426_i0) {
            int32_t v437_lead = v7_lead + (v426_i0 * 32);
            #pragma unroll
            for (int32_t v427_i1 = 0; v427_i1 < 4; ++v427_i1) {
              int32_t v428_a = v427_i1 * 2;
              int32_t v429_a = v426_i0 + v428_a;
              float v432_data = r1[(v426_i0 + v428_a)];
              glb_m0[(v437_lead + (v427_i1 * 35))] = v432_data;
            }
          }
          if (v7_lead < 3) {
            int32_t v451_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v441_i1 = 0; v441_i1 < 4; ++v441_i1) {
              int32_t v442_a = v441_i1 * 2;
              int32_t v443_a = 1 + v442_a;
              float v446_data = r1[(1 + v442_a)];
              glb_m0[(v451_lead + (v441_i1 * 35))] = v446_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

