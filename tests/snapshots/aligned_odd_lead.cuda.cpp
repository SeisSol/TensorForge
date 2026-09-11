// === base name ===
kernel_0bf462e48234fef6

// === header ===
void launcher_kernel_0bf462e48234fef6(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0bf462e48234fef6(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0bf462e48234fef6, block.x * block.y * block.z, 128 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_0bf462e48234fef6, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_0bf462e48234fef6<<<grid,block,128 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_0bf462e48234fef6(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 35×4(35×4) {0..35}×{0..4} strided
    // m1 35×8(35×8) {0..35}×{0..8} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 32);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
              float v28_data = __ldcg(&glb_m1[(v25_lead + (v20_i1 * 35))]);
              r0[(v19_i0 + (v20_i1 * 2))] = v28_data;
            }
          }
          if (v18_lead < 3) {
            int32_t v37_lead = v18_lead + 32_i32;
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
              float v40_data = __ldcg(&glb_m1[(v37_lead + (v32_i1 * 35))]);
              r0[(1 + (v32_i1 * 2))] = v40_data;
            }
          }
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
          float v49_data = r0[0];
          float v50_data = s0[0];
          float v52_data = ir1[0];
          ir1[0] = (v52_data + (v49_data * v50_data));
          float v55_data = s0[8];
          float v57_data = ir1[2];
          ir1[2] = (v57_data + (v49_data * v55_data));
          float v60_data = s0[16];
          float v62_data = ir1[4];
          ir1[4] = (v62_data + (v49_data * v60_data));
          float v65_data = s0[24];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + (v49_data * v65_data));
          if (v18_lead < 3) {
            float v70_data = r0[1];
            float v73_data = ir1[1];
            ir1[1] = (v73_data + (v70_data * v50_data));
            float v78_data = ir1[3];
            ir1[3] = (v78_data + (v70_data * v55_data));
            float v83_data = ir1[5];
            ir1[5] = (v83_data + (v70_data * v60_data));
            float v88_data = ir1[7];
            ir1[7] = (v88_data + (v70_data * v65_data));
          }
          float v93_data = r0[2];
          float v94_data = s0[1];
          float v96_data = ir1[0];
          ir1[0] = (v96_data + (v93_data * v94_data));
          float v99_data = s0[9];
          float v101_data = ir1[2];
          ir1[2] = (v101_data + (v93_data * v99_data));
          float v104_data = s0[17];
          float v106_data = ir1[4];
          ir1[4] = (v106_data + (v93_data * v104_data));
          float v109_data = s0[25];
          float v111_data = ir1[6];
          ir1[6] = (v111_data + (v93_data * v109_data));
          if (v18_lead < 3) {
            float v114_data = r0[3];
            float v117_data = ir1[1];
            ir1[1] = (v117_data + (v114_data * v94_data));
            float v122_data = ir1[3];
            ir1[3] = (v122_data + (v114_data * v99_data));
            float v127_data = ir1[5];
            ir1[5] = (v127_data + (v114_data * v104_data));
            float v132_data = ir1[7];
            ir1[7] = (v132_data + (v114_data * v109_data));
          }
          float v137_data = r0[4];
          float v138_data = s0[2];
          float v140_data = ir1[0];
          ir1[0] = (v140_data + (v137_data * v138_data));
          float v143_data = s0[10];
          float v145_data = ir1[2];
          ir1[2] = (v145_data + (v137_data * v143_data));
          float v148_data = s0[18];
          float v150_data = ir1[4];
          ir1[4] = (v150_data + (v137_data * v148_data));
          float v153_data = s0[26];
          float v155_data = ir1[6];
          ir1[6] = (v155_data + (v137_data * v153_data));
          if (v18_lead < 3) {
            float v158_data = r0[5];
            float v161_data = ir1[1];
            ir1[1] = (v161_data + (v158_data * v138_data));
            float v166_data = ir1[3];
            ir1[3] = (v166_data + (v158_data * v143_data));
            float v171_data = ir1[5];
            ir1[5] = (v171_data + (v158_data * v148_data));
            float v176_data = ir1[7];
            ir1[7] = (v176_data + (v158_data * v153_data));
          }
          float v181_data = r0[6];
          float v182_data = s0[3];
          float v184_data = ir1[0];
          ir1[0] = (v184_data + (v181_data * v182_data));
          float v187_data = s0[11];
          float v189_data = ir1[2];
          ir1[2] = (v189_data + (v181_data * v187_data));
          float v192_data = s0[19];
          float v194_data = ir1[4];
          ir1[4] = (v194_data + (v181_data * v192_data));
          float v197_data = s0[27];
          float v199_data = ir1[6];
          ir1[6] = (v199_data + (v181_data * v197_data));
          if (v18_lead < 3) {
            float v202_data = r0[7];
            float v205_data = ir1[1];
            ir1[1] = (v205_data + (v202_data * v182_data));
            float v210_data = ir1[3];
            ir1[3] = (v210_data + (v202_data * v187_data));
            float v215_data = ir1[5];
            ir1[5] = (v215_data + (v202_data * v192_data));
            float v220_data = ir1[7];
            ir1[7] = (v220_data + (v202_data * v197_data));
          }
          float v225_data = r0[8];
          float v226_data = s0[4];
          float v228_data = ir1[0];
          ir1[0] = (v228_data + (v225_data * v226_data));
          float v231_data = s0[12];
          float v233_data = ir1[2];
          ir1[2] = (v233_data + (v225_data * v231_data));
          float v236_data = s0[20];
          float v238_data = ir1[4];
          ir1[4] = (v238_data + (v225_data * v236_data));
          float v241_data = s0[28];
          float v243_data = ir1[6];
          ir1[6] = (v243_data + (v225_data * v241_data));
          if (v18_lead < 3) {
            float v246_data = r0[9];
            float v249_data = ir1[1];
            ir1[1] = (v249_data + (v246_data * v226_data));
            float v254_data = ir1[3];
            ir1[3] = (v254_data + (v246_data * v231_data));
            float v259_data = ir1[5];
            ir1[5] = (v259_data + (v246_data * v236_data));
            float v264_data = ir1[7];
            ir1[7] = (v264_data + (v246_data * v241_data));
          }
          float v269_data = r0[10];
          float v270_data = s0[5];
          float v272_data = ir1[0];
          ir1[0] = (v272_data + (v269_data * v270_data));
          float v275_data = s0[13];
          float v277_data = ir1[2];
          ir1[2] = (v277_data + (v269_data * v275_data));
          float v280_data = s0[21];
          float v282_data = ir1[4];
          ir1[4] = (v282_data + (v269_data * v280_data));
          float v285_data = s0[29];
          float v287_data = ir1[6];
          ir1[6] = (v287_data + (v269_data * v285_data));
          if (v18_lead < 3) {
            float v290_data = r0[11];
            float v293_data = ir1[1];
            ir1[1] = (v293_data + (v290_data * v270_data));
            float v298_data = ir1[3];
            ir1[3] = (v298_data + (v290_data * v275_data));
            float v303_data = ir1[5];
            ir1[5] = (v303_data + (v290_data * v280_data));
            float v308_data = ir1[7];
            ir1[7] = (v308_data + (v290_data * v285_data));
          }
          float v313_data = r0[12];
          float v314_data = s0[6];
          float v316_data = ir1[0];
          ir1[0] = (v316_data + (v313_data * v314_data));
          float v319_data = s0[14];
          float v321_data = ir1[2];
          ir1[2] = (v321_data + (v313_data * v319_data));
          float v324_data = s0[22];
          float v326_data = ir1[4];
          ir1[4] = (v326_data + (v313_data * v324_data));
          float v329_data = s0[30];
          float v331_data = ir1[6];
          ir1[6] = (v331_data + (v313_data * v329_data));
          if (v18_lead < 3) {
            float v334_data = r0[13];
            float v337_data = ir1[1];
            ir1[1] = (v337_data + (v334_data * v314_data));
            float v342_data = ir1[3];
            ir1[3] = (v342_data + (v334_data * v319_data));
            float v347_data = ir1[5];
            ir1[5] = (v347_data + (v334_data * v324_data));
            float v352_data = ir1[7];
            ir1[7] = (v352_data + (v334_data * v329_data));
          }
          float v357_data = r0[14];
          float v358_data = s0[7];
          float v360_data = ir1[0];
          ir1[0] = (v360_data + (v357_data * v358_data));
          float v363_data = s0[15];
          float v365_data = ir1[2];
          ir1[2] = (v365_data + (v357_data * v363_data));
          float v368_data = s0[23];
          float v370_data = ir1[4];
          ir1[4] = (v370_data + (v357_data * v368_data));
          float v373_data = s0[31];
          float v375_data = ir1[6];
          ir1[6] = (v375_data + (v357_data * v373_data));
          if (v18_lead < 3) {
            float v378_data = r0[15];
            float v381_data = ir1[1];
            ir1[1] = (v381_data + (v378_data * v358_data));
            float v386_data = ir1[3];
            ir1[3] = (v386_data + (v378_data * v363_data));
            float v391_data = ir1[5];
            ir1[5] = (v391_data + (v378_data * v368_data));
            float v396_data = ir1[7];
            ir1[7] = (v396_data + (v378_data * v373_data));
          }
          #pragma unroll
          for (int32_t v401_n0 = 0; v401_n0 < 1; ++v401_n0) {
            #pragma unroll
            for (int32_t v402_n1 = 0; v402_n1 < 4; ++v402_n1) {
              int32_t v404_a = v401_n0 + (v402_n1 * 2);
              float v405_data = ir1[v404_a];
              r1[v404_a] = v405_data;
            }
          }
          if (v18_lead < 3) {
            #pragma unroll
            for (int32_t v409_n1 = 0; v409_n1 < 4; ++v409_n1) {
              int32_t v411_a = 1 + (v409_n1 * 2);
              float v412_data = ir1[v411_a];
              r1[v411_a] = v412_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v418_i0 = 0; v418_i0 < 1; ++v418_i0) {
            int32_t v427_lead = v18_lead + (v418_i0 * 32);
            #pragma unroll
            for (int32_t v419_i1 = 0; v419_i1 < 4; ++v419_i1) {
              float v422_data = r1[(v418_i0 + (v419_i1 * 2))];
              glb_m0[(v427_lead + (v419_i1 * 35))] = v422_data;
            }
          }
          if (v18_lead < 3) {
            int32_t v439_lead = v18_lead + 32_i32;
            #pragma unroll
            for (int32_t v431_i1 = 0; v431_i1 < 4; ++v431_i1) {
              float v434_data = r1[(1 + (v431_i1 * 2))];
              glb_m0[(v439_lead + (v431_i1 * 35))] = v434_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

