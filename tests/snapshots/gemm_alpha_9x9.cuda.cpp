// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08a27dccde, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08a27dccde, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08a27dccde<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 9×9(9×9) {0..9}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 9×9(9×9) {0..9}×{0..9} strided
    // m3 ()  scalar
    // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 9) {
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 9; ++v8_i1) {
              int32_t v14_a = v8_i1 * 9;
              int32_t v15_a = v6_lead + v14_a;
              float v23_data = __ldcg(&glb_m1[(v6_lead + v14_a)]);
              int32_t v24_a = 0 + v8_i1;
              r0[v24_a] = v23_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
            if (threadIdx.x < 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          if (v6_lead < 9) {
            float v34_data = r0[0];
            float v35_data = s0[0];
            float v37_data = ir1[0];
            ir1[0] = (v37_data + (v34_data * v35_data));
            float v40_data = s0[9];
            float v42_data = ir1[1];
            ir1[1] = (v42_data + (v34_data * v40_data));
            float v45_data = s0[18];
            float v47_data = ir1[2];
            ir1[2] = (v47_data + (v34_data * v45_data));
            float v50_data = s0[27];
            float v52_data = ir1[3];
            ir1[3] = (v52_data + (v34_data * v50_data));
            float v55_data = s0[36];
            float v57_data = ir1[4];
            ir1[4] = (v57_data + (v34_data * v55_data));
            float v60_data = s0[45];
            float v62_data = ir1[5];
            ir1[5] = (v62_data + (v34_data * v60_data));
            float v65_data = s0[54];
            float v67_data = ir1[6];
            ir1[6] = (v67_data + (v34_data * v65_data));
            float v70_data = s0[63];
            float v72_data = ir1[7];
            ir1[7] = (v72_data + (v34_data * v70_data));
            float v75_data = s0[72];
            float v77_data = ir1[8];
            ir1[8] = (v77_data + (v34_data * v75_data));
          }
          if (v6_lead < 9) {
            float v83_data = r0[1];
            float v84_data = s0[1];
            float v86_data = ir1[0];
            ir1[0] = (v86_data + (v83_data * v84_data));
            float v89_data = s0[10];
            float v91_data = ir1[1];
            ir1[1] = (v91_data + (v83_data * v89_data));
            float v94_data = s0[19];
            float v96_data = ir1[2];
            ir1[2] = (v96_data + (v83_data * v94_data));
            float v99_data = s0[28];
            float v101_data = ir1[3];
            ir1[3] = (v101_data + (v83_data * v99_data));
            float v104_data = s0[37];
            float v106_data = ir1[4];
            ir1[4] = (v106_data + (v83_data * v104_data));
            float v109_data = s0[46];
            float v111_data = ir1[5];
            ir1[5] = (v111_data + (v83_data * v109_data));
            float v114_data = s0[55];
            float v116_data = ir1[6];
            ir1[6] = (v116_data + (v83_data * v114_data));
            float v119_data = s0[64];
            float v121_data = ir1[7];
            ir1[7] = (v121_data + (v83_data * v119_data));
            float v124_data = s0[73];
            float v126_data = ir1[8];
            ir1[8] = (v126_data + (v83_data * v124_data));
          }
          if (v6_lead < 9) {
            float v132_data = r0[2];
            float v133_data = s0[2];
            float v135_data = ir1[0];
            ir1[0] = (v135_data + (v132_data * v133_data));
            float v138_data = s0[11];
            float v140_data = ir1[1];
            ir1[1] = (v140_data + (v132_data * v138_data));
            float v143_data = s0[20];
            float v145_data = ir1[2];
            ir1[2] = (v145_data + (v132_data * v143_data));
            float v148_data = s0[29];
            float v150_data = ir1[3];
            ir1[3] = (v150_data + (v132_data * v148_data));
            float v153_data = s0[38];
            float v155_data = ir1[4];
            ir1[4] = (v155_data + (v132_data * v153_data));
            float v158_data = s0[47];
            float v160_data = ir1[5];
            ir1[5] = (v160_data + (v132_data * v158_data));
            float v163_data = s0[56];
            float v165_data = ir1[6];
            ir1[6] = (v165_data + (v132_data * v163_data));
            float v168_data = s0[65];
            float v170_data = ir1[7];
            ir1[7] = (v170_data + (v132_data * v168_data));
            float v173_data = s0[74];
            float v175_data = ir1[8];
            ir1[8] = (v175_data + (v132_data * v173_data));
          }
          if (v6_lead < 9) {
            float v181_data = r0[3];
            float v182_data = s0[3];
            float v184_data = ir1[0];
            ir1[0] = (v184_data + (v181_data * v182_data));
            float v187_data = s0[12];
            float v189_data = ir1[1];
            ir1[1] = (v189_data + (v181_data * v187_data));
            float v192_data = s0[21];
            float v194_data = ir1[2];
            ir1[2] = (v194_data + (v181_data * v192_data));
            float v197_data = s0[30];
            float v199_data = ir1[3];
            ir1[3] = (v199_data + (v181_data * v197_data));
            float v202_data = s0[39];
            float v204_data = ir1[4];
            ir1[4] = (v204_data + (v181_data * v202_data));
            float v207_data = s0[48];
            float v209_data = ir1[5];
            ir1[5] = (v209_data + (v181_data * v207_data));
            float v212_data = s0[57];
            float v214_data = ir1[6];
            ir1[6] = (v214_data + (v181_data * v212_data));
            float v217_data = s0[66];
            float v219_data = ir1[7];
            ir1[7] = (v219_data + (v181_data * v217_data));
            float v222_data = s0[75];
            float v224_data = ir1[8];
            ir1[8] = (v224_data + (v181_data * v222_data));
          }
          if (v6_lead < 9) {
            float v230_data = r0[4];
            float v231_data = s0[4];
            float v233_data = ir1[0];
            ir1[0] = (v233_data + (v230_data * v231_data));
            float v236_data = s0[13];
            float v238_data = ir1[1];
            ir1[1] = (v238_data + (v230_data * v236_data));
            float v241_data = s0[22];
            float v243_data = ir1[2];
            ir1[2] = (v243_data + (v230_data * v241_data));
            float v246_data = s0[31];
            float v248_data = ir1[3];
            ir1[3] = (v248_data + (v230_data * v246_data));
            float v251_data = s0[40];
            float v253_data = ir1[4];
            ir1[4] = (v253_data + (v230_data * v251_data));
            float v256_data = s0[49];
            float v258_data = ir1[5];
            ir1[5] = (v258_data + (v230_data * v256_data));
            float v261_data = s0[58];
            float v263_data = ir1[6];
            ir1[6] = (v263_data + (v230_data * v261_data));
            float v266_data = s0[67];
            float v268_data = ir1[7];
            ir1[7] = (v268_data + (v230_data * v266_data));
            float v271_data = s0[76];
            float v273_data = ir1[8];
            ir1[8] = (v273_data + (v230_data * v271_data));
          }
          if (v6_lead < 9) {
            float v279_data = r0[5];
            float v280_data = s0[5];
            float v282_data = ir1[0];
            ir1[0] = (v282_data + (v279_data * v280_data));
            float v285_data = s0[14];
            float v287_data = ir1[1];
            ir1[1] = (v287_data + (v279_data * v285_data));
            float v290_data = s0[23];
            float v292_data = ir1[2];
            ir1[2] = (v292_data + (v279_data * v290_data));
            float v295_data = s0[32];
            float v297_data = ir1[3];
            ir1[3] = (v297_data + (v279_data * v295_data));
            float v300_data = s0[41];
            float v302_data = ir1[4];
            ir1[4] = (v302_data + (v279_data * v300_data));
            float v305_data = s0[50];
            float v307_data = ir1[5];
            ir1[5] = (v307_data + (v279_data * v305_data));
            float v310_data = s0[59];
            float v312_data = ir1[6];
            ir1[6] = (v312_data + (v279_data * v310_data));
            float v315_data = s0[68];
            float v317_data = ir1[7];
            ir1[7] = (v317_data + (v279_data * v315_data));
            float v320_data = s0[77];
            float v322_data = ir1[8];
            ir1[8] = (v322_data + (v279_data * v320_data));
          }
          if (v6_lead < 9) {
            float v328_data = r0[6];
            float v329_data = s0[6];
            float v331_data = ir1[0];
            ir1[0] = (v331_data + (v328_data * v329_data));
            float v334_data = s0[15];
            float v336_data = ir1[1];
            ir1[1] = (v336_data + (v328_data * v334_data));
            float v339_data = s0[24];
            float v341_data = ir1[2];
            ir1[2] = (v341_data + (v328_data * v339_data));
            float v344_data = s0[33];
            float v346_data = ir1[3];
            ir1[3] = (v346_data + (v328_data * v344_data));
            float v349_data = s0[42];
            float v351_data = ir1[4];
            ir1[4] = (v351_data + (v328_data * v349_data));
            float v354_data = s0[51];
            float v356_data = ir1[5];
            ir1[5] = (v356_data + (v328_data * v354_data));
            float v359_data = s0[60];
            float v361_data = ir1[6];
            ir1[6] = (v361_data + (v328_data * v359_data));
            float v364_data = s0[69];
            float v366_data = ir1[7];
            ir1[7] = (v366_data + (v328_data * v364_data));
            float v369_data = s0[78];
            float v371_data = ir1[8];
            ir1[8] = (v371_data + (v328_data * v369_data));
          }
          if (v6_lead < 9) {
            float v377_data = r0[7];
            float v378_data = s0[7];
            float v380_data = ir1[0];
            ir1[0] = (v380_data + (v377_data * v378_data));
            float v383_data = s0[16];
            float v385_data = ir1[1];
            ir1[1] = (v385_data + (v377_data * v383_data));
            float v388_data = s0[25];
            float v390_data = ir1[2];
            ir1[2] = (v390_data + (v377_data * v388_data));
            float v393_data = s0[34];
            float v395_data = ir1[3];
            ir1[3] = (v395_data + (v377_data * v393_data));
            float v398_data = s0[43];
            float v400_data = ir1[4];
            ir1[4] = (v400_data + (v377_data * v398_data));
            float v403_data = s0[52];
            float v405_data = ir1[5];
            ir1[5] = (v405_data + (v377_data * v403_data));
            float v408_data = s0[61];
            float v410_data = ir1[6];
            ir1[6] = (v410_data + (v377_data * v408_data));
            float v413_data = s0[70];
            float v415_data = ir1[7];
            ir1[7] = (v415_data + (v377_data * v413_data));
            float v418_data = s0[79];
            float v420_data = ir1[8];
            ir1[8] = (v420_data + (v377_data * v418_data));
          }
          if (v6_lead < 9) {
            float v426_data = r0[8];
            float v427_data = s0[8];
            float v429_data = ir1[0];
            ir1[0] = (v429_data + (v426_data * v427_data));
            float v432_data = s0[17];
            float v434_data = ir1[1];
            ir1[1] = (v434_data + (v426_data * v432_data));
            float v437_data = s0[26];
            float v439_data = ir1[2];
            ir1[2] = (v439_data + (v426_data * v437_data));
            float v442_data = s0[35];
            float v444_data = ir1[3];
            ir1[3] = (v444_data + (v426_data * v442_data));
            float v447_data = s0[44];
            float v449_data = ir1[4];
            ir1[4] = (v449_data + (v426_data * v447_data));
            float v452_data = s0[53];
            float v454_data = ir1[5];
            ir1[5] = (v454_data + (v426_data * v452_data));
            float v457_data = s0[62];
            float v459_data = ir1[6];
            ir1[6] = (v459_data + (v426_data * v457_data));
            float v462_data = s0[71];
            float v464_data = ir1[7];
            ir1[7] = (v464_data + (v426_data * v462_data));
            float v467_data = s0[80];
            float v469_data = ir1[8];
            ir1[8] = (v469_data + (v426_data * v467_data));
          }
          if (v6_lead < 9) {
            #pragma unroll
            for (int32_t v476_n1 = 0; v476_n1 < 9; ++v476_n1) {
              int32_t v477_a = 0 + v476_n1;
              float v479_data = ir1[v476_n1];
              r1[v476_n1] = (v479_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v6_lead < 9) {
            #pragma unroll
            for (int32_t v486_i1 = 0; v486_i1 < 9; ++v486_i1) {
              int32_t v487_a = 0 + v486_i1;
              float v489_data = r1[v486_i1];
              glb_m0[(v6_lead + (v486_i1 * 9))] = v489_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

