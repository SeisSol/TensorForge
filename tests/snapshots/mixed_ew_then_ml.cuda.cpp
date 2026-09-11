// === base name ===
kernel_08eef3093f993dfa

// === header ===
void launcher_kernel_08eef3093f993dfa(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08eef3093f993dfa(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08eef3093f993dfa, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08eef3093f993dfa, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08eef3093f993dfa<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_08eef3093f993dfa(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // TMP = abs(A)
    // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 64 + 0 + m2_extraOffset];
          // s1 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v20_lead = threadIdx.x % 32;
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v22_k1 = 0; v22_k1 < 8; ++v22_k1) {
              float v30_data = glb_m0[(v20_lead + (v22_k1 * 8))];
              r0[v22_k1] = (fabsf(v30_data));
            }
          }
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir1[8]{};
          if (v20_lead < 8) {
            float v39_data = r0[0];
            float v40_data = s1[0];
            float v42_data = ir1[0];
            ir1[0] = (v42_data + (v39_data * v40_data));
            float v45_data = s1[8];
            float v47_data = ir1[1];
            ir1[1] = (v47_data + (v39_data * v45_data));
            float v50_data = s1[16];
            float v52_data = ir1[2];
            ir1[2] = (v52_data + (v39_data * v50_data));
            float v55_data = s1[24];
            float v57_data = ir1[3];
            ir1[3] = (v57_data + (v39_data * v55_data));
            float v60_data = s1[32];
            float v62_data = ir1[4];
            ir1[4] = (v62_data + (v39_data * v60_data));
            float v65_data = s1[40];
            float v67_data = ir1[5];
            ir1[5] = (v67_data + (v39_data * v65_data));
            float v70_data = s1[48];
            float v72_data = ir1[6];
            ir1[6] = (v72_data + (v39_data * v70_data));
            float v75_data = s1[56];
            float v77_data = ir1[7];
            ir1[7] = (v77_data + (v39_data * v75_data));
          }
          if (v20_lead < 8) {
            float v83_data = r0[1];
            float v84_data = s1[1];
            float v86_data = ir1[0];
            ir1[0] = (v86_data + (v83_data * v84_data));
            float v89_data = s1[9];
            float v91_data = ir1[1];
            ir1[1] = (v91_data + (v83_data * v89_data));
            float v94_data = s1[17];
            float v96_data = ir1[2];
            ir1[2] = (v96_data + (v83_data * v94_data));
            float v99_data = s1[25];
            float v101_data = ir1[3];
            ir1[3] = (v101_data + (v83_data * v99_data));
            float v104_data = s1[33];
            float v106_data = ir1[4];
            ir1[4] = (v106_data + (v83_data * v104_data));
            float v109_data = s1[41];
            float v111_data = ir1[5];
            ir1[5] = (v111_data + (v83_data * v109_data));
            float v114_data = s1[49];
            float v116_data = ir1[6];
            ir1[6] = (v116_data + (v83_data * v114_data));
            float v119_data = s1[57];
            float v121_data = ir1[7];
            ir1[7] = (v121_data + (v83_data * v119_data));
          }
          if (v20_lead < 8) {
            float v127_data = r0[2];
            float v128_data = s1[2];
            float v130_data = ir1[0];
            ir1[0] = (v130_data + (v127_data * v128_data));
            float v133_data = s1[10];
            float v135_data = ir1[1];
            ir1[1] = (v135_data + (v127_data * v133_data));
            float v138_data = s1[18];
            float v140_data = ir1[2];
            ir1[2] = (v140_data + (v127_data * v138_data));
            float v143_data = s1[26];
            float v145_data = ir1[3];
            ir1[3] = (v145_data + (v127_data * v143_data));
            float v148_data = s1[34];
            float v150_data = ir1[4];
            ir1[4] = (v150_data + (v127_data * v148_data));
            float v153_data = s1[42];
            float v155_data = ir1[5];
            ir1[5] = (v155_data + (v127_data * v153_data));
            float v158_data = s1[50];
            float v160_data = ir1[6];
            ir1[6] = (v160_data + (v127_data * v158_data));
            float v163_data = s1[58];
            float v165_data = ir1[7];
            ir1[7] = (v165_data + (v127_data * v163_data));
          }
          if (v20_lead < 8) {
            float v171_data = r0[3];
            float v172_data = s1[3];
            float v174_data = ir1[0];
            ir1[0] = (v174_data + (v171_data * v172_data));
            float v177_data = s1[11];
            float v179_data = ir1[1];
            ir1[1] = (v179_data + (v171_data * v177_data));
            float v182_data = s1[19];
            float v184_data = ir1[2];
            ir1[2] = (v184_data + (v171_data * v182_data));
            float v187_data = s1[27];
            float v189_data = ir1[3];
            ir1[3] = (v189_data + (v171_data * v187_data));
            float v192_data = s1[35];
            float v194_data = ir1[4];
            ir1[4] = (v194_data + (v171_data * v192_data));
            float v197_data = s1[43];
            float v199_data = ir1[5];
            ir1[5] = (v199_data + (v171_data * v197_data));
            float v202_data = s1[51];
            float v204_data = ir1[6];
            ir1[6] = (v204_data + (v171_data * v202_data));
            float v207_data = s1[59];
            float v209_data = ir1[7];
            ir1[7] = (v209_data + (v171_data * v207_data));
          }
          if (v20_lead < 8) {
            float v215_data = r0[4];
            float v216_data = s1[4];
            float v218_data = ir1[0];
            ir1[0] = (v218_data + (v215_data * v216_data));
            float v221_data = s1[12];
            float v223_data = ir1[1];
            ir1[1] = (v223_data + (v215_data * v221_data));
            float v226_data = s1[20];
            float v228_data = ir1[2];
            ir1[2] = (v228_data + (v215_data * v226_data));
            float v231_data = s1[28];
            float v233_data = ir1[3];
            ir1[3] = (v233_data + (v215_data * v231_data));
            float v236_data = s1[36];
            float v238_data = ir1[4];
            ir1[4] = (v238_data + (v215_data * v236_data));
            float v241_data = s1[44];
            float v243_data = ir1[5];
            ir1[5] = (v243_data + (v215_data * v241_data));
            float v246_data = s1[52];
            float v248_data = ir1[6];
            ir1[6] = (v248_data + (v215_data * v246_data));
            float v251_data = s1[60];
            float v253_data = ir1[7];
            ir1[7] = (v253_data + (v215_data * v251_data));
          }
          if (v20_lead < 8) {
            float v259_data = r0[5];
            float v260_data = s1[5];
            float v262_data = ir1[0];
            ir1[0] = (v262_data + (v259_data * v260_data));
            float v265_data = s1[13];
            float v267_data = ir1[1];
            ir1[1] = (v267_data + (v259_data * v265_data));
            float v270_data = s1[21];
            float v272_data = ir1[2];
            ir1[2] = (v272_data + (v259_data * v270_data));
            float v275_data = s1[29];
            float v277_data = ir1[3];
            ir1[3] = (v277_data + (v259_data * v275_data));
            float v280_data = s1[37];
            float v282_data = ir1[4];
            ir1[4] = (v282_data + (v259_data * v280_data));
            float v285_data = s1[45];
            float v287_data = ir1[5];
            ir1[5] = (v287_data + (v259_data * v285_data));
            float v290_data = s1[53];
            float v292_data = ir1[6];
            ir1[6] = (v292_data + (v259_data * v290_data));
            float v295_data = s1[61];
            float v297_data = ir1[7];
            ir1[7] = (v297_data + (v259_data * v295_data));
          }
          if (v20_lead < 8) {
            float v303_data = r0[6];
            float v304_data = s1[6];
            float v306_data = ir1[0];
            ir1[0] = (v306_data + (v303_data * v304_data));
            float v309_data = s1[14];
            float v311_data = ir1[1];
            ir1[1] = (v311_data + (v303_data * v309_data));
            float v314_data = s1[22];
            float v316_data = ir1[2];
            ir1[2] = (v316_data + (v303_data * v314_data));
            float v319_data = s1[30];
            float v321_data = ir1[3];
            ir1[3] = (v321_data + (v303_data * v319_data));
            float v324_data = s1[38];
            float v326_data = ir1[4];
            ir1[4] = (v326_data + (v303_data * v324_data));
            float v329_data = s1[46];
            float v331_data = ir1[5];
            ir1[5] = (v331_data + (v303_data * v329_data));
            float v334_data = s1[54];
            float v336_data = ir1[6];
            ir1[6] = (v336_data + (v303_data * v334_data));
            float v339_data = s1[62];
            float v341_data = ir1[7];
            ir1[7] = (v341_data + (v303_data * v339_data));
          }
          if (v20_lead < 8) {
            float v347_data = r0[7];
            float v348_data = s1[7];
            float v350_data = ir1[0];
            ir1[0] = (v350_data + (v347_data * v348_data));
            float v353_data = s1[15];
            float v355_data = ir1[1];
            ir1[1] = (v355_data + (v347_data * v353_data));
            float v358_data = s1[23];
            float v360_data = ir1[2];
            ir1[2] = (v360_data + (v347_data * v358_data));
            float v363_data = s1[31];
            float v365_data = ir1[3];
            ir1[3] = (v365_data + (v347_data * v363_data));
            float v368_data = s1[39];
            float v370_data = ir1[4];
            ir1[4] = (v370_data + (v347_data * v368_data));
            float v373_data = s1[47];
            float v375_data = ir1[5];
            ir1[5] = (v375_data + (v347_data * v373_data));
            float v378_data = s1[55];
            float v380_data = ir1[6];
            ir1[6] = (v380_data + (v347_data * v378_data));
            float v383_data = s1[63];
            float v385_data = ir1[7];
            ir1[7] = (v385_data + (v347_data * v383_data));
          }
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v391_n1 = 0; v391_n1 < 8; ++v391_n1) {
              float v393_data = ir1[v391_n1];
              r1[v391_n1] = v393_data;
            }
          }
          // glb_m1 = store{r>g}(r1);
          if (v20_lead < 8) {
            #pragma unroll
            for (int32_t v399_i1 = 0; v399_i1 < 8; ++v399_i1) {
              float v401_data = r1[v399_i1];
              glb_m1[(v20_lead + (v399_i1 * 8))] = v401_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

