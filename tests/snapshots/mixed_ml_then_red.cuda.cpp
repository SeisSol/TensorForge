// === base name ===
kernel_4b748443ff

// === header ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4b748443ff, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_4b748443ff, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_4b748443ff<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8(8) {0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // OUT = +(TMP, dims=[1])
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 8 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 32;
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 8; ++v17_i1) {
              float v25_data = __ldcg(&glb_m0[(v15_lead + (v17_i1 * 8))]);
              r0[v17_i1] = v25_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          if (v15_lead < 8) {
            float v34_data = r0[0];
            float v35_data = s0[0];
            float v37_data = r1[0];
            r1[0] = (v37_data + (v34_data * v35_data));
            float v40_data = s0[8];
            float v42_data = r1[1];
            r1[1] = (v42_data + (v34_data * v40_data));
            float v45_data = s0[16];
            float v47_data = r1[2];
            r1[2] = (v47_data + (v34_data * v45_data));
            float v50_data = s0[24];
            float v52_data = r1[3];
            r1[3] = (v52_data + (v34_data * v50_data));
            float v55_data = s0[32];
            float v57_data = r1[4];
            r1[4] = (v57_data + (v34_data * v55_data));
            float v60_data = s0[40];
            float v62_data = r1[5];
            r1[5] = (v62_data + (v34_data * v60_data));
            float v65_data = s0[48];
            float v67_data = r1[6];
            r1[6] = (v67_data + (v34_data * v65_data));
            float v70_data = s0[56];
            float v72_data = r1[7];
            r1[7] = (v72_data + (v34_data * v70_data));
          }
          if (v15_lead < 8) {
            float v78_data = r0[1];
            float v79_data = s0[1];
            float v81_data = r1[0];
            r1[0] = (v81_data + (v78_data * v79_data));
            float v84_data = s0[9];
            float v86_data = r1[1];
            r1[1] = (v86_data + (v78_data * v84_data));
            float v89_data = s0[17];
            float v91_data = r1[2];
            r1[2] = (v91_data + (v78_data * v89_data));
            float v94_data = s0[25];
            float v96_data = r1[3];
            r1[3] = (v96_data + (v78_data * v94_data));
            float v99_data = s0[33];
            float v101_data = r1[4];
            r1[4] = (v101_data + (v78_data * v99_data));
            float v104_data = s0[41];
            float v106_data = r1[5];
            r1[5] = (v106_data + (v78_data * v104_data));
            float v109_data = s0[49];
            float v111_data = r1[6];
            r1[6] = (v111_data + (v78_data * v109_data));
            float v114_data = s0[57];
            float v116_data = r1[7];
            r1[7] = (v116_data + (v78_data * v114_data));
          }
          if (v15_lead < 8) {
            float v122_data = r0[2];
            float v123_data = s0[2];
            float v125_data = r1[0];
            r1[0] = (v125_data + (v122_data * v123_data));
            float v128_data = s0[10];
            float v130_data = r1[1];
            r1[1] = (v130_data + (v122_data * v128_data));
            float v133_data = s0[18];
            float v135_data = r1[2];
            r1[2] = (v135_data + (v122_data * v133_data));
            float v138_data = s0[26];
            float v140_data = r1[3];
            r1[3] = (v140_data + (v122_data * v138_data));
            float v143_data = s0[34];
            float v145_data = r1[4];
            r1[4] = (v145_data + (v122_data * v143_data));
            float v148_data = s0[42];
            float v150_data = r1[5];
            r1[5] = (v150_data + (v122_data * v148_data));
            float v153_data = s0[50];
            float v155_data = r1[6];
            r1[6] = (v155_data + (v122_data * v153_data));
            float v158_data = s0[58];
            float v160_data = r1[7];
            r1[7] = (v160_data + (v122_data * v158_data));
          }
          if (v15_lead < 8) {
            float v166_data = r0[3];
            float v167_data = s0[3];
            float v169_data = r1[0];
            r1[0] = (v169_data + (v166_data * v167_data));
            float v172_data = s0[11];
            float v174_data = r1[1];
            r1[1] = (v174_data + (v166_data * v172_data));
            float v177_data = s0[19];
            float v179_data = r1[2];
            r1[2] = (v179_data + (v166_data * v177_data));
            float v182_data = s0[27];
            float v184_data = r1[3];
            r1[3] = (v184_data + (v166_data * v182_data));
            float v187_data = s0[35];
            float v189_data = r1[4];
            r1[4] = (v189_data + (v166_data * v187_data));
            float v192_data = s0[43];
            float v194_data = r1[5];
            r1[5] = (v194_data + (v166_data * v192_data));
            float v197_data = s0[51];
            float v199_data = r1[6];
            r1[6] = (v199_data + (v166_data * v197_data));
            float v202_data = s0[59];
            float v204_data = r1[7];
            r1[7] = (v204_data + (v166_data * v202_data));
          }
          if (v15_lead < 8) {
            float v210_data = r0[4];
            float v211_data = s0[4];
            float v213_data = r1[0];
            r1[0] = (v213_data + (v210_data * v211_data));
            float v216_data = s0[12];
            float v218_data = r1[1];
            r1[1] = (v218_data + (v210_data * v216_data));
            float v221_data = s0[20];
            float v223_data = r1[2];
            r1[2] = (v223_data + (v210_data * v221_data));
            float v226_data = s0[28];
            float v228_data = r1[3];
            r1[3] = (v228_data + (v210_data * v226_data));
            float v231_data = s0[36];
            float v233_data = r1[4];
            r1[4] = (v233_data + (v210_data * v231_data));
            float v236_data = s0[44];
            float v238_data = r1[5];
            r1[5] = (v238_data + (v210_data * v236_data));
            float v241_data = s0[52];
            float v243_data = r1[6];
            r1[6] = (v243_data + (v210_data * v241_data));
            float v246_data = s0[60];
            float v248_data = r1[7];
            r1[7] = (v248_data + (v210_data * v246_data));
          }
          if (v15_lead < 8) {
            float v254_data = r0[5];
            float v255_data = s0[5];
            float v257_data = r1[0];
            r1[0] = (v257_data + (v254_data * v255_data));
            float v260_data = s0[13];
            float v262_data = r1[1];
            r1[1] = (v262_data + (v254_data * v260_data));
            float v265_data = s0[21];
            float v267_data = r1[2];
            r1[2] = (v267_data + (v254_data * v265_data));
            float v270_data = s0[29];
            float v272_data = r1[3];
            r1[3] = (v272_data + (v254_data * v270_data));
            float v275_data = s0[37];
            float v277_data = r1[4];
            r1[4] = (v277_data + (v254_data * v275_data));
            float v280_data = s0[45];
            float v282_data = r1[5];
            r1[5] = (v282_data + (v254_data * v280_data));
            float v285_data = s0[53];
            float v287_data = r1[6];
            r1[6] = (v287_data + (v254_data * v285_data));
            float v290_data = s0[61];
            float v292_data = r1[7];
            r1[7] = (v292_data + (v254_data * v290_data));
          }
          if (v15_lead < 8) {
            float v298_data = r0[6];
            float v299_data = s0[6];
            float v301_data = r1[0];
            r1[0] = (v301_data + (v298_data * v299_data));
            float v304_data = s0[14];
            float v306_data = r1[1];
            r1[1] = (v306_data + (v298_data * v304_data));
            float v309_data = s0[22];
            float v311_data = r1[2];
            r1[2] = (v311_data + (v298_data * v309_data));
            float v314_data = s0[30];
            float v316_data = r1[3];
            r1[3] = (v316_data + (v298_data * v314_data));
            float v319_data = s0[38];
            float v321_data = r1[4];
            r1[4] = (v321_data + (v298_data * v319_data));
            float v324_data = s0[46];
            float v326_data = r1[5];
            r1[5] = (v326_data + (v298_data * v324_data));
            float v329_data = s0[54];
            float v331_data = r1[6];
            r1[6] = (v331_data + (v298_data * v329_data));
            float v334_data = s0[62];
            float v336_data = r1[7];
            r1[7] = (v336_data + (v298_data * v334_data));
          }
          if (v15_lead < 8) {
            float v342_data = r0[7];
            float v343_data = s0[7];
            float v345_data = r1[0];
            r1[0] = (v345_data + (v342_data * v343_data));
            float v348_data = s0[15];
            float v350_data = r1[1];
            r1[1] = (v350_data + (v342_data * v348_data));
            float v353_data = s0[23];
            float v355_data = r1[2];
            r1[2] = (v355_data + (v342_data * v353_data));
            float v358_data = s0[31];
            float v360_data = r1[3];
            r1[3] = (v360_data + (v342_data * v358_data));
            float v363_data = s0[39];
            float v365_data = r1[4];
            r1[4] = (v365_data + (v342_data * v363_data));
            float v368_data = s0[47];
            float v370_data = r1[5];
            r1[5] = (v370_data + (v342_data * v368_data));
            float v373_data = s0[55];
            float v375_data = r1[6];
            r1[6] = (v375_data + (v342_data * v373_data));
            float v378_data = s0[63];
            float v380_data = r1[7];
            r1[7] = (v380_data + (v342_data * v378_data));
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v386_i1 = 0; v386_i1 < 8; ++v386_i1) {
              float v388_data = r1[v386_i1];
              int32_t v395_a = v15_lead + (v386_i1 * 8);
              s1[(v395_a ^ ((v395_a >> 5) & 31))] = v388_data;
            }
          }
          __syncwarp();
          // glb_m2 = +(s1, dims=[1])
          if (v15_lead < 8) {
            float v404_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v403_r1 = 0; v403_r1 < 8; ++v403_r1) {
              int32_t v411_a = v15_lead + (v403_r1 * 8);
              float v415_data = s1[(v411_a ^ ((v411_a >> 5) & 31))];
              v404_acc0 = (v404_acc0 + v415_data);
            }
            glb_m2[v15_lead] = v404_acc0;
          }
          __syncwarp();
        }
      }
    }
  }
}

