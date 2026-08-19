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
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 16);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 16);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
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
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir1[16]{};
            float v17_data = r0[0];
            float v18_data;
            {
              v18_data = 0.0f;
              v18_data = s0[0];
            }
            float v20_data = ir1[0];
            ir1[0] = (v20_data + (v17_data * v18_data));
            float v22_data = r0[0];
            float v23_data;
            {
              v23_data = 0.0f;
              v23_data = s0[2];
            }
            float v25_data = ir1[1];
            ir1[1] = (v25_data + (v22_data * v23_data));
            float v30_data = r0[1];
            float v31_data;
            {
              v31_data = 0.0f;
              v31_data = s0[1];
            }
            float v33_data = ir1[0];
            ir1[0] = (v33_data + (v30_data * v31_data));
            float v35_data = r0[1];
            float v36_data;
            {
              v36_data = 0.0f;
              v36_data = s0[3];
            }
            float v38_data = ir1[1];
            ir1[1] = (v38_data + (v35_data * v36_data));
            float v40_data = r0[1];
            float v41_data;
            {
              v41_data = 0.0f;
              v41_data = s0[5];
            }
            float v43_data = ir1[2];
            ir1[2] = (v43_data + (v40_data * v41_data));
            float v48_data = r0[2];
            float v49_data;
            {
              v49_data = 0.0f;
              v49_data = s0[4];
            }
            float v51_data = ir1[1];
            ir1[1] = (v51_data + (v48_data * v49_data));
            float v53_data = r0[2];
            float v54_data;
            {
              v54_data = 0.0f;
              v54_data = s0[6];
            }
            float v56_data = ir1[2];
            ir1[2] = (v56_data + (v53_data * v54_data));
            float v58_data = r0[2];
            float v59_data;
            {
              v59_data = 0.0f;
              v59_data = s0[8];
            }
            float v61_data = ir1[3];
            ir1[3] = (v61_data + (v58_data * v59_data));
            float v66_data = r0[3];
            float v67_data;
            {
              v67_data = 0.0f;
              v67_data = s0[7];
            }
            float v69_data = ir1[2];
            ir1[2] = (v69_data + (v66_data * v67_data));
            float v71_data = r0[3];
            float v72_data;
            {
              v72_data = 0.0f;
              v72_data = s0[9];
            }
            float v74_data = ir1[3];
            ir1[3] = (v74_data + (v71_data * v72_data));
            float v76_data = r0[3];
            float v77_data;
            {
              v77_data = 0.0f;
              v77_data = s0[11];
            }
            float v79_data = ir1[4];
            ir1[4] = (v79_data + (v76_data * v77_data));
            float v84_data = r0[4];
            float v85_data;
            {
              v85_data = 0.0f;
              v85_data = s0[10];
            }
            float v87_data = ir1[3];
            ir1[3] = (v87_data + (v84_data * v85_data));
            float v89_data = r0[4];
            float v90_data;
            {
              v90_data = 0.0f;
              v90_data = s0[12];
            }
            float v92_data = ir1[4];
            ir1[4] = (v92_data + (v89_data * v90_data));
            float v94_data = r0[4];
            float v95_data;
            {
              v95_data = 0.0f;
              v95_data = s0[14];
            }
            float v97_data = ir1[5];
            ir1[5] = (v97_data + (v94_data * v95_data));
            float v102_data = r0[5];
            float v103_data;
            {
              v103_data = 0.0f;
              v103_data = s0[13];
            }
            float v105_data = ir1[4];
            ir1[4] = (v105_data + (v102_data * v103_data));
            float v107_data = r0[5];
            float v108_data;
            {
              v108_data = 0.0f;
              v108_data = s0[15];
            }
            float v110_data = ir1[5];
            ir1[5] = (v110_data + (v107_data * v108_data));
            float v112_data = r0[5];
            float v113_data;
            {
              v113_data = 0.0f;
              v113_data = s0[17];
            }
            float v115_data = ir1[6];
            ir1[6] = (v115_data + (v112_data * v113_data));
            float v120_data = r0[6];
            float v121_data;
            {
              v121_data = 0.0f;
              v121_data = s0[16];
            }
            float v123_data = ir1[5];
            ir1[5] = (v123_data + (v120_data * v121_data));
            float v125_data = r0[6];
            float v126_data;
            {
              v126_data = 0.0f;
              v126_data = s0[18];
            }
            float v128_data = ir1[6];
            ir1[6] = (v128_data + (v125_data * v126_data));
            float v130_data = r0[6];
            float v131_data;
            {
              v131_data = 0.0f;
              v131_data = s0[20];
            }
            float v133_data = ir1[7];
            ir1[7] = (v133_data + (v130_data * v131_data));
            float v138_data = r0[7];
            float v139_data;
            {
              v139_data = 0.0f;
              v139_data = s0[19];
            }
            float v141_data = ir1[6];
            ir1[6] = (v141_data + (v138_data * v139_data));
            float v143_data = r0[7];
            float v144_data;
            {
              v144_data = 0.0f;
              v144_data = s0[21];
            }
            float v146_data = ir1[7];
            ir1[7] = (v146_data + (v143_data * v144_data));
            float v148_data = r0[7];
            float v149_data;
            {
              v149_data = 0.0f;
              v149_data = s0[23];
            }
            float v151_data = ir1[8];
            ir1[8] = (v151_data + (v148_data * v149_data));
            float v156_data = r0[8];
            float v157_data;
            {
              v157_data = 0.0f;
              v157_data = s0[22];
            }
            float v159_data = ir1[7];
            ir1[7] = (v159_data + (v156_data * v157_data));
            float v161_data = r0[8];
            float v162_data;
            {
              v162_data = 0.0f;
              v162_data = s0[24];
            }
            float v164_data = ir1[8];
            ir1[8] = (v164_data + (v161_data * v162_data));
            float v166_data = r0[8];
            float v167_data;
            {
              v167_data = 0.0f;
              v167_data = s0[26];
            }
            float v169_data = ir1[9];
            ir1[9] = (v169_data + (v166_data * v167_data));
            float v174_data = r0[9];
            float v175_data;
            {
              v175_data = 0.0f;
              v175_data = s0[25];
            }
            float v177_data = ir1[8];
            ir1[8] = (v177_data + (v174_data * v175_data));
            float v179_data = r0[9];
            float v180_data;
            {
              v180_data = 0.0f;
              v180_data = s0[27];
            }
            float v182_data = ir1[9];
            ir1[9] = (v182_data + (v179_data * v180_data));
            float v184_data = r0[9];
            float v185_data;
            {
              v185_data = 0.0f;
              v185_data = s0[29];
            }
            float v187_data = ir1[10];
            ir1[10] = (v187_data + (v184_data * v185_data));
            float v192_data = r0[10];
            float v193_data;
            {
              v193_data = 0.0f;
              v193_data = s0[28];
            }
            float v195_data = ir1[9];
            ir1[9] = (v195_data + (v192_data * v193_data));
            float v197_data = r0[10];
            float v198_data;
            {
              v198_data = 0.0f;
              v198_data = s0[30];
            }
            float v200_data = ir1[10];
            ir1[10] = (v200_data + (v197_data * v198_data));
            float v202_data = r0[10];
            float v203_data;
            {
              v203_data = 0.0f;
              v203_data = s0[32];
            }
            float v205_data = ir1[11];
            ir1[11] = (v205_data + (v202_data * v203_data));
            float v210_data = r0[11];
            float v211_data;
            {
              v211_data = 0.0f;
              v211_data = s0[31];
            }
            float v213_data = ir1[10];
            ir1[10] = (v213_data + (v210_data * v211_data));
            float v215_data = r0[11];
            float v216_data;
            {
              v216_data = 0.0f;
              v216_data = s0[33];
            }
            float v218_data = ir1[11];
            ir1[11] = (v218_data + (v215_data * v216_data));
            float v220_data = r0[11];
            float v221_data;
            {
              v221_data = 0.0f;
              v221_data = s0[35];
            }
            float v223_data = ir1[12];
            ir1[12] = (v223_data + (v220_data * v221_data));
            float v228_data = r0[12];
            float v229_data;
            {
              v229_data = 0.0f;
              v229_data = s0[34];
            }
            float v231_data = ir1[11];
            ir1[11] = (v231_data + (v228_data * v229_data));
            float v233_data = r0[12];
            float v234_data;
            {
              v234_data = 0.0f;
              v234_data = s0[36];
            }
            float v236_data = ir1[12];
            ir1[12] = (v236_data + (v233_data * v234_data));
            float v238_data = r0[12];
            float v239_data;
            {
              v239_data = 0.0f;
              v239_data = s0[38];
            }
            float v241_data = ir1[13];
            ir1[13] = (v241_data + (v238_data * v239_data));
            float v246_data = r0[13];
            float v247_data;
            {
              v247_data = 0.0f;
              v247_data = s0[37];
            }
            float v249_data = ir1[12];
            ir1[12] = (v249_data + (v246_data * v247_data));
            float v251_data = r0[13];
            float v252_data;
            {
              v252_data = 0.0f;
              v252_data = s0[39];
            }
            float v254_data = ir1[13];
            ir1[13] = (v254_data + (v251_data * v252_data));
            float v256_data = r0[13];
            float v257_data;
            {
              v257_data = 0.0f;
              v257_data = s0[41];
            }
            float v259_data = ir1[14];
            ir1[14] = (v259_data + (v256_data * v257_data));
            float v264_data = r0[14];
            float v265_data;
            {
              v265_data = 0.0f;
              v265_data = s0[40];
            }
            float v267_data = ir1[13];
            ir1[13] = (v267_data + (v264_data * v265_data));
            float v269_data = r0[14];
            float v270_data;
            {
              v270_data = 0.0f;
              v270_data = s0[42];
            }
            float v272_data = ir1[14];
            ir1[14] = (v272_data + (v269_data * v270_data));
            float v274_data = r0[14];
            float v275_data;
            {
              v275_data = 0.0f;
              v275_data = s0[44];
            }
            float v277_data = ir1[15];
            ir1[15] = (v277_data + (v274_data * v275_data));
            float v282_data = r0[15];
            float v283_data;
            {
              v283_data = 0.0f;
              v283_data = s0[43];
            }
            float v285_data = ir1[14];
            ir1[14] = (v285_data + (v282_data * v283_data));
            float v287_data = r0[15];
            float v288_data;
            {
              v288_data = 0.0f;
              v288_data = s0[45];
            }
            float v290_data = ir1[15];
            ir1[15] = (v290_data + (v287_data * v288_data));
            #pragma unroll
            for (int32_t v295_n0 = 0; v295_n0 < 1; ++v295_n0) {
              #pragma unroll
              for (int32_t v296_n1 = 0; v296_n1 < 16; ++v296_n1) {
                int32_t v297_a = v295_n0 + v296_n1;
                int32_t v298_a = v295_n0 + v296_n1;
                float v299_data = ir1[v298_a];
                int32_t v300_a = v295_n0 + v296_n1;
                r1[v298_a] = v299_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v304_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v305_i0 = 0; v305_i0 < 1; ++v305_i0) {
            int32_t v314_lead = v304_lead + (v305_i0 * 16);
            #pragma unroll
            for (int32_t v306_i1 = 0; v306_i1 < 16; ++v306_i1) {
              int32_t v307_a = v305_i0 + v306_i1;
              float v309_data = r1[(v305_i0 + v306_i1)];
              int32_t v316_a = v314_lead + (v306_i1 * 16);
              glb_m0[v316_a] = v309_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

