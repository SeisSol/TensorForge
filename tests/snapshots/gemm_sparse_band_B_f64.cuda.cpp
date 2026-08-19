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
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 16);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 16);
              double v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<8>(8), pipeline);
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], cuda::aligned_size_t<8>(8), pipeline);
          if (threadIdx.x < 14) {
            cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<8>(8), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          double r1[16]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            double ir1[16]{};
            double v17_data = r0[0];
            double v18_data;
            {
              v18_data = 0.0;
              v18_data = s0[0];
            }
            double v20_data = ir1[0];
            ir1[0] = (v20_data + (v17_data * v18_data));
            double v22_data = r0[0];
            double v23_data;
            {
              v23_data = 0.0;
              v23_data = s0[2];
            }
            double v25_data = ir1[1];
            ir1[1] = (v25_data + (v22_data * v23_data));
            double v30_data = r0[1];
            double v31_data;
            {
              v31_data = 0.0;
              v31_data = s0[1];
            }
            double v33_data = ir1[0];
            ir1[0] = (v33_data + (v30_data * v31_data));
            double v35_data = r0[1];
            double v36_data;
            {
              v36_data = 0.0;
              v36_data = s0[3];
            }
            double v38_data = ir1[1];
            ir1[1] = (v38_data + (v35_data * v36_data));
            double v40_data = r0[1];
            double v41_data;
            {
              v41_data = 0.0;
              v41_data = s0[5];
            }
            double v43_data = ir1[2];
            ir1[2] = (v43_data + (v40_data * v41_data));
            double v48_data = r0[2];
            double v49_data;
            {
              v49_data = 0.0;
              v49_data = s0[4];
            }
            double v51_data = ir1[1];
            ir1[1] = (v51_data + (v48_data * v49_data));
            double v53_data = r0[2];
            double v54_data;
            {
              v54_data = 0.0;
              v54_data = s0[6];
            }
            double v56_data = ir1[2];
            ir1[2] = (v56_data + (v53_data * v54_data));
            double v58_data = r0[2];
            double v59_data;
            {
              v59_data = 0.0;
              v59_data = s0[8];
            }
            double v61_data = ir1[3];
            ir1[3] = (v61_data + (v58_data * v59_data));
            double v66_data = r0[3];
            double v67_data;
            {
              v67_data = 0.0;
              v67_data = s0[7];
            }
            double v69_data = ir1[2];
            ir1[2] = (v69_data + (v66_data * v67_data));
            double v71_data = r0[3];
            double v72_data;
            {
              v72_data = 0.0;
              v72_data = s0[9];
            }
            double v74_data = ir1[3];
            ir1[3] = (v74_data + (v71_data * v72_data));
            double v76_data = r0[3];
            double v77_data;
            {
              v77_data = 0.0;
              v77_data = s0[11];
            }
            double v79_data = ir1[4];
            ir1[4] = (v79_data + (v76_data * v77_data));
            double v84_data = r0[4];
            double v85_data;
            {
              v85_data = 0.0;
              v85_data = s0[10];
            }
            double v87_data = ir1[3];
            ir1[3] = (v87_data + (v84_data * v85_data));
            double v89_data = r0[4];
            double v90_data;
            {
              v90_data = 0.0;
              v90_data = s0[12];
            }
            double v92_data = ir1[4];
            ir1[4] = (v92_data + (v89_data * v90_data));
            double v94_data = r0[4];
            double v95_data;
            {
              v95_data = 0.0;
              v95_data = s0[14];
            }
            double v97_data = ir1[5];
            ir1[5] = (v97_data + (v94_data * v95_data));
            double v102_data = r0[5];
            double v103_data;
            {
              v103_data = 0.0;
              v103_data = s0[13];
            }
            double v105_data = ir1[4];
            ir1[4] = (v105_data + (v102_data * v103_data));
            double v107_data = r0[5];
            double v108_data;
            {
              v108_data = 0.0;
              v108_data = s0[15];
            }
            double v110_data = ir1[5];
            ir1[5] = (v110_data + (v107_data * v108_data));
            double v112_data = r0[5];
            double v113_data;
            {
              v113_data = 0.0;
              v113_data = s0[17];
            }
            double v115_data = ir1[6];
            ir1[6] = (v115_data + (v112_data * v113_data));
            double v120_data = r0[6];
            double v121_data;
            {
              v121_data = 0.0;
              v121_data = s0[16];
            }
            double v123_data = ir1[5];
            ir1[5] = (v123_data + (v120_data * v121_data));
            double v125_data = r0[6];
            double v126_data;
            {
              v126_data = 0.0;
              v126_data = s0[18];
            }
            double v128_data = ir1[6];
            ir1[6] = (v128_data + (v125_data * v126_data));
            double v130_data = r0[6];
            double v131_data;
            {
              v131_data = 0.0;
              v131_data = s0[20];
            }
            double v133_data = ir1[7];
            ir1[7] = (v133_data + (v130_data * v131_data));
            double v138_data = r0[7];
            double v139_data;
            {
              v139_data = 0.0;
              v139_data = s0[19];
            }
            double v141_data = ir1[6];
            ir1[6] = (v141_data + (v138_data * v139_data));
            double v143_data = r0[7];
            double v144_data;
            {
              v144_data = 0.0;
              v144_data = s0[21];
            }
            double v146_data = ir1[7];
            ir1[7] = (v146_data + (v143_data * v144_data));
            double v148_data = r0[7];
            double v149_data;
            {
              v149_data = 0.0;
              v149_data = s0[23];
            }
            double v151_data = ir1[8];
            ir1[8] = (v151_data + (v148_data * v149_data));
            double v156_data = r0[8];
            double v157_data;
            {
              v157_data = 0.0;
              v157_data = s0[22];
            }
            double v159_data = ir1[7];
            ir1[7] = (v159_data + (v156_data * v157_data));
            double v161_data = r0[8];
            double v162_data;
            {
              v162_data = 0.0;
              v162_data = s0[24];
            }
            double v164_data = ir1[8];
            ir1[8] = (v164_data + (v161_data * v162_data));
            double v166_data = r0[8];
            double v167_data;
            {
              v167_data = 0.0;
              v167_data = s0[26];
            }
            double v169_data = ir1[9];
            ir1[9] = (v169_data + (v166_data * v167_data));
            double v174_data = r0[9];
            double v175_data;
            {
              v175_data = 0.0;
              v175_data = s0[25];
            }
            double v177_data = ir1[8];
            ir1[8] = (v177_data + (v174_data * v175_data));
            double v179_data = r0[9];
            double v180_data;
            {
              v180_data = 0.0;
              v180_data = s0[27];
            }
            double v182_data = ir1[9];
            ir1[9] = (v182_data + (v179_data * v180_data));
            double v184_data = r0[9];
            double v185_data;
            {
              v185_data = 0.0;
              v185_data = s0[29];
            }
            double v187_data = ir1[10];
            ir1[10] = (v187_data + (v184_data * v185_data));
            double v192_data = r0[10];
            double v193_data;
            {
              v193_data = 0.0;
              v193_data = s0[28];
            }
            double v195_data = ir1[9];
            ir1[9] = (v195_data + (v192_data * v193_data));
            double v197_data = r0[10];
            double v198_data;
            {
              v198_data = 0.0;
              v198_data = s0[30];
            }
            double v200_data = ir1[10];
            ir1[10] = (v200_data + (v197_data * v198_data));
            double v202_data = r0[10];
            double v203_data;
            {
              v203_data = 0.0;
              v203_data = s0[32];
            }
            double v205_data = ir1[11];
            ir1[11] = (v205_data + (v202_data * v203_data));
            double v210_data = r0[11];
            double v211_data;
            {
              v211_data = 0.0;
              v211_data = s0[31];
            }
            double v213_data = ir1[10];
            ir1[10] = (v213_data + (v210_data * v211_data));
            double v215_data = r0[11];
            double v216_data;
            {
              v216_data = 0.0;
              v216_data = s0[33];
            }
            double v218_data = ir1[11];
            ir1[11] = (v218_data + (v215_data * v216_data));
            double v220_data = r0[11];
            double v221_data;
            {
              v221_data = 0.0;
              v221_data = s0[35];
            }
            double v223_data = ir1[12];
            ir1[12] = (v223_data + (v220_data * v221_data));
            double v228_data = r0[12];
            double v229_data;
            {
              v229_data = 0.0;
              v229_data = s0[34];
            }
            double v231_data = ir1[11];
            ir1[11] = (v231_data + (v228_data * v229_data));
            double v233_data = r0[12];
            double v234_data;
            {
              v234_data = 0.0;
              v234_data = s0[36];
            }
            double v236_data = ir1[12];
            ir1[12] = (v236_data + (v233_data * v234_data));
            double v238_data = r0[12];
            double v239_data;
            {
              v239_data = 0.0;
              v239_data = s0[38];
            }
            double v241_data = ir1[13];
            ir1[13] = (v241_data + (v238_data * v239_data));
            double v246_data = r0[13];
            double v247_data;
            {
              v247_data = 0.0;
              v247_data = s0[37];
            }
            double v249_data = ir1[12];
            ir1[12] = (v249_data + (v246_data * v247_data));
            double v251_data = r0[13];
            double v252_data;
            {
              v252_data = 0.0;
              v252_data = s0[39];
            }
            double v254_data = ir1[13];
            ir1[13] = (v254_data + (v251_data * v252_data));
            double v256_data = r0[13];
            double v257_data;
            {
              v257_data = 0.0;
              v257_data = s0[41];
            }
            double v259_data = ir1[14];
            ir1[14] = (v259_data + (v256_data * v257_data));
            double v264_data = r0[14];
            double v265_data;
            {
              v265_data = 0.0;
              v265_data = s0[40];
            }
            double v267_data = ir1[13];
            ir1[13] = (v267_data + (v264_data * v265_data));
            double v269_data = r0[14];
            double v270_data;
            {
              v270_data = 0.0;
              v270_data = s0[42];
            }
            double v272_data = ir1[14];
            ir1[14] = (v272_data + (v269_data * v270_data));
            double v274_data = r0[14];
            double v275_data;
            {
              v275_data = 0.0;
              v275_data = s0[44];
            }
            double v277_data = ir1[15];
            ir1[15] = (v277_data + (v274_data * v275_data));
            double v282_data = r0[15];
            double v283_data;
            {
              v283_data = 0.0;
              v283_data = s0[43];
            }
            double v285_data = ir1[14];
            ir1[14] = (v285_data + (v282_data * v283_data));
            double v287_data = r0[15];
            double v288_data;
            {
              v288_data = 0.0;
              v288_data = s0[45];
            }
            double v290_data = ir1[15];
            ir1[15] = (v290_data + (v287_data * v288_data));
            #pragma unroll
            for (int32_t v295_n0 = 0; v295_n0 < 1; ++v295_n0) {
              #pragma unroll
              for (int32_t v296_n1 = 0; v296_n1 < 16; ++v296_n1) {
                int32_t v297_a = v295_n0 + v296_n1;
                double v298_data = ir1[v297_a];
                int32_t v299_a = v295_n0 + v296_n1;
                r1[v299_a] = v298_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v302_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v303_i0 = 0; v303_i0 < 1; ++v303_i0) {
            int32_t v311_lead = v302_lead + (v303_i0 * 16);
            #pragma unroll
            for (int32_t v304_i1 = 0; v304_i1 < 16; ++v304_i1) {
              int32_t v305_a = v303_i0 + v304_i1;
              double v306_data = r1[v305_a];
              int32_t v313_a = v311_lead + (v304_i1 * 16);
              glb_m0[v313_a] = v306_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

