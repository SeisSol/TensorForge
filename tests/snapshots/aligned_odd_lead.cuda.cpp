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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 35))]);
              r0[(v11_i0 + (v12_i1 * 2))] = v20_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v29_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v24_i1 * 35))]);
              r0[(1 + (v24_i1 * 2))] = v32_data;
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
          float v42_data = r0[0];
          float v43_data = s0[0];
          float v45_data = ir1[0];
          ir1[0] = (v45_data + (v42_data * v43_data));
          float v48_data = s0[8];
          float v50_data = ir1[2];
          ir1[2] = (v50_data + (v42_data * v48_data));
          float v53_data = s0[16];
          float v55_data = ir1[4];
          ir1[4] = (v55_data + (v42_data * v53_data));
          float v58_data = s0[24];
          float v60_data = ir1[6];
          ir1[6] = (v60_data + (v42_data * v58_data));
          if (v10_lead < 3) {
            float v63_data = r0[1];
            float v66_data = ir1[1];
            ir1[1] = (v66_data + (v63_data * v43_data));
            float v71_data = ir1[3];
            ir1[3] = (v71_data + (v63_data * v48_data));
            float v76_data = ir1[5];
            ir1[5] = (v76_data + (v63_data * v53_data));
            float v81_data = ir1[7];
            ir1[7] = (v81_data + (v63_data * v58_data));
          }
          float v86_data = r0[2];
          float v87_data = s0[1];
          float v89_data = ir1[0];
          ir1[0] = (v89_data + (v86_data * v87_data));
          float v92_data = s0[9];
          float v94_data = ir1[2];
          ir1[2] = (v94_data + (v86_data * v92_data));
          float v97_data = s0[17];
          float v99_data = ir1[4];
          ir1[4] = (v99_data + (v86_data * v97_data));
          float v102_data = s0[25];
          float v104_data = ir1[6];
          ir1[6] = (v104_data + (v86_data * v102_data));
          if (v10_lead < 3) {
            float v107_data = r0[3];
            float v110_data = ir1[1];
            ir1[1] = (v110_data + (v107_data * v87_data));
            float v115_data = ir1[3];
            ir1[3] = (v115_data + (v107_data * v92_data));
            float v120_data = ir1[5];
            ir1[5] = (v120_data + (v107_data * v97_data));
            float v125_data = ir1[7];
            ir1[7] = (v125_data + (v107_data * v102_data));
          }
          float v130_data = r0[4];
          float v131_data = s0[2];
          float v133_data = ir1[0];
          ir1[0] = (v133_data + (v130_data * v131_data));
          float v136_data = s0[10];
          float v138_data = ir1[2];
          ir1[2] = (v138_data + (v130_data * v136_data));
          float v141_data = s0[18];
          float v143_data = ir1[4];
          ir1[4] = (v143_data + (v130_data * v141_data));
          float v146_data = s0[26];
          float v148_data = ir1[6];
          ir1[6] = (v148_data + (v130_data * v146_data));
          if (v10_lead < 3) {
            float v151_data = r0[5];
            float v154_data = ir1[1];
            ir1[1] = (v154_data + (v151_data * v131_data));
            float v159_data = ir1[3];
            ir1[3] = (v159_data + (v151_data * v136_data));
            float v164_data = ir1[5];
            ir1[5] = (v164_data + (v151_data * v141_data));
            float v169_data = ir1[7];
            ir1[7] = (v169_data + (v151_data * v146_data));
          }
          float v174_data = r0[6];
          float v175_data = s0[3];
          float v177_data = ir1[0];
          ir1[0] = (v177_data + (v174_data * v175_data));
          float v180_data = s0[11];
          float v182_data = ir1[2];
          ir1[2] = (v182_data + (v174_data * v180_data));
          float v185_data = s0[19];
          float v187_data = ir1[4];
          ir1[4] = (v187_data + (v174_data * v185_data));
          float v190_data = s0[27];
          float v192_data = ir1[6];
          ir1[6] = (v192_data + (v174_data * v190_data));
          if (v10_lead < 3) {
            float v195_data = r0[7];
            float v198_data = ir1[1];
            ir1[1] = (v198_data + (v195_data * v175_data));
            float v203_data = ir1[3];
            ir1[3] = (v203_data + (v195_data * v180_data));
            float v208_data = ir1[5];
            ir1[5] = (v208_data + (v195_data * v185_data));
            float v213_data = ir1[7];
            ir1[7] = (v213_data + (v195_data * v190_data));
          }
          float v218_data = r0[8];
          float v219_data = s0[4];
          float v221_data = ir1[0];
          ir1[0] = (v221_data + (v218_data * v219_data));
          float v224_data = s0[12];
          float v226_data = ir1[2];
          ir1[2] = (v226_data + (v218_data * v224_data));
          float v229_data = s0[20];
          float v231_data = ir1[4];
          ir1[4] = (v231_data + (v218_data * v229_data));
          float v234_data = s0[28];
          float v236_data = ir1[6];
          ir1[6] = (v236_data + (v218_data * v234_data));
          if (v10_lead < 3) {
            float v239_data = r0[9];
            float v242_data = ir1[1];
            ir1[1] = (v242_data + (v239_data * v219_data));
            float v247_data = ir1[3];
            ir1[3] = (v247_data + (v239_data * v224_data));
            float v252_data = ir1[5];
            ir1[5] = (v252_data + (v239_data * v229_data));
            float v257_data = ir1[7];
            ir1[7] = (v257_data + (v239_data * v234_data));
          }
          float v262_data = r0[10];
          float v263_data = s0[5];
          float v265_data = ir1[0];
          ir1[0] = (v265_data + (v262_data * v263_data));
          float v268_data = s0[13];
          float v270_data = ir1[2];
          ir1[2] = (v270_data + (v262_data * v268_data));
          float v273_data = s0[21];
          float v275_data = ir1[4];
          ir1[4] = (v275_data + (v262_data * v273_data));
          float v278_data = s0[29];
          float v280_data = ir1[6];
          ir1[6] = (v280_data + (v262_data * v278_data));
          if (v10_lead < 3) {
            float v283_data = r0[11];
            float v286_data = ir1[1];
            ir1[1] = (v286_data + (v283_data * v263_data));
            float v291_data = ir1[3];
            ir1[3] = (v291_data + (v283_data * v268_data));
            float v296_data = ir1[5];
            ir1[5] = (v296_data + (v283_data * v273_data));
            float v301_data = ir1[7];
            ir1[7] = (v301_data + (v283_data * v278_data));
          }
          float v306_data = r0[12];
          float v307_data = s0[6];
          float v309_data = ir1[0];
          ir1[0] = (v309_data + (v306_data * v307_data));
          float v312_data = s0[14];
          float v314_data = ir1[2];
          ir1[2] = (v314_data + (v306_data * v312_data));
          float v317_data = s0[22];
          float v319_data = ir1[4];
          ir1[4] = (v319_data + (v306_data * v317_data));
          float v322_data = s0[30];
          float v324_data = ir1[6];
          ir1[6] = (v324_data + (v306_data * v322_data));
          if (v10_lead < 3) {
            float v327_data = r0[13];
            float v330_data = ir1[1];
            ir1[1] = (v330_data + (v327_data * v307_data));
            float v335_data = ir1[3];
            ir1[3] = (v335_data + (v327_data * v312_data));
            float v340_data = ir1[5];
            ir1[5] = (v340_data + (v327_data * v317_data));
            float v345_data = ir1[7];
            ir1[7] = (v345_data + (v327_data * v322_data));
          }
          float v350_data = r0[14];
          float v351_data = s0[7];
          float v353_data = ir1[0];
          ir1[0] = (v353_data + (v350_data * v351_data));
          float v356_data = s0[15];
          float v358_data = ir1[2];
          ir1[2] = (v358_data + (v350_data * v356_data));
          float v361_data = s0[23];
          float v363_data = ir1[4];
          ir1[4] = (v363_data + (v350_data * v361_data));
          float v366_data = s0[31];
          float v368_data = ir1[6];
          ir1[6] = (v368_data + (v350_data * v366_data));
          if (v10_lead < 3) {
            float v371_data = r0[15];
            float v374_data = ir1[1];
            ir1[1] = (v374_data + (v371_data * v351_data));
            float v379_data = ir1[3];
            ir1[3] = (v379_data + (v371_data * v356_data));
            float v384_data = ir1[5];
            ir1[5] = (v384_data + (v371_data * v361_data));
            float v389_data = ir1[7];
            ir1[7] = (v389_data + (v371_data * v366_data));
          }
          #pragma unroll
          for (int32_t v394_n0 = 0; v394_n0 < 1; ++v394_n0) {
            #pragma unroll
            for (int32_t v395_n1 = 0; v395_n1 < 4; ++v395_n1) {
              int32_t v397_a = v394_n0 + (v395_n1 * 2);
              float v398_data = ir1[v397_a];
              r1[v397_a] = v398_data;
            }
          }
          if (v10_lead < 3) {
            #pragma unroll
            for (int32_t v402_n1 = 0; v402_n1 < 4; ++v402_n1) {
              int32_t v404_a = 1 + (v402_n1 * 2);
              float v405_data = ir1[v404_a];
              r1[v404_a] = v405_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v411_i0 = 0; v411_i0 < 1; ++v411_i0) {
            int32_t v420_lead = v10_lead + (v411_i0 * 32);
            #pragma unroll
            for (int32_t v412_i1 = 0; v412_i1 < 4; ++v412_i1) {
              float v415_data = r1[(v411_i0 + (v412_i1 * 2))];
              glb_m0[(v420_lead + (v412_i1 * 35))] = v415_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v432_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v424_i1 = 0; v424_i1 < 4; ++v424_i1) {
              float v427_data = r1[(1 + (v424_i1 * 2))];
              glb_m0[(v432_lead + (v424_i1 * 35))] = v427_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

