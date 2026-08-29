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
          int32_t v13_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
            int32_t v20_lead = v13_lead + (v14_i0 * 32);
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 8; ++v15_i1) {
              float v23_data = __ldcg(&glb_m1[(v20_lead + (v15_i1 * 35))]);
              r0[(v14_i0 + (v15_i1 * 2))] = v23_data;
            }
          }
          if (v13_lead < 3) {
            int32_t v32_lead = v13_lead + 32_i32;
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              float v35_data = __ldcg(&glb_m1[(v32_lead + (v27_i1 * 35))]);
              r0[(1 + (v27_i1 * 2))] = v35_data;
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
          float v45_data = r0[0];
          float v46_data = s0[0];
          float v48_data = ir1[0];
          ir1[0] = (v48_data + (v45_data * v46_data));
          float v51_data = s0[8];
          float v53_data = ir1[2];
          ir1[2] = (v53_data + (v45_data * v51_data));
          float v56_data = s0[16];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + (v45_data * v56_data));
          float v61_data = s0[24];
          float v63_data = ir1[6];
          ir1[6] = (v63_data + (v45_data * v61_data));
          if (v13_lead < 3) {
            float v66_data = r0[1];
            float v69_data = ir1[1];
            ir1[1] = (v69_data + (v66_data * v46_data));
            float v74_data = ir1[3];
            ir1[3] = (v74_data + (v66_data * v51_data));
            float v79_data = ir1[5];
            ir1[5] = (v79_data + (v66_data * v56_data));
            float v84_data = ir1[7];
            ir1[7] = (v84_data + (v66_data * v61_data));
          }
          float v89_data = r0[2];
          float v90_data = s0[1];
          float v92_data = ir1[0];
          ir1[0] = (v92_data + (v89_data * v90_data));
          float v95_data = s0[9];
          float v97_data = ir1[2];
          ir1[2] = (v97_data + (v89_data * v95_data));
          float v100_data = s0[17];
          float v102_data = ir1[4];
          ir1[4] = (v102_data + (v89_data * v100_data));
          float v105_data = s0[25];
          float v107_data = ir1[6];
          ir1[6] = (v107_data + (v89_data * v105_data));
          if (v13_lead < 3) {
            float v110_data = r0[3];
            float v113_data = ir1[1];
            ir1[1] = (v113_data + (v110_data * v90_data));
            float v118_data = ir1[3];
            ir1[3] = (v118_data + (v110_data * v95_data));
            float v123_data = ir1[5];
            ir1[5] = (v123_data + (v110_data * v100_data));
            float v128_data = ir1[7];
            ir1[7] = (v128_data + (v110_data * v105_data));
          }
          float v133_data = r0[4];
          float v134_data = s0[2];
          float v136_data = ir1[0];
          ir1[0] = (v136_data + (v133_data * v134_data));
          float v139_data = s0[10];
          float v141_data = ir1[2];
          ir1[2] = (v141_data + (v133_data * v139_data));
          float v144_data = s0[18];
          float v146_data = ir1[4];
          ir1[4] = (v146_data + (v133_data * v144_data));
          float v149_data = s0[26];
          float v151_data = ir1[6];
          ir1[6] = (v151_data + (v133_data * v149_data));
          if (v13_lead < 3) {
            float v154_data = r0[5];
            float v157_data = ir1[1];
            ir1[1] = (v157_data + (v154_data * v134_data));
            float v162_data = ir1[3];
            ir1[3] = (v162_data + (v154_data * v139_data));
            float v167_data = ir1[5];
            ir1[5] = (v167_data + (v154_data * v144_data));
            float v172_data = ir1[7];
            ir1[7] = (v172_data + (v154_data * v149_data));
          }
          float v177_data = r0[6];
          float v178_data = s0[3];
          float v180_data = ir1[0];
          ir1[0] = (v180_data + (v177_data * v178_data));
          float v183_data = s0[11];
          float v185_data = ir1[2];
          ir1[2] = (v185_data + (v177_data * v183_data));
          float v188_data = s0[19];
          float v190_data = ir1[4];
          ir1[4] = (v190_data + (v177_data * v188_data));
          float v193_data = s0[27];
          float v195_data = ir1[6];
          ir1[6] = (v195_data + (v177_data * v193_data));
          if (v13_lead < 3) {
            float v198_data = r0[7];
            float v201_data = ir1[1];
            ir1[1] = (v201_data + (v198_data * v178_data));
            float v206_data = ir1[3];
            ir1[3] = (v206_data + (v198_data * v183_data));
            float v211_data = ir1[5];
            ir1[5] = (v211_data + (v198_data * v188_data));
            float v216_data = ir1[7];
            ir1[7] = (v216_data + (v198_data * v193_data));
          }
          float v221_data = r0[8];
          float v222_data = s0[4];
          float v224_data = ir1[0];
          ir1[0] = (v224_data + (v221_data * v222_data));
          float v227_data = s0[12];
          float v229_data = ir1[2];
          ir1[2] = (v229_data + (v221_data * v227_data));
          float v232_data = s0[20];
          float v234_data = ir1[4];
          ir1[4] = (v234_data + (v221_data * v232_data));
          float v237_data = s0[28];
          float v239_data = ir1[6];
          ir1[6] = (v239_data + (v221_data * v237_data));
          if (v13_lead < 3) {
            float v242_data = r0[9];
            float v245_data = ir1[1];
            ir1[1] = (v245_data + (v242_data * v222_data));
            float v250_data = ir1[3];
            ir1[3] = (v250_data + (v242_data * v227_data));
            float v255_data = ir1[5];
            ir1[5] = (v255_data + (v242_data * v232_data));
            float v260_data = ir1[7];
            ir1[7] = (v260_data + (v242_data * v237_data));
          }
          float v265_data = r0[10];
          float v266_data = s0[5];
          float v268_data = ir1[0];
          ir1[0] = (v268_data + (v265_data * v266_data));
          float v271_data = s0[13];
          float v273_data = ir1[2];
          ir1[2] = (v273_data + (v265_data * v271_data));
          float v276_data = s0[21];
          float v278_data = ir1[4];
          ir1[4] = (v278_data + (v265_data * v276_data));
          float v281_data = s0[29];
          float v283_data = ir1[6];
          ir1[6] = (v283_data + (v265_data * v281_data));
          if (v13_lead < 3) {
            float v286_data = r0[11];
            float v289_data = ir1[1];
            ir1[1] = (v289_data + (v286_data * v266_data));
            float v294_data = ir1[3];
            ir1[3] = (v294_data + (v286_data * v271_data));
            float v299_data = ir1[5];
            ir1[5] = (v299_data + (v286_data * v276_data));
            float v304_data = ir1[7];
            ir1[7] = (v304_data + (v286_data * v281_data));
          }
          float v309_data = r0[12];
          float v310_data = s0[6];
          float v312_data = ir1[0];
          ir1[0] = (v312_data + (v309_data * v310_data));
          float v315_data = s0[14];
          float v317_data = ir1[2];
          ir1[2] = (v317_data + (v309_data * v315_data));
          float v320_data = s0[22];
          float v322_data = ir1[4];
          ir1[4] = (v322_data + (v309_data * v320_data));
          float v325_data = s0[30];
          float v327_data = ir1[6];
          ir1[6] = (v327_data + (v309_data * v325_data));
          if (v13_lead < 3) {
            float v330_data = r0[13];
            float v333_data = ir1[1];
            ir1[1] = (v333_data + (v330_data * v310_data));
            float v338_data = ir1[3];
            ir1[3] = (v338_data + (v330_data * v315_data));
            float v343_data = ir1[5];
            ir1[5] = (v343_data + (v330_data * v320_data));
            float v348_data = ir1[7];
            ir1[7] = (v348_data + (v330_data * v325_data));
          }
          float v353_data = r0[14];
          float v354_data = s0[7];
          float v356_data = ir1[0];
          ir1[0] = (v356_data + (v353_data * v354_data));
          float v359_data = s0[15];
          float v361_data = ir1[2];
          ir1[2] = (v361_data + (v353_data * v359_data));
          float v364_data = s0[23];
          float v366_data = ir1[4];
          ir1[4] = (v366_data + (v353_data * v364_data));
          float v369_data = s0[31];
          float v371_data = ir1[6];
          ir1[6] = (v371_data + (v353_data * v369_data));
          if (v13_lead < 3) {
            float v374_data = r0[15];
            float v377_data = ir1[1];
            ir1[1] = (v377_data + (v374_data * v354_data));
            float v382_data = ir1[3];
            ir1[3] = (v382_data + (v374_data * v359_data));
            float v387_data = ir1[5];
            ir1[5] = (v387_data + (v374_data * v364_data));
            float v392_data = ir1[7];
            ir1[7] = (v392_data + (v374_data * v369_data));
          }
          #pragma unroll
          for (int32_t v397_n0 = 0; v397_n0 < 1; ++v397_n0) {
            #pragma unroll
            for (int32_t v398_n1 = 0; v398_n1 < 4; ++v398_n1) {
              int32_t v400_a = v397_n0 + (v398_n1 * 2);
              float v401_data = ir1[v400_a];
              r1[v400_a] = v401_data;
            }
          }
          if (v13_lead < 3) {
            #pragma unroll
            for (int32_t v405_n1 = 0; v405_n1 < 4; ++v405_n1) {
              int32_t v407_a = 1 + (v405_n1 * 2);
              float v408_data = ir1[v407_a];
              r1[v407_a] = v408_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v414_i0 = 0; v414_i0 < 1; ++v414_i0) {
            int32_t v423_lead = v13_lead + (v414_i0 * 32);
            #pragma unroll
            for (int32_t v415_i1 = 0; v415_i1 < 4; ++v415_i1) {
              float v418_data = r1[(v414_i0 + (v415_i1 * 2))];
              glb_m0[(v423_lead + (v415_i1 * 35))] = v418_data;
            }
          }
          if (v13_lead < 3) {
            int32_t v435_lead = v13_lead + 32_i32;
            #pragma unroll
            for (int32_t v427_i1 = 0; v427_i1 < 4; ++v427_i1) {
              float v430_data = r1[(1 + (v427_i1 * 2))];
              glb_m0[(v435_lead + (v427_i1 * 35))] = v430_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

