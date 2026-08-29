// === base name ===
kernel_924fd3d329

// === header ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_924fd3d329, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_924fd3d329, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_924fd3d329<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×4(8×4) {0..8}×{0..4} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // C = abs(TMP)
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v14_lead = threadIdx.x % 32;
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
              float v24_data = __ldcg(&glb_m0[(v14_lead + (v16_i1 * 8))]);
              r0[v16_i1] = v24_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[4]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          if (v14_lead < 8) {
            float v33_data = r0[0];
            float v34_data = s0[0];
            float v36_data = r1[0];
            r1[0] = (v36_data + (v33_data * v34_data));
            float v39_data = s0[8];
            float v41_data = r1[1];
            r1[1] = (v41_data + (v33_data * v39_data));
            float v44_data = s0[16];
            float v46_data = r1[2];
            r1[2] = (v46_data + (v33_data * v44_data));
            float v49_data = s0[24];
            float v51_data = r1[3];
            r1[3] = (v51_data + (v33_data * v49_data));
          }
          if (v14_lead < 8) {
            float v57_data = r0[1];
            float v58_data = s0[1];
            float v60_data = r1[0];
            r1[0] = (v60_data + (v57_data * v58_data));
            float v63_data = s0[9];
            float v65_data = r1[1];
            r1[1] = (v65_data + (v57_data * v63_data));
            float v68_data = s0[17];
            float v70_data = r1[2];
            r1[2] = (v70_data + (v57_data * v68_data));
            float v73_data = s0[25];
            float v75_data = r1[3];
            r1[3] = (v75_data + (v57_data * v73_data));
          }
          if (v14_lead < 8) {
            float v81_data = r0[2];
            float v82_data = s0[2];
            float v84_data = r1[0];
            r1[0] = (v84_data + (v81_data * v82_data));
            float v87_data = s0[10];
            float v89_data = r1[1];
            r1[1] = (v89_data + (v81_data * v87_data));
            float v92_data = s0[18];
            float v94_data = r1[2];
            r1[2] = (v94_data + (v81_data * v92_data));
            float v97_data = s0[26];
            float v99_data = r1[3];
            r1[3] = (v99_data + (v81_data * v97_data));
          }
          if (v14_lead < 8) {
            float v105_data = r0[3];
            float v106_data = s0[3];
            float v108_data = r1[0];
            r1[0] = (v108_data + (v105_data * v106_data));
            float v111_data = s0[11];
            float v113_data = r1[1];
            r1[1] = (v113_data + (v105_data * v111_data));
            float v116_data = s0[19];
            float v118_data = r1[2];
            r1[2] = (v118_data + (v105_data * v116_data));
            float v121_data = s0[27];
            float v123_data = r1[3];
            r1[3] = (v123_data + (v105_data * v121_data));
          }
          if (v14_lead < 8) {
            float v129_data = r0[4];
            float v130_data = s0[4];
            float v132_data = r1[0];
            r1[0] = (v132_data + (v129_data * v130_data));
            float v135_data = s0[12];
            float v137_data = r1[1];
            r1[1] = (v137_data + (v129_data * v135_data));
            float v140_data = s0[20];
            float v142_data = r1[2];
            r1[2] = (v142_data + (v129_data * v140_data));
            float v145_data = s0[28];
            float v147_data = r1[3];
            r1[3] = (v147_data + (v129_data * v145_data));
          }
          if (v14_lead < 8) {
            float v153_data = r0[5];
            float v154_data = s0[5];
            float v156_data = r1[0];
            r1[0] = (v156_data + (v153_data * v154_data));
            float v159_data = s0[13];
            float v161_data = r1[1];
            r1[1] = (v161_data + (v153_data * v159_data));
            float v164_data = s0[21];
            float v166_data = r1[2];
            r1[2] = (v166_data + (v153_data * v164_data));
            float v169_data = s0[29];
            float v171_data = r1[3];
            r1[3] = (v171_data + (v153_data * v169_data));
          }
          if (v14_lead < 8) {
            float v177_data = r0[6];
            float v178_data = s0[6];
            float v180_data = r1[0];
            r1[0] = (v180_data + (v177_data * v178_data));
            float v183_data = s0[14];
            float v185_data = r1[1];
            r1[1] = (v185_data + (v177_data * v183_data));
            float v188_data = s0[22];
            float v190_data = r1[2];
            r1[2] = (v190_data + (v177_data * v188_data));
            float v193_data = s0[30];
            float v195_data = r1[3];
            r1[3] = (v195_data + (v177_data * v193_data));
          }
          if (v14_lead < 8) {
            float v201_data = r0[7];
            float v202_data = s0[7];
            float v204_data = r1[0];
            r1[0] = (v204_data + (v201_data * v202_data));
            float v207_data = s0[15];
            float v209_data = r1[1];
            r1[1] = (v209_data + (v201_data * v207_data));
            float v212_data = s0[23];
            float v214_data = r1[2];
            r1[2] = (v214_data + (v201_data * v212_data));
            float v217_data = s0[31];
            float v219_data = r1[3];
            r1[3] = (v219_data + (v201_data * v217_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v226_i1 = 0; v226_i1 < 4; ++v226_i1) {
              float v228_data = r1[v226_i1];
              int32_t v235_a = v14_lead + (v226_i1 * 8);
              s1[(v235_a ^ ((v235_a >> 5) & 31))] = v228_data;
            }
          }
          float* __restrict__ s2 = &localShrMem0[0];
          // s2 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(s2 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[4]{};
          __syncwarp();
          // r2 = +(r0 * s2) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float ir2[4]{};
          if (v14_lead < 8) {
            float v247_data = r0[0];
            float v248_data = s2[0];
            float v250_data = ir2[0];
            ir2[0] = (v250_data + (v247_data * v248_data));
            float v253_data = s2[8];
            float v255_data = ir2[1];
            ir2[1] = (v255_data + (v247_data * v253_data));
            float v258_data = s2[16];
            float v260_data = ir2[2];
            ir2[2] = (v260_data + (v247_data * v258_data));
            float v263_data = s2[24];
            float v265_data = ir2[3];
            ir2[3] = (v265_data + (v247_data * v263_data));
          }
          if (v14_lead < 8) {
            float v271_data = r0[1];
            float v272_data = s2[1];
            float v274_data = ir2[0];
            ir2[0] = (v274_data + (v271_data * v272_data));
            float v277_data = s2[9];
            float v279_data = ir2[1];
            ir2[1] = (v279_data + (v271_data * v277_data));
            float v282_data = s2[17];
            float v284_data = ir2[2];
            ir2[2] = (v284_data + (v271_data * v282_data));
            float v287_data = s2[25];
            float v289_data = ir2[3];
            ir2[3] = (v289_data + (v271_data * v287_data));
          }
          if (v14_lead < 8) {
            float v295_data = r0[2];
            float v296_data = s2[2];
            float v298_data = ir2[0];
            ir2[0] = (v298_data + (v295_data * v296_data));
            float v301_data = s2[10];
            float v303_data = ir2[1];
            ir2[1] = (v303_data + (v295_data * v301_data));
            float v306_data = s2[18];
            float v308_data = ir2[2];
            ir2[2] = (v308_data + (v295_data * v306_data));
            float v311_data = s2[26];
            float v313_data = ir2[3];
            ir2[3] = (v313_data + (v295_data * v311_data));
          }
          if (v14_lead < 8) {
            float v319_data = r0[3];
            float v320_data = s2[3];
            float v322_data = ir2[0];
            ir2[0] = (v322_data + (v319_data * v320_data));
            float v325_data = s2[11];
            float v327_data = ir2[1];
            ir2[1] = (v327_data + (v319_data * v325_data));
            float v330_data = s2[19];
            float v332_data = ir2[2];
            ir2[2] = (v332_data + (v319_data * v330_data));
            float v335_data = s2[27];
            float v337_data = ir2[3];
            ir2[3] = (v337_data + (v319_data * v335_data));
          }
          if (v14_lead < 8) {
            float v343_data = r0[4];
            float v344_data = s2[4];
            float v346_data = ir2[0];
            ir2[0] = (v346_data + (v343_data * v344_data));
            float v349_data = s2[12];
            float v351_data = ir2[1];
            ir2[1] = (v351_data + (v343_data * v349_data));
            float v354_data = s2[20];
            float v356_data = ir2[2];
            ir2[2] = (v356_data + (v343_data * v354_data));
            float v359_data = s2[28];
            float v361_data = ir2[3];
            ir2[3] = (v361_data + (v343_data * v359_data));
          }
          if (v14_lead < 8) {
            float v367_data = r0[5];
            float v368_data = s2[5];
            float v370_data = ir2[0];
            ir2[0] = (v370_data + (v367_data * v368_data));
            float v373_data = s2[13];
            float v375_data = ir2[1];
            ir2[1] = (v375_data + (v367_data * v373_data));
            float v378_data = s2[21];
            float v380_data = ir2[2];
            ir2[2] = (v380_data + (v367_data * v378_data));
            float v383_data = s2[29];
            float v385_data = ir2[3];
            ir2[3] = (v385_data + (v367_data * v383_data));
          }
          if (v14_lead < 8) {
            float v391_data = r0[6];
            float v392_data = s2[6];
            float v394_data = ir2[0];
            ir2[0] = (v394_data + (v391_data * v392_data));
            float v397_data = s2[14];
            float v399_data = ir2[1];
            ir2[1] = (v399_data + (v391_data * v397_data));
            float v402_data = s2[22];
            float v404_data = ir2[2];
            ir2[2] = (v404_data + (v391_data * v402_data));
            float v407_data = s2[30];
            float v409_data = ir2[3];
            ir2[3] = (v409_data + (v391_data * v407_data));
          }
          if (v14_lead < 8) {
            float v415_data = r0[7];
            float v416_data = s2[7];
            float v418_data = ir2[0];
            ir2[0] = (v418_data + (v415_data * v416_data));
            float v421_data = s2[15];
            float v423_data = ir2[1];
            ir2[1] = (v423_data + (v415_data * v421_data));
            float v426_data = s2[23];
            float v428_data = ir2[2];
            ir2[2] = (v428_data + (v415_data * v426_data));
            float v431_data = s2[31];
            float v433_data = ir2[3];
            ir2[3] = (v433_data + (v415_data * v431_data));
          }
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v439_n1 = 0; v439_n1 < 4; ++v439_n1) {
              float v441_data = ir2[v439_n1];
              r2[v439_n1] = v441_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r2);
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v447_i1 = 0; v447_i1 < 4; ++v447_i1) {
              float v449_data = r2[v447_i1];
              int32_t v457_a = v14_lead + ((v447_i1 + 4) * 8);
              s1[(v457_a ^ ((v457_a >> 5) & 31))] = v449_data;
            }
          }
          __syncwarp();
          // glb_m3 = abs(s1)
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v465_k1 = 0; v465_k1 < 8; ++v465_k1) {
              int32_t v471_a = v465_k1 * 8;
              int32_t v472_a = v14_lead + v471_a;
              float v476_data = s1[(v472_a ^ ((v472_a >> 5) & 31))];
              glb_m3[(v14_lead + v471_a)] = (fabsf(v476_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

