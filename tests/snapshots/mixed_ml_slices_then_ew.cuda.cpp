// === base name ===
kernel_aeb1216cf3baeb2c

// === header ===
void launcher_kernel_aeb1216cf3baeb2c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_aeb1216cf3baeb2c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_aeb1216cf3baeb2c, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_aeb1216cf3baeb2c, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_aeb1216cf3baeb2c<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_aeb1216cf3baeb2c(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
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
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
      float* __restrict__ s2 = &localShrMem0[0];
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
          int32_t v17_lead = threadIdx.x % 32;
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
              float v27_data = __ldcg(&glb_m0[(v17_lead + (v19_i1 * 8))]);
              r0[v19_i1] = v27_data;
            }
          }
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
          if (v17_lead < 8) {
            float v35_data = r0[0];
            float v36_data = s0[0];
            float v38_data = r1[0];
            r1[0] = (v38_data + (v35_data * v36_data));
            float v41_data = s0[8];
            float v43_data = r1[1];
            r1[1] = (v43_data + (v35_data * v41_data));
            float v46_data = s0[16];
            float v48_data = r1[2];
            r1[2] = (v48_data + (v35_data * v46_data));
            float v51_data = s0[24];
            float v53_data = r1[3];
            r1[3] = (v53_data + (v35_data * v51_data));
          }
          if (v17_lead < 8) {
            float v59_data = r0[1];
            float v60_data = s0[1];
            float v62_data = r1[0];
            r1[0] = (v62_data + (v59_data * v60_data));
            float v65_data = s0[9];
            float v67_data = r1[1];
            r1[1] = (v67_data + (v59_data * v65_data));
            float v70_data = s0[17];
            float v72_data = r1[2];
            r1[2] = (v72_data + (v59_data * v70_data));
            float v75_data = s0[25];
            float v77_data = r1[3];
            r1[3] = (v77_data + (v59_data * v75_data));
          }
          if (v17_lead < 8) {
            float v83_data = r0[2];
            float v84_data = s0[2];
            float v86_data = r1[0];
            r1[0] = (v86_data + (v83_data * v84_data));
            float v89_data = s0[10];
            float v91_data = r1[1];
            r1[1] = (v91_data + (v83_data * v89_data));
            float v94_data = s0[18];
            float v96_data = r1[2];
            r1[2] = (v96_data + (v83_data * v94_data));
            float v99_data = s0[26];
            float v101_data = r1[3];
            r1[3] = (v101_data + (v83_data * v99_data));
          }
          if (v17_lead < 8) {
            float v107_data = r0[3];
            float v108_data = s0[3];
            float v110_data = r1[0];
            r1[0] = (v110_data + (v107_data * v108_data));
            float v113_data = s0[11];
            float v115_data = r1[1];
            r1[1] = (v115_data + (v107_data * v113_data));
            float v118_data = s0[19];
            float v120_data = r1[2];
            r1[2] = (v120_data + (v107_data * v118_data));
            float v123_data = s0[27];
            float v125_data = r1[3];
            r1[3] = (v125_data + (v107_data * v123_data));
          }
          if (v17_lead < 8) {
            float v131_data = r0[4];
            float v132_data = s0[4];
            float v134_data = r1[0];
            r1[0] = (v134_data + (v131_data * v132_data));
            float v137_data = s0[12];
            float v139_data = r1[1];
            r1[1] = (v139_data + (v131_data * v137_data));
            float v142_data = s0[20];
            float v144_data = r1[2];
            r1[2] = (v144_data + (v131_data * v142_data));
            float v147_data = s0[28];
            float v149_data = r1[3];
            r1[3] = (v149_data + (v131_data * v147_data));
          }
          if (v17_lead < 8) {
            float v155_data = r0[5];
            float v156_data = s0[5];
            float v158_data = r1[0];
            r1[0] = (v158_data + (v155_data * v156_data));
            float v161_data = s0[13];
            float v163_data = r1[1];
            r1[1] = (v163_data + (v155_data * v161_data));
            float v166_data = s0[21];
            float v168_data = r1[2];
            r1[2] = (v168_data + (v155_data * v166_data));
            float v171_data = s0[29];
            float v173_data = r1[3];
            r1[3] = (v173_data + (v155_data * v171_data));
          }
          if (v17_lead < 8) {
            float v179_data = r0[6];
            float v180_data = s0[6];
            float v182_data = r1[0];
            r1[0] = (v182_data + (v179_data * v180_data));
            float v185_data = s0[14];
            float v187_data = r1[1];
            r1[1] = (v187_data + (v179_data * v185_data));
            float v190_data = s0[22];
            float v192_data = r1[2];
            r1[2] = (v192_data + (v179_data * v190_data));
            float v195_data = s0[30];
            float v197_data = r1[3];
            r1[3] = (v197_data + (v179_data * v195_data));
          }
          if (v17_lead < 8) {
            float v203_data = r0[7];
            float v204_data = s0[7];
            float v206_data = r1[0];
            r1[0] = (v206_data + (v203_data * v204_data));
            float v209_data = s0[15];
            float v211_data = r1[1];
            r1[1] = (v211_data + (v203_data * v209_data));
            float v214_data = s0[23];
            float v216_data = r1[2];
            r1[2] = (v216_data + (v203_data * v214_data));
            float v219_data = s0[31];
            float v221_data = r1[3];
            r1[3] = (v221_data + (v203_data * v219_data));
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v227_i1 = 0; v227_i1 < 4; ++v227_i1) {
              float v229_data = r1[v227_i1];
              int32_t v236_a = v17_lead + (v227_i1 * 8);
              s1[(v236_a ^ ((v236_a >> 5) & 31))] = v229_data;
            }
          }
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
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
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v439_n1 = 0; v439_n1 < 4; ++v439_n1) {
              float v441_data = ir2[v439_n1];
              r2[v439_n1] = v441_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r2);
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v447_i1 = 0; v447_i1 < 4; ++v447_i1) {
              float v449_data = r2[v447_i1];
              int32_t v457_a = v17_lead + ((v447_i1 + 4) * 8);
              s1[(v457_a ^ ((v457_a >> 5) & 31))] = v449_data;
            }
          }
          __syncwarp();
          // glb_m3 = abs(s1)
          if (v17_lead < 8) {
            #pragma unroll
            for (int32_t v465_k1 = 0; v465_k1 < 8; ++v465_k1) {
              int32_t v471_a = v465_k1 * 8;
              int32_t v472_a = v17_lead + v471_a;
              float v476_data = s1[(v472_a ^ ((v472_a >> 5) & 31))];
              glb_m3[(v17_lead + v471_a)] = (fabsf(v476_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

