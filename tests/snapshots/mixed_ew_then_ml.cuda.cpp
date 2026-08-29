// === base name ===
kernel_a587425bdd

// === header ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a587425bdd, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_a587425bdd, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_a587425bdd<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // TMP = abs(A)
    // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v16_lead = threadIdx.x % 32;
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v18_k1 = 0; v18_k1 < 8; ++v18_k1) {
              float v26_data = glb_m0[(v16_lead + (v18_k1 * 8))];
              r0[v18_k1] = (fabsf(v26_data));
            }
          }
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir1[8]{};
          if (v16_lead < 8) {
            float v35_data = r0[0];
            float v36_data = s1[0];
            float v38_data = ir1[0];
            ir1[0] = (v38_data + (v35_data * v36_data));
            float v41_data = s1[8];
            float v43_data = ir1[1];
            ir1[1] = (v43_data + (v35_data * v41_data));
            float v46_data = s1[16];
            float v48_data = ir1[2];
            ir1[2] = (v48_data + (v35_data * v46_data));
            float v51_data = s1[24];
            float v53_data = ir1[3];
            ir1[3] = (v53_data + (v35_data * v51_data));
            float v56_data = s1[33];
            float v58_data = ir1[4];
            ir1[4] = (v58_data + (v35_data * v56_data));
            float v61_data = s1[41];
            float v63_data = ir1[5];
            ir1[5] = (v63_data + (v35_data * v61_data));
            float v66_data = s1[49];
            float v68_data = ir1[6];
            ir1[6] = (v68_data + (v35_data * v66_data));
            float v71_data = s1[57];
            float v73_data = ir1[7];
            ir1[7] = (v73_data + (v35_data * v71_data));
          }
          if (v16_lead < 8) {
            float v79_data = r0[1];
            float v80_data = s1[1];
            float v82_data = ir1[0];
            ir1[0] = (v82_data + (v79_data * v80_data));
            float v85_data = s1[9];
            float v87_data = ir1[1];
            ir1[1] = (v87_data + (v79_data * v85_data));
            float v90_data = s1[17];
            float v92_data = ir1[2];
            ir1[2] = (v92_data + (v79_data * v90_data));
            float v95_data = s1[25];
            float v97_data = ir1[3];
            ir1[3] = (v97_data + (v79_data * v95_data));
            float v100_data = s1[32];
            float v102_data = ir1[4];
            ir1[4] = (v102_data + (v79_data * v100_data));
            float v105_data = s1[40];
            float v107_data = ir1[5];
            ir1[5] = (v107_data + (v79_data * v105_data));
            float v110_data = s1[48];
            float v112_data = ir1[6];
            ir1[6] = (v112_data + (v79_data * v110_data));
            float v115_data = s1[56];
            float v117_data = ir1[7];
            ir1[7] = (v117_data + (v79_data * v115_data));
          }
          if (v16_lead < 8) {
            float v123_data = r0[2];
            float v124_data = s1[2];
            float v126_data = ir1[0];
            ir1[0] = (v126_data + (v123_data * v124_data));
            float v129_data = s1[10];
            float v131_data = ir1[1];
            ir1[1] = (v131_data + (v123_data * v129_data));
            float v134_data = s1[18];
            float v136_data = ir1[2];
            ir1[2] = (v136_data + (v123_data * v134_data));
            float v139_data = s1[26];
            float v141_data = ir1[3];
            ir1[3] = (v141_data + (v123_data * v139_data));
            float v144_data = s1[35];
            float v146_data = ir1[4];
            ir1[4] = (v146_data + (v123_data * v144_data));
            float v149_data = s1[43];
            float v151_data = ir1[5];
            ir1[5] = (v151_data + (v123_data * v149_data));
            float v154_data = s1[51];
            float v156_data = ir1[6];
            ir1[6] = (v156_data + (v123_data * v154_data));
            float v159_data = s1[59];
            float v161_data = ir1[7];
            ir1[7] = (v161_data + (v123_data * v159_data));
          }
          if (v16_lead < 8) {
            float v167_data = r0[3];
            float v168_data = s1[3];
            float v170_data = ir1[0];
            ir1[0] = (v170_data + (v167_data * v168_data));
            float v173_data = s1[11];
            float v175_data = ir1[1];
            ir1[1] = (v175_data + (v167_data * v173_data));
            float v178_data = s1[19];
            float v180_data = ir1[2];
            ir1[2] = (v180_data + (v167_data * v178_data));
            float v183_data = s1[27];
            float v185_data = ir1[3];
            ir1[3] = (v185_data + (v167_data * v183_data));
            float v188_data = s1[34];
            float v190_data = ir1[4];
            ir1[4] = (v190_data + (v167_data * v188_data));
            float v193_data = s1[42];
            float v195_data = ir1[5];
            ir1[5] = (v195_data + (v167_data * v193_data));
            float v198_data = s1[50];
            float v200_data = ir1[6];
            ir1[6] = (v200_data + (v167_data * v198_data));
            float v203_data = s1[58];
            float v205_data = ir1[7];
            ir1[7] = (v205_data + (v167_data * v203_data));
          }
          if (v16_lead < 8) {
            float v211_data = r0[4];
            float v212_data = s1[4];
            float v214_data = ir1[0];
            ir1[0] = (v214_data + (v211_data * v212_data));
            float v217_data = s1[12];
            float v219_data = ir1[1];
            ir1[1] = (v219_data + (v211_data * v217_data));
            float v222_data = s1[20];
            float v224_data = ir1[2];
            ir1[2] = (v224_data + (v211_data * v222_data));
            float v227_data = s1[28];
            float v229_data = ir1[3];
            ir1[3] = (v229_data + (v211_data * v227_data));
            float v232_data = s1[37];
            float v234_data = ir1[4];
            ir1[4] = (v234_data + (v211_data * v232_data));
            float v237_data = s1[45];
            float v239_data = ir1[5];
            ir1[5] = (v239_data + (v211_data * v237_data));
            float v242_data = s1[53];
            float v244_data = ir1[6];
            ir1[6] = (v244_data + (v211_data * v242_data));
            float v247_data = s1[61];
            float v249_data = ir1[7];
            ir1[7] = (v249_data + (v211_data * v247_data));
          }
          if (v16_lead < 8) {
            float v255_data = r0[5];
            float v256_data = s1[5];
            float v258_data = ir1[0];
            ir1[0] = (v258_data + (v255_data * v256_data));
            float v261_data = s1[13];
            float v263_data = ir1[1];
            ir1[1] = (v263_data + (v255_data * v261_data));
            float v266_data = s1[21];
            float v268_data = ir1[2];
            ir1[2] = (v268_data + (v255_data * v266_data));
            float v271_data = s1[29];
            float v273_data = ir1[3];
            ir1[3] = (v273_data + (v255_data * v271_data));
            float v276_data = s1[36];
            float v278_data = ir1[4];
            ir1[4] = (v278_data + (v255_data * v276_data));
            float v281_data = s1[44];
            float v283_data = ir1[5];
            ir1[5] = (v283_data + (v255_data * v281_data));
            float v286_data = s1[52];
            float v288_data = ir1[6];
            ir1[6] = (v288_data + (v255_data * v286_data));
            float v291_data = s1[60];
            float v293_data = ir1[7];
            ir1[7] = (v293_data + (v255_data * v291_data));
          }
          if (v16_lead < 8) {
            float v299_data = r0[6];
            float v300_data = s1[6];
            float v302_data = ir1[0];
            ir1[0] = (v302_data + (v299_data * v300_data));
            float v305_data = s1[14];
            float v307_data = ir1[1];
            ir1[1] = (v307_data + (v299_data * v305_data));
            float v310_data = s1[22];
            float v312_data = ir1[2];
            ir1[2] = (v312_data + (v299_data * v310_data));
            float v315_data = s1[30];
            float v317_data = ir1[3];
            ir1[3] = (v317_data + (v299_data * v315_data));
            float v320_data = s1[39];
            float v322_data = ir1[4];
            ir1[4] = (v322_data + (v299_data * v320_data));
            float v325_data = s1[47];
            float v327_data = ir1[5];
            ir1[5] = (v327_data + (v299_data * v325_data));
            float v330_data = s1[55];
            float v332_data = ir1[6];
            ir1[6] = (v332_data + (v299_data * v330_data));
            float v335_data = s1[63];
            float v337_data = ir1[7];
            ir1[7] = (v337_data + (v299_data * v335_data));
          }
          if (v16_lead < 8) {
            float v343_data = r0[7];
            float v344_data = s1[7];
            float v346_data = ir1[0];
            ir1[0] = (v346_data + (v343_data * v344_data));
            float v349_data = s1[15];
            float v351_data = ir1[1];
            ir1[1] = (v351_data + (v343_data * v349_data));
            float v354_data = s1[23];
            float v356_data = ir1[2];
            ir1[2] = (v356_data + (v343_data * v354_data));
            float v359_data = s1[31];
            float v361_data = ir1[3];
            ir1[3] = (v361_data + (v343_data * v359_data));
            float v364_data = s1[38];
            float v366_data = ir1[4];
            ir1[4] = (v366_data + (v343_data * v364_data));
            float v369_data = s1[46];
            float v371_data = ir1[5];
            ir1[5] = (v371_data + (v343_data * v369_data));
            float v374_data = s1[54];
            float v376_data = ir1[6];
            ir1[6] = (v376_data + (v343_data * v374_data));
            float v379_data = s1[62];
            float v381_data = ir1[7];
            ir1[7] = (v381_data + (v343_data * v379_data));
          }
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v387_n1 = 0; v387_n1 < 8; ++v387_n1) {
              float v389_data = ir1[v387_n1];
              r1[v387_n1] = v389_data;
            }
          }
          // glb_m1 = store{r>g}(r1);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v395_i1 = 0; v395_i1 < 8; ++v395_i1) {
              float v397_data = r1[v395_i1];
              glb_m1[(v16_lead + (v395_i1 * 8))] = v397_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

