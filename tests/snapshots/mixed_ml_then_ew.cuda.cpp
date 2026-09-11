// === base name ===
kernel_192704bd18d1d249

// === header ===
void launcher_kernel_192704bd18d1d249(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_192704bd18d1d249(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_192704bd18d1d249, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_192704bd18d1d249, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_192704bd18d1d249<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_192704bd18d1d249(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v19_lead = threadIdx.x % 32;
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
              float v29_data = __ldcg(&glb_m0[(v19_lead + (v21_i1 * 8))]);
              r0[v21_i1] = v29_data;
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
          if (v19_lead < 8) {
            float v38_data = r0[0];
            float v39_data = s0[0];
            float v41_data = r1[0];
            r1[0] = (v41_data + (v38_data * v39_data));
            float v44_data = s0[8];
            float v46_data = r1[1];
            r1[1] = (v46_data + (v38_data * v44_data));
            float v49_data = s0[16];
            float v51_data = r1[2];
            r1[2] = (v51_data + (v38_data * v49_data));
            float v54_data = s0[24];
            float v56_data = r1[3];
            r1[3] = (v56_data + (v38_data * v54_data));
            float v59_data = s0[32];
            float v61_data = r1[4];
            r1[4] = (v61_data + (v38_data * v59_data));
            float v64_data = s0[40];
            float v66_data = r1[5];
            r1[5] = (v66_data + (v38_data * v64_data));
            float v69_data = s0[48];
            float v71_data = r1[6];
            r1[6] = (v71_data + (v38_data * v69_data));
            float v74_data = s0[56];
            float v76_data = r1[7];
            r1[7] = (v76_data + (v38_data * v74_data));
          }
          if (v19_lead < 8) {
            float v82_data = r0[1];
            float v83_data = s0[1];
            float v85_data = r1[0];
            r1[0] = (v85_data + (v82_data * v83_data));
            float v88_data = s0[9];
            float v90_data = r1[1];
            r1[1] = (v90_data + (v82_data * v88_data));
            float v93_data = s0[17];
            float v95_data = r1[2];
            r1[2] = (v95_data + (v82_data * v93_data));
            float v98_data = s0[25];
            float v100_data = r1[3];
            r1[3] = (v100_data + (v82_data * v98_data));
            float v103_data = s0[33];
            float v105_data = r1[4];
            r1[4] = (v105_data + (v82_data * v103_data));
            float v108_data = s0[41];
            float v110_data = r1[5];
            r1[5] = (v110_data + (v82_data * v108_data));
            float v113_data = s0[49];
            float v115_data = r1[6];
            r1[6] = (v115_data + (v82_data * v113_data));
            float v118_data = s0[57];
            float v120_data = r1[7];
            r1[7] = (v120_data + (v82_data * v118_data));
          }
          if (v19_lead < 8) {
            float v126_data = r0[2];
            float v127_data = s0[2];
            float v129_data = r1[0];
            r1[0] = (v129_data + (v126_data * v127_data));
            float v132_data = s0[10];
            float v134_data = r1[1];
            r1[1] = (v134_data + (v126_data * v132_data));
            float v137_data = s0[18];
            float v139_data = r1[2];
            r1[2] = (v139_data + (v126_data * v137_data));
            float v142_data = s0[26];
            float v144_data = r1[3];
            r1[3] = (v144_data + (v126_data * v142_data));
            float v147_data = s0[34];
            float v149_data = r1[4];
            r1[4] = (v149_data + (v126_data * v147_data));
            float v152_data = s0[42];
            float v154_data = r1[5];
            r1[5] = (v154_data + (v126_data * v152_data));
            float v157_data = s0[50];
            float v159_data = r1[6];
            r1[6] = (v159_data + (v126_data * v157_data));
            float v162_data = s0[58];
            float v164_data = r1[7];
            r1[7] = (v164_data + (v126_data * v162_data));
          }
          if (v19_lead < 8) {
            float v170_data = r0[3];
            float v171_data = s0[3];
            float v173_data = r1[0];
            r1[0] = (v173_data + (v170_data * v171_data));
            float v176_data = s0[11];
            float v178_data = r1[1];
            r1[1] = (v178_data + (v170_data * v176_data));
            float v181_data = s0[19];
            float v183_data = r1[2];
            r1[2] = (v183_data + (v170_data * v181_data));
            float v186_data = s0[27];
            float v188_data = r1[3];
            r1[3] = (v188_data + (v170_data * v186_data));
            float v191_data = s0[35];
            float v193_data = r1[4];
            r1[4] = (v193_data + (v170_data * v191_data));
            float v196_data = s0[43];
            float v198_data = r1[5];
            r1[5] = (v198_data + (v170_data * v196_data));
            float v201_data = s0[51];
            float v203_data = r1[6];
            r1[6] = (v203_data + (v170_data * v201_data));
            float v206_data = s0[59];
            float v208_data = r1[7];
            r1[7] = (v208_data + (v170_data * v206_data));
          }
          if (v19_lead < 8) {
            float v214_data = r0[4];
            float v215_data = s0[4];
            float v217_data = r1[0];
            r1[0] = (v217_data + (v214_data * v215_data));
            float v220_data = s0[12];
            float v222_data = r1[1];
            r1[1] = (v222_data + (v214_data * v220_data));
            float v225_data = s0[20];
            float v227_data = r1[2];
            r1[2] = (v227_data + (v214_data * v225_data));
            float v230_data = s0[28];
            float v232_data = r1[3];
            r1[3] = (v232_data + (v214_data * v230_data));
            float v235_data = s0[36];
            float v237_data = r1[4];
            r1[4] = (v237_data + (v214_data * v235_data));
            float v240_data = s0[44];
            float v242_data = r1[5];
            r1[5] = (v242_data + (v214_data * v240_data));
            float v245_data = s0[52];
            float v247_data = r1[6];
            r1[6] = (v247_data + (v214_data * v245_data));
            float v250_data = s0[60];
            float v252_data = r1[7];
            r1[7] = (v252_data + (v214_data * v250_data));
          }
          if (v19_lead < 8) {
            float v258_data = r0[5];
            float v259_data = s0[5];
            float v261_data = r1[0];
            r1[0] = (v261_data + (v258_data * v259_data));
            float v264_data = s0[13];
            float v266_data = r1[1];
            r1[1] = (v266_data + (v258_data * v264_data));
            float v269_data = s0[21];
            float v271_data = r1[2];
            r1[2] = (v271_data + (v258_data * v269_data));
            float v274_data = s0[29];
            float v276_data = r1[3];
            r1[3] = (v276_data + (v258_data * v274_data));
            float v279_data = s0[37];
            float v281_data = r1[4];
            r1[4] = (v281_data + (v258_data * v279_data));
            float v284_data = s0[45];
            float v286_data = r1[5];
            r1[5] = (v286_data + (v258_data * v284_data));
            float v289_data = s0[53];
            float v291_data = r1[6];
            r1[6] = (v291_data + (v258_data * v289_data));
            float v294_data = s0[61];
            float v296_data = r1[7];
            r1[7] = (v296_data + (v258_data * v294_data));
          }
          if (v19_lead < 8) {
            float v302_data = r0[6];
            float v303_data = s0[6];
            float v305_data = r1[0];
            r1[0] = (v305_data + (v302_data * v303_data));
            float v308_data = s0[14];
            float v310_data = r1[1];
            r1[1] = (v310_data + (v302_data * v308_data));
            float v313_data = s0[22];
            float v315_data = r1[2];
            r1[2] = (v315_data + (v302_data * v313_data));
            float v318_data = s0[30];
            float v320_data = r1[3];
            r1[3] = (v320_data + (v302_data * v318_data));
            float v323_data = s0[38];
            float v325_data = r1[4];
            r1[4] = (v325_data + (v302_data * v323_data));
            float v328_data = s0[46];
            float v330_data = r1[5];
            r1[5] = (v330_data + (v302_data * v328_data));
            float v333_data = s0[54];
            float v335_data = r1[6];
            r1[6] = (v335_data + (v302_data * v333_data));
            float v338_data = s0[62];
            float v340_data = r1[7];
            r1[7] = (v340_data + (v302_data * v338_data));
          }
          if (v19_lead < 8) {
            float v346_data = r0[7];
            float v347_data = s0[7];
            float v349_data = r1[0];
            r1[0] = (v349_data + (v346_data * v347_data));
            float v352_data = s0[15];
            float v354_data = r1[1];
            r1[1] = (v354_data + (v346_data * v352_data));
            float v357_data = s0[23];
            float v359_data = r1[2];
            r1[2] = (v359_data + (v346_data * v357_data));
            float v362_data = s0[31];
            float v364_data = r1[3];
            r1[3] = (v364_data + (v346_data * v362_data));
            float v367_data = s0[39];
            float v369_data = r1[4];
            r1[4] = (v369_data + (v346_data * v367_data));
            float v372_data = s0[47];
            float v374_data = r1[5];
            r1[5] = (v374_data + (v346_data * v372_data));
            float v377_data = s0[55];
            float v379_data = r1[6];
            r1[6] = (v379_data + (v346_data * v377_data));
            float v382_data = s0[63];
            float v384_data = r1[7];
            r1[7] = (v384_data + (v346_data * v382_data));
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v390_i1 = 0; v390_i1 < 8; ++v390_i1) {
              float v392_data = r1[v390_i1];
              int32_t v399_a = v19_lead + (v390_i1 * 8);
              s1[(v399_a ^ ((v399_a >> 5) & 31))] = v392_data;
            }
          }
          __syncwarp();
          // glb_m2 = abs(s1)
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v407_k1 = 0; v407_k1 < 8; ++v407_k1) {
              int32_t v413_a = v407_k1 * 8;
              int32_t v414_a = v19_lead + v413_a;
              float v418_data = s1[(v414_a ^ ((v414_a >> 5) & 31))];
              glb_m2[(v19_lead + v413_a)] = (fabsf(v418_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

