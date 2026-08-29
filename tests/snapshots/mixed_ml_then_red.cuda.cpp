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
          int32_t v13_lead = threadIdx.x % 32;
          if (v13_lead < 8) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 8; ++v15_i1) {
              float v23_data = __ldcg(&glb_m0[(v13_lead + (v15_i1 * 8))]);
              r0[v15_i1] = v23_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          if (v13_lead < 8) {
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
            float v54_data = s0[33];
            float v56_data = r1[4];
            r1[4] = (v56_data + (v33_data * v54_data));
            float v59_data = s0[41];
            float v61_data = r1[5];
            r1[5] = (v61_data + (v33_data * v59_data));
            float v64_data = s0[49];
            float v66_data = r1[6];
            r1[6] = (v66_data + (v33_data * v64_data));
            float v69_data = s0[57];
            float v71_data = r1[7];
            r1[7] = (v71_data + (v33_data * v69_data));
          }
          if (v13_lead < 8) {
            float v77_data = r0[1];
            float v78_data = s0[1];
            float v80_data = r1[0];
            r1[0] = (v80_data + (v77_data * v78_data));
            float v83_data = s0[9];
            float v85_data = r1[1];
            r1[1] = (v85_data + (v77_data * v83_data));
            float v88_data = s0[17];
            float v90_data = r1[2];
            r1[2] = (v90_data + (v77_data * v88_data));
            float v93_data = s0[25];
            float v95_data = r1[3];
            r1[3] = (v95_data + (v77_data * v93_data));
            float v98_data = s0[32];
            float v100_data = r1[4];
            r1[4] = (v100_data + (v77_data * v98_data));
            float v103_data = s0[40];
            float v105_data = r1[5];
            r1[5] = (v105_data + (v77_data * v103_data));
            float v108_data = s0[48];
            float v110_data = r1[6];
            r1[6] = (v110_data + (v77_data * v108_data));
            float v113_data = s0[56];
            float v115_data = r1[7];
            r1[7] = (v115_data + (v77_data * v113_data));
          }
          if (v13_lead < 8) {
            float v121_data = r0[2];
            float v122_data = s0[2];
            float v124_data = r1[0];
            r1[0] = (v124_data + (v121_data * v122_data));
            float v127_data = s0[10];
            float v129_data = r1[1];
            r1[1] = (v129_data + (v121_data * v127_data));
            float v132_data = s0[18];
            float v134_data = r1[2];
            r1[2] = (v134_data + (v121_data * v132_data));
            float v137_data = s0[26];
            float v139_data = r1[3];
            r1[3] = (v139_data + (v121_data * v137_data));
            float v142_data = s0[35];
            float v144_data = r1[4];
            r1[4] = (v144_data + (v121_data * v142_data));
            float v147_data = s0[43];
            float v149_data = r1[5];
            r1[5] = (v149_data + (v121_data * v147_data));
            float v152_data = s0[51];
            float v154_data = r1[6];
            r1[6] = (v154_data + (v121_data * v152_data));
            float v157_data = s0[59];
            float v159_data = r1[7];
            r1[7] = (v159_data + (v121_data * v157_data));
          }
          if (v13_lead < 8) {
            float v165_data = r0[3];
            float v166_data = s0[3];
            float v168_data = r1[0];
            r1[0] = (v168_data + (v165_data * v166_data));
            float v171_data = s0[11];
            float v173_data = r1[1];
            r1[1] = (v173_data + (v165_data * v171_data));
            float v176_data = s0[19];
            float v178_data = r1[2];
            r1[2] = (v178_data + (v165_data * v176_data));
            float v181_data = s0[27];
            float v183_data = r1[3];
            r1[3] = (v183_data + (v165_data * v181_data));
            float v186_data = s0[34];
            float v188_data = r1[4];
            r1[4] = (v188_data + (v165_data * v186_data));
            float v191_data = s0[42];
            float v193_data = r1[5];
            r1[5] = (v193_data + (v165_data * v191_data));
            float v196_data = s0[50];
            float v198_data = r1[6];
            r1[6] = (v198_data + (v165_data * v196_data));
            float v201_data = s0[58];
            float v203_data = r1[7];
            r1[7] = (v203_data + (v165_data * v201_data));
          }
          if (v13_lead < 8) {
            float v209_data = r0[4];
            float v210_data = s0[4];
            float v212_data = r1[0];
            r1[0] = (v212_data + (v209_data * v210_data));
            float v215_data = s0[12];
            float v217_data = r1[1];
            r1[1] = (v217_data + (v209_data * v215_data));
            float v220_data = s0[20];
            float v222_data = r1[2];
            r1[2] = (v222_data + (v209_data * v220_data));
            float v225_data = s0[28];
            float v227_data = r1[3];
            r1[3] = (v227_data + (v209_data * v225_data));
            float v230_data = s0[37];
            float v232_data = r1[4];
            r1[4] = (v232_data + (v209_data * v230_data));
            float v235_data = s0[45];
            float v237_data = r1[5];
            r1[5] = (v237_data + (v209_data * v235_data));
            float v240_data = s0[53];
            float v242_data = r1[6];
            r1[6] = (v242_data + (v209_data * v240_data));
            float v245_data = s0[61];
            float v247_data = r1[7];
            r1[7] = (v247_data + (v209_data * v245_data));
          }
          if (v13_lead < 8) {
            float v253_data = r0[5];
            float v254_data = s0[5];
            float v256_data = r1[0];
            r1[0] = (v256_data + (v253_data * v254_data));
            float v259_data = s0[13];
            float v261_data = r1[1];
            r1[1] = (v261_data + (v253_data * v259_data));
            float v264_data = s0[21];
            float v266_data = r1[2];
            r1[2] = (v266_data + (v253_data * v264_data));
            float v269_data = s0[29];
            float v271_data = r1[3];
            r1[3] = (v271_data + (v253_data * v269_data));
            float v274_data = s0[36];
            float v276_data = r1[4];
            r1[4] = (v276_data + (v253_data * v274_data));
            float v279_data = s0[44];
            float v281_data = r1[5];
            r1[5] = (v281_data + (v253_data * v279_data));
            float v284_data = s0[52];
            float v286_data = r1[6];
            r1[6] = (v286_data + (v253_data * v284_data));
            float v289_data = s0[60];
            float v291_data = r1[7];
            r1[7] = (v291_data + (v253_data * v289_data));
          }
          if (v13_lead < 8) {
            float v297_data = r0[6];
            float v298_data = s0[6];
            float v300_data = r1[0];
            r1[0] = (v300_data + (v297_data * v298_data));
            float v303_data = s0[14];
            float v305_data = r1[1];
            r1[1] = (v305_data + (v297_data * v303_data));
            float v308_data = s0[22];
            float v310_data = r1[2];
            r1[2] = (v310_data + (v297_data * v308_data));
            float v313_data = s0[30];
            float v315_data = r1[3];
            r1[3] = (v315_data + (v297_data * v313_data));
            float v318_data = s0[39];
            float v320_data = r1[4];
            r1[4] = (v320_data + (v297_data * v318_data));
            float v323_data = s0[47];
            float v325_data = r1[5];
            r1[5] = (v325_data + (v297_data * v323_data));
            float v328_data = s0[55];
            float v330_data = r1[6];
            r1[6] = (v330_data + (v297_data * v328_data));
            float v333_data = s0[63];
            float v335_data = r1[7];
            r1[7] = (v335_data + (v297_data * v333_data));
          }
          if (v13_lead < 8) {
            float v341_data = r0[7];
            float v342_data = s0[7];
            float v344_data = r1[0];
            r1[0] = (v344_data + (v341_data * v342_data));
            float v347_data = s0[15];
            float v349_data = r1[1];
            r1[1] = (v349_data + (v341_data * v347_data));
            float v352_data = s0[23];
            float v354_data = r1[2];
            r1[2] = (v354_data + (v341_data * v352_data));
            float v357_data = s0[31];
            float v359_data = r1[3];
            r1[3] = (v359_data + (v341_data * v357_data));
            float v362_data = s0[38];
            float v364_data = r1[4];
            r1[4] = (v364_data + (v341_data * v362_data));
            float v367_data = s0[46];
            float v369_data = r1[5];
            r1[5] = (v369_data + (v341_data * v367_data));
            float v372_data = s0[54];
            float v374_data = r1[6];
            r1[6] = (v374_data + (v341_data * v372_data));
            float v377_data = s0[62];
            float v379_data = r1[7];
            r1[7] = (v379_data + (v341_data * v377_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v13_lead < 8) {
            #pragma unroll
            for (int32_t v386_i1 = 0; v386_i1 < 8; ++v386_i1) {
              float v388_data = r1[v386_i1];
              int32_t v395_a = v13_lead + (v386_i1 * 8);
              s1[(v395_a ^ ((v395_a >> 5) & 31))] = v388_data;
            }
          }
          __syncwarp();
          // glb_m2 = +(s1, dims=[1])
          if (v13_lead < 8) {
            float v404_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v403_r1 = 0; v403_r1 < 8; ++v403_r1) {
              int32_t v411_a = v13_lead + (v403_r1 * 8);
              float v415_data = s1[(v411_a ^ ((v411_a >> 5) & 31))];
              v404_acc0 = (v404_acc0 + v415_data);
            }
            glb_m2[v13_lead] = v404_acc0;
          }
          __syncwarp();
        }
      }
    }
  }
}

