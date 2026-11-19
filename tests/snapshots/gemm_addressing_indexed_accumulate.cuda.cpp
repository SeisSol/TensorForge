// === base name ===
kernel_27cc0c436104a8d5

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_27cc0c436104a8d5 = {{32, 4, 1}, 32, 20, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_27cc0c436104a8d5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_27cc0c436104a8d5(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_27cc0c436104a8d5(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_27cc0c436104a8d5, block.x * block.y * block.z, 384 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 4;
  config.block[2] = 1;
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_27cc0c436104a8d5(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_27cc0c436104a8d5(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_27cc0c436104a8d5, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  float *const * m0Arg = (float *const *)m0;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_27cc0c436104a8d5<<<grid,block,config.sharedMemBytes,stream>>>(m0Arg, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_27cc0c436104a8d5(float *const * __restrict__ m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 4 per block = block 32x4x1, 1536 B shared, occupancy grid
    // operands:
    //   m0 20×9(20×9) {0..20}×{0..9} indexed_pointer(slots=2, slot=1)
    //   m1 20×9(20×9) {0..20}×{0..9} none
    //   m2 9×9(9×9) {0..9}×{0..9} indexed(offset=12)
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"indexed_pointer(slots=2, slot=1)","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"indexed(offset=12)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"indexed_pointer(slots=2, slot=1)","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},{"addressing":"indexed(offset=12)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[9]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v472_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v472_flagSecondAhead < numElements0) ? v472_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
      const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 81 + 0 + 12 + m2_extraOffset];
      bool v483_g = threadIdx.x < 17;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      if (allowed_peel) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      }
      if (allowed_peel) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      if (v483_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
        }
      }
      __pipeline_commit();
      float *const __restrict__ peel_glb_m0 = &m0[peel_elementIndex * 2 + 1][0 + m0_extraOffset];
      int32_t v501_lead = threadIdx.x % 32;
      bool v502_g = v501_lead < 20;
      bool v518_g = allowed_peel && v502_g;
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v510_i1 = 0; v510_i1 < 9; ++v510_i1) {
        float v513_data = v518_g ? (peel_glb_m0[(v501_lead + (v510_i1 * 20))]) : (0.0f);
        r0[v510_i1] = v513_data;
      }
      uint32_t v476_flagWord = flagWordFirst;
      uint32_t v477_flagWordNext = flagWordSecond;
      size_t v528_elementIndex = peel_elementIndex;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v12_batchId1]);
        const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 81 + 0 + 12 + m2_extraOffset];
        float *const __restrict__ wrap_glb_m0 = &m0[wrap_elementIndex * 2 + 1][0 + m0_extraOffset];
        size_t v523_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v523_flagAheadAhead < numElements0) ? v523_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v476_flagWord);
        const bool allowed_next = static_cast<bool>(v477_flagWordNext);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v528_elementIndex * 2 + 1][0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          // ir1 = +(glb_m1 * s0)
          // [(0, 20), (0, 9)] [(0, 9)]
          float ir1[9]{};
          float v33_data = v502_g ? (glb_m1[v501_lead]) : (0.0f);
          __syncwarp();
          float v34_data = s0[0];
          float v39_data = s0[9];
          float v44_data = s0[18];
          float v49_data = s0[27];
          float v54_data = s0[36];
          float v59_data = s0[45];
          float v64_data = s0[54];
          float v69_data = s0[63];
          float v74_data = s0[72];
          float v79_data = v502_g ? (glb_m1[(v501_lead + 20)]) : (0.0f);
          float v80_data = s0[1];
          float v85_data = s0[10];
          float v90_data = s0[19];
          float v95_data = s0[28];
          float v100_data = s0[37];
          float v105_data = s0[46];
          float v110_data = s0[55];
          float v115_data = s0[64];
          float v120_data = s0[73];
          float v125_data = v502_g ? (glb_m1[(v501_lead + 40)]) : (0.0f);
          float v126_data = s0[2];
          float v131_data = s0[11];
          float v136_data = s0[20];
          float v141_data = s0[29];
          float v146_data = s0[38];
          float v151_data = s0[47];
          float v156_data = s0[56];
          float v161_data = s0[65];
          float v166_data = s0[74];
          float v171_data = v502_g ? (glb_m1[(v501_lead + 60)]) : (0.0f);
          float v172_data = s0[3];
          float v177_data = s0[12];
          float v182_data = s0[21];
          float v187_data = s0[30];
          float v192_data = s0[39];
          float v197_data = s0[48];
          float v202_data = s0[57];
          float v207_data = s0[66];
          float v212_data = s0[75];
          float v217_data = v502_g ? (glb_m1[(v501_lead + 80)]) : (0.0f);
          float v218_data = s0[4];
          float v223_data = s0[13];
          float v228_data = s0[22];
          float v233_data = s0[31];
          float v238_data = s0[40];
          float v243_data = s0[49];
          float v248_data = s0[58];
          float v253_data = s0[67];
          float v258_data = s0[76];
          float v263_data = v502_g ? (glb_m1[(v501_lead + 100)]) : (0.0f);
          float v264_data = s0[5];
          float v269_data = s0[14];
          float v274_data = s0[23];
          float v279_data = s0[32];
          float v284_data = s0[41];
          float v289_data = s0[50];
          float v294_data = s0[59];
          float v299_data = s0[68];
          float v304_data = s0[77];
          float v309_data = v502_g ? (glb_m1[(v501_lead + 120)]) : (0.0f);
          float v310_data = s0[6];
          float v315_data = s0[15];
          float v320_data = s0[24];
          float v325_data = s0[33];
          float v330_data = s0[42];
          float v335_data = s0[51];
          float v340_data = s0[60];
          float v345_data = s0[69];
          float v350_data = s0[78];
          float v355_data = v502_g ? (glb_m1[(v501_lead + 140)]) : (0.0f);
          float v356_data = s0[7];
          float v361_data = s0[16];
          float v364_p = (((((((v33_data * v39_data) + (v79_data * v85_data)) + (v125_data * v131_data)) + (v171_data * v177_data)) + (v217_data * v223_data)) + (v263_data * v269_data)) + (v309_data * v315_data)) + (v355_data * v361_data);
          float v366_data = s0[25];
          float v369_p = (((((((v33_data * v44_data) + (v79_data * v90_data)) + (v125_data * v136_data)) + (v171_data * v182_data)) + (v217_data * v228_data)) + (v263_data * v274_data)) + (v309_data * v320_data)) + (v355_data * v366_data);
          float v371_data = s0[34];
          float v374_p = (((((((v33_data * v49_data) + (v79_data * v95_data)) + (v125_data * v141_data)) + (v171_data * v187_data)) + (v217_data * v233_data)) + (v263_data * v279_data)) + (v309_data * v325_data)) + (v355_data * v371_data);
          float v376_data = s0[43];
          float v379_p = (((((((v33_data * v54_data) + (v79_data * v100_data)) + (v125_data * v146_data)) + (v171_data * v192_data)) + (v217_data * v238_data)) + (v263_data * v284_data)) + (v309_data * v330_data)) + (v355_data * v376_data);
          float v381_data = s0[52];
          float v384_p = (((((((v33_data * v59_data) + (v79_data * v105_data)) + (v125_data * v151_data)) + (v171_data * v197_data)) + (v217_data * v243_data)) + (v263_data * v289_data)) + (v309_data * v335_data)) + (v355_data * v381_data);
          float v386_data = s0[61];
          float v389_p = (((((((v33_data * v64_data) + (v79_data * v110_data)) + (v125_data * v156_data)) + (v171_data * v202_data)) + (v217_data * v248_data)) + (v263_data * v294_data)) + (v309_data * v340_data)) + (v355_data * v386_data);
          float v391_data = s0[70];
          float v394_p = (((((((v33_data * v69_data) + (v79_data * v115_data)) + (v125_data * v161_data)) + (v171_data * v207_data)) + (v217_data * v253_data)) + (v263_data * v299_data)) + (v309_data * v345_data)) + (v355_data * v391_data);
          float v396_data = s0[79];
          float v399_p = (((((((v33_data * v74_data) + (v79_data * v120_data)) + (v125_data * v166_data)) + (v171_data * v212_data)) + (v217_data * v258_data)) + (v263_data * v304_data)) + (v309_data * v350_data)) + (v355_data * v396_data);
          float v401_data = v502_g ? (glb_m1[(v501_lead + 160)]) : (0.0f);
          float v402_data = s0[8];
          ir1[0] = (((((((((v33_data * v34_data) + (v79_data * v80_data)) + (v125_data * v126_data)) + (v171_data * v172_data)) + (v217_data * v218_data)) + (v263_data * v264_data)) + (v309_data * v310_data)) + (v355_data * v356_data)) + (v401_data * v402_data));
          float v407_data = s0[17];
          ir1[1] = (v364_p + (v401_data * v407_data));
          float v412_data = s0[26];
          ir1[2] = (v369_p + (v401_data * v412_data));
          float v417_data = s0[35];
          ir1[3] = (v374_p + (v401_data * v417_data));
          float v422_data = s0[44];
          ir1[4] = (v379_p + (v401_data * v422_data));
          float v427_data = s0[53];
          ir1[5] = (v384_p + (v401_data * v427_data));
          float v432_data = s0[62];
          ir1[6] = (v389_p + (v401_data * v432_data));
          float v437_data = s0[71];
          ir1[7] = (v394_p + (v401_data * v437_data));
          float v442_data = s0[80];
          ir1[8] = (v399_p + (v401_data * v442_data));
          // r1 = ir1 + r0
          if (v502_g) {
            #pragma unroll
            for (int32_t v458_n1 = 0; v458_n1 < 9; ++v458_n1) {
              float v460_data = ir1[v458_n1];
              float v461_data = r0[v458_n1];
              r1[v458_n1] = (v461_data + v460_data);
            }
          }
          bool v519_g = allowed_next && v502_g;
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v514_i1 = 0; v514_i1 < 9; ++v514_i1) {
            float v517_data = v519_g ? (wrap_glb_m0[(v501_lead + (v514_i1 * 20))]) : (0.0f);
            r0[v514_i1] = v517_data;
          }
          // glb_m0 = store{r>g}(r1);
          if (v502_g) {
            #pragma unroll
            for (int32_t v464_i1 = 0; v464_i1 < 9; ++v464_i1) {
              float v466_data = r1[v464_i1];
              glb_m0[(v501_lead + (v464_i1 * 20))] = v466_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
          bool v520_g = allowed_next && v502_g;
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v506_i1 = 0; v506_i1 < 9; ++v506_i1) {
            float v509_data = v520_g ? (wrap_glb_m0[(v501_lead + (v506_i1 * 20))]) : (0.0f);
            r0[v506_i1] = v509_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        if (allowed_next) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        }
        if (allowed_next) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        if (v483_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
          }
        }
        __pipeline_commit();
        v476_flagWord = v477_flagWordNext;
        v477_flagWordNext = flagWordAhead;
        v528_elementIndex = wrap_elementIndex;
      }
      __pipeline_wait_prior(0);
    }
  }
}

