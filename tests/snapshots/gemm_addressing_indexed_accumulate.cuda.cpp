// === base name ===
kernel_720374368d38122e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_720374368d38122e = {{32, 4, 1}, 32, 20, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_720374368d38122e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_720374368d38122e(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_720374368d38122e(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_720374368d38122e, block.x * block.y * block.z, 384 * sizeof(float));
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
void launcher_kernel_720374368d38122e(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_720374368d38122e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_720374368d38122e, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  float *const * m0Arg = (float *const *)m0;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_720374368d38122e<<<grid,block,config.sharedMemBytes,stream>>>(m0Arg, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_720374368d38122e(float *const * __restrict__ m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
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
      size_t v471_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v471_flagSecondAhead < numElements0) ? v471_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const float *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 12 + m2_extraOffset];
      bool v481_g = threadIdx.x < 17;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      if (allowed_peel) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      }
      if (allowed_peel) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      if (v481_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
        }
      }
      __pipeline_commit();
      float *const __restrict__ peel_glb_m0 = &m0[static_cast<size_t>(elementIndices[v3_batchId1]) * 2 + 1][0 + m0_extraOffset];
      int32_t v498_lead = threadIdx.x % 32;
      bool v499_g = v498_lead < 20;
      bool v515_g = allowed_peel && v499_g;
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v507_i1 = 0; v507_i1 < 9; ++v507_i1) {
        float v510_data = v515_g ? (peel_glb_m0[(v498_lead + (v507_i1 * 20))]) : (0.0f);
        r0[v507_i1] = v510_data;
      }
      uint32_t v475_flagWord = flagWordFirst;
      uint32_t v476_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v12_batchId1]) * 81 + 0 + 12 + m2_extraOffset];
        float *const __restrict__ wrap_glb_m0 = &m0[static_cast<size_t>(elementIndices[v12_batchId1]) * 2 + 1][0 + m0_extraOffset];
        size_t v520_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v520_flagAheadAhead < numElements0) ? v520_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v475_flagWord);
        const bool allowed_next = static_cast<bool>(v476_flagWordNext);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[static_cast<size_t>(elementIndices[v9_batchId0]) * 2 + 1][0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          // ir1 = +(glb_m1 * s0)
          // [(0, 20), (0, 9)] [(0, 9)]
          float ir1[9]{};
          float v32_data = v499_g ? (glb_m1[v498_lead]) : (0.0f);
          __syncwarp();
          float v33_data = s0[0];
          float v38_data = s0[9];
          float v43_data = s0[18];
          float v48_data = s0[27];
          float v53_data = s0[36];
          float v58_data = s0[45];
          float v63_data = s0[54];
          float v68_data = s0[63];
          float v73_data = s0[72];
          float v78_data = v499_g ? (glb_m1[(v498_lead + 20)]) : (0.0f);
          float v79_data = s0[1];
          float v84_data = s0[10];
          float v89_data = s0[19];
          float v94_data = s0[28];
          float v99_data = s0[37];
          float v104_data = s0[46];
          float v109_data = s0[55];
          float v114_data = s0[64];
          float v119_data = s0[73];
          float v124_data = v499_g ? (glb_m1[(v498_lead + 40)]) : (0.0f);
          float v125_data = s0[2];
          float v130_data = s0[11];
          float v135_data = s0[20];
          float v140_data = s0[29];
          float v145_data = s0[38];
          float v150_data = s0[47];
          float v155_data = s0[56];
          float v160_data = s0[65];
          float v165_data = s0[74];
          float v170_data = v499_g ? (glb_m1[(v498_lead + 60)]) : (0.0f);
          float v171_data = s0[3];
          float v176_data = s0[12];
          float v181_data = s0[21];
          float v186_data = s0[30];
          float v191_data = s0[39];
          float v196_data = s0[48];
          float v201_data = s0[57];
          float v206_data = s0[66];
          float v211_data = s0[75];
          float v216_data = v499_g ? (glb_m1[(v498_lead + 80)]) : (0.0f);
          float v217_data = s0[4];
          float v222_data = s0[13];
          float v227_data = s0[22];
          float v232_data = s0[31];
          float v237_data = s0[40];
          float v242_data = s0[49];
          float v247_data = s0[58];
          float v252_data = s0[67];
          float v257_data = s0[76];
          float v262_data = v499_g ? (glb_m1[(v498_lead + 100)]) : (0.0f);
          float v263_data = s0[5];
          float v268_data = s0[14];
          float v273_data = s0[23];
          float v278_data = s0[32];
          float v283_data = s0[41];
          float v288_data = s0[50];
          float v293_data = s0[59];
          float v298_data = s0[68];
          float v303_data = s0[77];
          float v308_data = v499_g ? (glb_m1[(v498_lead + 120)]) : (0.0f);
          float v309_data = s0[6];
          float v314_data = s0[15];
          float v319_data = s0[24];
          float v324_data = s0[33];
          float v329_data = s0[42];
          float v334_data = s0[51];
          float v339_data = s0[60];
          float v344_data = s0[69];
          float v349_data = s0[78];
          float v354_data = v499_g ? (glb_m1[(v498_lead + 140)]) : (0.0f);
          float v355_data = s0[7];
          float v360_data = s0[16];
          float v363_p = (((((((v32_data * v38_data) + (v78_data * v84_data)) + (v124_data * v130_data)) + (v170_data * v176_data)) + (v216_data * v222_data)) + (v262_data * v268_data)) + (v308_data * v314_data)) + (v354_data * v360_data);
          float v365_data = s0[25];
          float v368_p = (((((((v32_data * v43_data) + (v78_data * v89_data)) + (v124_data * v135_data)) + (v170_data * v181_data)) + (v216_data * v227_data)) + (v262_data * v273_data)) + (v308_data * v319_data)) + (v354_data * v365_data);
          float v370_data = s0[34];
          float v373_p = (((((((v32_data * v48_data) + (v78_data * v94_data)) + (v124_data * v140_data)) + (v170_data * v186_data)) + (v216_data * v232_data)) + (v262_data * v278_data)) + (v308_data * v324_data)) + (v354_data * v370_data);
          float v375_data = s0[43];
          float v378_p = (((((((v32_data * v53_data) + (v78_data * v99_data)) + (v124_data * v145_data)) + (v170_data * v191_data)) + (v216_data * v237_data)) + (v262_data * v283_data)) + (v308_data * v329_data)) + (v354_data * v375_data);
          float v380_data = s0[52];
          float v383_p = (((((((v32_data * v58_data) + (v78_data * v104_data)) + (v124_data * v150_data)) + (v170_data * v196_data)) + (v216_data * v242_data)) + (v262_data * v288_data)) + (v308_data * v334_data)) + (v354_data * v380_data);
          float v385_data = s0[61];
          float v388_p = (((((((v32_data * v63_data) + (v78_data * v109_data)) + (v124_data * v155_data)) + (v170_data * v201_data)) + (v216_data * v247_data)) + (v262_data * v293_data)) + (v308_data * v339_data)) + (v354_data * v385_data);
          float v390_data = s0[70];
          float v393_p = (((((((v32_data * v68_data) + (v78_data * v114_data)) + (v124_data * v160_data)) + (v170_data * v206_data)) + (v216_data * v252_data)) + (v262_data * v298_data)) + (v308_data * v344_data)) + (v354_data * v390_data);
          float v395_data = s0[79];
          float v398_p = (((((((v32_data * v73_data) + (v78_data * v119_data)) + (v124_data * v165_data)) + (v170_data * v211_data)) + (v216_data * v257_data)) + (v262_data * v303_data)) + (v308_data * v349_data)) + (v354_data * v395_data);
          float v400_data = v499_g ? (glb_m1[(v498_lead + 160)]) : (0.0f);
          float v401_data = s0[8];
          ir1[0] = (((((((((v32_data * v33_data) + (v78_data * v79_data)) + (v124_data * v125_data)) + (v170_data * v171_data)) + (v216_data * v217_data)) + (v262_data * v263_data)) + (v308_data * v309_data)) + (v354_data * v355_data)) + (v400_data * v401_data));
          float v406_data = s0[17];
          ir1[1] = (v363_p + (v400_data * v406_data));
          float v411_data = s0[26];
          ir1[2] = (v368_p + (v400_data * v411_data));
          float v416_data = s0[35];
          ir1[3] = (v373_p + (v400_data * v416_data));
          float v421_data = s0[44];
          ir1[4] = (v378_p + (v400_data * v421_data));
          float v426_data = s0[53];
          ir1[5] = (v383_p + (v400_data * v426_data));
          float v431_data = s0[62];
          ir1[6] = (v388_p + (v400_data * v431_data));
          float v436_data = s0[71];
          ir1[7] = (v393_p + (v400_data * v436_data));
          float v441_data = s0[80];
          ir1[8] = (v398_p + (v400_data * v441_data));
          // r1 = ir1 + r0
          if (v499_g) {
            #pragma unroll
            for (int32_t v457_n1 = 0; v457_n1 < 9; ++v457_n1) {
              float v459_data = ir1[v457_n1];
              float v460_data = r0[v457_n1];
              r1[v457_n1] = (v460_data + v459_data);
            }
          }
          bool v516_g = allowed_next && v499_g;
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v511_i1 = 0; v511_i1 < 9; ++v511_i1) {
            float v514_data = v516_g ? (wrap_glb_m0[(v498_lead + (v511_i1 * 20))]) : (0.0f);
            r0[v511_i1] = v514_data;
          }
          // glb_m0 = store{r>g}(r1);
          if (v499_g) {
            #pragma unroll
            for (int32_t v463_i1 = 0; v463_i1 < 9; ++v463_i1) {
              float v465_data = r1[v463_i1];
              glb_m0[(v498_lead + (v463_i1 * 20))] = v465_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
          bool v517_g = allowed_next && v499_g;
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v503_i1 = 0; v503_i1 < 9; ++v503_i1) {
            float v506_data = v517_g ? (wrap_glb_m0[(v498_lead + (v503_i1 * 20))]) : (0.0f);
            r0[v503_i1] = v506_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        if (allowed_next) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        }
        if (allowed_next) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        if (v481_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
          }
        }
        __pipeline_commit();
        v475_flagWord = v476_flagWordNext;
        v476_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

