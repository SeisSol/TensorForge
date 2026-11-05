// === base name ===
kernel_3dd2764d15b689a3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3dd2764d15b689a3 = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3dd2764d15b689a3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3dd2764d15b689a3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3dd2764d15b689a3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3dd2764d15b689a3, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.block[0] = 8;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3dd2764d15b689a3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3dd2764d15b689a3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3dd2764d15b689a3, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_3dd2764d15b689a3<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_3dd2764d15b689a3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 16 per block = block 8x16x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 8×9(8×9) {0..8}×{0..9} strided
    //   m1 8×16(8×4) {0..8}×{9..13} none
    //   m2 16×9(4×9) {9..13}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[8,9]],"name":"m0","ordered":false,"parts":1,"shape":[8,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,9],[8,13]],"name":"m1","ordered":false,"parts":1,"shape":[8,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[9,0],[13,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,9],[8,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,16]},{"addressing":"strided","bbox":[[9,0],[13,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v233_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v233_flagSecondAhead < numElements0) ? v233_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 36 + 0 + m2_extraOffset];
      bool v241_g = threadIdx.x < 4;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 8], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 8], 4);
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 24], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 24], 4);
      if (v241_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v237_flagWord = flagWordFirst;
      uint32_t v238_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 36 + 0 + m2_extraOffset];
        size_t v267_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v267_flagAheadAhead < numElements0) ? v267_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v237_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[9]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 8), (0, 9)] [(9, 13)]
          float ir0[9]{};
          int32_t v29_lead = threadIdx.x % 8;
          float v33_data = glb_m1[v29_lead];
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v34_data = s0[0];
          float v39_data = s0[4];
          float v44_data = s0[8];
          float v49_data = s0[12];
          float v54_data = s0[16];
          float v59_data = s0[20];
          float v64_data = s0[24];
          float v69_data = s0[28];
          float v74_data = s0[32];
          float v79_data = glb_m1[(v29_lead + 8)];
          float v80_data = s0[1];
          float v85_data = s0[5];
          float v90_data = s0[9];
          float v95_data = s0[13];
          float v100_data = s0[17];
          float v105_data = s0[21];
          float v110_data = s0[25];
          float v115_data = s0[29];
          float v120_data = s0[33];
          float v125_data = glb_m1[(v29_lead + 16)];
          float v126_data = s0[2];
          float v131_data = s0[6];
          float v134_p = ((v33_data * v39_data) + (v79_data * v85_data)) + (v125_data * v131_data);
          float v136_data = s0[10];
          float v139_p = ((v33_data * v44_data) + (v79_data * v90_data)) + (v125_data * v136_data);
          float v141_data = s0[14];
          float v144_p = ((v33_data * v49_data) + (v79_data * v95_data)) + (v125_data * v141_data);
          float v146_data = s0[18];
          float v149_p = ((v33_data * v54_data) + (v79_data * v100_data)) + (v125_data * v146_data);
          float v151_data = s0[22];
          float v154_p = ((v33_data * v59_data) + (v79_data * v105_data)) + (v125_data * v151_data);
          float v156_data = s0[26];
          float v159_p = ((v33_data * v64_data) + (v79_data * v110_data)) + (v125_data * v156_data);
          float v161_data = s0[30];
          float v164_p = ((v33_data * v69_data) + (v79_data * v115_data)) + (v125_data * v161_data);
          float v166_data = s0[34];
          float v169_p = ((v33_data * v74_data) + (v79_data * v120_data)) + (v125_data * v166_data);
          float v171_data = glb_m1[(v29_lead + 24)];
          float v172_data = s0[3];
          ir0[0] = ((((v33_data * v34_data) + (v79_data * v80_data)) + (v125_data * v126_data)) + (v171_data * v172_data));
          float v177_data = s0[7];
          ir0[1] = (v134_p + (v171_data * v177_data));
          float v182_data = s0[11];
          ir0[2] = (v139_p + (v171_data * v182_data));
          float v187_data = s0[15];
          ir0[3] = (v144_p + (v171_data * v187_data));
          float v192_data = s0[19];
          ir0[4] = (v149_p + (v171_data * v192_data));
          float v197_data = s0[23];
          ir0[5] = (v154_p + (v171_data * v197_data));
          float v202_data = s0[27];
          ir0[6] = (v159_p + (v171_data * v202_data));
          float v207_data = s0[31];
          ir0[7] = (v164_p + (v171_data * v207_data));
          float v212_data = s0[35];
          ir0[8] = (v169_p + (v171_data * v212_data));
          // r0 = ir0
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v224_i0 = 0; v224_i0 < 1; ++v224_i0) {
            int32_t v229_lead = v29_lead + (v224_i0 * 8);
            #pragma unroll
            for (int32_t v225_i1 = 0; v225_i1 < 9; ++v225_i1) {
              float v227_data = ir0[v225_i1];
              glb_m0[(v229_lead + (v225_i1 * 8))] = v227_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 8], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 8], 4);
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 24], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 24], 4);
        if (v241_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v237_flagWord = v238_flagWordNext;
        v238_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

