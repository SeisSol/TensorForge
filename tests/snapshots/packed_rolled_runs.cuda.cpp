// === base name ===
kernel_065b2fd2cc821022

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_065b2fd2cc821022 = {{32, 4, 1}, 32, 64, 1, 4, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_065b2fd2cc821022(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_065b2fd2cc821022(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_065b2fd2cc821022(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_065b2fd2cc821022, block.x * block.y * block.z, 2304 * sizeof(float));
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_065b2fd2cc821022(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_065b2fd2cc821022(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_065b2fd2cc821022, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  static const tensorforge::ValueArray<int32_t, 288> glb_m1_packArg{{3, 0, 14, 10, 11, 26, 17, 27, 38, 24, 48, 50, 31, 74, 45, 9, 88, 28, 16, 107, 40, 23, 131, 35, 30, 143, 47, 8, 160, 30, 15, 182, 42, 22, 209, 37, 29, 224, 49, 7, 244, 32, 14, 269, 27, 21, 282, 39, 28, 300, 51, 6, 323, 17, 13, 334, 29, 20, 350, 41, 27, 371, 53, 5, 397, 19, 12, 411, 31, 19, 430, 43, 26, 454, 38, 4, 466, 21, 11, 483, 33, 18, 505, 45, 25, 532, 40, 3, 547, 23, 10, 567, 35, 17, 592, 30, 24, 605, 42, 31, 623, 54, 9, 646, 20, 16, 657, 32, 23, 673, 44, 30, 694, 56, 8, 720, 22, 15, 734, 34, 22, 753, 46, 29, 777, 41, 7, 789, 24, 14, 806, 36, 21, 828, 48, 28, 855, 43, 6, 870, 26, 13, 890, 38, 20, 915, 33, 27, 928, 45, 5, 946, 28, 12, 969, 23, 19, 980, 35, 26, 996, 47, 4, 1017, 30, 11, 1043, 25, 18, 1057, 37, 25, 1076, 49, 3, 1100, 15, 10, 1112, 27, 17, 1129, 39, 24, 1151, 51, 31, 1178, 46, 9, 1193, 29, 16, 1213, 41, 23, 1238, 36, 30, 1251, 48, 8, 1269, 31, 15, 1292, 26, 22, 1303, 38, 29, 1319, 50, 7, 1340, 33, 14, 1366, 28, 21, 1380, 40, 28, 1399, 52, 6, 1423, 18, 13, 1435, 30, 20, 1452, 42, 27, 1474, 54, 5, 1501, 20, 12, 1516, 32, 19, 1536, 44, 26, 1561, 39, 4, 1574, 22, 11, 1592, 34, 18, 1615, 29, 25, 1626, 41, 3, 1642, 24, 10, 1663, 36, 17, 1689, 31, 24, 1703, 43, 31, 1722, 55, 9, 1746, 21, 16, 1758, 33, 23, 1775, 45, 30, 1797, 57}};
  kernel_kernel_065b2fd2cc821022<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, glb_m1_packArg, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_065b2fd2cc821022(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, __grid_constant__ const tensorforge::ValueArray<int32_t, 288> glb_m1_pack, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=runs
    // launch: 32 lanes (64 active) x 4 per block = block 32x4x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 64×6(64×6) {0..64}×{0..6} strided
    //   m1 64×96(64×96) {0..64}×{0..96} none ordered runs-packed
    //   m2 96×6(96×6) {0..96}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,96]],"name":"m1","ordered":true,"parts":1,"shape":[64,96],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[96,6]],"name":"m2","ordered":false,"parts":1,"shape":[96,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,96]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,96]},{"addressing":"strided","bbox":[[0,0],[96,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[96,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[576 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v226_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v226_flagSecondAhead < numElements0) ? v226_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 576 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v236_i = 0; v236_i < 18; ++v236_i) {
        int32_t v238_a = threadIdx.x + (v236_i * 32);
        __pipeline_memcpy_async(&s0[v238_a], &peel_glb_m2[v238_a], 4);
      }
      __pipeline_commit();
      uint32_t v230_flagWord = flagWordFirst;
      uint32_t v231_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 576 + 0 + m2_extraOffset];
        size_t v248_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v248_flagAheadAhead < numElements0) ? v248_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v230_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 64), (0, 6)] [(0, 96)]
          float ir0[12]{};
          int32_t v28_lead = threadIdx.x % 32;
          int32_t v122_row = v28_lead + 32;
          __syncwarp();
          #pragma unroll 4
          for (int32_t v26_k0 = 0; v26_k0 < 96; ++v26_k0) {
            int32_t v29_c = v26_k0 * 3;
            int32_t v30_start = glb_m1_pack[v29_c];
            int32_t v32_off = glb_m1_pack[(v29_c + 1)];
            int32_t v34_stop = glb_m1_pack[(v29_c + 2)];
            bool v37_g = (v28_lead >= v30_start) && (v28_lead < v34_stop);
            int32_t v39_slot = (v28_lead + v32_off) - v30_start;
            float v41_data = v37_g ? (glb_m1[v39_slot]) : (0.0f);
            float v43_data = s0[v26_k0];
            float v45_data = ir0[0];
            ir0[0] = (v45_data + (v41_data * v43_data));
            float v56_data = v37_g ? (glb_m1[v39_slot]) : (0.0f);
            float v58_data = s0[(v26_k0 + 96)];
            float v60_data = ir0[2];
            ir0[2] = (v60_data + (v56_data * v58_data));
            float v71_data = v37_g ? (glb_m1[v39_slot]) : (0.0f);
            float v73_data = s0[(v26_k0 + 192)];
            float v75_data = ir0[4];
            ir0[4] = (v75_data + (v71_data * v73_data));
            float v86_data = v37_g ? (glb_m1[v39_slot]) : (0.0f);
            float v88_data = s0[(v26_k0 + 288)];
            float v90_data = ir0[6];
            ir0[6] = (v90_data + (v86_data * v88_data));
            float v101_data = v37_g ? (glb_m1[v39_slot]) : (0.0f);
            float v103_data = s0[(v26_k0 + 384)];
            float v105_data = ir0[8];
            ir0[8] = (v105_data + (v101_data * v103_data));
            float v116_data = v37_g ? (glb_m1[v39_slot]) : (0.0f);
            float v118_data = s0[(v26_k0 + 480)];
            float v120_data = ir0[10];
            ir0[10] = (v120_data + (v116_data * v118_data));
            bool v128_g = (v122_row >= v30_start) && (v122_row < v34_stop);
            int32_t v130_slot = (v122_row + v32_off) - v30_start;
            float v132_data = v128_g ? (glb_m1[v130_slot]) : (0.0f);
            float v135_data = ir0[1];
            ir0[1] = (v135_data + (v132_data * v43_data));
            float v146_data = v128_g ? (glb_m1[v130_slot]) : (0.0f);
            float v149_data = ir0[3];
            ir0[3] = (v149_data + (v146_data * v58_data));
            float v160_data = v128_g ? (glb_m1[v130_slot]) : (0.0f);
            float v163_data = ir0[5];
            ir0[5] = (v163_data + (v160_data * v73_data));
            float v174_data = v128_g ? (glb_m1[v130_slot]) : (0.0f);
            float v177_data = ir0[7];
            ir0[7] = (v177_data + (v174_data * v88_data));
            float v188_data = v128_g ? (glb_m1[v130_slot]) : (0.0f);
            float v191_data = ir0[9];
            ir0[9] = (v191_data + (v188_data * v103_data));
            float v202_data = v128_g ? (glb_m1[v130_slot]) : (0.0f);
            float v205_data = ir0[11];
            ir0[11] = (v205_data + (v202_data * v118_data));
          }
          // r0 = ir0
          #pragma unroll
          for (int32_t v209_n0 = 0; v209_n0 < 2; ++v209_n0) {
            #pragma unroll
            for (int32_t v210_n1 = 0; v210_n1 < 6; ++v210_n1) {
              int32_t v212_a = v209_n0 + (v210_n1 * 2);
              float v213_data = ir0[v212_a];
              r0[v212_a] = v213_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v216_i0 = 0; v216_i0 < 2; ++v216_i0) {
            int32_t v222_lead = v28_lead + (v216_i0 * 32);
            #pragma unroll
            for (int32_t v217_i1 = 0; v217_i1 < 6; ++v217_i1) {
              float v220_data = r0[(v216_i0 + (v217_i1 * 2))];
              glb_m0[(v222_lead + (v217_i1 * 64))] = v220_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v240_i = 0; v240_i < 18; ++v240_i) {
          int32_t v242_a = threadIdx.x + (v240_i * 32);
          __pipeline_memcpy_async(&s0[v242_a], &wrap_glb_m2[v242_a], 4);
        }
        __pipeline_commit();
        v230_flagWord = v231_flagWordNext;
        v231_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

