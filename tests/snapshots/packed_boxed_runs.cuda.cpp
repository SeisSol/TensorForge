// === base name ===
kernel_441b42b1b6b541a1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_441b42b1b6b541a1 = {{32, 4, 1}, 32, 32, 1, 4, 7680, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_441b42b1b6b541a1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_441b42b1b6b541a1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_441b42b1b6b541a1(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_441b42b1b6b541a1, block.x * block.y * block.z, 1920 * sizeof(float));
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
  config.sharedMemBytes = 1920 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_441b42b1b6b541a1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_441b42b1b6b541a1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_441b42b1b6b541a1, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  static const tensorforge::ValueArray<int32_t, 288> glb_m1_packArg{{0, 0, 9, 3, 9, 17, 6, 23, 25, 2, 42, 13, 5, 53, 21, 1, 69, 22, 4, 90, 17, 0, 103, 18, 3, 121, 13, 6, 131, 21, 2, 146, 22, 5, 166, 17, 1, 178, 18, 4, 195, 13, 0, 204, 14, 3, 218, 22, 6, 237, 17, 2, 248, 18, 5, 264, 26, 1, 285, 14, 4, 298, 22, 0, 316, 10, 3, 326, 18, 6, 341, 26, 2, 361, 14, 5, 373, 22, 1, 390, 10, 4, 399, 18, 0, 413, 19, 3, 432, 14, 6, 443, 22, 2, 459, 23, 5, 480, 18, 1, 493, 19, 4, 511, 14, 0, 521, 15, 3, 536, 23, 6, 556, 18, 2, 568, 19, 5, 585, 14, 1, 594, 15, 4, 608, 23, 0, 627, 11, 3, 638, 19, 6, 654, 27, 2, 675, 15, 5, 688, 23, 1, 706, 11, 4, 716, 19, 0, 731, 20, 3, 751, 15, 6, 763, 23, 2, 780, 11, 5, 789, 19, 1, 803, 20, 4, 822, 15, 0, 833, 16, 3, 849, 24, 6, 870, 19, 2, 883, 20, 5, 901, 15, 1, 911, 16, 4, 926, 24, 0, 946, 12, 3, 958, 20, 6, 975, 15, 2, 984, 16, 5, 998, 24, 1, 1017, 12, 4, 1028, 20, 0, 1044, 21, 3, 1065, 16, 6, 1078, 24, 2, 1096, 12, 5, 1106, 20, 1, 1121, 21, 4, 1141, 16, 0, 1153, 17, 3, 1170, 12, 6, 1179, 20, 2, 1193, 21, 5, 1212, 16, 1, 1223, 17, 4, 1239, 25, 0, 1260, 13, 3, 1273, 21, 6, 1291, 16, 2, 1301, 17, 5, 1316, 25, 1, 1336, 13, 4, 1348, 21, 0, 1365, 9, 3, 1374, 17, 6, 1388, 25, 2, 1407, 13, 5, 1418, 21}};
  kernel_kernel_441b42b1b6b541a1<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, glb_m1_packArg, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_441b42b1b6b541a1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, __grid_constant__ const tensorforge::ValueArray<int32_t, 288> glb_m1_pack, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=runs
    // launch: 32 lanes x 4 per block = block 32x4x1, 7680 B shared, occupancy grid
    // operands:
    //   m0 56×5(32×5) {9..41}×{0..5} strided
    //   m1 56×96(32×96) {9..41}×{0..96} none ordered runs-packed
    //   m2 96×5(96×5) {0..96}×{0..5} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":1920}],"shared_bytes":7680,"shared_elements":1920,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[9,0],[41,5]],"name":"m0","ordered":false,"parts":1,"shape":[56,5],"variant":false},{"addressing":"none","alias":"A","bbox":[[9,0],[41,96]],"name":"m1","ordered":true,"parts":1,"shape":[56,96],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[96,5]],"name":"m2","ordered":false,"parts":1,"shape":[96,5],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[9,0],[41,5]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,5]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[9,0],[41,96]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,96]},{"addressing":"strided","bbox":[[0,0],[96,5]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[96,5]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[480 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v237_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v237_flagSecondAhead < numElements0) ? v237_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 480 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v247_i = 0; v247_i < 15; ++v247_i) {
        int32_t v249_a = threadIdx.x + (v247_i * 32);
        __pipeline_memcpy_async(&s0[v249_a], &peel_glb_m2[v249_a], 4);
      }
      __pipeline_commit();
      uint32_t v241_flagWord = flagWordFirst;
      uint32_t v242_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 480 + 0 + m2_extraOffset];
        size_t v259_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v259_flagAheadAhead < numElements0) ? v259_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v241_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 160 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[10]{};
          // ir0 = +(glb_m1 * s0)
          // [(9, 41), (0, 5)] [(0, 96)]
          float ir0[10]{};
          int32_t v28_lead = threadIdx.x % 32;
          int32_t v30_row = v28_lead + -9;
          bool v43_g = v28_lead >= 9;
          int32_t v119_row = v28_lead + 32;
          int32_t v120_row = v119_row + -9;
          bool v131_g = v28_lead < 9;
          __syncwarp();
          #pragma unroll 4
          for (int32_t v26_k0 = 0; v26_k0 < 96; ++v26_k0) {
            int32_t v29_c = v26_k0 * 3;
            int32_t v31_start = glb_m1_pack[v29_c];
            int32_t v33_off = glb_m1_pack[(v29_c + 1)];
            int32_t v35_stop = glb_m1_pack[(v29_c + 2)];
            int32_t v41_a = ((v28_lead + v33_off) - v31_start) - 9;
            bool v44_g = v43_g && ((v30_row >= v31_start) && (v30_row < v35_stop));
            float v45_data = v44_g ? (glb_m1[v41_a]) : (0.0f);
            float v47_data = s0[v26_k0];
            float v49_data = ir0[0];
            ir0[0] = (v49_data + (v45_data * v47_data));
            float v62_data = v44_g ? (glb_m1[v41_a]) : (0.0f);
            float v64_data = s0[(v26_k0 + 96)];
            float v66_data = ir0[2];
            ir0[2] = (v66_data + (v62_data * v64_data));
            float v79_data = v44_g ? (glb_m1[v41_a]) : (0.0f);
            float v81_data = s0[(v26_k0 + 192)];
            float v83_data = ir0[4];
            ir0[4] = (v83_data + (v79_data * v81_data));
            float v96_data = v44_g ? (glb_m1[v41_a]) : (0.0f);
            float v98_data = s0[(v26_k0 + 288)];
            float v100_data = ir0[6];
            ir0[6] = (v100_data + (v96_data * v98_data));
            float v113_data = v44_g ? (glb_m1[v41_a]) : (0.0f);
            float v115_data = s0[(v26_k0 + 384)];
            float v117_data = ir0[8];
            ir0[8] = (v117_data + (v113_data * v115_data));
            int32_t v129_a = ((v119_row + v33_off) - v31_start) - 9;
            bool v132_g = v131_g && ((v120_row >= v31_start) && (v120_row < v35_stop));
            float v133_data = v132_g ? (glb_m1[v129_a]) : (0.0f);
            float v136_data = ir0[1];
            ir0[1] = (v136_data + (v133_data * v47_data));
            float v149_data = v132_g ? (glb_m1[v129_a]) : (0.0f);
            float v152_data = ir0[3];
            ir0[3] = (v152_data + (v149_data * v64_data));
            float v165_data = v132_g ? (glb_m1[v129_a]) : (0.0f);
            float v168_data = ir0[5];
            ir0[5] = (v168_data + (v165_data * v81_data));
            float v181_data = v132_g ? (glb_m1[v129_a]) : (0.0f);
            float v184_data = ir0[7];
            ir0[7] = (v184_data + (v181_data * v98_data));
            float v197_data = v132_g ? (glb_m1[v129_a]) : (0.0f);
            float v200_data = ir0[9];
            ir0[9] = (v200_data + (v197_data * v115_data));
          }
          // r0 = ir0
          if (v43_g) {
            #pragma unroll
            for (int32_t v205_n1 = 0; v205_n1 < 5; ++v205_n1) {
              int32_t v206_a = v205_n1 * 2;
              float v208_data = ir0[v206_a];
              r0[v206_a] = v208_data;
            }
          }
          if (v131_g) {
            #pragma unroll
            for (int32_t v210_n1 = 0; v210_n1 < 5; ++v210_n1) {
              int32_t v212_a = 1 + (v210_n1 * 2);
              float v213_data = ir0[v212_a];
              r0[v212_a] = v213_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          if (v43_g) {
            int32_t v223_a = v28_lead - 9;
            #pragma unroll
            for (int32_t v217_i1 = 0; v217_i1 < 5; ++v217_i1) {
              float v220_data = r0[(v217_i1 * 2)];
              glb_m0[(v223_a + (v217_i1 * 32))] = v220_data;
            }
          }
          if (v131_g) {
            int32_t v233_a = (v28_lead + 32_i32) - 9;
            #pragma unroll
            for (int32_t v227_i1 = 0; v227_i1 < 5; ++v227_i1) {
              float v230_data = r0[(1 + (v227_i1 * 2))];
              glb_m0[(v233_a + (v227_i1 * 32))] = v230_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v251_i = 0; v251_i < 15; ++v251_i) {
          int32_t v253_a = threadIdx.x + (v251_i * 32);
          __pipeline_memcpy_async(&s0[v253_a], &wrap_glb_m2[v253_a], 4);
        }
        __pipeline_commit();
        v241_flagWord = v242_flagWordNext;
        v242_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

