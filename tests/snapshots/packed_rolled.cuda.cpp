// === base name ===
kernel_88a9f70f11ecce3d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_88a9f70f11ecce3d = {{32, 4, 1}, 32, 64, 1, 4, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_88a9f70f11ecce3d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_88a9f70f11ecce3d(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_88a9f70f11ecce3d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_88a9f70f11ecce3d, block.x * block.y * block.z, 2304 * sizeof(float));
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
void launcher_kernel_88a9f70f11ecce3d(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_88a9f70f11ecce3d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_88a9f70f11ecce3d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_88a9f70f11ecce3d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_88a9f70f11ecce3d(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=rows
    // launch: 32 lanes (64 active) x 4 per block = block 32x4x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 64×6(64×6) {0..64}×{0..6} strided
    //   m1 64×96(64×96) {0..64}×{0..96} none ordered rows-packed
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
      size_t v124_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v124_flagSecondAhead < numElements0) ? v124_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 576 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v134_i = 0; v134_i < 18; ++v134_i) {
        int32_t v136_a = threadIdx.x + (v134_i * 32);
        __pipeline_memcpy_async(&s0[v136_a], &peel_glb_m2[v136_a], 4);
      }
      __pipeline_commit();
      uint32_t v128_flagWord = flagWordFirst;
      uint32_t v129_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 576 + 0 + m2_extraOffset];
        size_t v146_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v146_flagAheadAhead < numElements0) ? v146_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v128_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 64), (0, 6)] [(0, 96)]
          float ir0[12]{};
          int32_t v28_lead = threadIdx.x % 32;
          bool v33_g = v28_lead >= 7;
          int32_t v70_row = v28_lead + 32;
          bool v74_g = v28_lead < 14;
          __syncwarp();
          #pragma unroll 4
          for (int32_t v26_k0 = 0; v26_k0 < 96; ++v26_k0) {
            int32_t v29_c = v26_k0 * 39;
            float v34_data = v33_g ? (glb_m1[((v28_lead + v29_c) + -7)]) : (0.0f);
            float v36_data = s0[v26_k0];
            float v38_data = ir0[0];
            ir0[0] = (v38_data + (v34_data * v36_data));
            float v42_data = s0[(v26_k0 + 96)];
            float v44_data = ir0[2];
            ir0[2] = (v44_data + (v34_data * v42_data));
            float v48_data = s0[(v26_k0 + 192)];
            float v50_data = ir0[4];
            ir0[4] = (v50_data + (v34_data * v48_data));
            float v54_data = s0[(v26_k0 + 288)];
            float v56_data = ir0[6];
            ir0[6] = (v56_data + (v34_data * v54_data));
            float v60_data = s0[(v26_k0 + 384)];
            float v62_data = ir0[8];
            ir0[8] = (v62_data + (v34_data * v60_data));
            float v66_data = s0[(v26_k0 + 480)];
            float v68_data = ir0[10];
            ir0[10] = (v68_data + (v34_data * v66_data));
            float v75_data = v74_g ? (glb_m1[((v70_row + v29_c) + -7)]) : (0.0f);
            float v78_data = ir0[1];
            ir0[1] = (v78_data + (v75_data * v36_data));
            float v83_data = ir0[3];
            ir0[3] = (v83_data + (v75_data * v42_data));
            float v88_data = ir0[5];
            ir0[5] = (v88_data + (v75_data * v48_data));
            float v93_data = ir0[7];
            ir0[7] = (v93_data + (v75_data * v54_data));
            float v98_data = ir0[9];
            ir0[9] = (v98_data + (v75_data * v60_data));
            float v103_data = ir0[11];
            ir0[11] = (v103_data + (v75_data * v66_data));
          }
          // r0 = ir0
          #pragma unroll
          for (int32_t v107_n0 = 0; v107_n0 < 2; ++v107_n0) {
            #pragma unroll
            for (int32_t v108_n1 = 0; v108_n1 < 6; ++v108_n1) {
              int32_t v110_a = v107_n0 + (v108_n1 * 2);
              float v111_data = ir0[v110_a];
              r0[v110_a] = v111_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v114_i0 = 0; v114_i0 < 2; ++v114_i0) {
            int32_t v120_lead = v28_lead + (v114_i0 * 32);
            #pragma unroll
            for (int32_t v115_i1 = 0; v115_i1 < 6; ++v115_i1) {
              float v118_data = r0[(v114_i0 + (v115_i1 * 2))];
              glb_m0[(v120_lead + (v115_i1 * 64))] = v118_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v138_i = 0; v138_i < 18; ++v138_i) {
          int32_t v140_a = threadIdx.x + (v138_i * 32);
          __pipeline_memcpy_async(&s0[v140_a], &wrap_glb_m2[v140_a], 4);
        }
        __pipeline_commit();
        v128_flagWord = v129_flagWordNext;
        v129_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

