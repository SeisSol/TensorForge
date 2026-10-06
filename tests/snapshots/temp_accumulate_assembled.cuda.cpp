// === base name ===
kernel_311d410a67983301

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_311d410a67983301 = {{16, 8, 1}, 16, 12, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_311d410a67983301(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_311d410a67983301(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_311d410a67983301(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_311d410a67983301, block.x * block.y * block.z, 384 * sizeof(float));
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
  config.block[0] = 16;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_311d410a67983301(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_311d410a67983301(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_311d410a67983301, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_311d410a67983301<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_311d410a67983301(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 1536 B shared, occupancy grid
    // operands:
    //   m0 6(6) {0..6} strided
    //   m1 6(6) {0..6} strided
    //   m2 12(12) {0..12} strided
    //   m3 12(12) {0..12} strided
    // operations:
    //   t0[i]@{0..6} = m0[i]
    //   t0[i]@{6..12} = m1[i]
    //   t0[i] += m2[i]
    //   m3[i] = t0[i]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"a","bbox":[[0],[6]],"name":"m0","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"strided","alias":"b","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"strided","alias":"w","bbox":[[0],[12]],"name":"m2","ordered":false,"parts":1,"shape":[12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0],[12]],"name":"m3","ordered":false,"parts":1,"shape":[12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m0","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[6],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0],[12]],"is_tmp":true,"name":"t0","offset":[0],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[12]],"is_tmp":false,"name":"m2","offset":[0],"shape":[12]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[12]],"is_tmp":false,"name":"m3","offset":[0],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[12]],"is_tmp":true,"name":"t0","offset":[0],"shape":[12]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[48 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 6 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 6 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 12 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v11_batchId0 * 12 + 0 + m3_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 16;
          bool v27_g = v26_lead < 6;
          if (v27_g) {
            float v30_data = __ldcg(&glb_m0[v26_lead]);
            r0[0] = v30_data;
          }
          float r2[1]{};
          // r2 = load{g>r}(glb_m1);
          if (v27_g) {
            float v34_data = __ldcg(&glb_m1[v26_lead]);
            r2[0] = v34_data;
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 6)] []
          float v36_data = r0[0];
          float v37_data = r1[0];
          r1[0] = (v37_data + v36_data);
          // s0 = store{r>s}(localShrMem0, r1);
          if (v27_g) {
            float v39_data = r1[0];
            s0[v26_lead] = v39_data;
          }
          float r4[1]{};
          // r4 = load{g>r}(glb_m2);
          bool v43_g = v26_lead < 12;
          if (v43_g) {
            float v46_data = __ldcg(&glb_m2[v26_lead]);
            r4[0] = v46_data;
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[1]{};
          // ir3 = +(r2)
          // [(0, 6)] []
          float ir3[1]{};
          float v49_data = r2[0];
          float v50_data = ir3[0];
          ir3[0] = (v50_data + v49_data);
          // r3 = ir3
          if (v27_g) {
            float v52_data = ir3[0];
            r3[0] = v52_data;
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v27_g) {
            float v53_data = r3[0];
            s0[(v26_lead + 6)] = v53_data;
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[1]{};
          // ir5 = +(r4)
          // [(0, 12)] []
          float ir5[1]{};
          float v59_data = r4[0];
          float v60_data = ir5[0];
          ir5[0] = (v60_data + v59_data);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r5 = ir5 + s0
          if (v43_g) {
            float v62_data = ir5[0];
            float v65_data = s0[v26_lead];
            r5[0] = (v65_data + v62_data);
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s0 = store{r>s}(localShrMem0, r5);
          if (v43_g) {
            float v67_data = r5[0];
            s0[v26_lead] = v67_data;
          }
          float r6[1]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir6 = +(s0)
          // [(0, 12)] []
          float ir6[1]{};
          float v74_data = v43_g ? (s0[v26_lead]) : (0.0f);
          float v75_data = ir6[0];
          ir6[0] = (v75_data + v74_data);
          // r6 = ir6
          if (v43_g) {
            float v77_data = ir6[0];
            r6[0] = v77_data;
          }
          // glb_m3 = store{r>g}(r6);
          if (v43_g) {
            float v78_data = r6[0];
            glb_m3[v26_lead] = v78_data;
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

