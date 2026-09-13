// === base name ===
kernel_4bba4ecc17892f5e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4bba4ecc17892f5e = {{2, 64, 1}, 2, 2, 1, 64, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4bba4ecc17892f5e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4bba4ecc17892f5e(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4bba4ecc17892f5e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (2, 64, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4bba4ecc17892f5e, block.x * block.y * block.z, 640 * sizeof(__float128));
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
  config.block[0] = 2;
  config.block[1] = 64;
  config.block[2] = 1;
  config.sharedMemBytes = 640 * sizeof(__float128);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4bba4ecc17892f5e(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4bba4ecc17892f5e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_4bba4ecc17892f5e, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_4bba4ecc17892f5e<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_4bba4ecc17892f5e(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 2 lanes x 64 per block = block 2x64x1, 10240 B shared, occupancy grid
    // operands:
    //   m0 2×2(2×2) {0..2}×{0..2} strided
    //   m1 2×2(2×2) {0..2}×{0..2} strided
    //   m2 2×2(2×2) {0..2}×{0..2} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[2,64,1],"cooperative":false,"lead_width":1,"mults_per_block":64,"persistent":true,"sections":[{"barrier":false,"mults_per_block":64,"shared_elements":640}],"shared_bytes":10240,"shared_elements":640,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[10 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[8];
      __float128 * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[v5_batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[v5_batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[v5_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v23_lead = v19_lead + (v20_i0 * 2);
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 2; ++v21_i1) {
              __float128 v26_data = glb_m1[(v23_lead + (v21_i1 * 2))];
              r0[(v20_i0 + v21_i1)] = v26_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 2], &glb_m2[0 + 0 + 1 * threadIdx.x + 2], 16);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          __float128 r1[2]{};
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
          // r1 = +(r0 * s0) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 ir1[2]{};
          __float128 v32_data = r0[0];
          __float128 v33_data = s0[0];
          __float128 v35_data = ir1[0];
          ir1[0] = (v35_data + (v32_data * v33_data));
          __float128 v38_data = s0[2];
          __float128 v40_data = ir1[1];
          ir1[1] = (v40_data + (v32_data * v38_data));
          __float128 v42_data = r0[1];
          __float128 v43_data = s0[1];
          __float128 v45_data = ir1[0];
          ir1[0] = (v45_data + (v42_data * v43_data));
          __float128 v48_data = s0[3];
          __float128 v50_data = ir1[1];
          ir1[1] = (v50_data + (v42_data * v48_data));
          #pragma unroll
          for (int32_t v52_n0 = 0; v52_n0 < 1; ++v52_n0) {
            #pragma unroll
            for (int32_t v53_n1 = 0; v53_n1 < 2; ++v53_n1) {
              int32_t v54_a = v52_n0 + v53_n1;
              __float128 v55_data = ir1[v54_a];
              r1[v54_a] = v55_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v56_i0 = 0; v56_i0 < 1; ++v56_i0) {
            int32_t v61_lead = v19_lead + (v56_i0 * 2);
            #pragma unroll
            for (int32_t v57_i1 = 0; v57_i1 < 2; ++v57_i1) {
              __float128 v59_data = r1[(v56_i0 + v57_i1)];
              glb_m0[(v61_lead + (v57_i1 * 2))] = v59_data;
            }
          }
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
        }
      }
    }
  }
}

