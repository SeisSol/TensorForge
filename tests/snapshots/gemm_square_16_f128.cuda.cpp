// === base name ===
kernel_1cb98dafce0184f9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1cb98dafce0184f9 = {{2, 64, 1}, 2, 2, 1, 64, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1cb98dafce0184f9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1cb98dafce0184f9(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1cb98dafce0184f9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1cb98dafce0184f9, block.x * block.y * block.z, 640 * sizeof(__float128));
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
void launcher_kernel_1cb98dafce0184f9(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1cb98dafce0184f9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1cb98dafce0184f9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_1cb98dafce0184f9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_1cb98dafce0184f9(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 2 lanes x 64 per block = block 2x64x1, 10240 B shared, occupancy grid
    // operands:
    //   m0 2×2(2×2) {0..2}×{0..2} strided
    //   m1 2×2(2×2) {0..2}×{0..2} strided
    //   m2 2×2(2×2) {0..2}×{0..2} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[2,64,1],"cooperative":false,"lead_width":1,"mults_per_block":64,"persistent":true,"sections":[{"barrier":false,"mults_per_block":64,"shared_elements":640}],"shared_bytes":10240,"shared_elements":640,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[10 * threadIdx.y + 0];
      __float128 * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[v8_batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[v8_batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[v8_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 2);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 2; ++v24_i1) {
              __float128 v29_data = glb_m1[(v26_lead + (v24_i1 * 2))];
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 2], &glb_m2[0 + 0 + 1 * threadIdx.x + 2], 16);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          __float128 r1[2]{};
          // ir1 = +(r0 * s0)
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 ir1[2]{};
          __float128 v35_data = r0[0];
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
          __float128 v36_data = s0[0];
          __float128 v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          __float128 v41_data = s0[2];
          __float128 v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          __float128 v45_data = r0[1];
          __float128 v46_data = s0[1];
          __float128 v48_data = ir1[0];
          ir1[0] = (v48_data + (v45_data * v46_data));
          __float128 v51_data = s0[3];
          __float128 v53_data = ir1[1];
          ir1[1] = (v53_data + (v45_data * v51_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v55_n0 = 0; v55_n0 < 1; ++v55_n0) {
            #pragma unroll
            for (int32_t v56_n1 = 0; v56_n1 < 2; ++v56_n1) {
              int32_t v57_a = v55_n0 + v56_n1;
              __float128 v58_data = ir1[v57_a];
              r1[v57_a] = v58_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v59_i0 = 0; v59_i0 < 1; ++v59_i0) {
            int32_t v64_lead = v22_lead + (v59_i0 * 2);
            #pragma unroll
            for (int32_t v60_i1 = 0; v60_i1 < 2; ++v60_i1) {
              __float128 v62_data = r1[(v59_i0 + v60_i1)];
              glb_m0[(v64_lead + (v60_i1 * 2))] = v62_data;
            }
          }
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
        }
      }
    }
  }
}

