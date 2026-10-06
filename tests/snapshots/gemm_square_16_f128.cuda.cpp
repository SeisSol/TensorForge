// === base name ===
kernel_42b694891ef726d7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_42b694891ef726d7 = {{2, 64, 1}, 2, 2, 1, 64, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_42b694891ef726d7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_42b694891ef726d7(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_42b694891ef726d7(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_42b694891ef726d7, block.x * block.y * block.z, 640 * sizeof(__float128));
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
void launcher_kernel_42b694891ef726d7(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_42b694891ef726d7(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_42b694891ef726d7, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_42b694891ef726d7<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_42b694891ef726d7(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      __float128* tempShrMem = &localShrMem0[8];
      __float128 * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[v11_batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[v11_batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[v11_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 2);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 2; ++v27_i1) {
              __float128 v32_data = glb_m1[(v29_lead + (v27_i1 * 2))];
              r0[(v26_i0 + v27_i1)] = v32_data;
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
          // ir1 = +(r0 * s0)
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 ir1[2]{};
          __float128 v38_data = r0[0];
          __float128 v39_data = s0[0];
          __float128 v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          __float128 v44_data = s0[2];
          __float128 v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          __float128 v48_data = r0[1];
          __float128 v49_data = s0[1];
          __float128 v51_data = ir1[0];
          ir1[0] = (v51_data + (v48_data * v49_data));
          __float128 v54_data = s0[3];
          __float128 v56_data = ir1[1];
          ir1[1] = (v56_data + (v48_data * v54_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v58_n0 = 0; v58_n0 < 1; ++v58_n0) {
            #pragma unroll
            for (int32_t v59_n1 = 0; v59_n1 < 2; ++v59_n1) {
              int32_t v60_a = v58_n0 + v59_n1;
              __float128 v61_data = ir1[v60_a];
              r1[v60_a] = v61_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v62_i0 = 0; v62_i0 < 1; ++v62_i0) {
            int32_t v67_lead = v25_lead + (v62_i0 * 2);
            #pragma unroll
            for (int32_t v63_i1 = 0; v63_i1 < 2; ++v63_i1) {
              __float128 v65_data = r1[(v62_i0 + v63_i1)];
              glb_m0[(v67_lead + (v63_i1 * 2))] = v65_data;
            }
          }
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
        }
      }
    }
  }
}

