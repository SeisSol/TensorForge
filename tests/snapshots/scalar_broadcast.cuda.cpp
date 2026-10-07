// === base name ===
kernel_ca00ff9dd074129f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ca00ff9dd074129f = {{32, 4, 1}, 32, 32, 1, 4, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ca00ff9dd074129f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ca00ff9dd074129f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ca00ff9dd074129f(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ca00ff9dd074129f, block.x * block.y * block.z, 0 * sizeof(float));
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
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ca00ff9dd074129f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ca00ff9dd074129f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ca00ff9dd074129f, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ca00ff9dd074129f<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ca00ff9dd074129f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32(32) {0..32} strided
    //   m1 32(32) {0..32} strided
    //   m2 ()  scalar
    //   m3 ()  scalar
    // operations:
    //   m0[i] = m1[i]
    //   m0[i] += m2[] × m3[]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"O","bbox":[[0],[32]],"name":"m0","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0],[32]],"name":"m1","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m2","ordered":false,"parts":1,"shape":[],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[32]],"is_tmp":false,"name":"m1","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m2","offset":[],"shape":[]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[],[]],"target":[[],[]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 32 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 32 + 0 + m1_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v20_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
            float v24_data = __ldcg(&glb_m1[(v20_lead + (v21_i0 * 32))]);
            r0[v21_i0] = v24_data;
          }
          float r1[1]{};
          // ir1 = +(r0)
          // [(0, 32)] []
          float ir1[1]{};
          float v27_data = r0[0];
          float v28_data = ir1[0];
          ir1[0] = (v28_data + v27_data);
          // r1 = ir1
          #pragma unroll
          for (int32_t v30_n0 = 0; v30_n0 < 1; ++v30_n0) {
            float v31_data = ir1[v30_n0];
            r1[v30_n0] = v31_data;
          }
          float r2[1]{};
          // ir2 = +()
          // [(0, 32)] []
          float ir2[1]{};
          // r2 = ir2 * glb_m2 * glb_m3 + r1
          #pragma unroll
          for (int32_t v37_n0 = 0; v37_n0 < 1; ++v37_n0) {
            float v40_data = r1[v37_n0];
            r2[v37_n0] = (v40_data + 6.0f);
          }
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v42_i0 = 0; v42_i0 < 1; ++v42_i0) {
            float v43_data = r2[v42_i0];
            glb_m0[(v20_lead + (v42_i0 * 32))] = v43_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

