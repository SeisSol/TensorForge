// === base name ===
kernel_dd4ced8b6ca64193

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dd4ced8b6ca64193 = {{32, 4, 1}, 32, 32, 1, 4, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dd4ced8b6ca64193(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dd4ced8b6ca64193(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dd4ced8b6ca64193(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_dd4ced8b6ca64193, block.x * block.y * block.z, 0 * sizeof(float));
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
void launcher_kernel_dd4ced8b6ca64193(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dd4ced8b6ca64193(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_dd4ced8b6ca64193, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_dd4ced8b6ca64193<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_dd4ced8b6ca64193(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 0 B shared, occupancy grid
    // operands:
    //   m0 16(16) {0..16} strided
    //   m1 16(16) {0..16} strided
    //   m2 16×16(4×16) {4..8}×{0..16} strided
    // operations:
    //   m0[i] = m1[i]
    //   V = +(N, dims=[1])
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"V","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0],[16]],"name":"m1","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m1","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4],[8]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[4,0],[8,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 16 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 64 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v15_lead = threadIdx.x % 32;
          bool v16_g = v15_lead < 16;
          if (v16_g) {
            float v19_data = __ldcg(&glb_m1[v15_lead]);
            r0[0] = v19_data;
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // ir1 = +(r0)
          // [(0, 16)] []
          float ir1[1]{};
          float v22_data = r0[0];
          float v23_data = ir1[0];
          ir1[0] = (v23_data + v22_data);
          // r1 = ir1
          if (v16_g) {
            float v25_data = ir1[0];
            r1[0] = v25_data;
          }
          // glb_m0 = store{r>g}(r1);
          if (v16_g) {
            float v26_data = r1[0];
            glb_m0[v15_lead] = v26_data;
          }
          float r2[1]{};
          // r2 = +(glb_m2, dims=[1])
          bool v30_g = v15_lead < 4;
          if (v30_g) {
            int32_t v36_a = (v15_lead + 4) - 4;
            float v32_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v31_r1 = 0; v31_r1 < 16; ++v31_r1) {
              float v39_data = glb_m2[(v36_a + (v31_r1 * 4))];
              v32_acc0 = (v32_acc0 + v39_data);
            }
            r2[0] = v32_acc0;
          }
          // glb_m0 = store{r>g}(r2);
          if (v30_g) {
            float v43_data = r2[0];
            glb_m0[(v15_lead + 4)] = v43_data;
          }
          if (v15_lead >= 28) {
            glb_m0[((v15_lead + -32_i32) + 4)] = 0.0f;
          }
          if ((v15_lead >= 4) && (v15_lead < 12)) {
            glb_m0[(v15_lead + 4)] = 0.0f;
          }
          __syncwarp();
        }
      }
    }
  }
}

