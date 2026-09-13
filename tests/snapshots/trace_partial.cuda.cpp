// === base name ===
kernel_508942059be6cbdd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_508942059be6cbdd = {{16, 8, 1}, 16, 16, 1, 8, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_508942059be6cbdd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_508942059be6cbdd(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_508942059be6cbdd(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_508942059be6cbdd, block.x * block.y * block.z, 128 * sizeof(float));
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
  config.sharedMemBytes = 128 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_508942059be6cbdd(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_508942059be6cbdd(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_508942059be6cbdd, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_508942059be6cbdd<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_508942059be6cbdd(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 512 B shared, occupancy grid
    // operands:
    //   m0 16(16) {0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i] = m1[i,k]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 256 + 0 + m1_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v21_lead = v17_lead + (v18_i0 * 16);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
              float v24_data = __ldcg(&glb_m1[(v21_lead + (v19_i1 * 16))]);
              r0[(v18_i0 + v19_i1)] = v24_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float ir1[1]{};
          float v28_data = r0[0];
          float v29_data = ir1[0];
          ir1[0] = (v29_data + v28_data);
          float v31_data = r0[1];
          float v32_data = ir1[0];
          ir1[0] = (v32_data + v31_data);
          float v34_data = r0[2];
          float v35_data = ir1[0];
          ir1[0] = (v35_data + v34_data);
          float v37_data = r0[3];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + v37_data);
          float v40_data = r0[4];
          float v41_data = ir1[0];
          ir1[0] = (v41_data + v40_data);
          float v43_data = r0[5];
          float v44_data = ir1[0];
          ir1[0] = (v44_data + v43_data);
          float v46_data = r0[6];
          float v47_data = ir1[0];
          ir1[0] = (v47_data + v46_data);
          float v49_data = r0[7];
          float v50_data = ir1[0];
          ir1[0] = (v50_data + v49_data);
          float v52_data = r0[8];
          float v53_data = ir1[0];
          ir1[0] = (v53_data + v52_data);
          float v55_data = r0[9];
          float v56_data = ir1[0];
          ir1[0] = (v56_data + v55_data);
          float v58_data = r0[10];
          float v59_data = ir1[0];
          ir1[0] = (v59_data + v58_data);
          float v61_data = r0[11];
          float v62_data = ir1[0];
          ir1[0] = (v62_data + v61_data);
          float v64_data = r0[12];
          float v65_data = ir1[0];
          ir1[0] = (v65_data + v64_data);
          float v67_data = r0[13];
          float v68_data = ir1[0];
          ir1[0] = (v68_data + v67_data);
          float v70_data = r0[14];
          float v71_data = ir1[0];
          ir1[0] = (v71_data + v70_data);
          float v73_data = r0[15];
          float v74_data = ir1[0];
          ir1[0] = (v74_data + v73_data);
          #pragma unroll
          for (int32_t v76_n0 = 0; v76_n0 < 1; ++v76_n0) {
            float v77_data = ir1[v76_n0];
            r1[v76_n0] = v77_data;
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v78_i0 = 0; v78_i0 < 1; ++v78_i0) {
            float v79_data = r1[v78_i0];
            glb_m0[(v17_lead + (v78_i0 * 16))] = v79_data;
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

