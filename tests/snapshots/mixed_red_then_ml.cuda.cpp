// === base name ===
kernel_2c4d3ff1d64dcb14

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2c4d3ff1d64dcb14 = {{32, 4, 1}, 32, 32, 1, 4, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2c4d3ff1d64dcb14(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2c4d3ff1d64dcb14(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2c4d3ff1d64dcb14(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_2c4d3ff1d64dcb14, block.x * block.y * block.z, 0 * sizeof(float));
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
void launcher_kernel_2c4d3ff1d64dcb14(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2c4d3ff1d64dcb14(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_2c4d3ff1d64dcb14, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_2c4d3ff1d64dcb14<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_2c4d3ff1d64dcb14(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 0 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   TMP = +(A, dims=[1])
    //   m1[i,j] = t0[i] × m2[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[8]],"is_tmp":true,"name":"t0","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[8]],"is_tmp":true,"name":"t0","offset":[0],"shape":[8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0],[0,1]],"target":[[0],[0,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v21_lead = threadIdx.x % 32;
          bool v22_g = v21_lead < 8;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
              float v28_data = __ldcg(&glb_m2[(v21_lead + (v23_i1 * 8))]);
              r1[v23_i1] = v28_data;
            }
          }
          float r0[1]{};
          // r0 = +(glb_m0, dims=[1])
          if (v22_g) {
            float v32_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v31_r1 = 0; v31_r1 < 8; ++v31_r1) {
              float v37_data = glb_m0[(v21_lead + (v31_r1 * 8))];
              v32_acc0 = (v32_acc0 + v37_data);
            }
            r0[0] = v32_acc0;
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // ir2 = +(r0 * r1)
          // [(0, 8), (0, 8)] []
          float ir2[8]{};
          float v42_data = r0[0];
          float v43_data = r1[0];
          float v45_data = ir2[0];
          ir2[0] = (v45_data + (v42_data * v43_data));
          float v48_data = r1[1];
          float v50_data = ir2[1];
          ir2[1] = (v50_data + (v42_data * v48_data));
          float v53_data = r1[2];
          float v55_data = ir2[2];
          ir2[2] = (v55_data + (v42_data * v53_data));
          float v58_data = r1[3];
          float v60_data = ir2[3];
          ir2[3] = (v60_data + (v42_data * v58_data));
          float v63_data = r1[4];
          float v65_data = ir2[4];
          ir2[4] = (v65_data + (v42_data * v63_data));
          float v68_data = r1[5];
          float v70_data = ir2[5];
          ir2[5] = (v70_data + (v42_data * v68_data));
          float v73_data = r1[6];
          float v75_data = ir2[6];
          ir2[6] = (v75_data + (v42_data * v73_data));
          float v78_data = r1[7];
          float v80_data = ir2[7];
          ir2[7] = (v80_data + (v42_data * v78_data));
          // r2 = ir2
          if (v22_g) {
            #pragma unroll
            for (int32_t v82_n1 = 0; v82_n1 < 8; ++v82_n1) {
              float v84_data = ir2[v82_n1];
              r2[v82_n1] = v84_data;
            }
          }
          // glb_m1 = store{r>g}(r2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v85_i1 = 0; v85_i1 < 8; ++v85_i1) {
              float v87_data = r2[v85_i1];
              glb_m1[(v21_lead + (v85_i1 * 8))] = v87_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

