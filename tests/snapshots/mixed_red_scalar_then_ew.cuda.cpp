// === base name ===
kernel_ee00f20553007a9c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ee00f20553007a9c = {{32, 4, 1}, 32, 32, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ee00f20553007a9c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ee00f20553007a9c(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ee00f20553007a9c(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ee00f20553007a9c, block.x * block.y * block.z, 256 * sizeof(float));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ee00f20553007a9c(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ee00f20553007a9c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ee00f20553007a9c, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ee00f20553007a9c<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ee00f20553007a9c(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 40(40) {0..40} strided
    //   m1 40(40) {0..40} strided
    // operations:
    //   P = abs(A)
    //   M = max(P, dims=[0])
    //   C = mul(A, M)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0],[40]],"name":"m0","ordered":false,"parts":1,"shape":[40],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0],[40]],"name":"m1","ordered":false,"parts":1,"shape":[40],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[40]],"is_tmp":true,"name":"t0","offset":[0],"shape":[40]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m0","offset":[0],"shape":[40]}],"permute":[[0]],"scalars":[],"target":[[0]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[],[]],"is_tmp":true,"name":"t1","offset":[],"shape":[]},"kind":"reduction","op":"max","ops":[{"addressing":"pointer_based","bbox":[[0],[40]],"is_tmp":true,"name":"t0","offset":[0],"shape":[40]}],"permute":[[0]],"target":[[-1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m1","offset":[0],"shape":[40]},"kind":"elementwise","op":"MUL","ops":[{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m0","offset":[0],"shape":[40]},{"addressing":"pointer_based","bbox":[[],[]],"is_tmp":true,"name":"t1","offset":[],"shape":[]}],"permute":[[0],[]],"scalars":[],"target":[[0],[]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 40 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 40 + 0 + m1_extraOffset];
          float r0[2]{};
          // r0 = abs(glb_m0)
          int32_t v19_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v20_k0 = 0; v20_k0 < 1; ++v20_k0) {
            float v23_data = glb_m0[(v19_lead + (v20_k0 * 32))];
            r0[v20_k0] = (fabsf(v23_data));
          }
          if (v19_lead < 8) {
            float v28_data = glb_m0[(v19_lead + 32_i32)];
            r0[1] = (fabsf(v28_data));
          }
          // s0 = store{r>s}(localShrMem0, r0);
          #pragma unroll
          for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
            float v34_data = r0[v33_i0];
            s0[(v19_lead + (v33_i0 * 32))] = v34_data;
          }
          bool v37_g = v19_lead < 8;
          if (v37_g) {
            float v38_data = r0[1];
            s0[(v19_lead + 32_i32)] = v38_data;
          }
          float r1[1]{};
          __syncwarp();
          // r1 = max(s0, dims=[0])
          float v44_data = s0[v19_lead];
          float v49_sel0;
          if (v37_g) {
            float v47_data = s0[(v19_lead + 32_i32)];
            v49_sel0 = v47_data;
          }
          else {
            v49_sel0 = -INFINITY;
          }
          r1[0] = (tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Max>, 32, 1, float>((fmaxf(v44_data, v49_sel0))));
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          float v52_data = r1[0];
          s1[0] = v52_data;
          __syncwarp();
          // glb_m1 = mul(glb_m0, s1)
          float v57_data = s1[0];
          #pragma unroll
          for (int32_t v53_k0 = 0; v53_k0 < 1; ++v53_k0) {
            int32_t v55_lead = v19_lead + (v53_k0 * 32);
            float v56_data = glb_m0[v55_lead];
            glb_m1[v55_lead] = ((v56_data * v57_data));
          }
          if (v37_g) {
            int32_t v60_lead = v19_lead + 32_i32;
            float v61_data = glb_m0[v60_lead];
            float v62_data = s1[0];
            glb_m1[v60_lead] = ((v61_data * v62_data));
          }
          __syncwarp();
        }
      }
    }
  }
}

