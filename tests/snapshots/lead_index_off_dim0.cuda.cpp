// === base name ===
kernel_0618c66fd68bb277

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0618c66fd68bb277 = {{32, 4, 1}, 32, 20, 1, 4, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0618c66fd68bb277(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0618c66fd68bb277(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0618c66fd68bb277(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0618c66fd68bb277, block.x * block.y * block.z, 128 * sizeof(float));
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
  config.sharedMemBytes = 128 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0618c66fd68bb277(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0618c66fd68bb277(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_0618c66fd68bb277, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_0618c66fd68bb277<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_0618c66fd68bb277(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 4 per block = block 32x4x1, 512 B shared, occupancy grid
    // operands:
    //   m0 20×9(20×9) {0..20}×{0..9} strided
    //   m1 1×20(1×20) {0..1}×{0..20} strided
    //   m2 1×9(1×9) {0..1}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[k,i] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[1,20]],"name":"m1","ordered":false,"parts":1,"shape":[1,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[1,9]],"name":"m2","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[1,20]},{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[1,9]}],"permute":[[0,1],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v26_lead = threadIdx.x % 32;
          bool v27_g = v26_lead < 20;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            if (v27_g) {
              float v31_data = __ldcg(&glb_m1[(v23_i0 + v26_lead)]);
              r0[v23_i0] = v31_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // ir1 = +(r0 * s0)
          // [(0, 20), (0, 9)] [(0, 1)]
          float ir1[9]{};
          int32_t v38_lead = threadIdx.x % 32;
          float v39_data = r0[0];
          float v40_data = s0[0];
          float v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          float v45_data = s0[1];
          float v47_data = ir1[1];
          ir1[1] = (v47_data + (v39_data * v45_data));
          float v50_data = s0[2];
          float v52_data = ir1[2];
          ir1[2] = (v52_data + (v39_data * v50_data));
          float v55_data = s0[3];
          float v57_data = ir1[3];
          ir1[3] = (v57_data + (v39_data * v55_data));
          float v60_data = s0[4];
          float v62_data = ir1[4];
          ir1[4] = (v62_data + (v39_data * v60_data));
          float v65_data = s0[5];
          float v67_data = ir1[5];
          ir1[5] = (v67_data + (v39_data * v65_data));
          float v70_data = s0[6];
          float v72_data = ir1[6];
          ir1[6] = (v72_data + (v39_data * v70_data));
          float v75_data = s0[7];
          float v77_data = ir1[7];
          ir1[7] = (v77_data + (v39_data * v75_data));
          float v80_data = s0[8];
          float v82_data = ir1[8];
          ir1[8] = (v82_data + (v39_data * v80_data));
          // r1 = ir1
          if (v38_lead < 20) {
            #pragma unroll
            for (int32_t v88_n1 = 0; v88_n1 < 9; ++v88_n1) {
              float v90_data = ir1[v88_n1];
              r1[v88_n1] = v90_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v38_lead < 20) {
            #pragma unroll
            for (int32_t v95_i1 = 0; v95_i1 < 9; ++v95_i1) {
              float v97_data = r1[v95_i1];
              glb_m0[(v38_lead + (v95_i1 * 20))] = v97_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

