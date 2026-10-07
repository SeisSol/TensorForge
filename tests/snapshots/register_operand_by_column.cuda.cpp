// === base name ===
kernel_890551fd091e4b52

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_890551fd091e4b52 = {{32, 4, 1}, 32, 21, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_890551fd091e4b52(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_890551fd091e4b52(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_890551fd091e4b52(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_890551fd091e4b52, block.x * block.y * block.z, 384 * sizeof(double));
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
  config.sharedMemBytes = 384 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_890551fd091e4b52(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_890551fd091e4b52(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_890551fd091e4b52, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_890551fd091e4b52<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_890551fd091e4b52(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (21 active) x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 32×3(32×3) {0..32}×{0..3} pointer_based
    //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
    //   m2 9×9(9×9) {0..9}×{0..9} pointer_based
    // operations:
    //   m0[i,j]@{0..21}×{0..3} = m1[i,k]@{0..21}×{0..3} × m2[j,k]@{6..9}×{6..9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":21,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":384}],"shared_bytes":3072,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,3]],"name":"m0","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"M0","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"T","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]},{"addressing":"pointer_based","bbox":[[0,0],[3,3]],"is_tmp":false,"name":"m2","offset":[6,6],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v8_batchId0][0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v8_batchId0][0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v8_batchId0][0 + m2_extraOffset];
          double r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 32;
          bool v23_g = v22_lead < 21;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 3; ++v24_i1) {
              double v29_data = __ldcg(&glb_m1[(v22_lead + (v24_i1 * 32))]);
              r0[v24_i1] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 8);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 8);
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &glb_m2[0 + 0 + 1 * threadIdx.x + 64], 8);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[3]{};
          // ir1 = +(r0 * s0)
          // [(0, 21), (0, 3)] [(0, 3)]
          double ir1[3]{};
          double v36_data = r0[0];
          __syncwarp();
          double v37_data = s0[60];
          double v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          double v42_data = s0[61];
          double v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          double v47_data = s0[62];
          double v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          double v51_data = r0[1];
          double v52_data = s0[69];
          double v54_data = ir1[0];
          ir1[0] = (v54_data + (v51_data * v52_data));
          double v57_data = s0[70];
          double v59_data = ir1[1];
          ir1[1] = (v59_data + (v51_data * v57_data));
          double v62_data = s0[71];
          double v64_data = ir1[2];
          ir1[2] = (v64_data + (v51_data * v62_data));
          double v66_data = r0[2];
          double v67_data = s0[78];
          double v69_data = ir1[0];
          ir1[0] = (v69_data + (v66_data * v67_data));
          double v72_data = s0[79];
          double v74_data = ir1[1];
          ir1[1] = (v74_data + (v66_data * v72_data));
          double v77_data = s0[80];
          double v79_data = ir1[2];
          ir1[2] = (v79_data + (v66_data * v77_data));
          // r1 = ir1
          if (v23_g) {
            #pragma unroll
            for (int32_t v81_n1 = 0; v81_n1 < 3; ++v81_n1) {
              double v83_data = ir1[v81_n1];
              r1[v81_n1] = v83_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v84_i1 = 0; v84_i1 < 3; ++v84_i1) {
            double v86_data = r1[v84_i1];
            glb_m0[(v22_lead + (v84_i1 * 32))] = (v23_g ? v86_data : 0.0);
          }
          __syncwarp();
        }
      }
    }
  }
}

