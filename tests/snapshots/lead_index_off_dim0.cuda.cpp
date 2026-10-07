// === base name ===
kernel_7020ccd28eb13811

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7020ccd28eb13811 = {{32, 4, 1}, 32, 20, 1, 4, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7020ccd28eb13811(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7020ccd28eb13811(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7020ccd28eb13811(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7020ccd28eb13811, block.x * block.y * block.z, 128 * sizeof(float));
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
void launcher_kernel_7020ccd28eb13811(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7020ccd28eb13811(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_7020ccd28eb13811, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7020ccd28eb13811<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_7020ccd28eb13811(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v23_lead = threadIdx.x % 32;
          bool v24_g = v23_lead < 20;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            if (v24_g) {
              float v28_data = __ldcg(&glb_m1[(v20_i0 + v23_lead)]);
              r0[v20_i0] = v28_data;
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
          // ir1 = +(r0 * s0)
          // [(0, 20), (0, 9)] [(0, 1)]
          float ir1[9]{};
          int32_t v35_lead = threadIdx.x % 32;
          float v36_data = r0[0];
          __syncwarp();
          float v37_data = s0[0];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v42_data = s0[1];
          float v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          float v47_data = s0[2];
          float v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          float v52_data = s0[3];
          float v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          float v57_data = s0[4];
          float v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          float v62_data = s0[5];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          float v67_data = s0[6];
          float v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          float v72_data = s0[7];
          float v74_data = ir1[7];
          ir1[7] = (v74_data + (v36_data * v72_data));
          float v77_data = s0[8];
          float v79_data = ir1[8];
          ir1[8] = (v79_data + (v36_data * v77_data));
          // r1 = ir1
          if (v35_lead < 20) {
            #pragma unroll
            for (int32_t v85_n1 = 0; v85_n1 < 9; ++v85_n1) {
              float v87_data = ir1[v85_n1];
              r1[v85_n1] = v87_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v35_lead < 20) {
            #pragma unroll
            for (int32_t v92_i1 = 0; v92_i1 < 9; ++v92_i1) {
              float v94_data = r1[v92_i1];
              glb_m0[(v35_lead + (v92_i1 * 20))] = v94_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

