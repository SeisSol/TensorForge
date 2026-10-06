// === base name ===
kernel_321cfac6f6f5bb98

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_321cfac6f6f5bb98 = {{32, 4, 1}, 32, 32, 1, 4, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_321cfac6f6f5bb98(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_321cfac6f6f5bb98(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_321cfac6f6f5bb98(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_321cfac6f6f5bb98, block.x * block.y * block.z, 128 * sizeof(float));
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
void launcher_kernel_321cfac6f6f5bb98(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_321cfac6f6f5bb98(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_321cfac6f6f5bb98, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_321cfac6f6f5bb98<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_321cfac6f6f5bb98(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 512 B shared, occupancy grid
    // operands:
    //   m0 16(16) {0..16} strided
    //   m1 24×16(24×6) {0..24}×{0..6} strided
    //   m2 16(16) {0..16} strided
    // operations:
    //   t0[i] = m0[i]
    //   TMP = +(A, dims=[0])
    //   m2[i] = t0[i]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m1","ordered":false,"parts":1,"shape":[24,16],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m2","ordered":false,"parts":1,"shape":[16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,16]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m2","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
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
          const float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 16 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 32;
          bool v26_g = v25_lead < 16;
          if (v26_g) {
            float v29_data = __ldcg(&glb_m0[v25_lead]);
            r0[0] = v29_data;
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] []
          float v31_data = r0[0];
          float v32_data = r1[0];
          r1[0] = (v32_data + v31_data);
          // s0 = store{r>s}(localShrMem0, r1);
          if (v26_g) {
            float v34_data = r1[0];
            s0[v25_lead] = v34_data;
          }
          float r2[1]{};
          // r2 = +(glb_m1, dims=[0])
          bool v38_own = v25_lead < 24;
          float v44_sel0;
          if (v38_own) {
            float v42_data = glb_m1[v25_lead];
            v44_sel0 = v42_data;
          }
          else {
            v44_sel0 = 0.0f;
          }
          float v45_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v44_sel0);
          if (threadIdx.x == 0) {
            r2[0] = v45_red;
          }
          float v51_sel0;
          if (v38_own) {
            float v49_data = glb_m1[(v25_lead + 24)];
            v51_sel0 = v49_data;
          }
          else {
            v51_sel0 = 0.0f;
          }
          float v52_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v51_sel0);
          if (threadIdx.x == 1) {
            r2[0] = v52_red;
          }
          float v58_sel0;
          if (v38_own) {
            float v56_data = glb_m1[(v25_lead + 48)];
            v58_sel0 = v56_data;
          }
          else {
            v58_sel0 = 0.0f;
          }
          float v59_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v58_sel0);
          if (threadIdx.x == 2) {
            r2[0] = v59_red;
          }
          float v65_sel0;
          if (v38_own) {
            float v63_data = glb_m1[(v25_lead + 72)];
            v65_sel0 = v63_data;
          }
          else {
            v65_sel0 = 0.0f;
          }
          float v66_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v65_sel0);
          if (threadIdx.x == 3) {
            r2[0] = v66_red;
          }
          float v72_sel0;
          if (v38_own) {
            float v70_data = glb_m1[(v25_lead + 96)];
            v72_sel0 = v70_data;
          }
          else {
            v72_sel0 = 0.0f;
          }
          float v73_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v72_sel0);
          if (threadIdx.x == 4) {
            r2[0] = v73_red;
          }
          float v79_sel0;
          if (v38_own) {
            float v77_data = glb_m1[(v25_lead + 120)];
            v79_sel0 = v77_data;
          }
          else {
            v79_sel0 = 0.0f;
          }
          float v80_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v79_sel0);
          if (threadIdx.x == 5) {
            r2[0] = v80_red;
          }
          // s0 = store{r>s, clear}(localShrMem0, r2);
          if ((v25_lead >= 6) && v26_g) {
            s0[v25_lead] = 0.0f;
          }
          if (v25_lead < 6) {
            float v87_data = r2[0];
            s0[v25_lead] = v87_data;
          }
          float r3[1]{};
          __syncwarp();
          // ir3 = +(s0)
          // [(0, 16)] []
          float ir3[1]{};
          float v94_data = v26_g ? (s0[v25_lead]) : (0.0f);
          float v95_data = ir3[0];
          ir3[0] = (v95_data + v94_data);
          // r3 = ir3
          if (v26_g) {
            float v97_data = ir3[0];
            r3[0] = v97_data;
          }
          // glb_m2 = store{r>g}(r3);
          if (v26_g) {
            float v98_data = r3[0];
            glb_m2[v25_lead] = v98_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

