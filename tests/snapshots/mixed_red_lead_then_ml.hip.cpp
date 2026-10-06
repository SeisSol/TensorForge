// === base name ===
kernel_05d46633d8f0c36e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_05d46633d8f0c36e = {{32, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_05d46633d8f0c36e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_05d46633d8f0c36e(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_05d46633d8f0c36e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_05d46633d8f0c36e, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_05d46633d8f0c36e, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_05d46633d8f0c36e(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_05d46633d8f0c36e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_05d46633d8f0c36e), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_05d46633d8f0c36e, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_05d46633d8f0c36e(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (64 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 24×6(24×6) {0..24}×{0..6} strided
    //   m1 6(6) {0..6} strided
    // operations:
    //   TMP = +(A, dims=[0])
    //   m1[i] = t0[i]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 6 + 0 + m1_extraOffset];
          float r0[1]{};
          // r0 = +(glb_m0, dims=[0])
          int32_t v20_lead = threadIdx.x % 32;
          bool v21_own = v20_lead < 24;
          float v27_sel0;
          if (v21_own) {
            float v25_data = glb_m0[v20_lead];
            v27_sel0 = v25_data;
          }
          else {
            v27_sel0 = 0.0f;
          }
          float v28_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v27_sel0);
          if (threadIdx.x == 0) {
            r0[0] = v28_red;
          }
          float v34_sel0;
          if (v21_own) {
            float v32_data = glb_m0[(v20_lead + 24)];
            v34_sel0 = v32_data;
          }
          else {
            v34_sel0 = 0.0f;
          }
          float v35_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v34_sel0);
          if (threadIdx.x == 1) {
            r0[0] = v35_red;
          }
          float v41_sel0;
          if (v21_own) {
            float v39_data = glb_m0[(v20_lead + 48)];
            v41_sel0 = v39_data;
          }
          else {
            v41_sel0 = 0.0f;
          }
          float v42_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v41_sel0);
          if (threadIdx.x == 2) {
            r0[0] = v42_red;
          }
          float v48_sel0;
          if (v21_own) {
            float v46_data = glb_m0[(v20_lead + 72)];
            v48_sel0 = v46_data;
          }
          else {
            v48_sel0 = 0.0f;
          }
          float v49_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v48_sel0);
          if (threadIdx.x == 3) {
            r0[0] = v49_red;
          }
          float v55_sel0;
          if (v21_own) {
            float v53_data = glb_m0[(v20_lead + 96)];
            v55_sel0 = v53_data;
          }
          else {
            v55_sel0 = 0.0f;
          }
          float v56_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v55_sel0);
          if (threadIdx.x == 4) {
            r0[0] = v56_red;
          }
          float v62_sel0;
          if (v21_own) {
            float v60_data = glb_m0[(v20_lead + 120)];
            v62_sel0 = v60_data;
          }
          else {
            v62_sel0 = 0.0f;
          }
          float v63_red = tensorforge::reduction<tensorforge::ReductionOperation<float, tensorforge::Operation::Add>, 32, 1, float>(v62_sel0);
          if (threadIdx.x == 5) {
            r0[0] = v63_red;
          }
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 6)] []
          float v68_data = r0[0];
          float v69_data = r1[0];
          r1[0] = (v69_data + v68_data);
          // glb_m1 = store{r>g}(r1);
          if (v20_lead < 6) {
            float v75_data = r1[0];
            glb_m1[v20_lead] = v75_data;
          }
        }
      }
    }
  }
}

