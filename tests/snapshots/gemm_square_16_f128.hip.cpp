// === base name ===
kernel_b9509dc5121edf2c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b9509dc5121edf2c = {{2, 128, 1}, 2, 2, 1, 128, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b9509dc5121edf2c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b9509dc5121edf2c(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b9509dc5121edf2c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (2, 128, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b9509dc5121edf2c, block.x * block.y * block.z, 256 * sizeof(__float128)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(__float128)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b9509dc5121edf2c, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(__float128)));
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
  config.block[0] = 2;
  config.block[1] = 128;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(__float128);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b9509dc5121edf2c(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b9509dc5121edf2c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b9509dc5121edf2c), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<__float128, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<__float128, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b9509dc5121edf2c, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b9509dc5121edf2c(tensorforge::SpacePtr<__float128, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 2 lanes x 128 per block = block 2x128x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 2×2(2×2) {0..2}×{0..2} strided
    //   m1 2×2(2×2) {0..2}×{0..2} strided
    //   m2 2×2(2×2) {0..2}×{0..2} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[2,128,1],"cooperative":false,"lead_width":1,"mults_per_block":128,"persistent":true,"sections":[{"barrier":false,"mults_per_block":128,"shared_elements":256}],"shared_bytes":4096,"shared_elements":256,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[2 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<__float128, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<__float128, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 4 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 4 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v22_lead = v18_lead + (v19_i0 * 2);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 2; ++v20_i1) {
              __float128 v25_data = __builtin_nontemporal_load(&glb_m1[(v22_lead + (v20_i1 * 2))]);
              r0[(v19_i0 + v20_i1)] = v25_data;
            }
          }
          __float128 r1[2]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v31_lead = v18_lead + (v28_i0 * 2);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 2; ++v29_i1) {
              __float128 v34_data = __builtin_nontemporal_load(&glb_m2[(v31_lead + (v29_i1 * 2))]);
              r1[(v28_i0 + v29_i1)] = v34_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          __float128 r2[2]{};
          // r2 = +(r0 * r1) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 v37_data = r0[0];
          __float128 v38_data = r0[1];
          __float128 v39_acc{};
          __float128 v40_acc{};
          __float128 v41_data = r1[0];
          __float128 v42_data = r1[1];
          v39_acc += ((tensorforge::broadcast<2, 1, 0>(v41_data)) * v37_data);
          v39_acc += ((tensorforge::broadcast<2, 1, 1>(v41_data)) * v38_data);
          v40_acc += ((tensorforge::broadcast<2, 1, 0>(v42_data)) * v37_data);
          v40_acc += ((tensorforge::broadcast<2, 1, 1>(v42_data)) * v38_data);
          r2[0] = v39_acc;
          r2[1] = v40_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
            int32_t v56_lead = v18_lead + (v51_i0 * 2);
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 2; ++v52_i1) {
              __float128 v54_data = r2[(v51_i0 + v52_i1)];
              glb_m0[(v56_lead + (v52_i1 * 2))] = v54_data;
            }
          }
        }
      }
    }
  }
}

