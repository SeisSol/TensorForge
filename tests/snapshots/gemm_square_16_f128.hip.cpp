// === base name ===
kernel_0269d9c02c92bea4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0269d9c02c92bea4 = {{2, 128, 1}, 2, 2, 1, 128, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0269d9c02c92bea4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0269d9c02c92bea4(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0269d9c02c92bea4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (2, 128, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0269d9c02c92bea4, block.x * block.y * block.z, 256 * sizeof(__float128)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(__float128)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0269d9c02c92bea4, block.x * block.y * block.z, 0));
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
void launcher_kernel_0269d9c02c92bea4(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0269d9c02c92bea4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0269d9c02c92bea4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<__float128, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<__float128, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0269d9c02c92bea4, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0269d9c02c92bea4(tensorforge::SpacePtr<__float128, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const __float128, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 2 lanes x 128 per block = block 2x128x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 2×2(2×2) {0..2}×{0..2} strided
    //   m1 2×2(2×2) {0..2}×{0..2} strided
    //   m2 2×2(2×2) {0..2}×{0..2} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[2,128,1],"cooperative":false,"lead_width":1,"mults_per_block":128,"persistent":true,"sections":[{"barrier":false,"mults_per_block":128,"shared_elements":256}],"shared_bytes":4096,"shared_elements":256,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[2 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[0];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<__float128, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<__float128, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 4 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace>)&m1[v10_batchId0 * 4 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v24_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
            int32_t v28_lead = v24_lead + (v25_i0 * 2);
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 2; ++v26_i1) {
              __float128 v31_data = __builtin_nontemporal_load(&glb_m1[(v28_lead + (v26_i1 * 2))]);
              r0[(v25_i0 + v26_i1)] = v31_data;
            }
          }
          __float128 r1[2]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
            int32_t v37_lead = v24_lead + (v34_i0 * 2);
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 2; ++v35_i1) {
              __float128 v40_data = __builtin_nontemporal_load(&glb_m2[(v37_lead + (v35_i1 * 2))]);
              r1[(v34_i0 + v35_i1)] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          __float128 r2[2]{};
          // r2 = +(r0 * r1) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 v43_data = r0[0];
          __float128 v44_data = r0[1];
          __float128 v45_acc{};
          __float128 v46_acc{};
          __float128 v47_data = r1[0];
          __float128 v48_data = r1[1];
          v45_acc += ((tensorforge::broadcast<2, 1, 0>(v47_data)) * v43_data);
          v45_acc += ((tensorforge::broadcast<2, 1, 1>(v47_data)) * v44_data);
          v46_acc += ((tensorforge::broadcast<2, 1, 0>(v48_data)) * v43_data);
          v46_acc += ((tensorforge::broadcast<2, 1, 1>(v48_data)) * v44_data);
          r2[0] = v45_acc;
          r2[1] = v46_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v57_i0 = 0; v57_i0 < 1; ++v57_i0) {
            int32_t v62_lead = v24_lead + (v57_i0 * 2);
            #pragma unroll
            for (int32_t v58_i1 = 0; v58_i1 < 2; ++v58_i1) {
              __float128 v60_data = r2[(v57_i0 + v58_i1)];
              glb_m0[(v62_lead + (v58_i1 * 2))] = v60_data;
            }
          }
        }
      }
    }
  }
}

