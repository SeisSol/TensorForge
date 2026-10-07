// === base name ===
kernel_0c115d045b2864c0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0c115d045b2864c0 = {{32, 8, 1}, 32, 21, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0c115d045b2864c0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0c115d045b2864c0(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0c115d045b2864c0(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0c115d045b2864c0, block.x * block.y * block.z, 0 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0c115d045b2864c0, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(double)));
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
  config.sharedMemBytes = 0 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0c115d045b2864c0(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0c115d045b2864c0(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0c115d045b2864c0), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0c115d045b2864c0, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0c115d045b2864c0(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (21 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32×3(32×3) {0..32}×{0..3} pointer_based
    //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
    //   m2 9×9(9×9) {0..9}×{0..9} pointer_based
    // operations:
    //   m0[i,j]@{0..21}×{0..3} = m1[i,k]@{0..21}×{0..3} × m2[j,k]@{6..9}×{6..9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":21,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,3]],"name":"m0","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"M0","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"T","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]},{"addressing":"pointer_based","bbox":[[0,0],[3,3]],"is_tmp":false,"name":"m2","offset":[6,6],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v7_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v7_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v7_batchId0][0 + m2_extraOffset];
          double r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v21_lead = threadIdx.x % 32;
          bool v22_g = v21_lead < 21;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 3; ++v23_i1) {
              double v28_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v23_i1 * 32))]);
              r0[v23_i1] = v28_data;
            }
          }
          double r1[3]{};
          // r1 = load{g>r}(glb_m2);
          bool v34_g = (v21_lead >= 6) && (v21_lead < 9);
          #pragma unroll
          for (int32_t v31_i0 = 6; v31_i0 < 9; ++v31_i0) {
            if (v34_g) {
              double v39_data = __builtin_nontemporal_load(&glb_m2[(v31_i0 + (v21_lead * 9))]);
              r1[(v31_i0 - 6)] = v39_data;
            }
          }
          double r2[3]{};
          // r2 = +(r0 * r1) + None
          // [(0, 21), (0, 3)] [(0, 3)]
          double v43_data = r0[0];
          double v44_data = r1[0];
          double v47_data = r2[0];
          r2[0] = (v47_data + (v43_data * (tensorforge::broadcast<32, 1, 6>(v44_data))));
          double v50_data = r1[1];
          double v53_data = r2[1];
          r2[1] = (v53_data + (v43_data * (tensorforge::broadcast<32, 1, 6>(v50_data))));
          double v56_data = r1[2];
          double v59_data = r2[2];
          r2[2] = (v59_data + (v43_data * (tensorforge::broadcast<32, 1, 6>(v56_data))));
          double v61_data = r0[1];
          double v65_data = r2[0];
          r2[0] = (v65_data + (v61_data * (tensorforge::broadcast<32, 1, 7>(v44_data))));
          double v71_data = r2[1];
          r2[1] = (v71_data + (v61_data * (tensorforge::broadcast<32, 1, 7>(v50_data))));
          double v77_data = r2[2];
          r2[2] = (v77_data + (v61_data * (tensorforge::broadcast<32, 1, 7>(v56_data))));
          double v79_data = r0[2];
          double v83_data = r2[0];
          r2[0] = (v83_data + (v79_data * (tensorforge::broadcast<32, 1, 8>(v44_data))));
          double v89_data = r2[1];
          r2[1] = (v89_data + (v79_data * (tensorforge::broadcast<32, 1, 8>(v50_data))));
          double v95_data = r2[2];
          r2[2] = (v95_data + (v79_data * (tensorforge::broadcast<32, 1, 8>(v56_data))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v97_i1 = 0; v97_i1 < 3; ++v97_i1) {
            double v99_data = r2[v97_i1];
            glb_m0[(v21_lead + (v97_i1 * 32))] = (v22_g ? v99_data : 0.0);
          }
        }
      }
    }
  }
}

