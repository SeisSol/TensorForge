// === base name ===
kernel_363afc6ea92d9f79

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_363afc6ea92d9f79 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_363afc6ea92d9f79(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_363afc6ea92d9f79(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_363afc6ea92d9f79(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_363afc6ea92d9f79, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_363afc6ea92d9f79, block.x * block.y * block.z, 0));
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
void launcher_kernel_363afc6ea92d9f79(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_363afc6ea92d9f79(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_363afc6ea92d9f79), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_363afc6ea92d9f79, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_363afc6ea92d9f79(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32(32) {0..32} pointer_based
    //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
    //   m2 32×3(32×3) {0..32}×{0..3} pointer_based
    // operations:
    //   t0[i] = m0[i]
    //   t1[i,j] = m1[i,j]
    //   t2[i,j] = t0[i]
    //   t2[i,j] += t1[i,j]
    //   m2[i,j] = t2[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0],[32]],"name":"m0","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"pointer_based","alias":"B","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,3]],"name":"m2","ordered":false,"parts":1,"shape":[32,3],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v1_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v1_batchId0][0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
            float v19_data = __builtin_nontemporal_load(&glb_m0[(v15_lead + (v16_i0 * 32))]);
            r0[v16_i0] = v19_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
            int32_t v24_lead = v15_lead + (v21_i0 * 32);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 3; ++v22_i1) {
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v22_i1 * 32))]);
              r2[(v21_i0 + v22_i1)] = v27_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v30_data = r0[0];
          float v31_data = r1[0];
          r1[0] = (v31_data + v30_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v34_data = r2[0];
          float v35_data = r3[0];
          r3[0] = (v35_data + v34_data);
          float v37_data = r2[1];
          float v38_data = r3[1];
          r3[1] = (v38_data + v37_data);
          float v40_data = r2[2];
          float v41_data = r3[2];
          r3[2] = (v41_data + v40_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v44_data = r1[0];
          float v45_data = r4[0];
          r4[0] = (v45_data + v44_data);
          float v48_data = r4[1];
          r4[1] = (v48_data + v44_data);
          float v51_data = r4[2];
          r4[2] = (v51_data + v44_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v55_data = r3[0];
          float v56_data = ir5[0];
          ir5[0] = (v56_data + v55_data);
          float v58_data = r3[1];
          float v59_data = ir5[1];
          ir5[1] = (v59_data + v58_data);
          float v61_data = r3[2];
          float v62_data = ir5[2];
          ir5[2] = (v62_data + v61_data);
          #pragma unroll
          for (int32_t v64_n0 = 0; v64_n0 < 1; ++v64_n0) {
            #pragma unroll
            for (int32_t v65_n1 = 0; v65_n1 < 3; ++v65_n1) {
              int32_t v66_a = v64_n0 + v65_n1;
              float v67_data = ir5[v66_a];
              float v68_data = r4[v66_a];
              r5[v66_a] = (v68_data + v67_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float v71_data = r5[0];
          float v72_data = r6[0];
          r6[0] = (v72_data + v71_data);
          float v74_data = r5[1];
          float v75_data = r6[1];
          r6[1] = (v75_data + v74_data);
          float v77_data = r5[2];
          float v78_data = r6[2];
          r6[2] = (v78_data + v77_data);
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v80_i0 = 0; v80_i0 < 1; ++v80_i0) {
            int32_t v85_lead = v15_lead + (v80_i0 * 32);
            #pragma unroll
            for (int32_t v81_i1 = 0; v81_i1 < 3; ++v81_i1) {
              float v83_data = r6[(v80_i0 + v81_i1)];
              glb_m2[(v85_lead + (v81_i1 * 32))] = v83_data;
            }
          }
        }
      }
    }
  }
}

