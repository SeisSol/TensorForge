// === base name ===
kernel_02026b03294f9189

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_02026b03294f9189 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_02026b03294f9189(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_02026b03294f9189(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_02026b03294f9189(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_02026b03294f9189, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_02026b03294f9189, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_02026b03294f9189(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_02026b03294f9189(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_02026b03294f9189), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_02026b03294f9189, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_02026b03294f9189(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 46 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v22_lead = v18_lead + (v19_i0 * 16);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v22_lead + (v20_i1 * 16))]);
              r0[(v19_i0 + v20_i1)] = v25_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v28_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v28_lin;
          float v29_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v29_lin;
          float v30_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v30_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v32_data = r0[0];
          float v33_data = r0[1];
          float v34_data = r0[2];
          float v35_data = r0[3];
          float v36_data = r0[4];
          float v37_data = r0[5];
          float v38_data = r0[6];
          float v39_data = r0[7];
          float v40_data = r0[8];
          float v41_data = r0[9];
          float v42_data = r0[10];
          float v43_data = r0[11];
          float v44_data = r0[12];
          float v45_data = r0[13];
          float v46_data = r0[14];
          float v47_data = r0[15];
          float v48_acc{};
          float v49_acc{};
          float v50_acc{};
          float v51_acc{};
          float v52_acc{};
          float v53_acc{};
          float v54_acc{};
          float v55_acc{};
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_acc{};
          float v63_acc{};
          float v64_lin = r1[0];
          tensorforge::fmacdpp16<0>(v48_acc, v64_lin, v32_data);
          tensorforge::fmacdpp16<1>(v48_acc, v64_lin, v33_data);
          tensorforge::fmacdpp16<2>(v49_acc, v64_lin, v32_data);
          tensorforge::fmacdpp16<3>(v49_acc, v64_lin, v33_data);
          tensorforge::fmacdpp16<4>(v49_acc, v64_lin, v34_data);
          tensorforge::fmacdpp16<5>(v50_acc, v64_lin, v33_data);
          tensorforge::fmacdpp16<6>(v50_acc, v64_lin, v34_data);
          tensorforge::fmacdpp16<7>(v50_acc, v64_lin, v35_data);
          tensorforge::fmacdpp16<8>(v51_acc, v64_lin, v34_data);
          tensorforge::fmacdpp16<9>(v51_acc, v64_lin, v35_data);
          tensorforge::fmacdpp16<10>(v51_acc, v64_lin, v36_data);
          tensorforge::fmacdpp16<11>(v52_acc, v64_lin, v35_data);
          tensorforge::fmacdpp16<12>(v52_acc, v64_lin, v36_data);
          tensorforge::fmacdpp16<13>(v52_acc, v64_lin, v37_data);
          tensorforge::fmacdpp16<14>(v53_acc, v64_lin, v36_data);
          tensorforge::fmacdpp16<15>(v53_acc, v64_lin, v37_data);
          float v65_lin = r1[1];
          tensorforge::fmacdpp16<0>(v53_acc, v65_lin, v38_data);
          tensorforge::fmacdpp16<1>(v54_acc, v65_lin, v37_data);
          tensorforge::fmacdpp16<2>(v54_acc, v65_lin, v38_data);
          tensorforge::fmacdpp16<3>(v54_acc, v65_lin, v39_data);
          tensorforge::fmacdpp16<4>(v55_acc, v65_lin, v38_data);
          tensorforge::fmacdpp16<5>(v55_acc, v65_lin, v39_data);
          tensorforge::fmacdpp16<6>(v55_acc, v65_lin, v40_data);
          tensorforge::fmacdpp16<7>(v56_acc, v65_lin, v39_data);
          tensorforge::fmacdpp16<8>(v56_acc, v65_lin, v40_data);
          tensorforge::fmacdpp16<9>(v56_acc, v65_lin, v41_data);
          tensorforge::fmacdpp16<10>(v57_acc, v65_lin, v40_data);
          tensorforge::fmacdpp16<11>(v57_acc, v65_lin, v41_data);
          tensorforge::fmacdpp16<12>(v57_acc, v65_lin, v42_data);
          tensorforge::fmacdpp16<13>(v58_acc, v65_lin, v41_data);
          tensorforge::fmacdpp16<14>(v58_acc, v65_lin, v42_data);
          tensorforge::fmacdpp16<15>(v58_acc, v65_lin, v43_data);
          float v66_lin = r1[2];
          tensorforge::fmacdpp16<0>(v59_acc, v66_lin, v42_data);
          tensorforge::fmacdpp16<1>(v59_acc, v66_lin, v43_data);
          tensorforge::fmacdpp16<2>(v59_acc, v66_lin, v44_data);
          tensorforge::fmacdpp16<3>(v60_acc, v66_lin, v43_data);
          tensorforge::fmacdpp16<4>(v60_acc, v66_lin, v44_data);
          tensorforge::fmacdpp16<5>(v60_acc, v66_lin, v45_data);
          tensorforge::fmacdpp16<6>(v61_acc, v66_lin, v44_data);
          tensorforge::fmacdpp16<7>(v61_acc, v66_lin, v45_data);
          tensorforge::fmacdpp16<8>(v61_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<9>(v62_acc, v66_lin, v45_data);
          tensorforge::fmacdpp16<10>(v62_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<11>(v62_acc, v66_lin, v47_data);
          tensorforge::fmacdpp16<12>(v63_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<13>(v63_acc, v66_lin, v47_data);
          r2[0] = v48_acc;
          r2[1] = v49_acc;
          r2[2] = v50_acc;
          r2[3] = v51_acc;
          r2[4] = v52_acc;
          r2[5] = v53_acc;
          r2[6] = v54_acc;
          r2[7] = v55_acc;
          r2[8] = v56_acc;
          r2[9] = v57_acc;
          r2[10] = v58_acc;
          r2[11] = v59_acc;
          r2[12] = v60_acc;
          r2[13] = v61_acc;
          r2[14] = v62_acc;
          r2[15] = v63_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
            int32_t v72_lead = v18_lead + (v67_i0 * 16);
            #pragma unroll
            for (int32_t v68_i1 = 0; v68_i1 < 16; ++v68_i1) {
              float v70_data = r2[(v67_i0 + v68_i1)];
              glb_m0[(v72_lead + (v68_i1 * 16))] = v70_data;
            }
          }
        }
      }
    }
  }
}

