// === base name ===
kernel_8c97002381d50b21

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8c97002381d50b21 = {{16, 16, 1}, 16, 12, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8c97002381d50b21(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8c97002381d50b21(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8c97002381d50b21(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8c97002381d50b21, block.x * block.y * block.z, 256 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_8c97002381d50b21, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(double)));
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
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8c97002381d50b21(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8c97002381d50b21(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8c97002381d50b21), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_8c97002381d50b21, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8c97002381d50b21(tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×16(12×16) {0..12}×{0..16} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 96 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 192 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          bool v19_g = v18_lead < 12;
          if (v19_g) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              double v25_data = __builtin_nontemporal_load(&glb_m1[(v18_lead + (v20_i1 * 12))]);
              r0[v20_i1] = v25_data;
            }
          }
          double r1[8]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v31_lead = v18_lead + (v28_i0 * 16);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
              double v34_data = __builtin_nontemporal_load(&glb_m2[(v31_lead + (v29_i1 * 16))]);
              r1[(v28_i0 + v29_i1)] = v34_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          double v37_data = r0[0];
          double v38_data = r0[1];
          double v39_data = r0[2];
          double v40_data = r0[3];
          double v41_data = r0[4];
          double v42_data = r0[5];
          double v43_data = r0[6];
          double v44_data = r0[7];
          double v45_data = r0[8];
          double v46_data = r0[9];
          double v47_data = r0[10];
          double v48_data = r0[11];
          double v49_data = r0[12];
          double v50_data = r0[13];
          double v51_data = r0[14];
          double v52_data = r0[15];
          double v53_acc{};
          double v54_acc{};
          double v55_acc{};
          double v56_acc{};
          double v57_acc{};
          double v58_acc{};
          double v59_acc{};
          double v60_acc{};
          double v61_data = r1[0];
          double v62_data = r1[1];
          double v63_data = r1[2];
          double v64_data = r1[3];
          double v65_data = r1[4];
          double v66_data = r1[5];
          double v67_data = r1[6];
          double v68_data = r1[7];
          tensorforge::fmacdpp16<0>(v53_acc, v61_data, v37_data);
          tensorforge::fmacdpp16<1>(v53_acc, v61_data, v38_data);
          tensorforge::fmacdpp16<2>(v53_acc, v61_data, v39_data);
          tensorforge::fmacdpp16<3>(v53_acc, v61_data, v40_data);
          tensorforge::fmacdpp16<4>(v53_acc, v61_data, v41_data);
          tensorforge::fmacdpp16<5>(v53_acc, v61_data, v42_data);
          tensorforge::fmacdpp16<6>(v53_acc, v61_data, v43_data);
          tensorforge::fmacdpp16<7>(v53_acc, v61_data, v44_data);
          tensorforge::fmacdpp16<8>(v53_acc, v61_data, v45_data);
          tensorforge::fmacdpp16<9>(v53_acc, v61_data, v46_data);
          tensorforge::fmacdpp16<10>(v53_acc, v61_data, v47_data);
          tensorforge::fmacdpp16<11>(v53_acc, v61_data, v48_data);
          tensorforge::fmacdpp16<12>(v53_acc, v61_data, v49_data);
          tensorforge::fmacdpp16<13>(v53_acc, v61_data, v50_data);
          tensorforge::fmacdpp16<14>(v53_acc, v61_data, v51_data);
          tensorforge::fmacdpp16<15>(v53_acc, v61_data, v52_data);
          tensorforge::fmacdpp16<0>(v54_acc, v62_data, v37_data);
          tensorforge::fmacdpp16<1>(v54_acc, v62_data, v38_data);
          tensorforge::fmacdpp16<2>(v54_acc, v62_data, v39_data);
          tensorforge::fmacdpp16<3>(v54_acc, v62_data, v40_data);
          tensorforge::fmacdpp16<4>(v54_acc, v62_data, v41_data);
          tensorforge::fmacdpp16<5>(v54_acc, v62_data, v42_data);
          tensorforge::fmacdpp16<6>(v54_acc, v62_data, v43_data);
          tensorforge::fmacdpp16<7>(v54_acc, v62_data, v44_data);
          tensorforge::fmacdpp16<8>(v54_acc, v62_data, v45_data);
          tensorforge::fmacdpp16<9>(v54_acc, v62_data, v46_data);
          tensorforge::fmacdpp16<10>(v54_acc, v62_data, v47_data);
          tensorforge::fmacdpp16<11>(v54_acc, v62_data, v48_data);
          tensorforge::fmacdpp16<12>(v54_acc, v62_data, v49_data);
          tensorforge::fmacdpp16<13>(v54_acc, v62_data, v50_data);
          tensorforge::fmacdpp16<14>(v54_acc, v62_data, v51_data);
          tensorforge::fmacdpp16<15>(v54_acc, v62_data, v52_data);
          tensorforge::fmacdpp16<0>(v55_acc, v63_data, v37_data);
          tensorforge::fmacdpp16<1>(v55_acc, v63_data, v38_data);
          tensorforge::fmacdpp16<2>(v55_acc, v63_data, v39_data);
          tensorforge::fmacdpp16<3>(v55_acc, v63_data, v40_data);
          tensorforge::fmacdpp16<4>(v55_acc, v63_data, v41_data);
          tensorforge::fmacdpp16<5>(v55_acc, v63_data, v42_data);
          tensorforge::fmacdpp16<6>(v55_acc, v63_data, v43_data);
          tensorforge::fmacdpp16<7>(v55_acc, v63_data, v44_data);
          tensorforge::fmacdpp16<8>(v55_acc, v63_data, v45_data);
          tensorforge::fmacdpp16<9>(v55_acc, v63_data, v46_data);
          tensorforge::fmacdpp16<10>(v55_acc, v63_data, v47_data);
          tensorforge::fmacdpp16<11>(v55_acc, v63_data, v48_data);
          tensorforge::fmacdpp16<12>(v55_acc, v63_data, v49_data);
          tensorforge::fmacdpp16<13>(v55_acc, v63_data, v50_data);
          tensorforge::fmacdpp16<14>(v55_acc, v63_data, v51_data);
          tensorforge::fmacdpp16<15>(v55_acc, v63_data, v52_data);
          tensorforge::fmacdpp16<0>(v56_acc, v64_data, v37_data);
          tensorforge::fmacdpp16<1>(v56_acc, v64_data, v38_data);
          tensorforge::fmacdpp16<2>(v56_acc, v64_data, v39_data);
          tensorforge::fmacdpp16<3>(v56_acc, v64_data, v40_data);
          tensorforge::fmacdpp16<4>(v56_acc, v64_data, v41_data);
          tensorforge::fmacdpp16<5>(v56_acc, v64_data, v42_data);
          tensorforge::fmacdpp16<6>(v56_acc, v64_data, v43_data);
          tensorforge::fmacdpp16<7>(v56_acc, v64_data, v44_data);
          tensorforge::fmacdpp16<8>(v56_acc, v64_data, v45_data);
          tensorforge::fmacdpp16<9>(v56_acc, v64_data, v46_data);
          tensorforge::fmacdpp16<10>(v56_acc, v64_data, v47_data);
          tensorforge::fmacdpp16<11>(v56_acc, v64_data, v48_data);
          tensorforge::fmacdpp16<12>(v56_acc, v64_data, v49_data);
          tensorforge::fmacdpp16<13>(v56_acc, v64_data, v50_data);
          tensorforge::fmacdpp16<14>(v56_acc, v64_data, v51_data);
          tensorforge::fmacdpp16<15>(v56_acc, v64_data, v52_data);
          tensorforge::fmacdpp16<0>(v57_acc, v65_data, v37_data);
          tensorforge::fmacdpp16<1>(v57_acc, v65_data, v38_data);
          tensorforge::fmacdpp16<2>(v57_acc, v65_data, v39_data);
          tensorforge::fmacdpp16<3>(v57_acc, v65_data, v40_data);
          tensorforge::fmacdpp16<4>(v57_acc, v65_data, v41_data);
          tensorforge::fmacdpp16<5>(v57_acc, v65_data, v42_data);
          tensorforge::fmacdpp16<6>(v57_acc, v65_data, v43_data);
          tensorforge::fmacdpp16<7>(v57_acc, v65_data, v44_data);
          tensorforge::fmacdpp16<8>(v57_acc, v65_data, v45_data);
          tensorforge::fmacdpp16<9>(v57_acc, v65_data, v46_data);
          tensorforge::fmacdpp16<10>(v57_acc, v65_data, v47_data);
          tensorforge::fmacdpp16<11>(v57_acc, v65_data, v48_data);
          tensorforge::fmacdpp16<12>(v57_acc, v65_data, v49_data);
          tensorforge::fmacdpp16<13>(v57_acc, v65_data, v50_data);
          tensorforge::fmacdpp16<14>(v57_acc, v65_data, v51_data);
          tensorforge::fmacdpp16<15>(v57_acc, v65_data, v52_data);
          tensorforge::fmacdpp16<0>(v58_acc, v66_data, v37_data);
          tensorforge::fmacdpp16<1>(v58_acc, v66_data, v38_data);
          tensorforge::fmacdpp16<2>(v58_acc, v66_data, v39_data);
          tensorforge::fmacdpp16<3>(v58_acc, v66_data, v40_data);
          tensorforge::fmacdpp16<4>(v58_acc, v66_data, v41_data);
          tensorforge::fmacdpp16<5>(v58_acc, v66_data, v42_data);
          tensorforge::fmacdpp16<6>(v58_acc, v66_data, v43_data);
          tensorforge::fmacdpp16<7>(v58_acc, v66_data, v44_data);
          tensorforge::fmacdpp16<8>(v58_acc, v66_data, v45_data);
          tensorforge::fmacdpp16<9>(v58_acc, v66_data, v46_data);
          tensorforge::fmacdpp16<10>(v58_acc, v66_data, v47_data);
          tensorforge::fmacdpp16<11>(v58_acc, v66_data, v48_data);
          tensorforge::fmacdpp16<12>(v58_acc, v66_data, v49_data);
          tensorforge::fmacdpp16<13>(v58_acc, v66_data, v50_data);
          tensorforge::fmacdpp16<14>(v58_acc, v66_data, v51_data);
          tensorforge::fmacdpp16<15>(v58_acc, v66_data, v52_data);
          tensorforge::fmacdpp16<0>(v59_acc, v67_data, v37_data);
          tensorforge::fmacdpp16<1>(v59_acc, v67_data, v38_data);
          tensorforge::fmacdpp16<2>(v59_acc, v67_data, v39_data);
          tensorforge::fmacdpp16<3>(v59_acc, v67_data, v40_data);
          tensorforge::fmacdpp16<4>(v59_acc, v67_data, v41_data);
          tensorforge::fmacdpp16<5>(v59_acc, v67_data, v42_data);
          tensorforge::fmacdpp16<6>(v59_acc, v67_data, v43_data);
          tensorforge::fmacdpp16<7>(v59_acc, v67_data, v44_data);
          tensorforge::fmacdpp16<8>(v59_acc, v67_data, v45_data);
          tensorforge::fmacdpp16<9>(v59_acc, v67_data, v46_data);
          tensorforge::fmacdpp16<10>(v59_acc, v67_data, v47_data);
          tensorforge::fmacdpp16<11>(v59_acc, v67_data, v48_data);
          tensorforge::fmacdpp16<12>(v59_acc, v67_data, v49_data);
          tensorforge::fmacdpp16<13>(v59_acc, v67_data, v50_data);
          tensorforge::fmacdpp16<14>(v59_acc, v67_data, v51_data);
          tensorforge::fmacdpp16<15>(v59_acc, v67_data, v52_data);
          tensorforge::fmacdpp16<0>(v60_acc, v68_data, v37_data);
          tensorforge::fmacdpp16<1>(v60_acc, v68_data, v38_data);
          tensorforge::fmacdpp16<2>(v60_acc, v68_data, v39_data);
          tensorforge::fmacdpp16<3>(v60_acc, v68_data, v40_data);
          tensorforge::fmacdpp16<4>(v60_acc, v68_data, v41_data);
          tensorforge::fmacdpp16<5>(v60_acc, v68_data, v42_data);
          tensorforge::fmacdpp16<6>(v60_acc, v68_data, v43_data);
          tensorforge::fmacdpp16<7>(v60_acc, v68_data, v44_data);
          tensorforge::fmacdpp16<8>(v60_acc, v68_data, v45_data);
          tensorforge::fmacdpp16<9>(v60_acc, v68_data, v46_data);
          tensorforge::fmacdpp16<10>(v60_acc, v68_data, v47_data);
          tensorforge::fmacdpp16<11>(v60_acc, v68_data, v48_data);
          tensorforge::fmacdpp16<12>(v60_acc, v68_data, v49_data);
          tensorforge::fmacdpp16<13>(v60_acc, v68_data, v50_data);
          tensorforge::fmacdpp16<14>(v60_acc, v68_data, v51_data);
          tensorforge::fmacdpp16<15>(v60_acc, v68_data, v52_data);
          r2[0] = v53_acc;
          r2[1] = v54_acc;
          r2[2] = v55_acc;
          r2[3] = v56_acc;
          r2[4] = v57_acc;
          r2[5] = v58_acc;
          r2[6] = v59_acc;
          r2[7] = v60_acc;
          // glb_m0 = store{r>g}(r2);
          if (v19_g) {
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 8; ++v69_i1) {
              double v71_data = r2[v69_i1];
              int32_t v75_a = v18_lead + (v69_i1 * 12);
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m0[v75_a], v71_data);
            }
          }
        }
      }
    }
  }
}

