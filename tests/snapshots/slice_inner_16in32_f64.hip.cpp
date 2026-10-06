// === base name ===
kernel_77a4369d47daf9bc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_77a4369d47daf9bc = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_77a4369d47daf9bc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_77a4369d47daf9bc(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_77a4369d47daf9bc(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_77a4369d47daf9bc, block.x * block.y * block.z, 256 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_77a4369d47daf9bc, block.x * block.y * block.z, 0));
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
void launcher_kernel_77a4369d47daf9bc(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_77a4369d47daf9bc(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_77a4369d47daf9bc), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_77a4369d47daf9bc, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_77a4369d47daf9bc(tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×8(16×8) {0..16}×{0..8} strided
    //   m1 32×32(32×32) {0..32}×{0..32} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[0];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 128 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v10_batchId0 * 1024 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v24_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
            int32_t v29_off = (v24_lead + (v25_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v26_i1 = 8; v26_i1 < 24; ++v26_i1) {
              double v32_data = __builtin_nontemporal_load(&glb_m1[(v29_off + (v26_i1 * 32))]);
              r0[(v25_i0 + (v26_i1 - 8))] = v32_data;
            }
          }
          double r1[8]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
            int32_t v39_lead = v24_lead + (v36_i0 * 16);
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
              double v42_data = __builtin_nontemporal_load(&glb_m2[(v39_lead + (v37_i1 * 16))]);
              r1[(v36_i0 + v37_i1)] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double v45_data = r0[0];
          double v46_data = r0[1];
          double v47_data = r0[2];
          double v48_data = r0[3];
          double v49_data = r0[4];
          double v50_data = r0[5];
          double v51_data = r0[6];
          double v52_data = r0[7];
          double v53_data = r0[8];
          double v54_data = r0[9];
          double v55_data = r0[10];
          double v56_data = r0[11];
          double v57_data = r0[12];
          double v58_data = r0[13];
          double v59_data = r0[14];
          double v60_data = r0[15];
          double v61_acc{};
          double v62_acc{};
          double v63_acc{};
          double v64_acc{};
          double v65_acc{};
          double v66_acc{};
          double v67_acc{};
          double v68_acc{};
          double v69_data = r1[0];
          double v70_data = r1[1];
          double v71_data = r1[2];
          double v72_data = r1[3];
          double v73_data = r1[4];
          double v74_data = r1[5];
          double v75_data = r1[6];
          double v76_data = r1[7];
          tensorforge::fmacdpp16<0>(v61_acc, v69_data, v45_data);
          tensorforge::fmacdpp16<1>(v61_acc, v69_data, v46_data);
          tensorforge::fmacdpp16<2>(v61_acc, v69_data, v47_data);
          tensorforge::fmacdpp16<3>(v61_acc, v69_data, v48_data);
          tensorforge::fmacdpp16<4>(v61_acc, v69_data, v49_data);
          tensorforge::fmacdpp16<5>(v61_acc, v69_data, v50_data);
          tensorforge::fmacdpp16<6>(v61_acc, v69_data, v51_data);
          tensorforge::fmacdpp16<7>(v61_acc, v69_data, v52_data);
          tensorforge::fmacdpp16<8>(v61_acc, v69_data, v53_data);
          tensorforge::fmacdpp16<9>(v61_acc, v69_data, v54_data);
          tensorforge::fmacdpp16<10>(v61_acc, v69_data, v55_data);
          tensorforge::fmacdpp16<11>(v61_acc, v69_data, v56_data);
          tensorforge::fmacdpp16<12>(v61_acc, v69_data, v57_data);
          tensorforge::fmacdpp16<13>(v61_acc, v69_data, v58_data);
          tensorforge::fmacdpp16<14>(v61_acc, v69_data, v59_data);
          tensorforge::fmacdpp16<15>(v61_acc, v69_data, v60_data);
          tensorforge::fmacdpp16<0>(v62_acc, v70_data, v45_data);
          tensorforge::fmacdpp16<1>(v62_acc, v70_data, v46_data);
          tensorforge::fmacdpp16<2>(v62_acc, v70_data, v47_data);
          tensorforge::fmacdpp16<3>(v62_acc, v70_data, v48_data);
          tensorforge::fmacdpp16<4>(v62_acc, v70_data, v49_data);
          tensorforge::fmacdpp16<5>(v62_acc, v70_data, v50_data);
          tensorforge::fmacdpp16<6>(v62_acc, v70_data, v51_data);
          tensorforge::fmacdpp16<7>(v62_acc, v70_data, v52_data);
          tensorforge::fmacdpp16<8>(v62_acc, v70_data, v53_data);
          tensorforge::fmacdpp16<9>(v62_acc, v70_data, v54_data);
          tensorforge::fmacdpp16<10>(v62_acc, v70_data, v55_data);
          tensorforge::fmacdpp16<11>(v62_acc, v70_data, v56_data);
          tensorforge::fmacdpp16<12>(v62_acc, v70_data, v57_data);
          tensorforge::fmacdpp16<13>(v62_acc, v70_data, v58_data);
          tensorforge::fmacdpp16<14>(v62_acc, v70_data, v59_data);
          tensorforge::fmacdpp16<15>(v62_acc, v70_data, v60_data);
          tensorforge::fmacdpp16<0>(v63_acc, v71_data, v45_data);
          tensorforge::fmacdpp16<1>(v63_acc, v71_data, v46_data);
          tensorforge::fmacdpp16<2>(v63_acc, v71_data, v47_data);
          tensorforge::fmacdpp16<3>(v63_acc, v71_data, v48_data);
          tensorforge::fmacdpp16<4>(v63_acc, v71_data, v49_data);
          tensorforge::fmacdpp16<5>(v63_acc, v71_data, v50_data);
          tensorforge::fmacdpp16<6>(v63_acc, v71_data, v51_data);
          tensorforge::fmacdpp16<7>(v63_acc, v71_data, v52_data);
          tensorforge::fmacdpp16<8>(v63_acc, v71_data, v53_data);
          tensorforge::fmacdpp16<9>(v63_acc, v71_data, v54_data);
          tensorforge::fmacdpp16<10>(v63_acc, v71_data, v55_data);
          tensorforge::fmacdpp16<11>(v63_acc, v71_data, v56_data);
          tensorforge::fmacdpp16<12>(v63_acc, v71_data, v57_data);
          tensorforge::fmacdpp16<13>(v63_acc, v71_data, v58_data);
          tensorforge::fmacdpp16<14>(v63_acc, v71_data, v59_data);
          tensorforge::fmacdpp16<15>(v63_acc, v71_data, v60_data);
          tensorforge::fmacdpp16<0>(v64_acc, v72_data, v45_data);
          tensorforge::fmacdpp16<1>(v64_acc, v72_data, v46_data);
          tensorforge::fmacdpp16<2>(v64_acc, v72_data, v47_data);
          tensorforge::fmacdpp16<3>(v64_acc, v72_data, v48_data);
          tensorforge::fmacdpp16<4>(v64_acc, v72_data, v49_data);
          tensorforge::fmacdpp16<5>(v64_acc, v72_data, v50_data);
          tensorforge::fmacdpp16<6>(v64_acc, v72_data, v51_data);
          tensorforge::fmacdpp16<7>(v64_acc, v72_data, v52_data);
          tensorforge::fmacdpp16<8>(v64_acc, v72_data, v53_data);
          tensorforge::fmacdpp16<9>(v64_acc, v72_data, v54_data);
          tensorforge::fmacdpp16<10>(v64_acc, v72_data, v55_data);
          tensorforge::fmacdpp16<11>(v64_acc, v72_data, v56_data);
          tensorforge::fmacdpp16<12>(v64_acc, v72_data, v57_data);
          tensorforge::fmacdpp16<13>(v64_acc, v72_data, v58_data);
          tensorforge::fmacdpp16<14>(v64_acc, v72_data, v59_data);
          tensorforge::fmacdpp16<15>(v64_acc, v72_data, v60_data);
          tensorforge::fmacdpp16<0>(v65_acc, v73_data, v45_data);
          tensorforge::fmacdpp16<1>(v65_acc, v73_data, v46_data);
          tensorforge::fmacdpp16<2>(v65_acc, v73_data, v47_data);
          tensorforge::fmacdpp16<3>(v65_acc, v73_data, v48_data);
          tensorforge::fmacdpp16<4>(v65_acc, v73_data, v49_data);
          tensorforge::fmacdpp16<5>(v65_acc, v73_data, v50_data);
          tensorforge::fmacdpp16<6>(v65_acc, v73_data, v51_data);
          tensorforge::fmacdpp16<7>(v65_acc, v73_data, v52_data);
          tensorforge::fmacdpp16<8>(v65_acc, v73_data, v53_data);
          tensorforge::fmacdpp16<9>(v65_acc, v73_data, v54_data);
          tensorforge::fmacdpp16<10>(v65_acc, v73_data, v55_data);
          tensorforge::fmacdpp16<11>(v65_acc, v73_data, v56_data);
          tensorforge::fmacdpp16<12>(v65_acc, v73_data, v57_data);
          tensorforge::fmacdpp16<13>(v65_acc, v73_data, v58_data);
          tensorforge::fmacdpp16<14>(v65_acc, v73_data, v59_data);
          tensorforge::fmacdpp16<15>(v65_acc, v73_data, v60_data);
          tensorforge::fmacdpp16<0>(v66_acc, v74_data, v45_data);
          tensorforge::fmacdpp16<1>(v66_acc, v74_data, v46_data);
          tensorforge::fmacdpp16<2>(v66_acc, v74_data, v47_data);
          tensorforge::fmacdpp16<3>(v66_acc, v74_data, v48_data);
          tensorforge::fmacdpp16<4>(v66_acc, v74_data, v49_data);
          tensorforge::fmacdpp16<5>(v66_acc, v74_data, v50_data);
          tensorforge::fmacdpp16<6>(v66_acc, v74_data, v51_data);
          tensorforge::fmacdpp16<7>(v66_acc, v74_data, v52_data);
          tensorforge::fmacdpp16<8>(v66_acc, v74_data, v53_data);
          tensorforge::fmacdpp16<9>(v66_acc, v74_data, v54_data);
          tensorforge::fmacdpp16<10>(v66_acc, v74_data, v55_data);
          tensorforge::fmacdpp16<11>(v66_acc, v74_data, v56_data);
          tensorforge::fmacdpp16<12>(v66_acc, v74_data, v57_data);
          tensorforge::fmacdpp16<13>(v66_acc, v74_data, v58_data);
          tensorforge::fmacdpp16<14>(v66_acc, v74_data, v59_data);
          tensorforge::fmacdpp16<15>(v66_acc, v74_data, v60_data);
          tensorforge::fmacdpp16<0>(v67_acc, v75_data, v45_data);
          tensorforge::fmacdpp16<1>(v67_acc, v75_data, v46_data);
          tensorforge::fmacdpp16<2>(v67_acc, v75_data, v47_data);
          tensorforge::fmacdpp16<3>(v67_acc, v75_data, v48_data);
          tensorforge::fmacdpp16<4>(v67_acc, v75_data, v49_data);
          tensorforge::fmacdpp16<5>(v67_acc, v75_data, v50_data);
          tensorforge::fmacdpp16<6>(v67_acc, v75_data, v51_data);
          tensorforge::fmacdpp16<7>(v67_acc, v75_data, v52_data);
          tensorforge::fmacdpp16<8>(v67_acc, v75_data, v53_data);
          tensorforge::fmacdpp16<9>(v67_acc, v75_data, v54_data);
          tensorforge::fmacdpp16<10>(v67_acc, v75_data, v55_data);
          tensorforge::fmacdpp16<11>(v67_acc, v75_data, v56_data);
          tensorforge::fmacdpp16<12>(v67_acc, v75_data, v57_data);
          tensorforge::fmacdpp16<13>(v67_acc, v75_data, v58_data);
          tensorforge::fmacdpp16<14>(v67_acc, v75_data, v59_data);
          tensorforge::fmacdpp16<15>(v67_acc, v75_data, v60_data);
          tensorforge::fmacdpp16<0>(v68_acc, v76_data, v45_data);
          tensorforge::fmacdpp16<1>(v68_acc, v76_data, v46_data);
          tensorforge::fmacdpp16<2>(v68_acc, v76_data, v47_data);
          tensorforge::fmacdpp16<3>(v68_acc, v76_data, v48_data);
          tensorforge::fmacdpp16<4>(v68_acc, v76_data, v49_data);
          tensorforge::fmacdpp16<5>(v68_acc, v76_data, v50_data);
          tensorforge::fmacdpp16<6>(v68_acc, v76_data, v51_data);
          tensorforge::fmacdpp16<7>(v68_acc, v76_data, v52_data);
          tensorforge::fmacdpp16<8>(v68_acc, v76_data, v53_data);
          tensorforge::fmacdpp16<9>(v68_acc, v76_data, v54_data);
          tensorforge::fmacdpp16<10>(v68_acc, v76_data, v55_data);
          tensorforge::fmacdpp16<11>(v68_acc, v76_data, v56_data);
          tensorforge::fmacdpp16<12>(v68_acc, v76_data, v57_data);
          tensorforge::fmacdpp16<13>(v68_acc, v76_data, v58_data);
          tensorforge::fmacdpp16<14>(v68_acc, v76_data, v59_data);
          tensorforge::fmacdpp16<15>(v68_acc, v76_data, v60_data);
          r2[0] = v61_acc;
          r2[1] = v62_acc;
          r2[2] = v63_acc;
          r2[3] = v64_acc;
          r2[4] = v65_acc;
          r2[5] = v66_acc;
          r2[6] = v67_acc;
          r2[7] = v68_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
            int32_t v82_lead = v24_lead + (v77_i0 * 16);
            #pragma unroll
            for (int32_t v78_i1 = 0; v78_i1 < 8; ++v78_i1) {
              double v80_data = r2[(v77_i0 + v78_i1)];
              glb_m0[(v82_lead + (v78_i1 * 16))] = v80_data;
            }
          }
        }
      }
    }
  }
}

