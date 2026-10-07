// === base name ===
kernel_0c645dfc8e4d3a50

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0c645dfc8e4d3a50 = {{16, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0c645dfc8e4d3a50(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0c645dfc8e4d3a50(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0c645dfc8e4d3a50(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0c645dfc8e4d3a50, block.x * block.y * block.z, 512 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0c645dfc8e4d3a50, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(double)));
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
  config.sharedMemBytes = 512 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0c645dfc8e4d3a50(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0c645dfc8e4d3a50(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0c645dfc8e4d3a50), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0c645dfc8e4d3a50, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0c645dfc8e4d3a50(tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} none
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":4096,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m1[0];
      double * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      double v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m2);
          int32_t v23_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 16);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
              double v30_data = __builtin_nontemporal_load(&glb_m2[(v27_lead + (v25_i1 * 16))]);
              r0[(v24_i0 + v25_i1)] = v30_data;
            }
          }
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double v36_data = glb_m1[v23_lead];
          double v38_data = glb_m1[(v23_lead + 16)];
          double v40_data = glb_m1[(v23_lead + 32)];
          double v42_data = glb_m1[(v23_lead + 48)];
          double v44_data = glb_m1[(v23_lead + 64)];
          double v46_data = glb_m1[(v23_lead + 80)];
          double v48_data = glb_m1[(v23_lead + 96)];
          double v50_data = glb_m1[(v23_lead + 112)];
          double v52_data = glb_m1[(v23_lead + 128)];
          double v54_data = glb_m1[(v23_lead + 144)];
          double v56_data = glb_m1[(v23_lead + 160)];
          double v58_data = glb_m1[(v23_lead + 176)];
          double v60_data = glb_m1[(v23_lead + 192)];
          double v62_data = glb_m1[(v23_lead + 208)];
          double v64_data = glb_m1[(v23_lead + 224)];
          double v66_data = glb_m1[(v23_lead + 240)];
          double v67_acc{};
          double v68_acc{};
          double v69_acc{};
          double v70_acc{};
          double v71_acc{};
          double v72_acc{};
          double v73_acc{};
          double v74_acc{};
          double v75_acc{};
          double v76_acc{};
          double v77_acc{};
          double v78_acc{};
          double v79_acc{};
          double v80_acc{};
          double v81_acc{};
          double v82_acc{};
          double v83_data = r0[0];
          double v84_data = r0[1];
          double v85_data = r0[2];
          double v86_data = r0[3];
          double v87_data = r0[4];
          double v88_data = r0[5];
          double v89_data = r0[6];
          double v90_data = r0[7];
          double v91_data = r0[8];
          double v92_data = r0[9];
          double v93_data = r0[10];
          double v94_data = r0[11];
          double v95_data = r0[12];
          double v96_data = r0[13];
          double v97_data = r0[14];
          double v98_data = r0[15];
          tensorforge::fmacdpp16<0>(v67_acc, v83_data, v36_data);
          tensorforge::fmacdpp16<1>(v67_acc, v83_data, v38_data);
          tensorforge::fmacdpp16<2>(v67_acc, v83_data, v40_data);
          tensorforge::fmacdpp16<3>(v67_acc, v83_data, v42_data);
          tensorforge::fmacdpp16<4>(v67_acc, v83_data, v44_data);
          tensorforge::fmacdpp16<5>(v67_acc, v83_data, v46_data);
          tensorforge::fmacdpp16<6>(v67_acc, v83_data, v48_data);
          tensorforge::fmacdpp16<7>(v67_acc, v83_data, v50_data);
          tensorforge::fmacdpp16<8>(v67_acc, v83_data, v52_data);
          tensorforge::fmacdpp16<9>(v67_acc, v83_data, v54_data);
          tensorforge::fmacdpp16<10>(v67_acc, v83_data, v56_data);
          tensorforge::fmacdpp16<11>(v67_acc, v83_data, v58_data);
          tensorforge::fmacdpp16<12>(v67_acc, v83_data, v60_data);
          tensorforge::fmacdpp16<13>(v67_acc, v83_data, v62_data);
          tensorforge::fmacdpp16<14>(v67_acc, v83_data, v64_data);
          tensorforge::fmacdpp16<15>(v67_acc, v83_data, v66_data);
          tensorforge::fmacdpp16<0>(v68_acc, v84_data, v36_data);
          tensorforge::fmacdpp16<1>(v68_acc, v84_data, v38_data);
          tensorforge::fmacdpp16<2>(v68_acc, v84_data, v40_data);
          tensorforge::fmacdpp16<3>(v68_acc, v84_data, v42_data);
          tensorforge::fmacdpp16<4>(v68_acc, v84_data, v44_data);
          tensorforge::fmacdpp16<5>(v68_acc, v84_data, v46_data);
          tensorforge::fmacdpp16<6>(v68_acc, v84_data, v48_data);
          tensorforge::fmacdpp16<7>(v68_acc, v84_data, v50_data);
          tensorforge::fmacdpp16<8>(v68_acc, v84_data, v52_data);
          tensorforge::fmacdpp16<9>(v68_acc, v84_data, v54_data);
          tensorforge::fmacdpp16<10>(v68_acc, v84_data, v56_data);
          tensorforge::fmacdpp16<11>(v68_acc, v84_data, v58_data);
          tensorforge::fmacdpp16<12>(v68_acc, v84_data, v60_data);
          tensorforge::fmacdpp16<13>(v68_acc, v84_data, v62_data);
          tensorforge::fmacdpp16<14>(v68_acc, v84_data, v64_data);
          tensorforge::fmacdpp16<15>(v68_acc, v84_data, v66_data);
          tensorforge::fmacdpp16<0>(v69_acc, v85_data, v36_data);
          tensorforge::fmacdpp16<1>(v69_acc, v85_data, v38_data);
          tensorforge::fmacdpp16<2>(v69_acc, v85_data, v40_data);
          tensorforge::fmacdpp16<3>(v69_acc, v85_data, v42_data);
          tensorforge::fmacdpp16<4>(v69_acc, v85_data, v44_data);
          tensorforge::fmacdpp16<5>(v69_acc, v85_data, v46_data);
          tensorforge::fmacdpp16<6>(v69_acc, v85_data, v48_data);
          tensorforge::fmacdpp16<7>(v69_acc, v85_data, v50_data);
          tensorforge::fmacdpp16<8>(v69_acc, v85_data, v52_data);
          tensorforge::fmacdpp16<9>(v69_acc, v85_data, v54_data);
          tensorforge::fmacdpp16<10>(v69_acc, v85_data, v56_data);
          tensorforge::fmacdpp16<11>(v69_acc, v85_data, v58_data);
          tensorforge::fmacdpp16<12>(v69_acc, v85_data, v60_data);
          tensorforge::fmacdpp16<13>(v69_acc, v85_data, v62_data);
          tensorforge::fmacdpp16<14>(v69_acc, v85_data, v64_data);
          tensorforge::fmacdpp16<15>(v69_acc, v85_data, v66_data);
          tensorforge::fmacdpp16<0>(v70_acc, v86_data, v36_data);
          tensorforge::fmacdpp16<1>(v70_acc, v86_data, v38_data);
          tensorforge::fmacdpp16<2>(v70_acc, v86_data, v40_data);
          tensorforge::fmacdpp16<3>(v70_acc, v86_data, v42_data);
          tensorforge::fmacdpp16<4>(v70_acc, v86_data, v44_data);
          tensorforge::fmacdpp16<5>(v70_acc, v86_data, v46_data);
          tensorforge::fmacdpp16<6>(v70_acc, v86_data, v48_data);
          tensorforge::fmacdpp16<7>(v70_acc, v86_data, v50_data);
          tensorforge::fmacdpp16<8>(v70_acc, v86_data, v52_data);
          tensorforge::fmacdpp16<9>(v70_acc, v86_data, v54_data);
          tensorforge::fmacdpp16<10>(v70_acc, v86_data, v56_data);
          tensorforge::fmacdpp16<11>(v70_acc, v86_data, v58_data);
          tensorforge::fmacdpp16<12>(v70_acc, v86_data, v60_data);
          tensorforge::fmacdpp16<13>(v70_acc, v86_data, v62_data);
          tensorforge::fmacdpp16<14>(v70_acc, v86_data, v64_data);
          tensorforge::fmacdpp16<15>(v70_acc, v86_data, v66_data);
          tensorforge::fmacdpp16<0>(v71_acc, v87_data, v36_data);
          tensorforge::fmacdpp16<1>(v71_acc, v87_data, v38_data);
          tensorforge::fmacdpp16<2>(v71_acc, v87_data, v40_data);
          tensorforge::fmacdpp16<3>(v71_acc, v87_data, v42_data);
          tensorforge::fmacdpp16<4>(v71_acc, v87_data, v44_data);
          tensorforge::fmacdpp16<5>(v71_acc, v87_data, v46_data);
          tensorforge::fmacdpp16<6>(v71_acc, v87_data, v48_data);
          tensorforge::fmacdpp16<7>(v71_acc, v87_data, v50_data);
          tensorforge::fmacdpp16<8>(v71_acc, v87_data, v52_data);
          tensorforge::fmacdpp16<9>(v71_acc, v87_data, v54_data);
          tensorforge::fmacdpp16<10>(v71_acc, v87_data, v56_data);
          tensorforge::fmacdpp16<11>(v71_acc, v87_data, v58_data);
          tensorforge::fmacdpp16<12>(v71_acc, v87_data, v60_data);
          tensorforge::fmacdpp16<13>(v71_acc, v87_data, v62_data);
          tensorforge::fmacdpp16<14>(v71_acc, v87_data, v64_data);
          tensorforge::fmacdpp16<15>(v71_acc, v87_data, v66_data);
          tensorforge::fmacdpp16<0>(v72_acc, v88_data, v36_data);
          tensorforge::fmacdpp16<1>(v72_acc, v88_data, v38_data);
          tensorforge::fmacdpp16<2>(v72_acc, v88_data, v40_data);
          tensorforge::fmacdpp16<3>(v72_acc, v88_data, v42_data);
          tensorforge::fmacdpp16<4>(v72_acc, v88_data, v44_data);
          tensorforge::fmacdpp16<5>(v72_acc, v88_data, v46_data);
          tensorforge::fmacdpp16<6>(v72_acc, v88_data, v48_data);
          tensorforge::fmacdpp16<7>(v72_acc, v88_data, v50_data);
          tensorforge::fmacdpp16<8>(v72_acc, v88_data, v52_data);
          tensorforge::fmacdpp16<9>(v72_acc, v88_data, v54_data);
          tensorforge::fmacdpp16<10>(v72_acc, v88_data, v56_data);
          tensorforge::fmacdpp16<11>(v72_acc, v88_data, v58_data);
          tensorforge::fmacdpp16<12>(v72_acc, v88_data, v60_data);
          tensorforge::fmacdpp16<13>(v72_acc, v88_data, v62_data);
          tensorforge::fmacdpp16<14>(v72_acc, v88_data, v64_data);
          tensorforge::fmacdpp16<15>(v72_acc, v88_data, v66_data);
          tensorforge::fmacdpp16<0>(v73_acc, v89_data, v36_data);
          tensorforge::fmacdpp16<1>(v73_acc, v89_data, v38_data);
          tensorforge::fmacdpp16<2>(v73_acc, v89_data, v40_data);
          tensorforge::fmacdpp16<3>(v73_acc, v89_data, v42_data);
          tensorforge::fmacdpp16<4>(v73_acc, v89_data, v44_data);
          tensorforge::fmacdpp16<5>(v73_acc, v89_data, v46_data);
          tensorforge::fmacdpp16<6>(v73_acc, v89_data, v48_data);
          tensorforge::fmacdpp16<7>(v73_acc, v89_data, v50_data);
          tensorforge::fmacdpp16<8>(v73_acc, v89_data, v52_data);
          tensorforge::fmacdpp16<9>(v73_acc, v89_data, v54_data);
          tensorforge::fmacdpp16<10>(v73_acc, v89_data, v56_data);
          tensorforge::fmacdpp16<11>(v73_acc, v89_data, v58_data);
          tensorforge::fmacdpp16<12>(v73_acc, v89_data, v60_data);
          tensorforge::fmacdpp16<13>(v73_acc, v89_data, v62_data);
          tensorforge::fmacdpp16<14>(v73_acc, v89_data, v64_data);
          tensorforge::fmacdpp16<15>(v73_acc, v89_data, v66_data);
          tensorforge::fmacdpp16<0>(v74_acc, v90_data, v36_data);
          tensorforge::fmacdpp16<1>(v74_acc, v90_data, v38_data);
          tensorforge::fmacdpp16<2>(v74_acc, v90_data, v40_data);
          tensorforge::fmacdpp16<3>(v74_acc, v90_data, v42_data);
          tensorforge::fmacdpp16<4>(v74_acc, v90_data, v44_data);
          tensorforge::fmacdpp16<5>(v74_acc, v90_data, v46_data);
          tensorforge::fmacdpp16<6>(v74_acc, v90_data, v48_data);
          tensorforge::fmacdpp16<7>(v74_acc, v90_data, v50_data);
          tensorforge::fmacdpp16<8>(v74_acc, v90_data, v52_data);
          tensorforge::fmacdpp16<9>(v74_acc, v90_data, v54_data);
          tensorforge::fmacdpp16<10>(v74_acc, v90_data, v56_data);
          tensorforge::fmacdpp16<11>(v74_acc, v90_data, v58_data);
          tensorforge::fmacdpp16<12>(v74_acc, v90_data, v60_data);
          tensorforge::fmacdpp16<13>(v74_acc, v90_data, v62_data);
          tensorforge::fmacdpp16<14>(v74_acc, v90_data, v64_data);
          tensorforge::fmacdpp16<15>(v74_acc, v90_data, v66_data);
          tensorforge::fmacdpp16<0>(v75_acc, v91_data, v36_data);
          tensorforge::fmacdpp16<1>(v75_acc, v91_data, v38_data);
          tensorforge::fmacdpp16<2>(v75_acc, v91_data, v40_data);
          tensorforge::fmacdpp16<3>(v75_acc, v91_data, v42_data);
          tensorforge::fmacdpp16<4>(v75_acc, v91_data, v44_data);
          tensorforge::fmacdpp16<5>(v75_acc, v91_data, v46_data);
          tensorforge::fmacdpp16<6>(v75_acc, v91_data, v48_data);
          tensorforge::fmacdpp16<7>(v75_acc, v91_data, v50_data);
          tensorforge::fmacdpp16<8>(v75_acc, v91_data, v52_data);
          tensorforge::fmacdpp16<9>(v75_acc, v91_data, v54_data);
          tensorforge::fmacdpp16<10>(v75_acc, v91_data, v56_data);
          tensorforge::fmacdpp16<11>(v75_acc, v91_data, v58_data);
          tensorforge::fmacdpp16<12>(v75_acc, v91_data, v60_data);
          tensorforge::fmacdpp16<13>(v75_acc, v91_data, v62_data);
          tensorforge::fmacdpp16<14>(v75_acc, v91_data, v64_data);
          tensorforge::fmacdpp16<15>(v75_acc, v91_data, v66_data);
          tensorforge::fmacdpp16<0>(v76_acc, v92_data, v36_data);
          tensorforge::fmacdpp16<1>(v76_acc, v92_data, v38_data);
          tensorforge::fmacdpp16<2>(v76_acc, v92_data, v40_data);
          tensorforge::fmacdpp16<3>(v76_acc, v92_data, v42_data);
          tensorforge::fmacdpp16<4>(v76_acc, v92_data, v44_data);
          tensorforge::fmacdpp16<5>(v76_acc, v92_data, v46_data);
          tensorforge::fmacdpp16<6>(v76_acc, v92_data, v48_data);
          tensorforge::fmacdpp16<7>(v76_acc, v92_data, v50_data);
          tensorforge::fmacdpp16<8>(v76_acc, v92_data, v52_data);
          tensorforge::fmacdpp16<9>(v76_acc, v92_data, v54_data);
          tensorforge::fmacdpp16<10>(v76_acc, v92_data, v56_data);
          tensorforge::fmacdpp16<11>(v76_acc, v92_data, v58_data);
          tensorforge::fmacdpp16<12>(v76_acc, v92_data, v60_data);
          tensorforge::fmacdpp16<13>(v76_acc, v92_data, v62_data);
          tensorforge::fmacdpp16<14>(v76_acc, v92_data, v64_data);
          tensorforge::fmacdpp16<15>(v76_acc, v92_data, v66_data);
          tensorforge::fmacdpp16<0>(v77_acc, v93_data, v36_data);
          tensorforge::fmacdpp16<1>(v77_acc, v93_data, v38_data);
          tensorforge::fmacdpp16<2>(v77_acc, v93_data, v40_data);
          tensorforge::fmacdpp16<3>(v77_acc, v93_data, v42_data);
          tensorforge::fmacdpp16<4>(v77_acc, v93_data, v44_data);
          tensorforge::fmacdpp16<5>(v77_acc, v93_data, v46_data);
          tensorforge::fmacdpp16<6>(v77_acc, v93_data, v48_data);
          tensorforge::fmacdpp16<7>(v77_acc, v93_data, v50_data);
          tensorforge::fmacdpp16<8>(v77_acc, v93_data, v52_data);
          tensorforge::fmacdpp16<9>(v77_acc, v93_data, v54_data);
          tensorforge::fmacdpp16<10>(v77_acc, v93_data, v56_data);
          tensorforge::fmacdpp16<11>(v77_acc, v93_data, v58_data);
          tensorforge::fmacdpp16<12>(v77_acc, v93_data, v60_data);
          tensorforge::fmacdpp16<13>(v77_acc, v93_data, v62_data);
          tensorforge::fmacdpp16<14>(v77_acc, v93_data, v64_data);
          tensorforge::fmacdpp16<15>(v77_acc, v93_data, v66_data);
          tensorforge::fmacdpp16<0>(v78_acc, v94_data, v36_data);
          tensorforge::fmacdpp16<1>(v78_acc, v94_data, v38_data);
          tensorforge::fmacdpp16<2>(v78_acc, v94_data, v40_data);
          tensorforge::fmacdpp16<3>(v78_acc, v94_data, v42_data);
          tensorforge::fmacdpp16<4>(v78_acc, v94_data, v44_data);
          tensorforge::fmacdpp16<5>(v78_acc, v94_data, v46_data);
          tensorforge::fmacdpp16<6>(v78_acc, v94_data, v48_data);
          tensorforge::fmacdpp16<7>(v78_acc, v94_data, v50_data);
          tensorforge::fmacdpp16<8>(v78_acc, v94_data, v52_data);
          tensorforge::fmacdpp16<9>(v78_acc, v94_data, v54_data);
          tensorforge::fmacdpp16<10>(v78_acc, v94_data, v56_data);
          tensorforge::fmacdpp16<11>(v78_acc, v94_data, v58_data);
          tensorforge::fmacdpp16<12>(v78_acc, v94_data, v60_data);
          tensorforge::fmacdpp16<13>(v78_acc, v94_data, v62_data);
          tensorforge::fmacdpp16<14>(v78_acc, v94_data, v64_data);
          tensorforge::fmacdpp16<15>(v78_acc, v94_data, v66_data);
          tensorforge::fmacdpp16<0>(v79_acc, v95_data, v36_data);
          tensorforge::fmacdpp16<1>(v79_acc, v95_data, v38_data);
          tensorforge::fmacdpp16<2>(v79_acc, v95_data, v40_data);
          tensorforge::fmacdpp16<3>(v79_acc, v95_data, v42_data);
          tensorforge::fmacdpp16<4>(v79_acc, v95_data, v44_data);
          tensorforge::fmacdpp16<5>(v79_acc, v95_data, v46_data);
          tensorforge::fmacdpp16<6>(v79_acc, v95_data, v48_data);
          tensorforge::fmacdpp16<7>(v79_acc, v95_data, v50_data);
          tensorforge::fmacdpp16<8>(v79_acc, v95_data, v52_data);
          tensorforge::fmacdpp16<9>(v79_acc, v95_data, v54_data);
          tensorforge::fmacdpp16<10>(v79_acc, v95_data, v56_data);
          tensorforge::fmacdpp16<11>(v79_acc, v95_data, v58_data);
          tensorforge::fmacdpp16<12>(v79_acc, v95_data, v60_data);
          tensorforge::fmacdpp16<13>(v79_acc, v95_data, v62_data);
          tensorforge::fmacdpp16<14>(v79_acc, v95_data, v64_data);
          tensorforge::fmacdpp16<15>(v79_acc, v95_data, v66_data);
          tensorforge::fmacdpp16<0>(v80_acc, v96_data, v36_data);
          tensorforge::fmacdpp16<1>(v80_acc, v96_data, v38_data);
          tensorforge::fmacdpp16<2>(v80_acc, v96_data, v40_data);
          tensorforge::fmacdpp16<3>(v80_acc, v96_data, v42_data);
          tensorforge::fmacdpp16<4>(v80_acc, v96_data, v44_data);
          tensorforge::fmacdpp16<5>(v80_acc, v96_data, v46_data);
          tensorforge::fmacdpp16<6>(v80_acc, v96_data, v48_data);
          tensorforge::fmacdpp16<7>(v80_acc, v96_data, v50_data);
          tensorforge::fmacdpp16<8>(v80_acc, v96_data, v52_data);
          tensorforge::fmacdpp16<9>(v80_acc, v96_data, v54_data);
          tensorforge::fmacdpp16<10>(v80_acc, v96_data, v56_data);
          tensorforge::fmacdpp16<11>(v80_acc, v96_data, v58_data);
          tensorforge::fmacdpp16<12>(v80_acc, v96_data, v60_data);
          tensorforge::fmacdpp16<13>(v80_acc, v96_data, v62_data);
          tensorforge::fmacdpp16<14>(v80_acc, v96_data, v64_data);
          tensorforge::fmacdpp16<15>(v80_acc, v96_data, v66_data);
          tensorforge::fmacdpp16<0>(v81_acc, v97_data, v36_data);
          tensorforge::fmacdpp16<1>(v81_acc, v97_data, v38_data);
          tensorforge::fmacdpp16<2>(v81_acc, v97_data, v40_data);
          tensorforge::fmacdpp16<3>(v81_acc, v97_data, v42_data);
          tensorforge::fmacdpp16<4>(v81_acc, v97_data, v44_data);
          tensorforge::fmacdpp16<5>(v81_acc, v97_data, v46_data);
          tensorforge::fmacdpp16<6>(v81_acc, v97_data, v48_data);
          tensorforge::fmacdpp16<7>(v81_acc, v97_data, v50_data);
          tensorforge::fmacdpp16<8>(v81_acc, v97_data, v52_data);
          tensorforge::fmacdpp16<9>(v81_acc, v97_data, v54_data);
          tensorforge::fmacdpp16<10>(v81_acc, v97_data, v56_data);
          tensorforge::fmacdpp16<11>(v81_acc, v97_data, v58_data);
          tensorforge::fmacdpp16<12>(v81_acc, v97_data, v60_data);
          tensorforge::fmacdpp16<13>(v81_acc, v97_data, v62_data);
          tensorforge::fmacdpp16<14>(v81_acc, v97_data, v64_data);
          tensorforge::fmacdpp16<15>(v81_acc, v97_data, v66_data);
          tensorforge::fmacdpp16<0>(v82_acc, v98_data, v36_data);
          tensorforge::fmacdpp16<1>(v82_acc, v98_data, v38_data);
          tensorforge::fmacdpp16<2>(v82_acc, v98_data, v40_data);
          tensorforge::fmacdpp16<3>(v82_acc, v98_data, v42_data);
          tensorforge::fmacdpp16<4>(v82_acc, v98_data, v44_data);
          tensorforge::fmacdpp16<5>(v82_acc, v98_data, v46_data);
          tensorforge::fmacdpp16<6>(v82_acc, v98_data, v48_data);
          tensorforge::fmacdpp16<7>(v82_acc, v98_data, v50_data);
          tensorforge::fmacdpp16<8>(v82_acc, v98_data, v52_data);
          tensorforge::fmacdpp16<9>(v82_acc, v98_data, v54_data);
          tensorforge::fmacdpp16<10>(v82_acc, v98_data, v56_data);
          tensorforge::fmacdpp16<11>(v82_acc, v98_data, v58_data);
          tensorforge::fmacdpp16<12>(v82_acc, v98_data, v60_data);
          tensorforge::fmacdpp16<13>(v82_acc, v98_data, v62_data);
          tensorforge::fmacdpp16<14>(v82_acc, v98_data, v64_data);
          tensorforge::fmacdpp16<15>(v82_acc, v98_data, v66_data);
          r1[0] = v67_acc;
          r1[1] = v68_acc;
          r1[2] = v69_acc;
          r1[3] = v70_acc;
          r1[4] = v71_acc;
          r1[5] = v72_acc;
          r1[6] = v73_acc;
          r1[7] = v74_acc;
          r1[8] = v75_acc;
          r1[9] = v76_acc;
          r1[10] = v77_acc;
          r1[11] = v78_acc;
          r1[12] = v79_acc;
          r1[13] = v80_acc;
          r1[14] = v81_acc;
          r1[15] = v82_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v99_i0 = 0; v99_i0 < 1; ++v99_i0) {
            int32_t v104_lead = v23_lead + (v99_i0 * 16);
            #pragma unroll
            for (int32_t v100_i1 = 0; v100_i1 < 16; ++v100_i1) {
              double v102_data = r1[(v99_i0 + v100_i1)];
              glb_m0[(v104_lead + (v100_i1 * 16))] = v102_data;
            }
          }
        }
      }
    }
  }
}

