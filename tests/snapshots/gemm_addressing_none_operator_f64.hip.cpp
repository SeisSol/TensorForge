// === base name ===
kernel_a68163b072b36fd0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a68163b072b36fd0 = {{16, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a68163b072b36fd0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a68163b072b36fd0(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a68163b072b36fd0(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a68163b072b36fd0, block.x * block.y * block.z, 512 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_a68163b072b36fd0, block.x * block.y * block.z, 0));
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
void launcher_kernel_a68163b072b36fd0(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a68163b072b36fd0(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a68163b072b36fd0), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_a68163b072b36fd0, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a68163b072b36fd0(tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} none
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":4096,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      double* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[0];
      double * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      double v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m2);
          int32_t v20_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
            int32_t v24_lead = v20_lead + (v21_i0 * 16);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 16; ++v22_i1) {
              double v27_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + (v22_i1 * 16))]);
              r0[(v21_i0 + v22_i1)] = v27_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double v33_data = glb_m1[v20_lead];
          double v35_data = glb_m1[(v20_lead + 16)];
          double v37_data = glb_m1[(v20_lead + 32)];
          double v39_data = glb_m1[(v20_lead + 48)];
          double v41_data = glb_m1[(v20_lead + 64)];
          double v43_data = glb_m1[(v20_lead + 80)];
          double v45_data = glb_m1[(v20_lead + 96)];
          double v47_data = glb_m1[(v20_lead + 112)];
          double v49_data = glb_m1[(v20_lead + 128)];
          double v51_data = glb_m1[(v20_lead + 144)];
          double v53_data = glb_m1[(v20_lead + 160)];
          double v55_data = glb_m1[(v20_lead + 176)];
          double v57_data = glb_m1[(v20_lead + 192)];
          double v59_data = glb_m1[(v20_lead + 208)];
          double v61_data = glb_m1[(v20_lead + 224)];
          double v63_data = glb_m1[(v20_lead + 240)];
          double v64_acc{};
          double v65_acc{};
          double v66_acc{};
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
          double v80_data = r0[0];
          double v81_data = r0[1];
          double v82_data = r0[2];
          double v83_data = r0[3];
          double v84_data = r0[4];
          double v85_data = r0[5];
          double v86_data = r0[6];
          double v87_data = r0[7];
          double v88_data = r0[8];
          double v89_data = r0[9];
          double v90_data = r0[10];
          double v91_data = r0[11];
          double v92_data = r0[12];
          double v93_data = r0[13];
          double v94_data = r0[14];
          double v95_data = r0[15];
          tensorforge::fmacdpp16<0>(v64_acc, v80_data, v33_data);
          tensorforge::fmacdpp16<1>(v64_acc, v80_data, v35_data);
          tensorforge::fmacdpp16<2>(v64_acc, v80_data, v37_data);
          tensorforge::fmacdpp16<3>(v64_acc, v80_data, v39_data);
          tensorforge::fmacdpp16<4>(v64_acc, v80_data, v41_data);
          tensorforge::fmacdpp16<5>(v64_acc, v80_data, v43_data);
          tensorforge::fmacdpp16<6>(v64_acc, v80_data, v45_data);
          tensorforge::fmacdpp16<7>(v64_acc, v80_data, v47_data);
          tensorforge::fmacdpp16<8>(v64_acc, v80_data, v49_data);
          tensorforge::fmacdpp16<9>(v64_acc, v80_data, v51_data);
          tensorforge::fmacdpp16<10>(v64_acc, v80_data, v53_data);
          tensorforge::fmacdpp16<11>(v64_acc, v80_data, v55_data);
          tensorforge::fmacdpp16<12>(v64_acc, v80_data, v57_data);
          tensorforge::fmacdpp16<13>(v64_acc, v80_data, v59_data);
          tensorforge::fmacdpp16<14>(v64_acc, v80_data, v61_data);
          tensorforge::fmacdpp16<15>(v64_acc, v80_data, v63_data);
          tensorforge::fmacdpp16<0>(v65_acc, v81_data, v33_data);
          tensorforge::fmacdpp16<1>(v65_acc, v81_data, v35_data);
          tensorforge::fmacdpp16<2>(v65_acc, v81_data, v37_data);
          tensorforge::fmacdpp16<3>(v65_acc, v81_data, v39_data);
          tensorforge::fmacdpp16<4>(v65_acc, v81_data, v41_data);
          tensorforge::fmacdpp16<5>(v65_acc, v81_data, v43_data);
          tensorforge::fmacdpp16<6>(v65_acc, v81_data, v45_data);
          tensorforge::fmacdpp16<7>(v65_acc, v81_data, v47_data);
          tensorforge::fmacdpp16<8>(v65_acc, v81_data, v49_data);
          tensorforge::fmacdpp16<9>(v65_acc, v81_data, v51_data);
          tensorforge::fmacdpp16<10>(v65_acc, v81_data, v53_data);
          tensorforge::fmacdpp16<11>(v65_acc, v81_data, v55_data);
          tensorforge::fmacdpp16<12>(v65_acc, v81_data, v57_data);
          tensorforge::fmacdpp16<13>(v65_acc, v81_data, v59_data);
          tensorforge::fmacdpp16<14>(v65_acc, v81_data, v61_data);
          tensorforge::fmacdpp16<15>(v65_acc, v81_data, v63_data);
          tensorforge::fmacdpp16<0>(v66_acc, v82_data, v33_data);
          tensorforge::fmacdpp16<1>(v66_acc, v82_data, v35_data);
          tensorforge::fmacdpp16<2>(v66_acc, v82_data, v37_data);
          tensorforge::fmacdpp16<3>(v66_acc, v82_data, v39_data);
          tensorforge::fmacdpp16<4>(v66_acc, v82_data, v41_data);
          tensorforge::fmacdpp16<5>(v66_acc, v82_data, v43_data);
          tensorforge::fmacdpp16<6>(v66_acc, v82_data, v45_data);
          tensorforge::fmacdpp16<7>(v66_acc, v82_data, v47_data);
          tensorforge::fmacdpp16<8>(v66_acc, v82_data, v49_data);
          tensorforge::fmacdpp16<9>(v66_acc, v82_data, v51_data);
          tensorforge::fmacdpp16<10>(v66_acc, v82_data, v53_data);
          tensorforge::fmacdpp16<11>(v66_acc, v82_data, v55_data);
          tensorforge::fmacdpp16<12>(v66_acc, v82_data, v57_data);
          tensorforge::fmacdpp16<13>(v66_acc, v82_data, v59_data);
          tensorforge::fmacdpp16<14>(v66_acc, v82_data, v61_data);
          tensorforge::fmacdpp16<15>(v66_acc, v82_data, v63_data);
          tensorforge::fmacdpp16<0>(v67_acc, v83_data, v33_data);
          tensorforge::fmacdpp16<1>(v67_acc, v83_data, v35_data);
          tensorforge::fmacdpp16<2>(v67_acc, v83_data, v37_data);
          tensorforge::fmacdpp16<3>(v67_acc, v83_data, v39_data);
          tensorforge::fmacdpp16<4>(v67_acc, v83_data, v41_data);
          tensorforge::fmacdpp16<5>(v67_acc, v83_data, v43_data);
          tensorforge::fmacdpp16<6>(v67_acc, v83_data, v45_data);
          tensorforge::fmacdpp16<7>(v67_acc, v83_data, v47_data);
          tensorforge::fmacdpp16<8>(v67_acc, v83_data, v49_data);
          tensorforge::fmacdpp16<9>(v67_acc, v83_data, v51_data);
          tensorforge::fmacdpp16<10>(v67_acc, v83_data, v53_data);
          tensorforge::fmacdpp16<11>(v67_acc, v83_data, v55_data);
          tensorforge::fmacdpp16<12>(v67_acc, v83_data, v57_data);
          tensorforge::fmacdpp16<13>(v67_acc, v83_data, v59_data);
          tensorforge::fmacdpp16<14>(v67_acc, v83_data, v61_data);
          tensorforge::fmacdpp16<15>(v67_acc, v83_data, v63_data);
          tensorforge::fmacdpp16<0>(v68_acc, v84_data, v33_data);
          tensorforge::fmacdpp16<1>(v68_acc, v84_data, v35_data);
          tensorforge::fmacdpp16<2>(v68_acc, v84_data, v37_data);
          tensorforge::fmacdpp16<3>(v68_acc, v84_data, v39_data);
          tensorforge::fmacdpp16<4>(v68_acc, v84_data, v41_data);
          tensorforge::fmacdpp16<5>(v68_acc, v84_data, v43_data);
          tensorforge::fmacdpp16<6>(v68_acc, v84_data, v45_data);
          tensorforge::fmacdpp16<7>(v68_acc, v84_data, v47_data);
          tensorforge::fmacdpp16<8>(v68_acc, v84_data, v49_data);
          tensorforge::fmacdpp16<9>(v68_acc, v84_data, v51_data);
          tensorforge::fmacdpp16<10>(v68_acc, v84_data, v53_data);
          tensorforge::fmacdpp16<11>(v68_acc, v84_data, v55_data);
          tensorforge::fmacdpp16<12>(v68_acc, v84_data, v57_data);
          tensorforge::fmacdpp16<13>(v68_acc, v84_data, v59_data);
          tensorforge::fmacdpp16<14>(v68_acc, v84_data, v61_data);
          tensorforge::fmacdpp16<15>(v68_acc, v84_data, v63_data);
          tensorforge::fmacdpp16<0>(v69_acc, v85_data, v33_data);
          tensorforge::fmacdpp16<1>(v69_acc, v85_data, v35_data);
          tensorforge::fmacdpp16<2>(v69_acc, v85_data, v37_data);
          tensorforge::fmacdpp16<3>(v69_acc, v85_data, v39_data);
          tensorforge::fmacdpp16<4>(v69_acc, v85_data, v41_data);
          tensorforge::fmacdpp16<5>(v69_acc, v85_data, v43_data);
          tensorforge::fmacdpp16<6>(v69_acc, v85_data, v45_data);
          tensorforge::fmacdpp16<7>(v69_acc, v85_data, v47_data);
          tensorforge::fmacdpp16<8>(v69_acc, v85_data, v49_data);
          tensorforge::fmacdpp16<9>(v69_acc, v85_data, v51_data);
          tensorforge::fmacdpp16<10>(v69_acc, v85_data, v53_data);
          tensorforge::fmacdpp16<11>(v69_acc, v85_data, v55_data);
          tensorforge::fmacdpp16<12>(v69_acc, v85_data, v57_data);
          tensorforge::fmacdpp16<13>(v69_acc, v85_data, v59_data);
          tensorforge::fmacdpp16<14>(v69_acc, v85_data, v61_data);
          tensorforge::fmacdpp16<15>(v69_acc, v85_data, v63_data);
          tensorforge::fmacdpp16<0>(v70_acc, v86_data, v33_data);
          tensorforge::fmacdpp16<1>(v70_acc, v86_data, v35_data);
          tensorforge::fmacdpp16<2>(v70_acc, v86_data, v37_data);
          tensorforge::fmacdpp16<3>(v70_acc, v86_data, v39_data);
          tensorforge::fmacdpp16<4>(v70_acc, v86_data, v41_data);
          tensorforge::fmacdpp16<5>(v70_acc, v86_data, v43_data);
          tensorforge::fmacdpp16<6>(v70_acc, v86_data, v45_data);
          tensorforge::fmacdpp16<7>(v70_acc, v86_data, v47_data);
          tensorforge::fmacdpp16<8>(v70_acc, v86_data, v49_data);
          tensorforge::fmacdpp16<9>(v70_acc, v86_data, v51_data);
          tensorforge::fmacdpp16<10>(v70_acc, v86_data, v53_data);
          tensorforge::fmacdpp16<11>(v70_acc, v86_data, v55_data);
          tensorforge::fmacdpp16<12>(v70_acc, v86_data, v57_data);
          tensorforge::fmacdpp16<13>(v70_acc, v86_data, v59_data);
          tensorforge::fmacdpp16<14>(v70_acc, v86_data, v61_data);
          tensorforge::fmacdpp16<15>(v70_acc, v86_data, v63_data);
          tensorforge::fmacdpp16<0>(v71_acc, v87_data, v33_data);
          tensorforge::fmacdpp16<1>(v71_acc, v87_data, v35_data);
          tensorforge::fmacdpp16<2>(v71_acc, v87_data, v37_data);
          tensorforge::fmacdpp16<3>(v71_acc, v87_data, v39_data);
          tensorforge::fmacdpp16<4>(v71_acc, v87_data, v41_data);
          tensorforge::fmacdpp16<5>(v71_acc, v87_data, v43_data);
          tensorforge::fmacdpp16<6>(v71_acc, v87_data, v45_data);
          tensorforge::fmacdpp16<7>(v71_acc, v87_data, v47_data);
          tensorforge::fmacdpp16<8>(v71_acc, v87_data, v49_data);
          tensorforge::fmacdpp16<9>(v71_acc, v87_data, v51_data);
          tensorforge::fmacdpp16<10>(v71_acc, v87_data, v53_data);
          tensorforge::fmacdpp16<11>(v71_acc, v87_data, v55_data);
          tensorforge::fmacdpp16<12>(v71_acc, v87_data, v57_data);
          tensorforge::fmacdpp16<13>(v71_acc, v87_data, v59_data);
          tensorforge::fmacdpp16<14>(v71_acc, v87_data, v61_data);
          tensorforge::fmacdpp16<15>(v71_acc, v87_data, v63_data);
          tensorforge::fmacdpp16<0>(v72_acc, v88_data, v33_data);
          tensorforge::fmacdpp16<1>(v72_acc, v88_data, v35_data);
          tensorforge::fmacdpp16<2>(v72_acc, v88_data, v37_data);
          tensorforge::fmacdpp16<3>(v72_acc, v88_data, v39_data);
          tensorforge::fmacdpp16<4>(v72_acc, v88_data, v41_data);
          tensorforge::fmacdpp16<5>(v72_acc, v88_data, v43_data);
          tensorforge::fmacdpp16<6>(v72_acc, v88_data, v45_data);
          tensorforge::fmacdpp16<7>(v72_acc, v88_data, v47_data);
          tensorforge::fmacdpp16<8>(v72_acc, v88_data, v49_data);
          tensorforge::fmacdpp16<9>(v72_acc, v88_data, v51_data);
          tensorforge::fmacdpp16<10>(v72_acc, v88_data, v53_data);
          tensorforge::fmacdpp16<11>(v72_acc, v88_data, v55_data);
          tensorforge::fmacdpp16<12>(v72_acc, v88_data, v57_data);
          tensorforge::fmacdpp16<13>(v72_acc, v88_data, v59_data);
          tensorforge::fmacdpp16<14>(v72_acc, v88_data, v61_data);
          tensorforge::fmacdpp16<15>(v72_acc, v88_data, v63_data);
          tensorforge::fmacdpp16<0>(v73_acc, v89_data, v33_data);
          tensorforge::fmacdpp16<1>(v73_acc, v89_data, v35_data);
          tensorforge::fmacdpp16<2>(v73_acc, v89_data, v37_data);
          tensorforge::fmacdpp16<3>(v73_acc, v89_data, v39_data);
          tensorforge::fmacdpp16<4>(v73_acc, v89_data, v41_data);
          tensorforge::fmacdpp16<5>(v73_acc, v89_data, v43_data);
          tensorforge::fmacdpp16<6>(v73_acc, v89_data, v45_data);
          tensorforge::fmacdpp16<7>(v73_acc, v89_data, v47_data);
          tensorforge::fmacdpp16<8>(v73_acc, v89_data, v49_data);
          tensorforge::fmacdpp16<9>(v73_acc, v89_data, v51_data);
          tensorforge::fmacdpp16<10>(v73_acc, v89_data, v53_data);
          tensorforge::fmacdpp16<11>(v73_acc, v89_data, v55_data);
          tensorforge::fmacdpp16<12>(v73_acc, v89_data, v57_data);
          tensorforge::fmacdpp16<13>(v73_acc, v89_data, v59_data);
          tensorforge::fmacdpp16<14>(v73_acc, v89_data, v61_data);
          tensorforge::fmacdpp16<15>(v73_acc, v89_data, v63_data);
          tensorforge::fmacdpp16<0>(v74_acc, v90_data, v33_data);
          tensorforge::fmacdpp16<1>(v74_acc, v90_data, v35_data);
          tensorforge::fmacdpp16<2>(v74_acc, v90_data, v37_data);
          tensorforge::fmacdpp16<3>(v74_acc, v90_data, v39_data);
          tensorforge::fmacdpp16<4>(v74_acc, v90_data, v41_data);
          tensorforge::fmacdpp16<5>(v74_acc, v90_data, v43_data);
          tensorforge::fmacdpp16<6>(v74_acc, v90_data, v45_data);
          tensorforge::fmacdpp16<7>(v74_acc, v90_data, v47_data);
          tensorforge::fmacdpp16<8>(v74_acc, v90_data, v49_data);
          tensorforge::fmacdpp16<9>(v74_acc, v90_data, v51_data);
          tensorforge::fmacdpp16<10>(v74_acc, v90_data, v53_data);
          tensorforge::fmacdpp16<11>(v74_acc, v90_data, v55_data);
          tensorforge::fmacdpp16<12>(v74_acc, v90_data, v57_data);
          tensorforge::fmacdpp16<13>(v74_acc, v90_data, v59_data);
          tensorforge::fmacdpp16<14>(v74_acc, v90_data, v61_data);
          tensorforge::fmacdpp16<15>(v74_acc, v90_data, v63_data);
          tensorforge::fmacdpp16<0>(v75_acc, v91_data, v33_data);
          tensorforge::fmacdpp16<1>(v75_acc, v91_data, v35_data);
          tensorforge::fmacdpp16<2>(v75_acc, v91_data, v37_data);
          tensorforge::fmacdpp16<3>(v75_acc, v91_data, v39_data);
          tensorforge::fmacdpp16<4>(v75_acc, v91_data, v41_data);
          tensorforge::fmacdpp16<5>(v75_acc, v91_data, v43_data);
          tensorforge::fmacdpp16<6>(v75_acc, v91_data, v45_data);
          tensorforge::fmacdpp16<7>(v75_acc, v91_data, v47_data);
          tensorforge::fmacdpp16<8>(v75_acc, v91_data, v49_data);
          tensorforge::fmacdpp16<9>(v75_acc, v91_data, v51_data);
          tensorforge::fmacdpp16<10>(v75_acc, v91_data, v53_data);
          tensorforge::fmacdpp16<11>(v75_acc, v91_data, v55_data);
          tensorforge::fmacdpp16<12>(v75_acc, v91_data, v57_data);
          tensorforge::fmacdpp16<13>(v75_acc, v91_data, v59_data);
          tensorforge::fmacdpp16<14>(v75_acc, v91_data, v61_data);
          tensorforge::fmacdpp16<15>(v75_acc, v91_data, v63_data);
          tensorforge::fmacdpp16<0>(v76_acc, v92_data, v33_data);
          tensorforge::fmacdpp16<1>(v76_acc, v92_data, v35_data);
          tensorforge::fmacdpp16<2>(v76_acc, v92_data, v37_data);
          tensorforge::fmacdpp16<3>(v76_acc, v92_data, v39_data);
          tensorforge::fmacdpp16<4>(v76_acc, v92_data, v41_data);
          tensorforge::fmacdpp16<5>(v76_acc, v92_data, v43_data);
          tensorforge::fmacdpp16<6>(v76_acc, v92_data, v45_data);
          tensorforge::fmacdpp16<7>(v76_acc, v92_data, v47_data);
          tensorforge::fmacdpp16<8>(v76_acc, v92_data, v49_data);
          tensorforge::fmacdpp16<9>(v76_acc, v92_data, v51_data);
          tensorforge::fmacdpp16<10>(v76_acc, v92_data, v53_data);
          tensorforge::fmacdpp16<11>(v76_acc, v92_data, v55_data);
          tensorforge::fmacdpp16<12>(v76_acc, v92_data, v57_data);
          tensorforge::fmacdpp16<13>(v76_acc, v92_data, v59_data);
          tensorforge::fmacdpp16<14>(v76_acc, v92_data, v61_data);
          tensorforge::fmacdpp16<15>(v76_acc, v92_data, v63_data);
          tensorforge::fmacdpp16<0>(v77_acc, v93_data, v33_data);
          tensorforge::fmacdpp16<1>(v77_acc, v93_data, v35_data);
          tensorforge::fmacdpp16<2>(v77_acc, v93_data, v37_data);
          tensorforge::fmacdpp16<3>(v77_acc, v93_data, v39_data);
          tensorforge::fmacdpp16<4>(v77_acc, v93_data, v41_data);
          tensorforge::fmacdpp16<5>(v77_acc, v93_data, v43_data);
          tensorforge::fmacdpp16<6>(v77_acc, v93_data, v45_data);
          tensorforge::fmacdpp16<7>(v77_acc, v93_data, v47_data);
          tensorforge::fmacdpp16<8>(v77_acc, v93_data, v49_data);
          tensorforge::fmacdpp16<9>(v77_acc, v93_data, v51_data);
          tensorforge::fmacdpp16<10>(v77_acc, v93_data, v53_data);
          tensorforge::fmacdpp16<11>(v77_acc, v93_data, v55_data);
          tensorforge::fmacdpp16<12>(v77_acc, v93_data, v57_data);
          tensorforge::fmacdpp16<13>(v77_acc, v93_data, v59_data);
          tensorforge::fmacdpp16<14>(v77_acc, v93_data, v61_data);
          tensorforge::fmacdpp16<15>(v77_acc, v93_data, v63_data);
          tensorforge::fmacdpp16<0>(v78_acc, v94_data, v33_data);
          tensorforge::fmacdpp16<1>(v78_acc, v94_data, v35_data);
          tensorforge::fmacdpp16<2>(v78_acc, v94_data, v37_data);
          tensorforge::fmacdpp16<3>(v78_acc, v94_data, v39_data);
          tensorforge::fmacdpp16<4>(v78_acc, v94_data, v41_data);
          tensorforge::fmacdpp16<5>(v78_acc, v94_data, v43_data);
          tensorforge::fmacdpp16<6>(v78_acc, v94_data, v45_data);
          tensorforge::fmacdpp16<7>(v78_acc, v94_data, v47_data);
          tensorforge::fmacdpp16<8>(v78_acc, v94_data, v49_data);
          tensorforge::fmacdpp16<9>(v78_acc, v94_data, v51_data);
          tensorforge::fmacdpp16<10>(v78_acc, v94_data, v53_data);
          tensorforge::fmacdpp16<11>(v78_acc, v94_data, v55_data);
          tensorforge::fmacdpp16<12>(v78_acc, v94_data, v57_data);
          tensorforge::fmacdpp16<13>(v78_acc, v94_data, v59_data);
          tensorforge::fmacdpp16<14>(v78_acc, v94_data, v61_data);
          tensorforge::fmacdpp16<15>(v78_acc, v94_data, v63_data);
          tensorforge::fmacdpp16<0>(v79_acc, v95_data, v33_data);
          tensorforge::fmacdpp16<1>(v79_acc, v95_data, v35_data);
          tensorforge::fmacdpp16<2>(v79_acc, v95_data, v37_data);
          tensorforge::fmacdpp16<3>(v79_acc, v95_data, v39_data);
          tensorforge::fmacdpp16<4>(v79_acc, v95_data, v41_data);
          tensorforge::fmacdpp16<5>(v79_acc, v95_data, v43_data);
          tensorforge::fmacdpp16<6>(v79_acc, v95_data, v45_data);
          tensorforge::fmacdpp16<7>(v79_acc, v95_data, v47_data);
          tensorforge::fmacdpp16<8>(v79_acc, v95_data, v49_data);
          tensorforge::fmacdpp16<9>(v79_acc, v95_data, v51_data);
          tensorforge::fmacdpp16<10>(v79_acc, v95_data, v53_data);
          tensorforge::fmacdpp16<11>(v79_acc, v95_data, v55_data);
          tensorforge::fmacdpp16<12>(v79_acc, v95_data, v57_data);
          tensorforge::fmacdpp16<13>(v79_acc, v95_data, v59_data);
          tensorforge::fmacdpp16<14>(v79_acc, v95_data, v61_data);
          tensorforge::fmacdpp16<15>(v79_acc, v95_data, v63_data);
          r1[0] = v64_acc;
          r1[1] = v65_acc;
          r1[2] = v66_acc;
          r1[3] = v67_acc;
          r1[4] = v68_acc;
          r1[5] = v69_acc;
          r1[6] = v70_acc;
          r1[7] = v71_acc;
          r1[8] = v72_acc;
          r1[9] = v73_acc;
          r1[10] = v74_acc;
          r1[11] = v75_acc;
          r1[12] = v76_acc;
          r1[13] = v77_acc;
          r1[14] = v78_acc;
          r1[15] = v79_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v96_i0 = 0; v96_i0 < 1; ++v96_i0) {
            int32_t v101_lead = v20_lead + (v96_i0 * 16);
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 16; ++v97_i1) {
              double v99_data = r1[(v96_i0 + v97_i1)];
              glb_m0[(v101_lead + (v97_i1 * 16))] = v99_data;
            }
          }
        }
      }
    }
  }
}

