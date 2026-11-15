// === base name ===
kernel_5529e89525a0132d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5529e89525a0132d = {{16, 8, 1}, 16, 20, 1, 8, 9024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5529e89525a0132d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5529e89525a0132d(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5529e89525a0132d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5529e89525a0132d, block.x * block.y * block.z, 1128 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1128 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5529e89525a0132d, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (1128 * sizeof(double)));
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
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 1128 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5529e89525a0132d(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5529e89525a0132d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5529e89525a0132d), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  const double *const * m0Arg = (const double *const *)m0;
  const double *const * m1Arg = (const double *const *)m1;
  double *const * m2Arg = (double *const *)m2;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m3;
  const double *const * m4Arg = (const double *const *)m4;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m5Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m5;
  const double *const * m6Arg = (const double *const *)m6;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m7Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m7;
  const double *const * m8Arg = (const double *const *)m8;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m9Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m9;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5529e89525a0132d, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, m5Arg, m6Arg, m6_extraOffset, m7Arg, m8Arg, m8_extraOffset, m9Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_5529e89525a0132d(const double *const * __restrict__ m0, size_t m0_extraOffset, const double *const * __restrict__ m1, size_t m1_extraOffset, double *const * __restrict__ m2, size_t m2_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m3, const double *const * __restrict__ m4, size_t m4_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m5, const double *const * __restrict__ m6, size_t m6_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m7, const double *const * __restrict__ m8, size_t m8_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m9, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16,merge_variants=1,mma_d_layout=1,mma_d_layout_loops=1
    // launch: 16 lanes (20 active) x 8 per block = block 16x8x1, 9024 B shared, occupancy grid
    // operands:
    //   v0 9×9(9×9) {0..9}×{0..9} pointer_based
    //   v1 20×10(20×10) {0..20}×{0..10} none
    //   m0 20×9(20×9) {0..20}×{0..9} pointer_based
    //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m2 20×9(20×9) {0..20}×{0..9} pointer_based
    //   m3 20×10(20×10) {0..20}×{0..10} none
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m5 20×10(20×10) {0..20}×{0..10} none
    //   m6 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m7 20×10(20×10) {0..20}×{0..10} none
    //   m8 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m9 20×10(20×10) {0..20}×{0..10} none
    // operations:
    //   for 4 iterations (any order):
    //     t0[i,j] = m0[i,k]@{0..10}×{0..9} × v0[k,j]
    //     m2[i,j] += v1[i,k] × t0[k,j]
    //     v0 ∈ {m1, m4, m6, m8}
    //     v1 ∈ {m3, m5, m7, m9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":20,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1128}],"shared_bytes":9024,"shared_elements":1128,"threads_per_mult":16},"loops":[{"body":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"v0","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"v1","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"holes":[{"members":["m1","m4","m6","m8"],"stand_in":"v0"},{"members":["m3","m5","m7","m9"],"stand_in":"v1"}],"iterations":4,"kind":"for","sequential":false}],"operands":[{"addressing":"pointer_based","alias":"variant0","bbox":[[0,0],[9,9]],"name":"v0","ordered":false,"parts":1,"shape":[9,9],"variant":true},{"addressing":"none","alias":"variant1","bbox":[[0,0],[20,10]],"name":"v1","ordered":false,"parts":1,"shape":[20,10],"variant":true},{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"pointer_based","alias":"S0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[20,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"K0","bbox":[[0,0],[20,10]],"name":"m3","ordered":false,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K1","bbox":[[0,0],[20,10]],"name":"m5","ordered":false,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K2","bbox":[[0,0],[20,10]],"name":"m7","ordered":false,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S3","bbox":[[0,0],[9,9]],"name":"m8","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K3","bbox":[[0,0],[20,10]],"name":"m9","ordered":false,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m9","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[116 * threadIdx.y + 200];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m3[0];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m5[0];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m7[0];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const glb_m9 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m9[0];
      size_t v1640_batchIdLane0 = threadIdx.y % 8;
      size_t v1643_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 8) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 8) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v1742_lead = threadIdx.x % 16;
      bool v1743_g = v1742_lead < 10;
      int32_t v1663_m = threadIdx.y % 4;
      int32_t v1671_a = threadIdx.x + 20_i32;
      int32_t v1675_a = threadIdx.x + 40_i32;
      int32_t v1679_a = threadIdx.x + 60_i32;
      int32_t v1684_a = threadIdx.x + 80_i32;
      int32_t v1688_a = threadIdx.x + 100_i32;
      int32_t v1692_a = threadIdx.x + 120_i32;
      int32_t v1696_a = threadIdx.x + 140_i32;
      int32_t v1701_a = threadIdx.x + 160_i32;
      int32_t v1704_dr = threadIdx.x + 16;
      bool v1705_g = v1704_dr < 20;
      int32_t v1709_a = v1704_dr + 20_i32;
      int32_t v1711_a = v1704_dr + 40_i32;
      int32_t v1713_a = v1704_dr + 60_i32;
      int32_t v1716_a = v1704_dr + 80_i32;
      int32_t v1718_a = v1704_dr + 100_i32;
      int32_t v1720_a = v1704_dr + 120_i32;
      int32_t v1722_a = v1704_dr + 140_i32;
      int32_t v1725_a = v1704_dr + 160_i32;
      bool v1757_g = v1742_lead < 9;
      int32_t v1799_c = threadIdx.x / 4;
      int32_t v1801_sh = v1799_c * 10;
      int32_t v1803_sh = ((threadIdx.x % 4) - v1663_m) * 116;
      int32_t v1804_sh = (v1663_m + v1801_sh) + v1803_sh;
      int32_t v1808_a = v1804_sh + 40;
      bool v1810_g = v1799_c < 1;
      int32_t v1812_sh = (min(v1799_c, 0)) * 10;
      int32_t v1814_sh = (v1663_m + v1812_sh) + v1803_sh;
      int32_t v1816_a = v1814_sh + 80;
      int32_t v1819_r = v1663_m * 20;
      int32_t v1822_a = v1742_lead + v1819_r;
      int32_t v1829_lead = v1742_lead + 16_i32;
      int32_t v1830_a = v1829_lead + v1819_r;
      int32_t v1836_a = v1804_sh + 4;
      int32_t v1839_a = v1836_a + 40;
      int32_t v1842_a = (v1814_sh + 4) + 80;
      int32_t v1845_a = v1822_a + 80;
      int32_t v1850_a = v1830_a + 80;
      int32_t v1858_ks = (min((max((v1663_m + 8), 0)), 9)) - 8;
      int32_t v1861_a = ((v1858_ks + v1801_sh) + v1803_sh) + 8;
      double v1865_ks = (v1663_m == 2) ? 0.0 : 0.0;
      bool v1866_g = v1663_m == 1;
      bool v1868_g = v1663_m == 0;
      int32_t v1870_a = v1861_a + 40;
      int32_t v1877_a = (((v1858_ks + v1812_sh) + v1803_sh) + 8) + 80;
      int32_t v1882_a = v1742_lead + 160;
      int32_t v1884_a = v1742_lead + 180;
      int32_t v1891_a = v1829_lead + 160;
      int32_t v1893_a = v1829_lead + 180;
      int32_t v2004_flat = threadIdx.x + (threadIdx.y * 16);
      double * v2005_shard = &totalShrMem[0];
      double r0[9]{};
      size_t v2034_row = v1643_batchIdGroup0First + v1640_batchIdLane0;
      const bool peel_batchIdActive0 = v2034_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v2034_row]));
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[(peel_batchIdActive0 ? v2034_row : v1643_batchIdGroup0First)][0 + m0_extraOffset];
      // r0 = load{g>r}(peel_glb_m0);
      if (v1743_g) {
        #pragma unroll
        for (int32_t v2042_i1 = 0; v2042_i1 < 9; ++v2042_i1) {
          double v2045_data = __builtin_nontemporal_load(&peel_glb_m0[(v1742_lead + (v2042_i1 * 20))]);
          r0[v2042_i1] = v2045_data;
        }
      }
      int32_t v2012_s = v2004_flat + 128;
      int32_t v2014_s = (v2012_s < 200) ? v2012_s : v2004_flat;
      for (size_t v1644_batchIdGroup0 = (threadIdx.y - threadIdx.y % 8) + blockDim.y * (blockIdx.x); v1644_batchIdGroup0 < numElements0; v1644_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v2031_groupNextAhead = v1644_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v2033_groupNext = (v2031_groupNextAhead < numElements0) ? v2031_groupNextAhead : v1644_batchIdGroup0;
        size_t v2038_row = v2033_groupNext + v1640_batchIdLane0;
        const bool wrap_batchIdActive0 = v2038_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v2038_row]));
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[(wrap_batchIdActive0 ? v2038_row : v2033_groupNext)][0 + m0_extraOffset];
        size_t v1645_row = v1644_batchIdGroup0 + v1640_batchIdLane0;
        const bool batchIdActive0 = v1645_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1645_row]));
        size_t v1647_batchId0 = batchIdActive0 ? v1645_row : v1644_batchIdGroup0;
        size_t v1648_ahead1 = v1647_batchId0 + (gridDim.x * blockDim.y);
        size_t v1650_batchId1 = (v1648_ahead1 < numElements0) ? v1648_ahead1 : v1647_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v1647_batchId0][0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m2[v1647_batchId0][0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m4[v1647_batchId0][0 + m4_extraOffset];
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m6[v1647_batchId0][0 + m6_extraOffset];
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m8[v1647_batchId0][0 + m8_extraOffset];
        double * __restrict__ s0 = &localShrMem0[0];
        // ir4 = +(glb_v1 * s0)
        // [(0, 20), (0, 9)] [(0, 10)]
        double v1668_q = glb_m2[threadIdx.x];
        double v1672_q = glb_m2[v1671_a];
        double v1676_q = glb_m2[v1675_a];
        double v1680_q = glb_m2[v1679_a];
        tensorforge::VectorT<double, 4> v1681_acc = {v1668_q, v1672_q, v1676_q, v1680_q};
        double v1685_q = glb_m2[v1684_a];
        double v1689_q = glb_m2[v1688_a];
        double v1693_q = glb_m2[v1692_a];
        double v1697_q = glb_m2[v1696_a];
        tensorforge::VectorT<double, 4> v1698_acc = {v1685_q, v1689_q, v1693_q, v1697_q};
        double v1702_q = glb_m2[v1701_a];
        tensorforge::VectorT<double, 4> v1703_acc = {v1702_q, 0.0, 0.0, 0.0};
        double v1708_q = v1705_g ? (glb_m2[v1704_dr]) : (0.0);
        double v1710_q = v1705_g ? (glb_m2[v1709_a]) : (0.0);
        double v1712_q = v1705_g ? (glb_m2[v1711_a]) : (0.0);
        double v1714_q = v1705_g ? (glb_m2[v1713_a]) : (0.0);
        tensorforge::VectorT<double, 4> v1715_acc = {v1708_q, v1710_q, v1712_q, v1714_q};
        double v1717_q = v1705_g ? (glb_m2[v1716_a]) : (0.0);
        double v1719_q = v1705_g ? (glb_m2[v1718_a]) : (0.0);
        double v1721_q = v1705_g ? (glb_m2[v1720_a]) : (0.0);
        double v1723_q = v1705_g ? (glb_m2[v1722_a]) : (0.0);
        tensorforge::VectorT<double, 4> v1724_acc = {v1717_q, v1719_q, v1721_q, v1723_q};
        double v1726_q = v1705_g ? (glb_m2[v1725_a]) : (0.0);
        tensorforge::VectorT<double, 4> v1727_acc = {v1726_q, 0.0, 0.0, 0.0};
        double r1[9]{};
        const double *const peel_v0Table = (0 == 0) ? (const double *)glb_m1 : (0 == 1) ? (const double *)glb_m4 : (0 == 2) ? (const double *)glb_m6 : (const double *)glb_m8;
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_v0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)peel_v0Table;
        #pragma unroll
        for (int32_t v2023_i1 = 0; v2023_i1 < 9; ++v2023_i1) {
          double v2026_data = v1757_g ? (__builtin_nontemporal_load(&peel_glb_v0[(v1742_lead + (v2023_i1 * 9))])) : (0.0);
          r1[v2023_i1] = v2026_data;
        }
        tensorforge::VectorT<double, 4> v1729_acc0 = v1681_acc;
        tensorforge::VectorT<double, 4> v1730_acc1 = v1698_acc;
        tensorforge::VectorT<double, 4> v1731_acc2 = v1703_acc;
        tensorforge::VectorT<double, 4> v1732_acc3 = v1715_acc;
        tensorforge::VectorT<double, 4> v1733_acc4 = v1724_acc;
        tensorforge::VectorT<double, 4> v1734_acc5 = v1727_acc;
        for (int32_t batchIdv6 = 0; batchIdv6 < 4; ++batchIdv6) {
          int32_t v2016_tripAhead = batchIdv6 + 1;
          int32_t v2018_trip = (v2016_tripAhead < 4) ? v2016_tripAhead : batchIdv6;
          const double *const wrap_v0Table = (v2018_trip == 0) ? (const double *)glb_m1 : (v2018_trip == 1) ? (const double *)glb_m4 : (v2018_trip == 2) ? (const double *)glb_m6 : (const double *)glb_m8;
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_v0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)wrap_v0Table;
          const double *const v1Table = (batchIdv6 == 0) ? (const double *)glb_m3 : (batchIdv6 == 1) ? (const double *)glb_m5 : (batchIdv6 == 2) ? (const double *)glb_m7 : (const double *)glb_m9;
          tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const glb_v1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)v1Table;
          // r1 = load{g>r}(glb_v0);
          double v2011_sh = glb_v1[v2004_flat];
          double v2015_sh = glb_v1[v2014_s];
          double r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 10), (0, 9)] [(0, 9)]
          double v1761_data = r0[0];
          double v1762_data = r0[1];
          double v1763_data = r0[2];
          double v1764_data = r0[3];
          double v1765_data = r0[4];
          double v1766_data = r0[5];
          double v1767_data = r0[6];
          double v1768_data = r0[7];
          double v1769_data = r0[8];
          double v1770_acc{};
          double v1771_acc{};
          double v1772_acc{};
          double v1773_acc{};
          double v1774_acc{};
          double v1775_acc{};
          double v1776_acc{};
          double v1777_acc{};
          double v1778_acc{};
          double v1779_data = r1[0];
          double v1780_data = r1[1];
          double v1781_data = r1[2];
          double v1782_data = r1[3];
          double v1783_data = r1[4];
          double v1784_data = r1[5];
          double v1785_data = r1[6];
          double v1786_data = r1[7];
          double v1787_data = r1[8];
          #pragma unroll
          for (int32_t v2027_i1 = 0; v2027_i1 < 9; ++v2027_i1) {
            double v2030_data = v1757_g ? (__builtin_nontemporal_load(&wrap_glb_v0[(v1742_lead + (v2027_i1 * 9))])) : (0.0);
            r1[v2027_i1] = v2030_data;
          }
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %10 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %12 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %4, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %10 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %12 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %14, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %10 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %12 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %15, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %16, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %16, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %16, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %16, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %16, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v1770_acc), "+&v"(v1771_acc), "+&v"(v1772_acc), "+&v"(v1773_acc)
          : "v"(v1779_data), "v"(v1761_data), "v"(v1762_data), "v"(v1763_data), "v"(v1764_data), "v"(v1765_data), "v"(v1766_data), "v"(v1767_data), "v"(v1768_data), "v"(v1769_data), "v"(v1780_data), "v"(v1781_data), "v"(v1782_data)
          );
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %5, %6 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %5, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %5, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %5, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %11 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %6 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %10, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %11 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %6 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %2, %16, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %11 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %6 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %3, %17, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %4, %18, %11 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v1773_acc), "+&v"(v1774_acc), "+&v"(v1775_acc), "+&v"(v1776_acc), "+&v"(v1777_acc)
          : "v"(v1782_data), "v"(v1766_data), "v"(v1767_data), "v"(v1768_data), "v"(v1769_data), "v"(v1783_data), "v"(v1761_data), "v"(v1762_data), "v"(v1763_data), "v"(v1764_data), "v"(v1765_data), "v"(v1784_data), "v"(v1785_data), "v"(v1786_data)
          );
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %2, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %2, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %1, %11, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v1777_acc), "+&v"(v1778_acc)
          : "v"(v1786_data), "v"(v1762_data), "v"(v1763_data), "v"(v1764_data), "v"(v1765_data), "v"(v1766_data), "v"(v1767_data), "v"(v1768_data), "v"(v1769_data), "v"(v1787_data), "v"(v1761_data)
          );
          r2[0] = v1770_acc;
          r2[1] = v1771_acc;
          r2[2] = v1772_acc;
          r2[3] = v1773_acc;
          r2[4] = v1774_acc;
          r2[5] = v1775_acc;
          r2[6] = v1776_acc;
          r2[7] = v1777_acc;
          r2[8] = v1778_acc;
          // r2 = 0 past (0, 10)
          // s0 = store{r>s}(localShrMem0, r2);
          if (v1743_g) {
            #pragma unroll
            for (int32_t v1789_i1 = 0; v1789_i1 < 9; ++v1789_i1) {
              double v1791_data = r2[v1789_i1];
              s0[(v1742_lead + (v1789_i1 * 10))] = v1791_data;
            }
          }
          // ir4 = +(glb_v1 * s0)
          // [(0, 20), (0, 9)] [(0, 10)]
          __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
          double v1807_data = s0[v1804_sh];
          double v1809_data = s0[v1808_a];
          double v1817_data = s0[v1816_a];
          double v1818_af = v1810_g ? v1817_data : 0.0;
          __syncthreads();
          v2005_shard[v2004_flat] = v2011_sh;
          v2005_shard[v2014_s] = v2015_sh;
          __syncthreads();
          double v1824_data = v2005_shard[v1822_a];
          tensorforge::VectorT<double, 4> v1825_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1807_data, v1824_data, v1729_acc0, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1826_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1809_data, v1824_data, v1730_acc1, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1827_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1818_af, v1824_data, v1731_acc2, 0, 0, 0);
          double v1832_data = v2005_shard[v1830_a];
          tensorforge::VectorT<double, 4> v1833_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1807_data, v1832_data, v1732_acc3, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1834_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1809_data, v1832_data, v1733_acc4, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1835_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1818_af, v1832_data, v1734_acc5, 0, 0, 0);
          double v1838_data = s0[v1836_a];
          double v1840_data = s0[v1839_a];
          double v1843_data = s0[v1842_a];
          double v1844_af = v1810_g ? v1843_data : 0.0;
          double v1846_data = v2005_shard[v1845_a];
          tensorforge::VectorT<double, 4> v1847_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1838_data, v1846_data, v1825_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1848_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1840_data, v1846_data, v1826_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1849_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1844_af, v1846_data, v1827_acc, 0, 0, 0);
          double v1851_data = v2005_shard[v1850_a];
          tensorforge::VectorT<double, 4> v1852_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1838_data, v1851_data, v1833_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1853_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1840_data, v1851_data, v1834_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1854_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1844_af, v1851_data, v1835_acc, 0, 0, 0);
          double v1863_data = s0[v1861_a];
          double v1869_ks = v1868_g ? v1863_data : (v1866_g ? v1863_data : v1865_ks);
          double v1871_data = s0[v1870_a];
          double v1873_ks = v1868_g ? v1871_data : (v1866_g ? v1871_data : v1865_ks);
          double v1878_data = s0[v1877_a];
          double v1879_af = v1810_g ? v1878_data : 0.0;
          double v1881_ks = v1868_g ? v1879_af : (v1866_g ? v1879_af : v1865_ks);
          double v1883_data = v2005_shard[v1882_a];
          double v1885_data = v2005_shard[v1884_a];
          double v1887_ks = v1868_g ? v1883_data : (v1866_g ? v1885_data : v1865_ks);
          tensorforge::VectorT<double, 4> v1888_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1869_ks, v1887_ks, v1847_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1889_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1873_ks, v1887_ks, v1848_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1890_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1881_ks, v1887_ks, v1849_acc, 0, 0, 0);
          double v1892_data = v2005_shard[v1891_a];
          double v1894_data = v2005_shard[v1893_a];
          double v1896_ks = v1868_g ? v1892_data : (v1866_g ? v1894_data : v1865_ks);
          tensorforge::VectorT<double, 4> v1897_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1869_ks, v1896_ks, v1852_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1898_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1873_ks, v1896_ks, v1853_acc, 0, 0, 0);
          tensorforge::VectorT<double, 4> v1899_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1881_ks, v1896_ks, v1854_acc, 0, 0, 0);
          v1729_acc0 = v1888_acc;
          v1730_acc1 = v1889_acc;
          v1731_acc2 = v1890_acc;
          v1732_acc3 = v1897_acc;
          v1733_acc4 = v1898_acc;
          v1734_acc5 = v1899_acc;
        }
        // r0 = load{g>r}(wrap_glb_m0);
        if (v1743_g) {
          #pragma unroll
          for (int32_t v2046_i1 = 0; v2046_i1 < 9; ++v2046_i1) {
            double v2049_data = __builtin_nontemporal_load(&wrap_glb_m0[(v1742_lead + (v2046_i1 * 20))]);
            r0[v2046_i1] = v2049_data;
          }
        }
        // ir4 = +(glb_v1 * s0)
        // [(0, 20), (0, 9)] [(0, 10)]
        double v1909_el = v1729_acc0[0];
        if (batchIdActive0) {
          glb_m2[threadIdx.x] = v1909_el;
        }
        double v1914_el = v1729_acc0[1];
        if (batchIdActive0) {
          glb_m2[v1671_a] = v1914_el;
        }
        double v1919_el = v1729_acc0[2];
        if (batchIdActive0) {
          glb_m2[v1675_a] = v1919_el;
        }
        double v1924_el = v1729_acc0[3];
        if (batchIdActive0) {
          glb_m2[v1679_a] = v1924_el;
        }
        double v1929_el = v1730_acc1[0];
        if (batchIdActive0) {
          glb_m2[v1684_a] = v1929_el;
        }
        double v1934_el = v1730_acc1[1];
        if (batchIdActive0) {
          glb_m2[v1688_a] = v1934_el;
        }
        double v1939_el = v1730_acc1[2];
        if (batchIdActive0) {
          glb_m2[v1692_a] = v1939_el;
        }
        double v1944_el = v1730_acc1[3];
        if (batchIdActive0) {
          glb_m2[v1696_a] = v1944_el;
        }
        double v1949_el = v1731_acc2[0];
        if (batchIdActive0) {
          glb_m2[v1701_a] = v1949_el;
        }
        double v1956_el = v1732_acc3[0];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1704_dr] = v1956_el;
          }
        }
        double v1961_el = v1732_acc3[1];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1709_a] = v1961_el;
          }
        }
        double v1966_el = v1732_acc3[2];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1711_a] = v1966_el;
          }
        }
        double v1971_el = v1732_acc3[3];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1713_a] = v1971_el;
          }
        }
        double v1976_el = v1733_acc4[0];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1716_a] = v1976_el;
          }
        }
        double v1981_el = v1733_acc4[1];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1718_a] = v1981_el;
          }
        }
        double v1986_el = v1733_acc4[2];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1720_a] = v1986_el;
          }
        }
        double v1991_el = v1733_acc4[3];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1722_a] = v1991_el;
          }
        }
        double v1996_el = v1734_acc5[0];
        if (v1705_g) {
          if (batchIdActive0) {
            glb_m2[v1725_a] = v1996_el;
          }
        }
      }
    }
  }
}

