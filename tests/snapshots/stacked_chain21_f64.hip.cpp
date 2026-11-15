// === base name ===
kernel_be059ee8b833456b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_be059ee8b833456b = {{16, 16, 1}, 16, 56, 1, 16, 45568, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_be059ee8b833456b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_be059ee8b833456b(const double * m0, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, double ** m3, size_t m3_extraOffset, const double * m4, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_be059ee8b833456b(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_be059ee8b833456b, block.x * block.y * block.z, 5696 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (5696 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_be059ee8b833456b, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (5696 * sizeof(double)));
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
  config.sharedMemBytes = 5696 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_be059ee8b833456b(const double * m0, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, double ** m3, size_t m3_extraOffset, const double * m4, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_be059ee8b833456b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_be059ee8b833456b), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m0Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m0;
  const double *const * m1Arg = (const double *const *)m1;
  const double *const * m2Arg = (const double *const *)m2;
  double *const * m3Arg = (double *const *)m3;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m4Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m4;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_be059ee8b833456b, grid, block, config.sharedMemBytes, stream, m0Arg, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_be059ee8b833456b(tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m0, const double *const * __restrict__ m1, size_t m1_extraOffset, const double *const * __restrict__ m2, size_t m2_extraOffset, double *const * __restrict__ m3, size_t m3_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m4, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (56 active) x 16 per block = block 16x16x1, 45568 B shared, occupancy grid
    // operands:
    //   m0 21×56(21×56) {0..21}×{0..56} none
    //   m1 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m2 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m3 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m4 56×21(56×21) {0..56}×{0..21} none
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = t0[i,k] × m2[k,j]
    //   m3[i,j] += m4[i,k] × t1[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":56,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5696}],"shared_bytes":45568,"shared_elements":5696,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"fMrT","bbox":[[0,0],[21,56]],"name":"m0","ordered":false,"parts":1,"shape":[21,56],"variant":false},{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[56,9]],"name":"m1","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"pointer_based","alias":"F","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"R","bbox":[[0,0],[56,9]],"name":"m3","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"none","alias":"rDivM","bbox":[[0,0],[56,21]],"name":"m4","ordered":false,"parts":1,"shape":[56,21],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[21,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[21,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[21,56]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[21,56]},{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[21,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[21,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[21,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[21,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,21]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[56,21]},{"addressing":"strided","bbox":[[0,0],[21,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[21,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[208 * threadIdx.y + 2368];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m0[0];
      double * __restrict__ glb_m0 = &totalShrMem[0];
      // glb_m0 = load{g>s}(ptr_glb_m0[0, 1])
      tensorforge::VectorT<double, 2> v1335_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m0[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<double, 2>*)&glb_m0[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v1335_ld;
      tensorforge::VectorT<double, 2> v1336_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m0[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512];
      *(tensorforge::VectorT<double, 2>*)&glb_m0[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512] = v1336_ld;
      bool v1341_g = (threadIdx.x + (threadIdx.y * 16)) < 152;
      if (v1341_g) {
        double v1342_ld = ptr_glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
        glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v1342_ld;
      }
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m4 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m4[0];
      double * __restrict__ glb_m4 = &totalShrMem[1184];
      // glb_m4 = load{g>s}(ptr_glb_m4[0, 1])
      tensorforge::VectorT<double, 2> v1345_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m4[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<double, 2>*)&glb_m4[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v1345_ld;
      tensorforge::VectorT<double, 2> v1346_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m4[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512];
      *(tensorforge::VectorT<double, 2>*)&glb_m4[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512] = v1346_ld;
      if (v1341_g) {
        double v1347_ld = ptr_glb_m4[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
        glb_m4[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v1347_ld;
      }
      __syncthreads();
      double * __restrict__ s1 = &localShrMem0[0];
      size_t v1351_batchIdLane0 = threadIdx.y % 4;
      size_t v1354_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v1371_lead = threadIdx.x % 16;
      int32_t v1383_lead = v1371_lead + 48_i32;
      bool v1386_g = v1371_lead < 8;
      bool v2128_g = v1371_lead < 9;
      int32_t v1392_m = threadIdx.y % 4;
      int32_t v1393_e = threadIdx.x % 4;
      int32_t v1394_c = threadIdx.x / 4;
      int32_t v1396_src = (v1393_e * 16) + v1392_m;
      bool v1410_g = v1394_c == 2;
      bool v1412_g = v1394_c == 1;
      bool v1414_g = v1394_c == 0;
      int32_t v1427_r = v1392_m * 21;
      int32_t v1430_a = v1371_lead + v1427_r;
      int32_t v1436_lead = v1371_lead + 16_i32;
      int32_t v1437_a = v1436_lead + v1427_r;
      int32_t v1442_src = v1396_src + 4;
      int32_t v1465_a = v1430_a + 84;
      int32_t v1469_a = v1437_a + 84;
      int32_t v1473_src = v1396_src + 8;
      int32_t v1496_a = v1430_a + 168;
      int32_t v1500_a = v1437_a + 168;
      int32_t v1504_src = v1396_src + 12;
      int32_t v1527_a = v1430_a + 252;
      int32_t v1531_a = v1437_a + 252;
      int32_t v1557_a = v1430_a + 336;
      int32_t v1561_a = v1437_a + 336;
      int32_t v1587_a = v1430_a + 420;
      int32_t v1591_a = v1437_a + 420;
      int32_t v1617_a = v1430_a + 504;
      int32_t v1621_a = v1437_a + 504;
      int32_t v1647_a = v1430_a + 588;
      int32_t v1651_a = v1437_a + 588;
      int32_t v1677_a = v1430_a + 672;
      int32_t v1681_a = v1437_a + 672;
      int32_t v1707_a = v1430_a + 756;
      int32_t v1711_a = v1437_a + 756;
      int32_t v1737_a = v1430_a + 840;
      int32_t v1741_a = v1437_a + 840;
      int32_t v1767_a = v1430_a + 924;
      int32_t v1771_a = v1437_a + 924;
      int32_t v1797_a = v1430_a + 1008;
      int32_t v1801_a = v1437_a + 1008;
      int32_t v1827_a = v1430_a + 1092;
      int32_t v1831_a = v1437_a + 1092;
      int32_t v1859_a = v1371_lead + 21;
      int32_t v1861_a = v1436_lead + 21;
      int32_t v1863_a = v1371_lead + 42;
      int32_t v1865_a = v1436_lead + 42;
      int32_t v1867_a = v1371_lead + 63;
      int32_t v1869_a = v1436_lead + 63;
      int32_t v1871_a = v1371_lead + 84;
      int32_t v1873_a = v1436_lead + 84;
      int32_t v1875_a = v1371_lead + 105;
      int32_t v1877_a = v1436_lead + 105;
      int32_t v1879_a = v1371_lead + 126;
      int32_t v1881_a = v1436_lead + 126;
      int32_t v1883_a = v1371_lead + 147;
      int32_t v1885_a = v1436_lead + 147;
      int32_t v1887_a = v1371_lead + 168;
      int32_t v1889_a = v1436_lead + 168;
      int32_t v1891_a = v1371_lead + 189;
      int32_t v1893_a = v1436_lead + 189;
      int32_t v1895_a = v1371_lead + 210;
      int32_t v1897_a = v1436_lead + 210;
      int32_t v1899_a = v1371_lead + 231;
      int32_t v1901_a = v1436_lead + 231;
      int32_t v1903_a = v1371_lead + 252;
      int32_t v1905_a = v1436_lead + 252;
      int32_t v1907_a = v1371_lead + 273;
      int32_t v1909_a = v1436_lead + 273;
      int32_t v1911_a = v1371_lead + 294;
      int32_t v1913_a = v1436_lead + 294;
      int32_t v1915_a = v1371_lead + 315;
      int32_t v1917_a = v1436_lead + 315;
      int32_t v1919_a = v1371_lead + 336;
      int32_t v1921_a = v1436_lead + 336;
      int32_t v1923_a = v1371_lead + 357;
      int32_t v1925_a = v1436_lead + 357;
      int32_t v1927_a = v1371_lead + 378;
      int32_t v1929_a = v1436_lead + 378;
      int32_t v1931_a = v1371_lead + 399;
      int32_t v1933_a = v1436_lead + 399;
      int32_t v1935_a = v1371_lead + 420;
      int32_t v1937_a = v1436_lead + 420;
      int32_t v1939_a = v1371_lead + 441;
      int32_t v1941_a = v1436_lead + 441;
      int32_t v1943_a = v1371_lead + 462;
      int32_t v1945_a = v1436_lead + 462;
      int32_t v1947_a = v1371_lead + 483;
      int32_t v1949_a = v1436_lead + 483;
      int32_t v1951_a = v1371_lead + 504;
      int32_t v1953_a = v1436_lead + 504;
      int32_t v1955_a = v1371_lead + 525;
      int32_t v1957_a = v1436_lead + 525;
      int32_t v1959_a = v1371_lead + 546;
      int32_t v1961_a = v1436_lead + 546;
      int32_t v1963_a = v1371_lead + 567;
      int32_t v1965_a = v1436_lead + 567;
      int32_t v1967_a = v1371_lead + 588;
      int32_t v1969_a = v1436_lead + 588;
      int32_t v1971_a = v1371_lead + 609;
      int32_t v1973_a = v1436_lead + 609;
      int32_t v1975_a = v1371_lead + 630;
      int32_t v1977_a = v1436_lead + 630;
      int32_t v1979_a = v1371_lead + 651;
      int32_t v1981_a = v1436_lead + 651;
      int32_t v1983_a = v1371_lead + 672;
      int32_t v1985_a = v1436_lead + 672;
      int32_t v1987_a = v1371_lead + 693;
      int32_t v1989_a = v1436_lead + 693;
      int32_t v1991_a = v1371_lead + 714;
      int32_t v1993_a = v1436_lead + 714;
      int32_t v1995_a = v1371_lead + 735;
      int32_t v1997_a = v1436_lead + 735;
      int32_t v1999_a = v1371_lead + 756;
      int32_t v2001_a = v1436_lead + 756;
      int32_t v2003_a = v1371_lead + 777;
      int32_t v2005_a = v1436_lead + 777;
      int32_t v2007_a = v1371_lead + 798;
      int32_t v2009_a = v1436_lead + 798;
      int32_t v2011_a = v1371_lead + 819;
      int32_t v2013_a = v1436_lead + 819;
      int32_t v2015_a = v1371_lead + 840;
      int32_t v2017_a = v1436_lead + 840;
      int32_t v2019_a = v1371_lead + 861;
      int32_t v2021_a = v1436_lead + 861;
      int32_t v2023_a = v1371_lead + 882;
      int32_t v2025_a = v1436_lead + 882;
      int32_t v2027_a = v1371_lead + 903;
      int32_t v2029_a = v1436_lead + 903;
      int32_t v2031_a = v1371_lead + 924;
      int32_t v2033_a = v1436_lead + 924;
      int32_t v2035_a = v1371_lead + 945;
      int32_t v2037_a = v1436_lead + 945;
      int32_t v2039_a = v1371_lead + 966;
      int32_t v2041_a = v1436_lead + 966;
      int32_t v2043_a = v1371_lead + 987;
      int32_t v2045_a = v1436_lead + 987;
      int32_t v2047_a = v1371_lead + 1008;
      int32_t v2049_a = v1436_lead + 1008;
      int32_t v2051_a = v1371_lead + 1029;
      int32_t v2053_a = v1436_lead + 1029;
      int32_t v2055_a = v1371_lead + 1050;
      int32_t v2057_a = v1436_lead + 1050;
      int32_t v2059_a = v1371_lead + 1071;
      int32_t v2061_a = v1436_lead + 1071;
      int32_t v2063_a = v1371_lead + 1092;
      int32_t v2065_a = v1436_lead + 1092;
      int32_t v2067_a = v1371_lead + 1113;
      int32_t v2069_a = v1436_lead + 1113;
      int32_t v2071_a = v1371_lead + 1134;
      int32_t v2073_a = v1436_lead + 1134;
      int32_t v2075_a = v1371_lead + 1155;
      int32_t v2077_a = v1436_lead + 1155;
      bool v2095_g = v1371_lead < 5;
      int32_t v2210_sh = v1394_c * 21;
      int32_t v2212_sh = (v1393_e - v1392_m) * 208;
      int32_t v2213_sh = (v1392_m + v2210_sh) + v2212_sh;
      int32_t v2217_a = v2213_sh + 84;
      int32_t v2219_r = v1392_m * 56;
      int32_t v2222_a = v1371_lead + v2219_r;
      int32_t v2229_a = v1436_lead + v2219_r;
      int32_t v2235_lead = v1371_lead + 32_i32;
      int32_t v2236_a = v2235_lead + v2219_r;
      int32_t v2243_a = v1383_lead + v2219_r;
      int32_t v2248_a = v2213_sh + 4;
      int32_t v2251_a = v2248_a + 84;
      int32_t v2253_a = v2222_a + 224;
      int32_t v2257_a = v2229_a + 224;
      int32_t v2261_a = v2236_a + 224;
      int32_t v2265_a = v2243_a + 224;
      int32_t v2269_a = v2213_sh + 8;
      int32_t v2272_a = v2269_a + 84;
      int32_t v2274_a = v2222_a + 448;
      int32_t v2278_a = v2229_a + 448;
      int32_t v2282_a = v2236_a + 448;
      int32_t v2286_a = v2243_a + 448;
      int32_t v2290_a = v2213_sh + 12;
      int32_t v2293_a = v2290_a + 84;
      int32_t v2295_a = v2229_a + 672;
      int32_t v2299_a = v2236_a + 672;
      int32_t v2303_a = v2243_a + 672;
      int32_t v2307_a = v2213_sh + 16;
      int32_t v2310_a = v2307_a + 84;
      int32_t v2312_a = v2229_a + 896;
      int32_t v2316_a = v2236_a + 896;
      int32_t v2320_a = v2243_a + 896;
      int32_t v2330_a = ((((min((max((v1392_m + 20), 0)), 20)) - 20) + v2210_sh) + v2212_sh) + 20;
      double v2336_ks = (v1392_m == 1) ? 0.0 : ((v1392_m == 2) ? 0.0 : 0.0);
      bool v2337_g = v1392_m == 0;
      int32_t v2339_a = v2330_a + 84;
      int32_t v2342_a = v1436_lead + 1120;
      int32_t v2347_a = v2235_lead + 1120;
      int32_t v2352_a = v1383_lead + 1120;
      int32_t v2405_a = v1371_lead + 56;
      int32_t v2407_a = v1436_lead + 56;
      int32_t v2409_a = v2235_lead + 56;
      int32_t v2411_a = v1383_lead + 56;
      int32_t v2413_a = v1371_lead + 112;
      int32_t v2415_a = v1436_lead + 112;
      int32_t v2417_a = v2235_lead + 112;
      int32_t v2419_a = v1383_lead + 112;
      int32_t v2425_a = v2235_lead + 168;
      int32_t v2427_a = v1383_lead + 168;
      int32_t v2429_a = v1371_lead + 224;
      int32_t v2431_a = v1436_lead + 224;
      int32_t v2433_a = v2235_lead + 224;
      int32_t v2435_a = v1383_lead + 224;
      int32_t v2437_a = v1371_lead + 280;
      int32_t v2439_a = v1436_lead + 280;
      int32_t v2441_a = v2235_lead + 280;
      int32_t v2443_a = v1383_lead + 280;
      int32_t v2449_a = v2235_lead + 336;
      int32_t v2451_a = v1383_lead + 336;
      int32_t v2453_a = v1371_lead + 392;
      int32_t v2455_a = v1436_lead + 392;
      int32_t v2457_a = v2235_lead + 392;
      int32_t v2459_a = v1383_lead + 392;
      int32_t v2461_a = v1371_lead + 448;
      int32_t v2463_a = v1436_lead + 448;
      int32_t v2465_a = v2235_lead + 448;
      int32_t v2467_a = v1383_lead + 448;
      int32_t v2473_a = v2235_lead + 504;
      int32_t v2475_a = v1383_lead + 504;
      int32_t v2477_a = v1371_lead + 560;
      int32_t v2479_a = v1436_lead + 560;
      int32_t v2481_a = v2235_lead + 560;
      int32_t v2483_a = v1383_lead + 560;
      int32_t v2485_a = v1371_lead + 616;
      int32_t v2487_a = v1436_lead + 616;
      int32_t v2489_a = v2235_lead + 616;
      int32_t v2491_a = v1383_lead + 616;
      int32_t v2495_a = v2235_lead + 672;
      int32_t v2497_a = v1383_lead + 672;
      int32_t v2499_a = v1436_lead + 728;
      int32_t v2501_a = v2235_lead + 728;
      int32_t v2503_a = v1383_lead + 728;
      int32_t v2505_a = v1436_lead + 784;
      int32_t v2507_a = v2235_lead + 784;
      int32_t v2509_a = v1383_lead + 784;
      int32_t v2513_a = v2235_lead + 840;
      int32_t v2515_a = v1383_lead + 840;
      int32_t v2517_a = v1436_lead + 896;
      int32_t v2519_a = v2235_lead + 896;
      int32_t v2521_a = v1383_lead + 896;
      int32_t v2523_a = v1436_lead + 952;
      int32_t v2525_a = v2235_lead + 952;
      int32_t v2527_a = v1383_lead + 952;
      int32_t v2531_a = v2235_lead + 1008;
      int32_t v2533_a = v1383_lead + 1008;
      int32_t v2535_a = v1436_lead + 1064;
      int32_t v2537_a = v2235_lead + 1064;
      int32_t v2539_a = v1383_lead + 1064;
      double r0[36]{};
      size_t v2610_row = v1354_batchIdGroup0First + v1351_batchIdLane0;
      const bool peel_batchIdActive0 = v2610_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v2610_row]));
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[(peel_batchIdActive0 ? v2610_row : v1354_batchIdGroup0First)][0 + m1_extraOffset];
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v2618_i0 = 0; v2618_i0 < 3; ++v2618_i0) {
        int32_t v2620_lead = v1371_lead + (v2618_i0 * 16);
        #pragma unroll
        for (int32_t v2621_i1 = 0; v2621_i1 < 9; ++v2621_i1) {
          double v2624_data = __builtin_nontemporal_load(&peel_glb_m1[(v2620_lead + (v2621_i1 * 56))]);
          r0[(v2618_i0 + (v2621_i1 * 4))] = v2624_data;
        }
      }
      #pragma unroll
      for (int32_t v2627_i1 = 0; v2627_i1 < 9; ++v2627_i1) {
        double v2630_data = v1386_g ? (__builtin_nontemporal_load(&peel_glb_m1[(v1383_lead + (v2627_i1 * 56))])) : (0.0);
        r0[(3 + (v2627_i1 * 4))] = v2630_data;
      }
      for (size_t v1355_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v1355_batchIdGroup0 < numElements0; v1355_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v2607_groupNextAhead = v1355_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v2609_groupNext = (v2607_groupNextAhead < numElements0) ? v2607_groupNextAhead : v1355_batchIdGroup0;
        size_t v2614_row = v2609_groupNext + v1351_batchIdLane0;
        const bool wrap_batchIdActive0 = v2614_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v2614_row]));
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[(wrap_batchIdActive0 ? v2614_row : v2609_groupNext)][0 + m1_extraOffset];
        size_t v1356_row = v1355_batchIdGroup0 + v1351_batchIdLane0;
        const bool batchIdActive0 = v1356_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1356_row]));
        size_t v1358_batchId0 = batchIdActive0 ? v1356_row : v1355_batchIdGroup0;
        size_t v1359_ahead1 = v1358_batchId0 + (gridDim.x * blockDim.y);
        size_t v1361_batchId1 = (v1359_ahead1 < numElements0) ? v1359_ahead1 : v1358_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v1358_batchId0][0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m3[v1358_batchId0][0 + m3_extraOffset];
        double r3[9]{};
        double r1[18]{};
        // r1 = +(glb_m0 * r0) + None
        // [(0, 21), (0, 9)] [(0, 56)]
        tensorforge::VectorT<double, 4> v1397_acc{};
        tensorforge::VectorT<double, 4> v1398_acc{};
        tensorforge::VectorT<double, 4> v1399_acc{};
        tensorforge::VectorT<double, 4> v1400_acc{};
        double v1402_data = r0[0];
        double v1403_ga = tensorforge::gather<64>(v1402_data, v1396_src);
        double v1404_data = r0[4];
        double v1405_ga = tensorforge::gather<64>(v1404_data, v1396_src);
        double v1406_data = r0[8];
        double v1407_ga = tensorforge::gather<64>(v1406_data, v1396_src);
        double v1408_data = r0[12];
        double v1409_ga = tensorforge::gather<64>(v1408_data, v1396_src);
        double v1415_af = v1414_g ? v1403_ga : (v1412_g ? v1405_ga : (v1410_g ? v1407_ga : v1409_ga));
        double v1416_data = r0[16];
        double v1417_ga = tensorforge::gather<64>(v1416_data, v1396_src);
        double v1418_data = r0[20];
        double v1419_ga = tensorforge::gather<64>(v1418_data, v1396_src);
        double v1420_data = r0[24];
        double v1421_ga = tensorforge::gather<64>(v1420_data, v1396_src);
        double v1422_data = r0[28];
        double v1423_ga = tensorforge::gather<64>(v1422_data, v1396_src);
        double v1426_af = v1414_g ? v1417_ga : (v1412_g ? v1419_ga : (v1410_g ? v1421_ga : v1423_ga));
        double v1432_data = glb_m0[v1430_a];
        tensorforge::VectorT<double, 4> v1433_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1415_af, v1432_data, v1397_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1434_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1426_af, v1432_data, v1398_acc, 0, 0, 0);
        double v1439_data = glb_m0[v1437_a];
        tensorforge::VectorT<double, 4> v1440_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1415_af, v1439_data, v1399_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1441_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1426_af, v1439_data, v1400_acc, 0, 0, 0);
        double v1444_ga = tensorforge::gather<64>(v1402_data, v1442_src);
        double v1446_ga = tensorforge::gather<64>(v1404_data, v1442_src);
        double v1448_ga = tensorforge::gather<64>(v1406_data, v1442_src);
        double v1450_ga = tensorforge::gather<64>(v1408_data, v1442_src);
        double v1453_af = v1414_g ? v1444_ga : (v1412_g ? v1446_ga : (v1410_g ? v1448_ga : v1450_ga));
        double v1455_ga = tensorforge::gather<64>(v1416_data, v1442_src);
        double v1457_ga = tensorforge::gather<64>(v1418_data, v1442_src);
        double v1459_ga = tensorforge::gather<64>(v1420_data, v1442_src);
        double v1461_ga = tensorforge::gather<64>(v1422_data, v1442_src);
        double v1464_af = v1414_g ? v1455_ga : (v1412_g ? v1457_ga : (v1410_g ? v1459_ga : v1461_ga));
        double v1466_data = glb_m0[v1465_a];
        tensorforge::VectorT<double, 4> v1467_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1453_af, v1466_data, v1433_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1468_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1464_af, v1466_data, v1434_acc, 0, 0, 0);
        double v1470_data = glb_m0[v1469_a];
        tensorforge::VectorT<double, 4> v1471_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1453_af, v1470_data, v1440_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1472_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1464_af, v1470_data, v1441_acc, 0, 0, 0);
        double v1475_ga = tensorforge::gather<64>(v1402_data, v1473_src);
        double v1477_ga = tensorforge::gather<64>(v1404_data, v1473_src);
        double v1479_ga = tensorforge::gather<64>(v1406_data, v1473_src);
        double v1481_ga = tensorforge::gather<64>(v1408_data, v1473_src);
        double v1484_af = v1414_g ? v1475_ga : (v1412_g ? v1477_ga : (v1410_g ? v1479_ga : v1481_ga));
        double v1486_ga = tensorforge::gather<64>(v1416_data, v1473_src);
        double v1488_ga = tensorforge::gather<64>(v1418_data, v1473_src);
        double v1490_ga = tensorforge::gather<64>(v1420_data, v1473_src);
        double v1492_ga = tensorforge::gather<64>(v1422_data, v1473_src);
        double v1495_af = v1414_g ? v1486_ga : (v1412_g ? v1488_ga : (v1410_g ? v1490_ga : v1492_ga));
        double v1497_data = glb_m0[v1496_a];
        tensorforge::VectorT<double, 4> v1498_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1484_af, v1497_data, v1467_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1499_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1495_af, v1497_data, v1468_acc, 0, 0, 0);
        double v1501_data = glb_m0[v1500_a];
        tensorforge::VectorT<double, 4> v1502_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1484_af, v1501_data, v1471_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1503_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1495_af, v1501_data, v1472_acc, 0, 0, 0);
        double v1506_ga = tensorforge::gather<64>(v1402_data, v1504_src);
        double v1508_ga = tensorforge::gather<64>(v1404_data, v1504_src);
        double v1510_ga = tensorforge::gather<64>(v1406_data, v1504_src);
        double v1512_ga = tensorforge::gather<64>(v1408_data, v1504_src);
        double v1515_af = v1414_g ? v1506_ga : (v1412_g ? v1508_ga : (v1410_g ? v1510_ga : v1512_ga));
        double v1517_ga = tensorforge::gather<64>(v1416_data, v1504_src);
        double v1519_ga = tensorforge::gather<64>(v1418_data, v1504_src);
        double v1521_ga = tensorforge::gather<64>(v1420_data, v1504_src);
        double v1523_ga = tensorforge::gather<64>(v1422_data, v1504_src);
        double v1526_af = v1414_g ? v1517_ga : (v1412_g ? v1519_ga : (v1410_g ? v1521_ga : v1523_ga));
        double v1528_data = glb_m0[v1527_a];
        tensorforge::VectorT<double, 4> v1529_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1515_af, v1528_data, v1498_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1530_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1526_af, v1528_data, v1499_acc, 0, 0, 0);
        double v1532_data = glb_m0[v1531_a];
        tensorforge::VectorT<double, 4> v1533_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1515_af, v1532_data, v1502_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1534_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1526_af, v1532_data, v1503_acc, 0, 0, 0);
        double v1535_data = r0[1];
        double v1536_ga = tensorforge::gather<64>(v1535_data, v1396_src);
        double v1537_data = r0[5];
        double v1538_ga = tensorforge::gather<64>(v1537_data, v1396_src);
        double v1539_data = r0[9];
        double v1540_ga = tensorforge::gather<64>(v1539_data, v1396_src);
        double v1541_data = r0[13];
        double v1542_ga = tensorforge::gather<64>(v1541_data, v1396_src);
        double v1545_af = v1414_g ? v1536_ga : (v1412_g ? v1538_ga : (v1410_g ? v1540_ga : v1542_ga));
        double v1546_data = r0[17];
        double v1547_ga = tensorforge::gather<64>(v1546_data, v1396_src);
        double v1548_data = r0[21];
        double v1549_ga = tensorforge::gather<64>(v1548_data, v1396_src);
        double v1550_data = r0[25];
        double v1551_ga = tensorforge::gather<64>(v1550_data, v1396_src);
        double v1552_data = r0[29];
        double v1553_ga = tensorforge::gather<64>(v1552_data, v1396_src);
        double v1556_af = v1414_g ? v1547_ga : (v1412_g ? v1549_ga : (v1410_g ? v1551_ga : v1553_ga));
        double v1558_data = glb_m0[v1557_a];
        tensorforge::VectorT<double, 4> v1559_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1545_af, v1558_data, v1529_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1560_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1556_af, v1558_data, v1530_acc, 0, 0, 0);
        double v1562_data = glb_m0[v1561_a];
        tensorforge::VectorT<double, 4> v1563_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1545_af, v1562_data, v1533_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1564_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1556_af, v1562_data, v1534_acc, 0, 0, 0);
        double v1566_ga = tensorforge::gather<64>(v1535_data, v1442_src);
        double v1568_ga = tensorforge::gather<64>(v1537_data, v1442_src);
        double v1570_ga = tensorforge::gather<64>(v1539_data, v1442_src);
        double v1572_ga = tensorforge::gather<64>(v1541_data, v1442_src);
        double v1575_af = v1414_g ? v1566_ga : (v1412_g ? v1568_ga : (v1410_g ? v1570_ga : v1572_ga));
        double v1577_ga = tensorforge::gather<64>(v1546_data, v1442_src);
        double v1579_ga = tensorforge::gather<64>(v1548_data, v1442_src);
        double v1581_ga = tensorforge::gather<64>(v1550_data, v1442_src);
        double v1583_ga = tensorforge::gather<64>(v1552_data, v1442_src);
        double v1586_af = v1414_g ? v1577_ga : (v1412_g ? v1579_ga : (v1410_g ? v1581_ga : v1583_ga));
        double v1588_data = glb_m0[v1587_a];
        tensorforge::VectorT<double, 4> v1589_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1575_af, v1588_data, v1559_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1590_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1586_af, v1588_data, v1560_acc, 0, 0, 0);
        double v1592_data = glb_m0[v1591_a];
        tensorforge::VectorT<double, 4> v1593_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1575_af, v1592_data, v1563_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1594_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1586_af, v1592_data, v1564_acc, 0, 0, 0);
        double v1596_ga = tensorforge::gather<64>(v1535_data, v1473_src);
        double v1598_ga = tensorforge::gather<64>(v1537_data, v1473_src);
        double v1600_ga = tensorforge::gather<64>(v1539_data, v1473_src);
        double v1602_ga = tensorforge::gather<64>(v1541_data, v1473_src);
        double v1605_af = v1414_g ? v1596_ga : (v1412_g ? v1598_ga : (v1410_g ? v1600_ga : v1602_ga));
        double v1607_ga = tensorforge::gather<64>(v1546_data, v1473_src);
        double v1609_ga = tensorforge::gather<64>(v1548_data, v1473_src);
        double v1611_ga = tensorforge::gather<64>(v1550_data, v1473_src);
        double v1613_ga = tensorforge::gather<64>(v1552_data, v1473_src);
        double v1616_af = v1414_g ? v1607_ga : (v1412_g ? v1609_ga : (v1410_g ? v1611_ga : v1613_ga));
        double v1618_data = glb_m0[v1617_a];
        tensorforge::VectorT<double, 4> v1619_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1605_af, v1618_data, v1589_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1620_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1616_af, v1618_data, v1590_acc, 0, 0, 0);
        double v1622_data = glb_m0[v1621_a];
        tensorforge::VectorT<double, 4> v1623_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1605_af, v1622_data, v1593_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1624_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1616_af, v1622_data, v1594_acc, 0, 0, 0);
        double v1626_ga = tensorforge::gather<64>(v1535_data, v1504_src);
        double v1628_ga = tensorforge::gather<64>(v1537_data, v1504_src);
        double v1630_ga = tensorforge::gather<64>(v1539_data, v1504_src);
        double v1632_ga = tensorforge::gather<64>(v1541_data, v1504_src);
        double v1635_af = v1414_g ? v1626_ga : (v1412_g ? v1628_ga : (v1410_g ? v1630_ga : v1632_ga));
        double v1637_ga = tensorforge::gather<64>(v1546_data, v1504_src);
        double v1639_ga = tensorforge::gather<64>(v1548_data, v1504_src);
        double v1641_ga = tensorforge::gather<64>(v1550_data, v1504_src);
        double v1643_ga = tensorforge::gather<64>(v1552_data, v1504_src);
        double v1646_af = v1414_g ? v1637_ga : (v1412_g ? v1639_ga : (v1410_g ? v1641_ga : v1643_ga));
        double v1648_data = glb_m0[v1647_a];
        // r3 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v2123_i1 = 0; v2123_i1 < 9; ++v2123_i1) {
          double v2129_data = v2128_g ? (__builtin_nontemporal_load(&glb_m2[(v1371_lead + (v2123_i1 * 9))])) : (0.0);
          r3[v2123_i1] = v2129_data;
        }
        tensorforge::VectorT<double, 4> v1649_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1635_af, v1648_data, v1619_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1650_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1646_af, v1648_data, v1620_acc, 0, 0, 0);
        double v1652_data = glb_m0[v1651_a];
        tensorforge::VectorT<double, 4> v1653_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1635_af, v1652_data, v1623_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1654_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1646_af, v1652_data, v1624_acc, 0, 0, 0);
        double v1655_data = r0[2];
        double v1656_ga = tensorforge::gather<64>(v1655_data, v1396_src);
        double v1657_data = r0[6];
        double v1658_ga = tensorforge::gather<64>(v1657_data, v1396_src);
        double v1659_data = r0[10];
        double v1660_ga = tensorforge::gather<64>(v1659_data, v1396_src);
        double v1661_data = r0[14];
        double v1662_ga = tensorforge::gather<64>(v1661_data, v1396_src);
        double v1665_af = v1414_g ? v1656_ga : (v1412_g ? v1658_ga : (v1410_g ? v1660_ga : v1662_ga));
        double v1666_data = r0[18];
        double v1667_ga = tensorforge::gather<64>(v1666_data, v1396_src);
        double v1668_data = r0[22];
        double v1669_ga = tensorforge::gather<64>(v1668_data, v1396_src);
        double v1670_data = r0[26];
        double v1671_ga = tensorforge::gather<64>(v1670_data, v1396_src);
        double v1672_data = r0[30];
        double v1673_ga = tensorforge::gather<64>(v1672_data, v1396_src);
        double v1676_af = v1414_g ? v1667_ga : (v1412_g ? v1669_ga : (v1410_g ? v1671_ga : v1673_ga));
        double v1678_data = glb_m0[v1677_a];
        tensorforge::VectorT<double, 4> v1679_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1665_af, v1678_data, v1649_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1680_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1676_af, v1678_data, v1650_acc, 0, 0, 0);
        double v1682_data = glb_m0[v1681_a];
        tensorforge::VectorT<double, 4> v1683_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1665_af, v1682_data, v1653_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1684_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1676_af, v1682_data, v1654_acc, 0, 0, 0);
        double v1686_ga = tensorforge::gather<64>(v1655_data, v1442_src);
        double v1688_ga = tensorforge::gather<64>(v1657_data, v1442_src);
        double v1690_ga = tensorforge::gather<64>(v1659_data, v1442_src);
        double v1692_ga = tensorforge::gather<64>(v1661_data, v1442_src);
        double v1695_af = v1414_g ? v1686_ga : (v1412_g ? v1688_ga : (v1410_g ? v1690_ga : v1692_ga));
        double v1697_ga = tensorforge::gather<64>(v1666_data, v1442_src);
        double v1699_ga = tensorforge::gather<64>(v1668_data, v1442_src);
        double v1701_ga = tensorforge::gather<64>(v1670_data, v1442_src);
        double v1703_ga = tensorforge::gather<64>(v1672_data, v1442_src);
        double v1706_af = v1414_g ? v1697_ga : (v1412_g ? v1699_ga : (v1410_g ? v1701_ga : v1703_ga));
        double v1708_data = glb_m0[v1707_a];
        tensorforge::VectorT<double, 4> v1709_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1695_af, v1708_data, v1679_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1710_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1706_af, v1708_data, v1680_acc, 0, 0, 0);
        double v1712_data = glb_m0[v1711_a];
        tensorforge::VectorT<double, 4> v1713_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1695_af, v1712_data, v1683_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1714_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1706_af, v1712_data, v1684_acc, 0, 0, 0);
        double v1716_ga = tensorforge::gather<64>(v1655_data, v1473_src);
        double v1718_ga = tensorforge::gather<64>(v1657_data, v1473_src);
        double v1720_ga = tensorforge::gather<64>(v1659_data, v1473_src);
        double v1722_ga = tensorforge::gather<64>(v1661_data, v1473_src);
        double v1725_af = v1414_g ? v1716_ga : (v1412_g ? v1718_ga : (v1410_g ? v1720_ga : v1722_ga));
        double v1727_ga = tensorforge::gather<64>(v1666_data, v1473_src);
        double v1729_ga = tensorforge::gather<64>(v1668_data, v1473_src);
        double v1731_ga = tensorforge::gather<64>(v1670_data, v1473_src);
        double v1733_ga = tensorforge::gather<64>(v1672_data, v1473_src);
        double v1736_af = v1414_g ? v1727_ga : (v1412_g ? v1729_ga : (v1410_g ? v1731_ga : v1733_ga));
        double v1738_data = glb_m0[v1737_a];
        tensorforge::VectorT<double, 4> v1739_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1725_af, v1738_data, v1709_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1740_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1736_af, v1738_data, v1710_acc, 0, 0, 0);
        double v1742_data = glb_m0[v1741_a];
        tensorforge::VectorT<double, 4> v1743_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1725_af, v1742_data, v1713_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1744_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1736_af, v1742_data, v1714_acc, 0, 0, 0);
        double v1746_ga = tensorforge::gather<64>(v1655_data, v1504_src);
        double v1748_ga = tensorforge::gather<64>(v1657_data, v1504_src);
        double v1750_ga = tensorforge::gather<64>(v1659_data, v1504_src);
        double v1752_ga = tensorforge::gather<64>(v1661_data, v1504_src);
        double v1755_af = v1414_g ? v1746_ga : (v1412_g ? v1748_ga : (v1410_g ? v1750_ga : v1752_ga));
        double v1757_ga = tensorforge::gather<64>(v1666_data, v1504_src);
        double v1759_ga = tensorforge::gather<64>(v1668_data, v1504_src);
        double v1761_ga = tensorforge::gather<64>(v1670_data, v1504_src);
        double v1763_ga = tensorforge::gather<64>(v1672_data, v1504_src);
        double v1766_af = v1414_g ? v1757_ga : (v1412_g ? v1759_ga : (v1410_g ? v1761_ga : v1763_ga));
        double v1768_data = glb_m0[v1767_a];
        tensorforge::VectorT<double, 4> v1769_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1755_af, v1768_data, v1739_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1770_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1766_af, v1768_data, v1740_acc, 0, 0, 0);
        double v1772_data = glb_m0[v1771_a];
        tensorforge::VectorT<double, 4> v1773_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1755_af, v1772_data, v1743_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1774_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1766_af, v1772_data, v1744_acc, 0, 0, 0);
        double v1775_data = r0[3];
        double v1776_ga = tensorforge::gather<64>(v1775_data, v1396_src);
        double v1777_data = r0[7];
        double v1778_ga = tensorforge::gather<64>(v1777_data, v1396_src);
        double v1779_data = r0[11];
        double v1780_ga = tensorforge::gather<64>(v1779_data, v1396_src);
        double v1781_data = r0[15];
        double v1782_ga = tensorforge::gather<64>(v1781_data, v1396_src);
        double v1785_af = v1414_g ? v1776_ga : (v1412_g ? v1778_ga : (v1410_g ? v1780_ga : v1782_ga));
        double v1786_data = r0[19];
        double v1787_ga = tensorforge::gather<64>(v1786_data, v1396_src);
        double v1788_data = r0[23];
        double v1789_ga = tensorforge::gather<64>(v1788_data, v1396_src);
        double v1790_data = r0[27];
        double v1791_ga = tensorforge::gather<64>(v1790_data, v1396_src);
        double v1792_data = r0[31];
        double v1793_ga = tensorforge::gather<64>(v1792_data, v1396_src);
        double v1796_af = v1414_g ? v1787_ga : (v1412_g ? v1789_ga : (v1410_g ? v1791_ga : v1793_ga));
        double v1798_data = glb_m0[v1797_a];
        tensorforge::VectorT<double, 4> v1799_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1785_af, v1798_data, v1769_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1800_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1796_af, v1798_data, v1770_acc, 0, 0, 0);
        double v1802_data = glb_m0[v1801_a];
        tensorforge::VectorT<double, 4> v1803_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1785_af, v1802_data, v1773_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1804_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1796_af, v1802_data, v1774_acc, 0, 0, 0);
        double v1806_ga = tensorforge::gather<64>(v1775_data, v1442_src);
        double v1808_ga = tensorforge::gather<64>(v1777_data, v1442_src);
        double v1810_ga = tensorforge::gather<64>(v1779_data, v1442_src);
        double v1812_ga = tensorforge::gather<64>(v1781_data, v1442_src);
        double v1815_af = v1414_g ? v1806_ga : (v1412_g ? v1808_ga : (v1410_g ? v1810_ga : v1812_ga));
        double v1817_ga = tensorforge::gather<64>(v1786_data, v1442_src);
        double v1819_ga = tensorforge::gather<64>(v1788_data, v1442_src);
        double v1821_ga = tensorforge::gather<64>(v1790_data, v1442_src);
        double v1823_ga = tensorforge::gather<64>(v1792_data, v1442_src);
        double v1826_af = v1414_g ? v1817_ga : (v1412_g ? v1819_ga : (v1410_g ? v1821_ga : v1823_ga));
        double v1828_data = glb_m0[v1827_a];
        tensorforge::VectorT<double, 4> v1829_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1815_af, v1828_data, v1799_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1830_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1826_af, v1828_data, v1800_acc, 0, 0, 0);
        double v1832_data = glb_m0[v1831_a];
        tensorforge::VectorT<double, 4> v1833_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1815_af, v1832_data, v1803_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v1834_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v1826_af, v1832_data, v1804_acc, 0, 0, 0);
        double v1835_el = v1829_acc[0];
        r1[0] = v1835_el;
        double v1836_el = v1829_acc[1];
        r1[2] = v1836_el;
        double v1837_el = v1829_acc[2];
        r1[4] = v1837_el;
        double v1838_el = v1829_acc[3];
        r1[6] = v1838_el;
        double v1839_el = v1830_acc[0];
        r1[8] = v1839_el;
        double v1840_el = v1830_acc[1];
        r1[10] = v1840_el;
        double v1841_el = v1830_acc[2];
        r1[12] = v1841_el;
        double v1842_el = v1830_acc[3];
        r1[14] = v1842_el;
        r1[1] = (v1833_acc[0]);
        r1[3] = (v1833_acc[1]);
        r1[5] = (v1833_acc[2]);
        r1[7] = (v1833_acc[3]);
        r1[9] = (v1834_acc[0]);
        r1[11] = (v1834_acc[1]);
        r1[13] = (v1834_acc[2]);
        r1[15] = (v1834_acc[3]);
        double v1854_data = glb_m0[v1371_lead];
        double v1858_data = glb_m0[v1436_lead];
        double v1860_data = glb_m0[v1859_a];
        double v1862_data = glb_m0[v1861_a];
        double v1864_data = glb_m0[v1863_a];
        double v1866_data = glb_m0[v1865_a];
        double v1868_data = glb_m0[v1867_a];
        double v1870_data = glb_m0[v1869_a];
        double v1872_data = glb_m0[v1871_a];
        double v1874_data = glb_m0[v1873_a];
        double v1876_data = glb_m0[v1875_a];
        double v1878_data = glb_m0[v1877_a];
        double v1880_data = glb_m0[v1879_a];
        double v1882_data = glb_m0[v1881_a];
        double v1884_data = glb_m0[v1883_a];
        double v1886_data = glb_m0[v1885_a];
        double v1888_data = glb_m0[v1887_a];
        double v1890_data = glb_m0[v1889_a];
        double v1892_data = glb_m0[v1891_a];
        double v1894_data = glb_m0[v1893_a];
        double v1896_data = glb_m0[v1895_a];
        double v1898_data = glb_m0[v1897_a];
        double v1900_data = glb_m0[v1899_a];
        double v1902_data = glb_m0[v1901_a];
        double v1904_data = glb_m0[v1903_a];
        double v1906_data = glb_m0[v1905_a];
        double v1908_data = glb_m0[v1907_a];
        double v1910_data = glb_m0[v1909_a];
        double v1912_data = glb_m0[v1911_a];
        double v1914_data = glb_m0[v1913_a];
        double v1916_data = glb_m0[v1915_a];
        double v1918_data = glb_m0[v1917_a];
        double v1920_data = glb_m0[v1919_a];
        double v1922_data = glb_m0[v1921_a];
        double v1924_data = glb_m0[v1923_a];
        double v1926_data = glb_m0[v1925_a];
        double v1928_data = glb_m0[v1927_a];
        double v1930_data = glb_m0[v1929_a];
        double v1932_data = glb_m0[v1931_a];
        double v1934_data = glb_m0[v1933_a];
        double v1936_data = glb_m0[v1935_a];
        double v1938_data = glb_m0[v1937_a];
        double v1940_data = glb_m0[v1939_a];
        double v1942_data = glb_m0[v1941_a];
        double v1944_data = glb_m0[v1943_a];
        double v1946_data = glb_m0[v1945_a];
        double v1948_data = glb_m0[v1947_a];
        double v1950_data = glb_m0[v1949_a];
        double v1952_data = glb_m0[v1951_a];
        double v1954_data = glb_m0[v1953_a];
        double v1956_data = glb_m0[v1955_a];
        double v1958_data = glb_m0[v1957_a];
        double v1960_data = glb_m0[v1959_a];
        double v1962_data = glb_m0[v1961_a];
        double v1964_data = glb_m0[v1963_a];
        double v1966_data = glb_m0[v1965_a];
        double v1968_data = glb_m0[v1967_a];
        double v1970_data = glb_m0[v1969_a];
        double v1972_data = glb_m0[v1971_a];
        double v1974_data = glb_m0[v1973_a];
        double v1976_data = glb_m0[v1975_a];
        double v1978_data = glb_m0[v1977_a];
        double v1980_data = glb_m0[v1979_a];
        double v1982_data = glb_m0[v1981_a];
        double v1984_data = glb_m0[v1983_a];
        double v1986_data = glb_m0[v1985_a];
        double v1988_data = glb_m0[v1987_a];
        double v1990_data = glb_m0[v1989_a];
        double v1992_data = glb_m0[v1991_a];
        double v1994_data = glb_m0[v1993_a];
        double v1996_data = glb_m0[v1995_a];
        double v1998_data = glb_m0[v1997_a];
        double v2000_data = glb_m0[v1999_a];
        double v2002_data = glb_m0[v2001_a];
        double v2004_data = glb_m0[v2003_a];
        double v2006_data = glb_m0[v2005_a];
        double v2008_data = glb_m0[v2007_a];
        double v2010_data = glb_m0[v2009_a];
        double v2012_data = glb_m0[v2011_a];
        double v2014_data = glb_m0[v2013_a];
        double v2016_data = glb_m0[v2015_a];
        double v2018_data = glb_m0[v2017_a];
        double v2020_data = glb_m0[v2019_a];
        double v2022_data = glb_m0[v2021_a];
        double v2024_data = glb_m0[v2023_a];
        double v2026_data = glb_m0[v2025_a];
        double v2028_data = glb_m0[v2027_a];
        double v2030_data = glb_m0[v2029_a];
        double v2032_data = glb_m0[v2031_a];
        double v2034_data = glb_m0[v2033_a];
        double v2036_data = glb_m0[v2035_a];
        double v2038_data = glb_m0[v2037_a];
        double v2040_data = glb_m0[v2039_a];
        double v2042_data = glb_m0[v2041_a];
        double v2044_data = glb_m0[v2043_a];
        double v2046_data = glb_m0[v2045_a];
        double v2048_data = glb_m0[v2047_a];
        double v2050_data = glb_m0[v2049_a];
        double v2052_data = glb_m0[v2051_a];
        double v2054_data = glb_m0[v2053_a];
        double v2056_data = glb_m0[v2055_a];
        double v2058_data = glb_m0[v2057_a];
        double v2060_data = glb_m0[v2059_a];
        double v2062_data = glb_m0[v2061_a];
        double v2064_data = glb_m0[v2063_a];
        double v2066_data = glb_m0[v2065_a];
        double v2068_data = glb_m0[v2067_a];
        double v2070_data = glb_m0[v2069_a];
        double v2072_data = glb_m0[v2071_a];
        double v2074_data = glb_m0[v2073_a];
        double v2076_data = glb_m0[v2075_a];
        double v2078_data = glb_m0[v2077_a];
        double v2079_acc{};
        double v2080_acc{};
        double v2081_data = r0[32];
        double v2082_data = r0[33];
        double v2083_data = r0[34];
        double v2084_data = r0[35];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %19 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %21 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %22 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %23 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %24 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %25 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %26 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %27 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %28 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %29 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %30 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2079_acc), "+&v"(v2080_acc)
        : "v"(v2081_data), "v"(v1854_data), "v"(v1858_data), "v"(v1860_data), "v"(v1862_data), "v"(v1864_data), "v"(v1866_data), "v"(v1868_data), "v"(v1870_data), "v"(v1872_data), "v"(v1874_data), "v"(v1876_data), "v"(v1878_data), "v"(v1880_data), "v"(v1882_data), "v"(v1884_data), "v"(v1886_data), "v"(v1888_data), "v"(v1890_data), "v"(v1892_data), "v"(v1894_data), "v"(v1896_data), "v"(v1898_data), "v"(v1900_data), "v"(v1902_data), "v"(v1904_data), "v"(v1906_data), "v"(v1908_data), "v"(v1910_data), "v"(v1912_data), "v"(v1914_data), "v"(v1916_data), "v"(v1918_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %19 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %21 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %22 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %23 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %24 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %25 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %26 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %27 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %28 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %29 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %30 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2079_acc), "+&v"(v2080_acc)
        : "v"(v2082_data), "v"(v1920_data), "v"(v1922_data), "v"(v1924_data), "v"(v1926_data), "v"(v1928_data), "v"(v1930_data), "v"(v1932_data), "v"(v1934_data), "v"(v1936_data), "v"(v1938_data), "v"(v1940_data), "v"(v1942_data), "v"(v1944_data), "v"(v1946_data), "v"(v1948_data), "v"(v1950_data), "v"(v1952_data), "v"(v1954_data), "v"(v1956_data), "v"(v1958_data), "v"(v1960_data), "v"(v1962_data), "v"(v1964_data), "v"(v1966_data), "v"(v1968_data), "v"(v1970_data), "v"(v1972_data), "v"(v1974_data), "v"(v1976_data), "v"(v1978_data), "v"(v1980_data), "v"(v1982_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %19 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %21 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %22 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %23 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %24 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %25 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %26 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %27 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %28 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %29 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %30 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2079_acc), "+&v"(v2080_acc)
        : "v"(v2083_data), "v"(v1984_data), "v"(v1986_data), "v"(v1988_data), "v"(v1990_data), "v"(v1992_data), "v"(v1994_data), "v"(v1996_data), "v"(v1998_data), "v"(v2000_data), "v"(v2002_data), "v"(v2004_data), "v"(v2006_data), "v"(v2008_data), "v"(v2010_data), "v"(v2012_data), "v"(v2014_data), "v"(v2016_data), "v"(v2018_data), "v"(v2020_data), "v"(v2022_data), "v"(v2024_data), "v"(v2026_data), "v"(v2028_data), "v"(v2030_data), "v"(v2032_data), "v"(v2034_data), "v"(v2036_data), "v"(v2038_data), "v"(v2040_data), "v"(v2042_data), "v"(v2044_data), "v"(v2046_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2079_acc), "+&v"(v2080_acc)
        : "v"(v2084_data), "v"(v2048_data), "v"(v2050_data), "v"(v2052_data), "v"(v2054_data), "v"(v2056_data), "v"(v2058_data), "v"(v2060_data), "v"(v2062_data), "v"(v2064_data), "v"(v2066_data), "v"(v2068_data), "v"(v2070_data), "v"(v2072_data), "v"(v2074_data), "v"(v2076_data), "v"(v2078_data)
        );
        r1[16] = v2079_acc;
        r1[17] = v2080_acc;
        // r1 = 0 past (0, 21)
        // s0 = store{r>s}(localShrMem0, r1);
        double r2[18]{};
        // r2 = load{g>r}(s0);
        double r4[18]{};
        // r4 = +(r2 * r3) + None
        // [(0, 21), (0, 9)] [(0, 9)]
        double v2133_data = r1[1];
        double v2135_data = r1[3];
        double v2137_data = r1[5];
        double v2139_data = r1[7];
        double v2141_data = r1[9];
        double v2143_data = r1[11];
        double v2145_data = r1[13];
        double v2147_data = r1[15];
        double v2149_data = r1[17];
        double v2150_acc{};
        double v2151_acc{};
        double v2152_acc{};
        double v2153_acc{};
        double v2154_acc{};
        double v2155_acc{};
        double v2156_acc{};
        double v2157_acc{};
        double v2158_acc{};
        double v2159_acc{};
        double v2160_acc{};
        double v2161_acc{};
        double v2162_acc{};
        double v2163_acc{};
        double v2164_acc{};
        double v2165_acc{};
        double v2166_acc{};
        double v2167_acc{};
        double v2168_data = r3[0];
        double v2169_data = r3[1];
        double v2170_data = r3[2];
        double v2171_data = r3[3];
        double v2172_data = r3[4];
        double v2173_data = r3[5];
        double v2174_data = r3[6];
        double v2175_data = r3[7];
        double v2176_data = r3[8];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %12 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %13 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %14 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %15 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %16 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %18 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %20 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %12 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %13 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %14 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %15 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %16 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %23, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %23, %18 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2150_acc), "+&v"(v2151_acc), "+&v"(v2152_acc), "+&v"(v2153_acc)
        : "v"(v2168_data), "v"(v1835_el), "v"(v2133_data), "v"(v1836_el), "v"(v2135_data), "v"(v1837_el), "v"(v2137_data), "v"(v1838_el), "v"(v2139_data), "v"(v1839_el), "v"(v2141_data), "v"(v1840_el), "v"(v2143_data), "v"(v1841_el), "v"(v2145_data), "v"(v1842_el), "v"(v2147_data), "v"(v2079_acc), "v"(v2149_data), "v"(v2169_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %14 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %15 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %17 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %20 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %22 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %23 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %25 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %7 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %14 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %15 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %17 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %20 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2152_acc), "+&v"(v2153_acc), "+&v"(v2154_acc), "+&v"(v2155_acc), "+&v"(v2156_acc), "+&v"(v2157_acc)
        : "v"(v2169_data), "v"(v1842_el), "v"(v2147_data), "v"(v2079_acc), "v"(v2149_data), "v"(v2170_data), "v"(v1835_el), "v"(v2133_data), "v"(v1836_el), "v"(v2135_data), "v"(v1837_el), "v"(v2137_data), "v"(v1838_el), "v"(v2139_data), "v"(v1839_el), "v"(v2141_data), "v"(v1840_el), "v"(v2143_data), "v"(v1841_el), "v"(v2145_data), "v"(v2171_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %9 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %11 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %12 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %16 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %17 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %18 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %19 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %20 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %21 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %22 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %23 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %25 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %9 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %11 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %12 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %15, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %15, %14 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %16 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %17 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %18 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %19 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %20 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %21 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2156_acc), "+&v"(v2157_acc), "+&v"(v2158_acc), "+&v"(v2159_acc), "+&v"(v2160_acc), "+&v"(v2161_acc)
        : "v"(v2171_data), "v"(v1840_el), "v"(v2143_data), "v"(v1841_el), "v"(v2145_data), "v"(v1842_el), "v"(v2147_data), "v"(v2079_acc), "v"(v2149_data), "v"(v2172_data), "v"(v1835_el), "v"(v2133_data), "v"(v1836_el), "v"(v2135_data), "v"(v1837_el), "v"(v2137_data), "v"(v1838_el), "v"(v2139_data), "v"(v1839_el), "v"(v2141_data), "v"(v2173_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %11 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %15 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %16 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %17 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %18 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %22 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %24 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %25 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %11 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %13 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %14 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %15 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %16 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %19, %17 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %18 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %26, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %26, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2160_acc), "+&v"(v2161_acc), "+&v"(v2162_acc), "+&v"(v2163_acc), "+&v"(v2164_acc), "+&v"(v2165_acc)
        : "v"(v2173_data), "v"(v1838_el), "v"(v2139_data), "v"(v1839_el), "v"(v2141_data), "v"(v1840_el), "v"(v2143_data), "v"(v1841_el), "v"(v2145_data), "v"(v1842_el), "v"(v2147_data), "v"(v2079_acc), "v"(v2149_data), "v"(v2174_data), "v"(v1835_el), "v"(v2133_data), "v"(v1836_el), "v"(v2135_data), "v"(v1837_el), "v"(v2137_data), "v"(v2175_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %4, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %19 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %22 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %23 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %21, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %21, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2164_acc), "+&v"(v2165_acc), "+&v"(v2166_acc), "+&v"(v2167_acc)
        : "v"(v2175_data), "v"(v1836_el), "v"(v2135_data), "v"(v1837_el), "v"(v2137_data), "v"(v1838_el), "v"(v2139_data), "v"(v1839_el), "v"(v2141_data), "v"(v1840_el), "v"(v2143_data), "v"(v1841_el), "v"(v2145_data), "v"(v1842_el), "v"(v2147_data), "v"(v2079_acc), "v"(v2149_data), "v"(v2176_data), "v"(v1835_el), "v"(v2133_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2166_acc), "+&v"(v2167_acc)
        : "v"(v2176_data), "v"(v2079_acc), "v"(v2149_data)
        );
        r4[0] = v2150_acc;
        r4[1] = v2151_acc;
        r4[2] = v2152_acc;
        r4[3] = v2153_acc;
        r4[4] = v2154_acc;
        r4[5] = v2155_acc;
        r4[6] = v2156_acc;
        r4[7] = v2157_acc;
        r4[8] = v2158_acc;
        r4[9] = v2159_acc;
        r4[10] = v2160_acc;
        r4[11] = v2161_acc;
        r4[12] = v2162_acc;
        r4[13] = v2163_acc;
        r4[14] = v2164_acc;
        r4[15] = v2165_acc;
        r4[16] = v2166_acc;
        r4[17] = v2167_acc;
        // r4 = 0 past (0, 21)
        // s1 = store{r>s}(localShrMem0, r4);
        #pragma unroll
        for (int32_t v2178_i0 = 0; v2178_i0 < 1; ++v2178_i0) {
          int32_t v2184_lead = v1371_lead + (v2178_i0 * 16);
          #pragma unroll
          for (int32_t v2179_i1 = 0; v2179_i1 < 9; ++v2179_i1) {
            double v2182_data = r4[(v2178_i0 + (v2179_i1 * 2))];
            s1[(v2184_lead + (v2179_i1 * 21))] = v2182_data;
          }
        }
        if (v2095_g) {
          #pragma unroll
          for (int32_t v2187_i1 = 0; v2187_i1 < 9; ++v2187_i1) {
            double v2190_data = r4[(1 + (v2187_i1 * 2))];
            s1[(v1436_lead + (v2187_i1 * 21))] = v2190_data;
          }
        }
        double r6[36]{};
        // ir6 = +(glb_m4 * s1)
        // [(0, 56), (0, 9)] [(0, 21)]
        double ir6[36]{};
        tensorforge::VectorT<double, 4> v2202_acc{};
        tensorforge::VectorT<double, 4> v2203_acc{};
        tensorforge::VectorT<double, 4> v2204_acc{};
        tensorforge::VectorT<double, 4> v2205_acc{};
        tensorforge::VectorT<double, 4> v2206_acc{};
        tensorforge::VectorT<double, 4> v2207_acc{};
        tensorforge::VectorT<double, 4> v2208_acc{};
        tensorforge::VectorT<double, 4> v2209_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        double v2216_data = s1[v2213_sh];
        double v2218_data = s1[v2217_a];
        double v2224_data = glb_m4[v2222_a];
        tensorforge::VectorT<double, 4> v2225_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2216_data, v2224_data, v2202_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2226_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2218_data, v2224_data, v2203_acc, 0, 0, 0);
        double v2231_data = glb_m4[v2229_a];
        tensorforge::VectorT<double, 4> v2232_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2216_data, v2231_data, v2204_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2233_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2218_data, v2231_data, v2205_acc, 0, 0, 0);
        double v2238_data = glb_m4[v2236_a];
        tensorforge::VectorT<double, 4> v2239_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2216_data, v2238_data, v2206_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2240_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2218_data, v2238_data, v2207_acc, 0, 0, 0);
        double v2245_data = glb_m4[v2243_a];
        tensorforge::VectorT<double, 4> v2246_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2216_data, v2245_data, v2208_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2247_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2218_data, v2245_data, v2209_acc, 0, 0, 0);
        double v2250_data = s1[v2248_a];
        double v2252_data = s1[v2251_a];
        double v2254_data = glb_m4[v2253_a];
        tensorforge::VectorT<double, 4> v2255_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2250_data, v2254_data, v2225_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2256_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2252_data, v2254_data, v2226_acc, 0, 0, 0);
        double v2258_data = glb_m4[v2257_a];
        tensorforge::VectorT<double, 4> v2259_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2250_data, v2258_data, v2232_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2260_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2252_data, v2258_data, v2233_acc, 0, 0, 0);
        double v2262_data = glb_m4[v2261_a];
        tensorforge::VectorT<double, 4> v2263_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2250_data, v2262_data, v2239_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2264_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2252_data, v2262_data, v2240_acc, 0, 0, 0);
        double v2266_data = glb_m4[v2265_a];
        tensorforge::VectorT<double, 4> v2267_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2250_data, v2266_data, v2246_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2268_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2252_data, v2266_data, v2247_acc, 0, 0, 0);
        double v2271_data = s1[v2269_a];
        double v2273_data = s1[v2272_a];
        double v2275_data = glb_m4[v2274_a];
        tensorforge::VectorT<double, 4> v2276_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2271_data, v2275_data, v2255_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2277_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2273_data, v2275_data, v2256_acc, 0, 0, 0);
        double v2279_data = glb_m4[v2278_a];
        tensorforge::VectorT<double, 4> v2280_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2271_data, v2279_data, v2259_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2281_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2273_data, v2279_data, v2260_acc, 0, 0, 0);
        double v2283_data = glb_m4[v2282_a];
        tensorforge::VectorT<double, 4> v2284_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2271_data, v2283_data, v2263_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2285_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2273_data, v2283_data, v2264_acc, 0, 0, 0);
        double v2287_data = glb_m4[v2286_a];
        tensorforge::VectorT<double, 4> v2288_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2271_data, v2287_data, v2267_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2289_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2273_data, v2287_data, v2268_acc, 0, 0, 0);
        double v2292_data = s1[v2290_a];
        double v2294_data = s1[v2293_a];
        double v2296_data = glb_m4[v2295_a];
        tensorforge::VectorT<double, 4> v2297_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2292_data, v2296_data, v2280_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2298_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2294_data, v2296_data, v2281_acc, 0, 0, 0);
        double v2300_data = glb_m4[v2299_a];
        tensorforge::VectorT<double, 4> v2301_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2292_data, v2300_data, v2284_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2302_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2294_data, v2300_data, v2285_acc, 0, 0, 0);
        double v2304_data = glb_m4[v2303_a];
        tensorforge::VectorT<double, 4> v2305_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2292_data, v2304_data, v2288_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2306_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2294_data, v2304_data, v2289_acc, 0, 0, 0);
        double v2309_data = s1[v2307_a];
        double v2311_data = s1[v2310_a];
        double v2313_data = glb_m4[v2312_a];
        tensorforge::VectorT<double, 4> v2314_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2309_data, v2313_data, v2297_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2315_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2311_data, v2313_data, v2298_acc, 0, 0, 0);
        double v2317_data = glb_m4[v2316_a];
        tensorforge::VectorT<double, 4> v2318_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2309_data, v2317_data, v2301_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2319_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2311_data, v2317_data, v2302_acc, 0, 0, 0);
        double v2321_data = glb_m4[v2320_a];
        tensorforge::VectorT<double, 4> v2322_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2309_data, v2321_data, v2305_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2323_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2311_data, v2321_data, v2306_acc, 0, 0, 0);
        double v2332_data = s1[v2330_a];
        double v2338_ks = v2337_g ? v2332_data : v2336_ks;
        double v2340_data = s1[v2339_a];
        double v2341_ks = v2337_g ? v2340_data : v2336_ks;
        double v2343_data = glb_m4[v2342_a];
        double v2344_ks = v2337_g ? v2343_data : v2336_ks;
        tensorforge::VectorT<double, 4> v2345_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2338_ks, v2344_ks, v2314_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2346_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2341_ks, v2344_ks, v2315_acc, 0, 0, 0);
        double v2348_data = glb_m4[v2347_a];
        double v2349_ks = v2337_g ? v2348_data : v2336_ks;
        tensorforge::VectorT<double, 4> v2350_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2338_ks, v2349_ks, v2318_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2351_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2341_ks, v2349_ks, v2319_acc, 0, 0, 0);
        double v2353_data = glb_m4[v2352_a];
        double v2354_ks = v2337_g ? v2353_data : v2336_ks;
        tensorforge::VectorT<double, 4> v2355_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2338_ks, v2354_ks, v2322_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v2356_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v2341_ks, v2354_ks, v2323_acc, 0, 0, 0);
        ir6[0] = (v2276_acc[0]);
        ir6[4] = (v2276_acc[1]);
        ir6[8] = (v2276_acc[2]);
        ir6[12] = (v2276_acc[3]);
        ir6[16] = (v2277_acc[0]);
        ir6[20] = (v2277_acc[1]);
        ir6[24] = (v2277_acc[2]);
        ir6[28] = (v2277_acc[3]);
        ir6[1] = (v2345_acc[0]);
        ir6[5] = (v2345_acc[1]);
        ir6[9] = (v2345_acc[2]);
        ir6[13] = (v2345_acc[3]);
        ir6[17] = (v2346_acc[0]);
        ir6[21] = (v2346_acc[1]);
        ir6[25] = (v2346_acc[2]);
        ir6[29] = (v2346_acc[3]);
        ir6[2] = (v2350_acc[0]);
        ir6[6] = (v2350_acc[1]);
        ir6[10] = (v2350_acc[2]);
        ir6[14] = (v2350_acc[3]);
        ir6[18] = (v2351_acc[0]);
        ir6[22] = (v2351_acc[1]);
        ir6[26] = (v2351_acc[2]);
        ir6[30] = (v2351_acc[3]);
        ir6[3] = (v2355_acc[0]);
        ir6[7] = (v2355_acc[1]);
        ir6[11] = (v2355_acc[2]);
        ir6[15] = (v2355_acc[3]);
        ir6[19] = (v2356_acc[0]);
        ir6[23] = (v2356_acc[1]);
        ir6[27] = (v2356_acc[2]);
        ir6[31] = (v2356_acc[3]);
        double v2392_data = glb_m4[v1371_lead];
        double v2396_data = glb_m4[v1436_lead];
        double v2400_data = glb_m4[v2235_lead];
        double v2404_data = glb_m4[v1383_lead];
        double v2406_data = glb_m4[v2405_a];
        double v2408_data = glb_m4[v2407_a];
        double v2410_data = glb_m4[v2409_a];
        double v2412_data = glb_m4[v2411_a];
        double v2414_data = glb_m4[v2413_a];
        double v2416_data = glb_m4[v2415_a];
        double v2418_data = glb_m4[v2417_a];
        double v2420_data = glb_m4[v2419_a];
        double v2422_data = glb_m4[v1887_a];
        double v2424_data = glb_m4[v1889_a];
        double v2426_data = glb_m4[v2425_a];
        double v2428_data = glb_m4[v2427_a];
        double v2430_data = glb_m4[v2429_a];
        double v2432_data = glb_m4[v2431_a];
        double v2434_data = glb_m4[v2433_a];
        double v2436_data = glb_m4[v2435_a];
        double v2438_data = glb_m4[v2437_a];
        double v2440_data = glb_m4[v2439_a];
        double v2442_data = glb_m4[v2441_a];
        double v2444_data = glb_m4[v2443_a];
        double v2446_data = glb_m4[v1919_a];
        double v2448_data = glb_m4[v1921_a];
        double v2450_data = glb_m4[v2449_a];
        double v2452_data = glb_m4[v2451_a];
        double v2454_data = glb_m4[v2453_a];
        double v2456_data = glb_m4[v2455_a];
        double v2458_data = glb_m4[v2457_a];
        double v2460_data = glb_m4[v2459_a];
        double v2462_data = glb_m4[v2461_a];
        double v2464_data = glb_m4[v2463_a];
        double v2466_data = glb_m4[v2465_a];
        double v2468_data = glb_m4[v2467_a];
        double v2470_data = glb_m4[v1951_a];
        double v2472_data = glb_m4[v1953_a];
        double v2474_data = glb_m4[v2473_a];
        double v2476_data = glb_m4[v2475_a];
        double v2478_data = glb_m4[v2477_a];
        double v2480_data = glb_m4[v2479_a];
        double v2482_data = glb_m4[v2481_a];
        double v2484_data = glb_m4[v2483_a];
        double v2486_data = glb_m4[v2485_a];
        double v2488_data = glb_m4[v2487_a];
        double v2490_data = glb_m4[v2489_a];
        double v2492_data = glb_m4[v2491_a];
        double v2494_data = glb_m4[v1985_a];
        double v2496_data = glb_m4[v2495_a];
        double v2498_data = glb_m4[v2497_a];
        double v2500_data = glb_m4[v2499_a];
        double v2502_data = glb_m4[v2501_a];
        double v2504_data = glb_m4[v2503_a];
        double v2506_data = glb_m4[v2505_a];
        double v2508_data = glb_m4[v2507_a];
        double v2510_data = glb_m4[v2509_a];
        double v2512_data = glb_m4[v2017_a];
        double v2514_data = glb_m4[v2513_a];
        double v2516_data = glb_m4[v2515_a];
        double v2518_data = glb_m4[v2517_a];
        double v2520_data = glb_m4[v2519_a];
        double v2522_data = glb_m4[v2521_a];
        double v2524_data = glb_m4[v2523_a];
        double v2526_data = glb_m4[v2525_a];
        double v2528_data = glb_m4[v2527_a];
        double v2530_data = glb_m4[v2049_a];
        double v2532_data = glb_m4[v2531_a];
        double v2534_data = glb_m4[v2533_a];
        double v2536_data = glb_m4[v2535_a];
        double v2538_data = glb_m4[v2537_a];
        double v2540_data = glb_m4[v2539_a];
        double v2547_acc{};
        double v2548_acc{};
        double v2549_acc{};
        double v2550_acc{};
        double v2552_data = r4[17];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %20 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %22 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %23 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %25 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %26 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %29 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %36 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2547_acc), "+&v"(v2548_acc), "+&v"(v2549_acc), "+&v"(v2550_acc)
        : "v"(v2166_acc), "v"(v2392_data), "v"(v2396_data), "v"(v2400_data), "v"(v2404_data), "v"(v2406_data), "v"(v2408_data), "v"(v2410_data), "v"(v2412_data), "v"(v2414_data), "v"(v2416_data), "v"(v2418_data), "v"(v2420_data), "v"(v2422_data), "v"(v2424_data), "v"(v2426_data), "v"(v2428_data), "v"(v2430_data), "v"(v2432_data), "v"(v2434_data), "v"(v2436_data), "v"(v2438_data), "v"(v2440_data), "v"(v2442_data), "v"(v2444_data), "v"(v2446_data), "v"(v2448_data), "v"(v2450_data), "v"(v2452_data), "v"(v2454_data), "v"(v2456_data), "v"(v2458_data), "v"(v2460_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %4, %5 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %6 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %7 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %8 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %9 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %10 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %12 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %13 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %14 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %15 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %16 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %17 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %18 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2547_acc), "+&v"(v2548_acc), "+&v"(v2549_acc), "+&v"(v2550_acc)
        : "v"(v2166_acc), "v"(v2462_data), "v"(v2464_data), "v"(v2466_data), "v"(v2468_data), "v"(v2470_data), "v"(v2472_data), "v"(v2474_data), "v"(v2476_data), "v"(v2478_data), "v"(v2480_data), "v"(v2482_data), "v"(v2484_data), "v"(v2486_data), "v"(v2488_data)
        );
        // r0 = load{g>r}(wrap_glb_m1);
        #pragma unroll
        for (int32_t v2633_i0 = 0; v2633_i0 < 3; ++v2633_i0) {
          int32_t v2635_lead = v1371_lead + (v2633_i0 * 16);
          #pragma unroll
          for (int32_t v2636_i1 = 0; v2636_i1 < 9; ++v2636_i1) {
            double v2639_data = __builtin_nontemporal_load(&wrap_glb_m1[(v2635_lead + (v2636_i1 * 56))]);
            r0[(v2633_i0 + (v2636_i1 * 4))] = v2639_data;
          }
        }
        #pragma unroll
        for (int32_t v2642_i1 = 0; v2642_i1 < 9; ++v2642_i1) {
          double v2645_data = v1386_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v1383_lead + (v2642_i1 * 56))])) : (0.0);
          r0[(3 + (v2642_i1 * 4))] = v2645_data;
        }
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %18, %19 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %18, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %18, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %18, %22 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %18, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %18, %24 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %18, %25 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %18, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %18, %27 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %18, %28 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %18, %29 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %18, %30 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %18, %31 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %18, %32 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %18, %33 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2549_acc), "+&v"(v2550_acc), "+&v"(v2548_acc)
        : "v"(v2166_acc), "v"(v2490_data), "v"(v2492_data), "v"(v2494_data), "v"(v2496_data), "v"(v2498_data), "v"(v2500_data), "v"(v2502_data), "v"(v2504_data), "v"(v2506_data), "v"(v2508_data), "v"(v2510_data), "v"(v2512_data), "v"(v2514_data), "v"(v2516_data), "v"(v2552_data), "v"(v2518_data), "v"(v2520_data), "v"(v2522_data), "v"(v2524_data), "v"(v2526_data), "v"(v2528_data), "v"(v2530_data), "v"(v2532_data), "v"(v2534_data), "v"(v2536_data), "v"(v2538_data), "v"(v2540_data), "v"(v2343_data), "v"(v2348_data), "v"(v2353_data)
        );
        ir6[32] = v2547_acc;
        ir6[33] = v2548_acc;
        ir6[34] = v2549_acc;
        ir6[35] = v2550_acc;
        double r5[36]{};
        // r5 = load{g>r}(glb_m3);
        // r6 = ir6 + r5
        // glb_m3 = store{r>g}(r6);
        #pragma unroll
        for (int32_t v2586_i0 = 0; v2586_i0 < 3; ++v2586_i0) {
          #pragma unroll
          for (int32_t v2587_i1 = 0; v2587_i1 < 9; ++v2587_i1) {
            double v2590_data = ir6[(v2586_i0 + (v2587_i1 * 4))];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m3[((v1371_lead + (v2586_i0 * 16)) + (v2587_i1 * 56))], v2590_data);
            }
          }
        }
        if (v1386_g) {
          #pragma unroll
          for (int32_t v2596_i1 = 0; v2596_i1 < 9; ++v2596_i1) {
            double v2599_data = ir6[((v2596_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m3[(v1383_lead + (v2596_i1 * 56))], v2599_data);
            }
          }
        }
      }
    }
  }
}

