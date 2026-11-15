// === base name ===
kernel_948d1b3e8a55faf1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_948d1b3e8a55faf1 = {{16, 8, 1}, 16, 48, 1, 8, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_948d1b3e8a55faf1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_948d1b3e8a55faf1(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_948d1b3e8a55faf1(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_948d1b3e8a55faf1, block.x * block.y * block.z, 1664 * sizeof(double));
        CHECK_ERR;
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
  config.sharedMemBytes = 1664 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_948d1b3e8a55faf1(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_948d1b3e8a55faf1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_948d1b3e8a55faf1, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_948d1b3e8a55faf1<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_948d1b3e8a55faf1(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (48 active) x 8 per block = block 16x8x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 48×8(48×8) {0..48}×{0..8} strided
    //   m1 48×24(48×24) {0..48}×{0..24} none ordered
    //   m2 24×8(24×8) {0..24}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":48,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1664}],"shared_bytes":13312,"shared_elements":1664,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[48,8]],"name":"m0","ordered":false,"parts":1,"shape":[48,8],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[48,24]],"name":"m1","ordered":true,"parts":1,"shape":[48,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,8]],"name":"m2","ordered":false,"parts":1,"shape":[24,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[48,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[48,8]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[48,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[48,24]},{"addressing":"strided","bbox":[[0,0],[24,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const double *const __restrict__ glb_m1 = &m1[0];
      double * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v3553_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v3553_flagSecondAhead < numElements0) ? v3553_flagSecondAhead : v3_batchId1)];
      const double *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 192 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v3563_i = 0; v3563_i < 12; ++v3563_i) {
        int32_t v3565_a = threadIdx.x + (v3563_i * 16);
        __pipeline_memcpy_async(&s0[v3565_a], &peel_glb_m2[v3565_a], 8);
      }
      __pipeline_commit();
      uint32_t v3557_flagWord = flagWordFirst;
      uint32_t v3558_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const double *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 192 + 0 + m2_extraOffset];
        size_t v3575_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v3575_flagAheadAhead < numElements0) ? v3575_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v3557_flagWord);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r0[24]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 48), (0, 8)] [(0, 24)]
          double ir0[24]{};
          int32_t v27_lead = threadIdx.x % 16;
          int32_t v28_a = v27_lead * 2;
          tensorforge::VectorT<double, 2> v30_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[v28_a];
          double v31_p = v30_data[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v32_data = s0[0];
          double v38_data = s0[24];
          double v44_data = s0[48];
          double v50_data = s0[72];
          double v56_data = s0[96];
          double v62_data = s0[120];
          double v68_data = s0[144];
          double v74_data = s0[168];
          double v79_p = v30_data[1];
          int32_t v126_a = v28_a + 32;
          tensorforge::VectorT<double, 2> v128_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[v126_a];
          double v129_p = v128_data[0];
          tensorforge::VectorT<double, 2> v177_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(64 + v28_a)];
          double v178_p = v177_data[0];
          double v179_data = s0[1];
          double v185_data = s0[25];
          double v191_data = s0[49];
          double v197_data = s0[73];
          double v203_data = s0[97];
          double v209_data = s0[121];
          double v215_data = s0[145];
          double v221_data = s0[169];
          double v226_p = v177_data[1];
          tensorforge::VectorT<double, 2> v274_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(64 + v126_a)];
          double v275_p = v274_data[0];
          tensorforge::VectorT<double, 2> v323_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(128 + v28_a)];
          double v324_p = v323_data[0];
          double v325_data = s0[2];
          double v331_data = s0[26];
          double v337_data = s0[50];
          double v343_data = s0[74];
          double v349_data = s0[98];
          double v355_data = s0[122];
          double v361_data = s0[146];
          double v367_data = s0[170];
          double v372_p = v323_data[1];
          tensorforge::VectorT<double, 2> v420_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(128 + v126_a)];
          double v421_p = v420_data[0];
          tensorforge::VectorT<double, 2> v469_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(192 + v28_a)];
          double v470_p = v469_data[0];
          double v471_data = s0[3];
          double v477_data = s0[27];
          double v483_data = s0[51];
          double v489_data = s0[75];
          double v495_data = s0[99];
          double v501_data = s0[123];
          double v507_data = s0[147];
          double v513_data = s0[171];
          double v518_p = v469_data[1];
          tensorforge::VectorT<double, 2> v566_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(192 + v126_a)];
          double v567_p = v566_data[0];
          tensorforge::VectorT<double, 2> v615_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(256 + v28_a)];
          double v616_p = v615_data[0];
          double v617_data = s0[4];
          double v623_data = s0[28];
          double v629_data = s0[52];
          double v635_data = s0[76];
          double v641_data = s0[100];
          double v647_data = s0[124];
          double v653_data = s0[148];
          double v659_data = s0[172];
          double v664_p = v615_data[1];
          tensorforge::VectorT<double, 2> v712_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(256 + v126_a)];
          double v713_p = v712_data[0];
          tensorforge::VectorT<double, 2> v761_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(320 + v28_a)];
          double v762_p = v761_data[0];
          double v763_data = s0[5];
          double v769_data = s0[29];
          double v775_data = s0[53];
          double v781_data = s0[77];
          double v787_data = s0[101];
          double v793_data = s0[125];
          double v799_data = s0[149];
          double v805_data = s0[173];
          double v810_p = v761_data[1];
          tensorforge::VectorT<double, 2> v858_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(320 + v126_a)];
          double v859_p = v858_data[0];
          tensorforge::VectorT<double, 2> v907_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(384 + v28_a)];
          double v908_p = v907_data[0];
          double v909_data = s0[6];
          double v915_data = s0[30];
          double v921_data = s0[54];
          double v927_data = s0[78];
          double v933_data = s0[102];
          double v939_data = s0[126];
          double v945_data = s0[150];
          double v951_data = s0[174];
          double v956_p = v907_data[1];
          tensorforge::VectorT<double, 2> v1004_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(384 + v126_a)];
          double v1005_p = v1004_data[0];
          tensorforge::VectorT<double, 2> v1053_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(448 + v28_a)];
          double v1054_p = v1053_data[0];
          double v1055_data = s0[7];
          double v1061_data = s0[31];
          double v1067_data = s0[55];
          double v1073_data = s0[79];
          double v1079_data = s0[103];
          double v1085_data = s0[127];
          double v1091_data = s0[151];
          double v1097_data = s0[175];
          double v1102_p = v1053_data[1];
          tensorforge::VectorT<double, 2> v1150_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(448 + v126_a)];
          double v1151_p = v1150_data[0];
          tensorforge::VectorT<double, 2> v1199_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(512 + v28_a)];
          double v1200_p = v1199_data[0];
          double v1201_data = s0[8];
          double v1207_data = s0[32];
          double v1213_data = s0[56];
          double v1219_data = s0[80];
          double v1225_data = s0[104];
          double v1231_data = s0[128];
          double v1237_data = s0[152];
          double v1243_data = s0[176];
          double v1248_p = v1199_data[1];
          tensorforge::VectorT<double, 2> v1296_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(512 + v126_a)];
          double v1297_p = v1296_data[0];
          tensorforge::VectorT<double, 2> v1345_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(576 + v28_a)];
          double v1346_p = v1345_data[0];
          double v1347_data = s0[9];
          double v1353_data = s0[33];
          double v1359_data = s0[57];
          double v1365_data = s0[81];
          double v1371_data = s0[105];
          double v1377_data = s0[129];
          double v1383_data = s0[153];
          double v1389_data = s0[177];
          double v1394_p = v1345_data[1];
          tensorforge::VectorT<double, 2> v1442_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(576 + v126_a)];
          double v1443_p = v1442_data[0];
          tensorforge::VectorT<double, 2> v1491_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(640 + v28_a)];
          double v1492_p = v1491_data[0];
          double v1493_data = s0[10];
          double v1499_data = s0[34];
          double v1505_data = s0[58];
          double v1511_data = s0[82];
          double v1517_data = s0[106];
          double v1523_data = s0[130];
          double v1529_data = s0[154];
          double v1535_data = s0[178];
          double v1540_p = v1491_data[1];
          tensorforge::VectorT<double, 2> v1588_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(640 + v126_a)];
          double v1589_p = v1588_data[0];
          tensorforge::VectorT<double, 2> v1637_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(704 + v28_a)];
          double v1638_p = v1637_data[0];
          double v1639_data = s0[11];
          double v1645_data = s0[35];
          double v1651_data = s0[59];
          double v1657_data = s0[83];
          double v1663_data = s0[107];
          double v1669_data = s0[131];
          double v1675_data = s0[155];
          double v1681_data = s0[179];
          double v1686_p = v1637_data[1];
          tensorforge::VectorT<double, 2> v1734_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(704 + v126_a)];
          double v1735_p = v1734_data[0];
          tensorforge::VectorT<double, 2> v1783_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(768 + v28_a)];
          double v1784_p = v1783_data[0];
          double v1785_data = s0[12];
          double v1791_data = s0[36];
          double v1797_data = s0[60];
          double v1803_data = s0[84];
          double v1809_data = s0[108];
          double v1815_data = s0[132];
          double v1821_data = s0[156];
          double v1827_data = s0[180];
          double v1832_p = v1783_data[1];
          tensorforge::VectorT<double, 2> v1880_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(768 + v126_a)];
          double v1881_p = v1880_data[0];
          tensorforge::VectorT<double, 2> v1929_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(832 + v28_a)];
          double v1930_p = v1929_data[0];
          double v1931_data = s0[13];
          double v1937_data = s0[37];
          double v1943_data = s0[61];
          double v1949_data = s0[85];
          double v1955_data = s0[109];
          double v1961_data = s0[133];
          double v1967_data = s0[157];
          double v1973_data = s0[181];
          double v1978_p = v1929_data[1];
          tensorforge::VectorT<double, 2> v2026_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(832 + v126_a)];
          double v2027_p = v2026_data[0];
          tensorforge::VectorT<double, 2> v2075_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(896 + v28_a)];
          double v2076_p = v2075_data[0];
          double v2077_data = s0[14];
          double v2083_data = s0[38];
          double v2089_data = s0[62];
          double v2095_data = s0[86];
          double v2101_data = s0[110];
          double v2107_data = s0[134];
          double v2113_data = s0[158];
          double v2119_data = s0[182];
          double v2124_p = v2075_data[1];
          tensorforge::VectorT<double, 2> v2172_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(896 + v126_a)];
          double v2173_p = v2172_data[0];
          tensorforge::VectorT<double, 2> v2221_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(960 + v28_a)];
          double v2222_p = v2221_data[0];
          double v2223_data = s0[15];
          double v2229_data = s0[39];
          double v2235_data = s0[63];
          double v2241_data = s0[87];
          double v2247_data = s0[111];
          double v2253_data = s0[135];
          double v2259_data = s0[159];
          double v2265_data = s0[183];
          double v2270_p = v2221_data[1];
          tensorforge::VectorT<double, 2> v2318_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(960 + v126_a)];
          double v2319_p = v2318_data[0];
          tensorforge::VectorT<double, 2> v2367_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1024 + v28_a)];
          double v2368_p = v2367_data[0];
          double v2369_data = s0[16];
          double v2375_data = s0[40];
          double v2381_data = s0[64];
          double v2387_data = s0[88];
          double v2393_data = s0[112];
          double v2399_data = s0[136];
          double v2405_data = s0[160];
          double v2411_data = s0[184];
          double v2416_p = v2367_data[1];
          tensorforge::VectorT<double, 2> v2464_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1024 + v126_a)];
          double v2465_p = v2464_data[0];
          tensorforge::VectorT<double, 2> v2513_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1088 + v28_a)];
          double v2514_p = v2513_data[0];
          double v2515_data = s0[17];
          double v2521_data = s0[41];
          double v2527_data = s0[65];
          double v2533_data = s0[89];
          double v2539_data = s0[113];
          double v2545_data = s0[137];
          double v2551_data = s0[161];
          double v2557_data = s0[185];
          double v2562_p = v2513_data[1];
          tensorforge::VectorT<double, 2> v2610_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1088 + v126_a)];
          double v2611_p = v2610_data[0];
          tensorforge::VectorT<double, 2> v2659_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1152 + v28_a)];
          double v2660_p = v2659_data[0];
          double v2661_data = s0[18];
          double v2667_data = s0[42];
          double v2673_data = s0[66];
          double v2679_data = s0[90];
          double v2685_data = s0[114];
          double v2691_data = s0[138];
          double v2697_data = s0[162];
          double v2703_data = s0[186];
          double v2708_p = v2659_data[1];
          tensorforge::VectorT<double, 2> v2756_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1152 + v126_a)];
          double v2757_p = v2756_data[0];
          tensorforge::VectorT<double, 2> v2805_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1216 + v28_a)];
          double v2806_p = v2805_data[0];
          double v2807_data = s0[19];
          double v2813_data = s0[43];
          double v2819_data = s0[67];
          double v2825_data = s0[91];
          double v2831_data = s0[115];
          double v2837_data = s0[139];
          double v2843_data = s0[163];
          double v2849_data = s0[187];
          double v2854_p = v2805_data[1];
          tensorforge::VectorT<double, 2> v2902_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1216 + v126_a)];
          double v2903_p = v2902_data[0];
          tensorforge::VectorT<double, 2> v2951_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1280 + v28_a)];
          double v2952_p = v2951_data[0];
          double v2953_data = s0[20];
          double v2959_data = s0[44];
          double v2965_data = s0[68];
          double v2971_data = s0[92];
          double v2977_data = s0[116];
          double v2983_data = s0[140];
          double v2989_data = s0[164];
          double v2995_data = s0[188];
          double v3000_p = v2951_data[1];
          tensorforge::VectorT<double, 2> v3048_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1280 + v126_a)];
          double v3049_p = v3048_data[0];
          tensorforge::VectorT<double, 2> v3097_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1344 + v28_a)];
          double v3098_p = v3097_data[0];
          double v3099_data = s0[21];
          double v3105_data = s0[45];
          double v3111_data = s0[69];
          double v3117_data = s0[93];
          double v3123_data = s0[117];
          double v3129_data = s0[141];
          double v3135_data = s0[165];
          double v3141_data = s0[189];
          double v3146_p = v3097_data[1];
          tensorforge::VectorT<double, 2> v3194_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1344 + v126_a)];
          double v3195_p = v3194_data[0];
          tensorforge::VectorT<double, 2> v3243_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1408 + v28_a)];
          double v3244_p = v3243_data[0];
          double v3245_data = s0[22];
          double v3251_data = s0[46];
          double v3254_p = ((((((((((((((((((((((v31_p * v38_data) + (v178_p * v185_data)) + (v324_p * v331_data)) + (v470_p * v477_data)) + (v616_p * v623_data)) + (v762_p * v769_data)) + (v908_p * v915_data)) + (v1054_p * v1061_data)) + (v1200_p * v1207_data)) + (v1346_p * v1353_data)) + (v1492_p * v1499_data)) + (v1638_p * v1645_data)) + (v1784_p * v1791_data)) + (v1930_p * v1937_data)) + (v2076_p * v2083_data)) + (v2222_p * v2229_data)) + (v2368_p * v2375_data)) + (v2514_p * v2521_data)) + (v2660_p * v2667_data)) + (v2806_p * v2813_data)) + (v2952_p * v2959_data)) + (v3098_p * v3105_data)) + (v3244_p * v3251_data);
          double v3257_data = s0[70];
          double v3260_p = ((((((((((((((((((((((v31_p * v44_data) + (v178_p * v191_data)) + (v324_p * v337_data)) + (v470_p * v483_data)) + (v616_p * v629_data)) + (v762_p * v775_data)) + (v908_p * v921_data)) + (v1054_p * v1067_data)) + (v1200_p * v1213_data)) + (v1346_p * v1359_data)) + (v1492_p * v1505_data)) + (v1638_p * v1651_data)) + (v1784_p * v1797_data)) + (v1930_p * v1943_data)) + (v2076_p * v2089_data)) + (v2222_p * v2235_data)) + (v2368_p * v2381_data)) + (v2514_p * v2527_data)) + (v2660_p * v2673_data)) + (v2806_p * v2819_data)) + (v2952_p * v2965_data)) + (v3098_p * v3111_data)) + (v3244_p * v3257_data);
          double v3263_data = s0[94];
          double v3266_p = ((((((((((((((((((((((v31_p * v50_data) + (v178_p * v197_data)) + (v324_p * v343_data)) + (v470_p * v489_data)) + (v616_p * v635_data)) + (v762_p * v781_data)) + (v908_p * v927_data)) + (v1054_p * v1073_data)) + (v1200_p * v1219_data)) + (v1346_p * v1365_data)) + (v1492_p * v1511_data)) + (v1638_p * v1657_data)) + (v1784_p * v1803_data)) + (v1930_p * v1949_data)) + (v2076_p * v2095_data)) + (v2222_p * v2241_data)) + (v2368_p * v2387_data)) + (v2514_p * v2533_data)) + (v2660_p * v2679_data)) + (v2806_p * v2825_data)) + (v2952_p * v2971_data)) + (v3098_p * v3117_data)) + (v3244_p * v3263_data);
          double v3269_data = s0[118];
          double v3272_p = ((((((((((((((((((((((v31_p * v56_data) + (v178_p * v203_data)) + (v324_p * v349_data)) + (v470_p * v495_data)) + (v616_p * v641_data)) + (v762_p * v787_data)) + (v908_p * v933_data)) + (v1054_p * v1079_data)) + (v1200_p * v1225_data)) + (v1346_p * v1371_data)) + (v1492_p * v1517_data)) + (v1638_p * v1663_data)) + (v1784_p * v1809_data)) + (v1930_p * v1955_data)) + (v2076_p * v2101_data)) + (v2222_p * v2247_data)) + (v2368_p * v2393_data)) + (v2514_p * v2539_data)) + (v2660_p * v2685_data)) + (v2806_p * v2831_data)) + (v2952_p * v2977_data)) + (v3098_p * v3123_data)) + (v3244_p * v3269_data);
          double v3275_data = s0[142];
          double v3278_p = ((((((((((((((((((((((v31_p * v62_data) + (v178_p * v209_data)) + (v324_p * v355_data)) + (v470_p * v501_data)) + (v616_p * v647_data)) + (v762_p * v793_data)) + (v908_p * v939_data)) + (v1054_p * v1085_data)) + (v1200_p * v1231_data)) + (v1346_p * v1377_data)) + (v1492_p * v1523_data)) + (v1638_p * v1669_data)) + (v1784_p * v1815_data)) + (v1930_p * v1961_data)) + (v2076_p * v2107_data)) + (v2222_p * v2253_data)) + (v2368_p * v2399_data)) + (v2514_p * v2545_data)) + (v2660_p * v2691_data)) + (v2806_p * v2837_data)) + (v2952_p * v2983_data)) + (v3098_p * v3129_data)) + (v3244_p * v3275_data);
          double v3281_data = s0[166];
          double v3284_p = ((((((((((((((((((((((v31_p * v68_data) + (v178_p * v215_data)) + (v324_p * v361_data)) + (v470_p * v507_data)) + (v616_p * v653_data)) + (v762_p * v799_data)) + (v908_p * v945_data)) + (v1054_p * v1091_data)) + (v1200_p * v1237_data)) + (v1346_p * v1383_data)) + (v1492_p * v1529_data)) + (v1638_p * v1675_data)) + (v1784_p * v1821_data)) + (v1930_p * v1967_data)) + (v2076_p * v2113_data)) + (v2222_p * v2259_data)) + (v2368_p * v2405_data)) + (v2514_p * v2551_data)) + (v2660_p * v2697_data)) + (v2806_p * v2843_data)) + (v2952_p * v2989_data)) + (v3098_p * v3135_data)) + (v3244_p * v3281_data);
          double v3287_data = s0[190];
          double v3290_p = ((((((((((((((((((((((v31_p * v74_data) + (v178_p * v221_data)) + (v324_p * v367_data)) + (v470_p * v513_data)) + (v616_p * v659_data)) + (v762_p * v805_data)) + (v908_p * v951_data)) + (v1054_p * v1097_data)) + (v1200_p * v1243_data)) + (v1346_p * v1389_data)) + (v1492_p * v1535_data)) + (v1638_p * v1681_data)) + (v1784_p * v1827_data)) + (v1930_p * v1973_data)) + (v2076_p * v2119_data)) + (v2222_p * v2265_data)) + (v2368_p * v2411_data)) + (v2514_p * v2557_data)) + (v2660_p * v2703_data)) + (v2806_p * v2849_data)) + (v2952_p * v2995_data)) + (v3098_p * v3141_data)) + (v3244_p * v3287_data);
          double v3292_p = v3243_data[1];
          double v3296_p = ((((((((((((((((((((((v79_p * v32_data) + (v226_p * v179_data)) + (v372_p * v325_data)) + (v518_p * v471_data)) + (v664_p * v617_data)) + (v810_p * v763_data)) + (v956_p * v909_data)) + (v1102_p * v1055_data)) + (v1248_p * v1201_data)) + (v1394_p * v1347_data)) + (v1540_p * v1493_data)) + (v1686_p * v1639_data)) + (v1832_p * v1785_data)) + (v1978_p * v1931_data)) + (v2124_p * v2077_data)) + (v2270_p * v2223_data)) + (v2416_p * v2369_data)) + (v2562_p * v2515_data)) + (v2708_p * v2661_data)) + (v2854_p * v2807_data)) + (v3000_p * v2953_data)) + (v3146_p * v3099_data)) + (v3292_p * v3245_data);
          double v3302_p = ((((((((((((((((((((((v79_p * v38_data) + (v226_p * v185_data)) + (v372_p * v331_data)) + (v518_p * v477_data)) + (v664_p * v623_data)) + (v810_p * v769_data)) + (v956_p * v915_data)) + (v1102_p * v1061_data)) + (v1248_p * v1207_data)) + (v1394_p * v1353_data)) + (v1540_p * v1499_data)) + (v1686_p * v1645_data)) + (v1832_p * v1791_data)) + (v1978_p * v1937_data)) + (v2124_p * v2083_data)) + (v2270_p * v2229_data)) + (v2416_p * v2375_data)) + (v2562_p * v2521_data)) + (v2708_p * v2667_data)) + (v2854_p * v2813_data)) + (v3000_p * v2959_data)) + (v3146_p * v3105_data)) + (v3292_p * v3251_data);
          double v3308_p = ((((((((((((((((((((((v79_p * v44_data) + (v226_p * v191_data)) + (v372_p * v337_data)) + (v518_p * v483_data)) + (v664_p * v629_data)) + (v810_p * v775_data)) + (v956_p * v921_data)) + (v1102_p * v1067_data)) + (v1248_p * v1213_data)) + (v1394_p * v1359_data)) + (v1540_p * v1505_data)) + (v1686_p * v1651_data)) + (v1832_p * v1797_data)) + (v1978_p * v1943_data)) + (v2124_p * v2089_data)) + (v2270_p * v2235_data)) + (v2416_p * v2381_data)) + (v2562_p * v2527_data)) + (v2708_p * v2673_data)) + (v2854_p * v2819_data)) + (v3000_p * v2965_data)) + (v3146_p * v3111_data)) + (v3292_p * v3257_data);
          double v3314_p = ((((((((((((((((((((((v79_p * v50_data) + (v226_p * v197_data)) + (v372_p * v343_data)) + (v518_p * v489_data)) + (v664_p * v635_data)) + (v810_p * v781_data)) + (v956_p * v927_data)) + (v1102_p * v1073_data)) + (v1248_p * v1219_data)) + (v1394_p * v1365_data)) + (v1540_p * v1511_data)) + (v1686_p * v1657_data)) + (v1832_p * v1803_data)) + (v1978_p * v1949_data)) + (v2124_p * v2095_data)) + (v2270_p * v2241_data)) + (v2416_p * v2387_data)) + (v2562_p * v2533_data)) + (v2708_p * v2679_data)) + (v2854_p * v2825_data)) + (v3000_p * v2971_data)) + (v3146_p * v3117_data)) + (v3292_p * v3263_data);
          double v3320_p = ((((((((((((((((((((((v79_p * v56_data) + (v226_p * v203_data)) + (v372_p * v349_data)) + (v518_p * v495_data)) + (v664_p * v641_data)) + (v810_p * v787_data)) + (v956_p * v933_data)) + (v1102_p * v1079_data)) + (v1248_p * v1225_data)) + (v1394_p * v1371_data)) + (v1540_p * v1517_data)) + (v1686_p * v1663_data)) + (v1832_p * v1809_data)) + (v1978_p * v1955_data)) + (v2124_p * v2101_data)) + (v2270_p * v2247_data)) + (v2416_p * v2393_data)) + (v2562_p * v2539_data)) + (v2708_p * v2685_data)) + (v2854_p * v2831_data)) + (v3000_p * v2977_data)) + (v3146_p * v3123_data)) + (v3292_p * v3269_data);
          double v3326_p = ((((((((((((((((((((((v79_p * v62_data) + (v226_p * v209_data)) + (v372_p * v355_data)) + (v518_p * v501_data)) + (v664_p * v647_data)) + (v810_p * v793_data)) + (v956_p * v939_data)) + (v1102_p * v1085_data)) + (v1248_p * v1231_data)) + (v1394_p * v1377_data)) + (v1540_p * v1523_data)) + (v1686_p * v1669_data)) + (v1832_p * v1815_data)) + (v1978_p * v1961_data)) + (v2124_p * v2107_data)) + (v2270_p * v2253_data)) + (v2416_p * v2399_data)) + (v2562_p * v2545_data)) + (v2708_p * v2691_data)) + (v2854_p * v2837_data)) + (v3000_p * v2983_data)) + (v3146_p * v3129_data)) + (v3292_p * v3275_data);
          double v3332_p = ((((((((((((((((((((((v79_p * v68_data) + (v226_p * v215_data)) + (v372_p * v361_data)) + (v518_p * v507_data)) + (v664_p * v653_data)) + (v810_p * v799_data)) + (v956_p * v945_data)) + (v1102_p * v1091_data)) + (v1248_p * v1237_data)) + (v1394_p * v1383_data)) + (v1540_p * v1529_data)) + (v1686_p * v1675_data)) + (v1832_p * v1821_data)) + (v1978_p * v1967_data)) + (v2124_p * v2113_data)) + (v2270_p * v2259_data)) + (v2416_p * v2405_data)) + (v2562_p * v2551_data)) + (v2708_p * v2697_data)) + (v2854_p * v2843_data)) + (v3000_p * v2989_data)) + (v3146_p * v3135_data)) + (v3292_p * v3281_data);
          double v3338_p = ((((((((((((((((((((((v79_p * v74_data) + (v226_p * v221_data)) + (v372_p * v367_data)) + (v518_p * v513_data)) + (v664_p * v659_data)) + (v810_p * v805_data)) + (v956_p * v951_data)) + (v1102_p * v1097_data)) + (v1248_p * v1243_data)) + (v1394_p * v1389_data)) + (v1540_p * v1535_data)) + (v1686_p * v1681_data)) + (v1832_p * v1827_data)) + (v1978_p * v1973_data)) + (v2124_p * v2119_data)) + (v2270_p * v2265_data)) + (v2416_p * v2411_data)) + (v2562_p * v2557_data)) + (v2708_p * v2703_data)) + (v2854_p * v2849_data)) + (v3000_p * v2995_data)) + (v3146_p * v3141_data)) + (v3292_p * v3287_data);
          tensorforge::VectorT<double, 2> v3340_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1408 + v126_a)];
          double v3341_p = v3340_data[0];
          double v3345_p = ((((((((((((((((((((((v129_p * v32_data) + (v275_p * v179_data)) + (v421_p * v325_data)) + (v567_p * v471_data)) + (v713_p * v617_data)) + (v859_p * v763_data)) + (v1005_p * v909_data)) + (v1151_p * v1055_data)) + (v1297_p * v1201_data)) + (v1443_p * v1347_data)) + (v1589_p * v1493_data)) + (v1735_p * v1639_data)) + (v1881_p * v1785_data)) + (v2027_p * v1931_data)) + (v2173_p * v2077_data)) + (v2319_p * v2223_data)) + (v2465_p * v2369_data)) + (v2611_p * v2515_data)) + (v2757_p * v2661_data)) + (v2903_p * v2807_data)) + (v3049_p * v2953_data)) + (v3195_p * v3099_data)) + (v3341_p * v3245_data);
          double v3351_p = ((((((((((((((((((((((v129_p * v38_data) + (v275_p * v185_data)) + (v421_p * v331_data)) + (v567_p * v477_data)) + (v713_p * v623_data)) + (v859_p * v769_data)) + (v1005_p * v915_data)) + (v1151_p * v1061_data)) + (v1297_p * v1207_data)) + (v1443_p * v1353_data)) + (v1589_p * v1499_data)) + (v1735_p * v1645_data)) + (v1881_p * v1791_data)) + (v2027_p * v1937_data)) + (v2173_p * v2083_data)) + (v2319_p * v2229_data)) + (v2465_p * v2375_data)) + (v2611_p * v2521_data)) + (v2757_p * v2667_data)) + (v2903_p * v2813_data)) + (v3049_p * v2959_data)) + (v3195_p * v3105_data)) + (v3341_p * v3251_data);
          double v3357_p = ((((((((((((((((((((((v129_p * v44_data) + (v275_p * v191_data)) + (v421_p * v337_data)) + (v567_p * v483_data)) + (v713_p * v629_data)) + (v859_p * v775_data)) + (v1005_p * v921_data)) + (v1151_p * v1067_data)) + (v1297_p * v1213_data)) + (v1443_p * v1359_data)) + (v1589_p * v1505_data)) + (v1735_p * v1651_data)) + (v1881_p * v1797_data)) + (v2027_p * v1943_data)) + (v2173_p * v2089_data)) + (v2319_p * v2235_data)) + (v2465_p * v2381_data)) + (v2611_p * v2527_data)) + (v2757_p * v2673_data)) + (v2903_p * v2819_data)) + (v3049_p * v2965_data)) + (v3195_p * v3111_data)) + (v3341_p * v3257_data);
          double v3363_p = ((((((((((((((((((((((v129_p * v50_data) + (v275_p * v197_data)) + (v421_p * v343_data)) + (v567_p * v489_data)) + (v713_p * v635_data)) + (v859_p * v781_data)) + (v1005_p * v927_data)) + (v1151_p * v1073_data)) + (v1297_p * v1219_data)) + (v1443_p * v1365_data)) + (v1589_p * v1511_data)) + (v1735_p * v1657_data)) + (v1881_p * v1803_data)) + (v2027_p * v1949_data)) + (v2173_p * v2095_data)) + (v2319_p * v2241_data)) + (v2465_p * v2387_data)) + (v2611_p * v2533_data)) + (v2757_p * v2679_data)) + (v2903_p * v2825_data)) + (v3049_p * v2971_data)) + (v3195_p * v3117_data)) + (v3341_p * v3263_data);
          double v3369_p = ((((((((((((((((((((((v129_p * v56_data) + (v275_p * v203_data)) + (v421_p * v349_data)) + (v567_p * v495_data)) + (v713_p * v641_data)) + (v859_p * v787_data)) + (v1005_p * v933_data)) + (v1151_p * v1079_data)) + (v1297_p * v1225_data)) + (v1443_p * v1371_data)) + (v1589_p * v1517_data)) + (v1735_p * v1663_data)) + (v1881_p * v1809_data)) + (v2027_p * v1955_data)) + (v2173_p * v2101_data)) + (v2319_p * v2247_data)) + (v2465_p * v2393_data)) + (v2611_p * v2539_data)) + (v2757_p * v2685_data)) + (v2903_p * v2831_data)) + (v3049_p * v2977_data)) + (v3195_p * v3123_data)) + (v3341_p * v3269_data);
          double v3375_p = ((((((((((((((((((((((v129_p * v62_data) + (v275_p * v209_data)) + (v421_p * v355_data)) + (v567_p * v501_data)) + (v713_p * v647_data)) + (v859_p * v793_data)) + (v1005_p * v939_data)) + (v1151_p * v1085_data)) + (v1297_p * v1231_data)) + (v1443_p * v1377_data)) + (v1589_p * v1523_data)) + (v1735_p * v1669_data)) + (v1881_p * v1815_data)) + (v2027_p * v1961_data)) + (v2173_p * v2107_data)) + (v2319_p * v2253_data)) + (v2465_p * v2399_data)) + (v2611_p * v2545_data)) + (v2757_p * v2691_data)) + (v2903_p * v2837_data)) + (v3049_p * v2983_data)) + (v3195_p * v3129_data)) + (v3341_p * v3275_data);
          double v3381_p = ((((((((((((((((((((((v129_p * v68_data) + (v275_p * v215_data)) + (v421_p * v361_data)) + (v567_p * v507_data)) + (v713_p * v653_data)) + (v859_p * v799_data)) + (v1005_p * v945_data)) + (v1151_p * v1091_data)) + (v1297_p * v1237_data)) + (v1443_p * v1383_data)) + (v1589_p * v1529_data)) + (v1735_p * v1675_data)) + (v1881_p * v1821_data)) + (v2027_p * v1967_data)) + (v2173_p * v2113_data)) + (v2319_p * v2259_data)) + (v2465_p * v2405_data)) + (v2611_p * v2551_data)) + (v2757_p * v2697_data)) + (v2903_p * v2843_data)) + (v3049_p * v2989_data)) + (v3195_p * v3135_data)) + (v3341_p * v3281_data);
          double v3387_p = ((((((((((((((((((((((v129_p * v74_data) + (v275_p * v221_data)) + (v421_p * v367_data)) + (v567_p * v513_data)) + (v713_p * v659_data)) + (v859_p * v805_data)) + (v1005_p * v951_data)) + (v1151_p * v1097_data)) + (v1297_p * v1243_data)) + (v1443_p * v1389_data)) + (v1589_p * v1535_data)) + (v1735_p * v1681_data)) + (v1881_p * v1827_data)) + (v2027_p * v1973_data)) + (v2173_p * v2119_data)) + (v2319_p * v2265_data)) + (v2465_p * v2411_data)) + (v2611_p * v2557_data)) + (v2757_p * v2703_data)) + (v2903_p * v2849_data)) + (v3049_p * v2995_data)) + (v3195_p * v3141_data)) + (v3341_p * v3287_data);
          tensorforge::VectorT<double, 2> v3389_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1472 + v28_a)];
          double v3390_p = v3389_data[0];
          double v3391_data = s0[23];
          ir0[0] = ((((((((((((((((((((((((v31_p * v32_data) + (v178_p * v179_data)) + (v324_p * v325_data)) + (v470_p * v471_data)) + (v616_p * v617_data)) + (v762_p * v763_data)) + (v908_p * v909_data)) + (v1054_p * v1055_data)) + (v1200_p * v1201_data)) + (v1346_p * v1347_data)) + (v1492_p * v1493_data)) + (v1638_p * v1639_data)) + (v1784_p * v1785_data)) + (v1930_p * v1931_data)) + (v2076_p * v2077_data)) + (v2222_p * v2223_data)) + (v2368_p * v2369_data)) + (v2514_p * v2515_data)) + (v2660_p * v2661_data)) + (v2806_p * v2807_data)) + (v2952_p * v2953_data)) + (v3098_p * v3099_data)) + (v3244_p * v3245_data)) + (v3390_p * v3391_data));
          double v3397_data = s0[47];
          ir0[3] = (v3254_p + (v3390_p * v3397_data));
          double v3403_data = s0[71];
          ir0[6] = (v3260_p + (v3390_p * v3403_data));
          double v3409_data = s0[95];
          ir0[9] = (v3266_p + (v3390_p * v3409_data));
          double v3415_data = s0[119];
          ir0[12] = (v3272_p + (v3390_p * v3415_data));
          double v3421_data = s0[143];
          ir0[15] = (v3278_p + (v3390_p * v3421_data));
          double v3427_data = s0[167];
          ir0[18] = (v3284_p + (v3390_p * v3427_data));
          double v3433_data = s0[191];
          ir0[21] = (v3290_p + (v3390_p * v3433_data));
          double v3438_p = v3389_data[1];
          ir0[1] = (v3296_p + (v3438_p * v3391_data));
          ir0[4] = (v3302_p + (v3438_p * v3397_data));
          ir0[7] = (v3308_p + (v3438_p * v3403_data));
          ir0[10] = (v3314_p + (v3438_p * v3409_data));
          ir0[13] = (v3320_p + (v3438_p * v3415_data));
          ir0[16] = (v3326_p + (v3438_p * v3421_data));
          ir0[19] = (v3332_p + (v3438_p * v3427_data));
          ir0[22] = (v3338_p + (v3438_p * v3433_data));
          tensorforge::VectorT<double, 2> v3486_data = *(tensorforge::VectorT<double, 2>*)&glb_m1[(1472 + v126_a)];
          double v3487_p = v3486_data[0];
          ir0[2] = (v3345_p + (v3487_p * v3391_data));
          ir0[5] = (v3351_p + (v3487_p * v3397_data));
          ir0[8] = (v3357_p + (v3487_p * v3403_data));
          ir0[11] = (v3363_p + (v3487_p * v3409_data));
          ir0[14] = (v3369_p + (v3487_p * v3415_data));
          ir0[17] = (v3375_p + (v3487_p * v3421_data));
          ir0[20] = (v3381_p + (v3487_p * v3427_data));
          ir0[23] = (v3387_p + (v3487_p * v3433_data));
          // r0 = ir0
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v3543_i0 = 0; v3543_i0 < 3; ++v3543_i0) {
            int32_t v3549_lead = v27_lead + (v3543_i0 * 16);
            #pragma unroll
            for (int32_t v3544_i1 = 0; v3544_i1 < 8; ++v3544_i1) {
              double v3547_data = ir0[(v3543_i0 + (v3544_i1 * 3))];
              glb_m0[(v3549_lead + (v3544_i1 * 48))] = v3547_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v3567_i = 0; v3567_i < 12; ++v3567_i) {
          int32_t v3569_a = threadIdx.x + (v3567_i * 16);
          __pipeline_memcpy_async(&s0[v3569_a], &wrap_glb_m2[v3569_a], 8);
        }
        __pipeline_commit();
        v3557_flagWord = v3558_flagWordNext;
        v3558_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

