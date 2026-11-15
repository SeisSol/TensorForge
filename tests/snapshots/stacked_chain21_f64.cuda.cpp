// === base name ===
kernel_a7794e32d2623609

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a7794e32d2623609 = {{16, 8, 1}, 16, 56, 1, 8, 33792, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a7794e32d2623609(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a7794e32d2623609(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double * m2, double ** m3, size_t m3_extraOffset, const double * m4, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a7794e32d2623609(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a7794e32d2623609, block.x * block.y * block.z, 4224 * sizeof(double));
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
  config.sharedMemBytes = 4224 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a7794e32d2623609(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double * m2, double ** m3, size_t m3_extraOffset, const double * m4, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a7794e32d2623609(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_a7794e32d2623609, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const double *const * m0Arg = (const double *const *)m0;
  const double *const * m1Arg = (const double *const *)m1;
  double *const * m3Arg = (double *const *)m3;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_a7794e32d2623609<<<grid,block,config.sharedMemBytes,stream>>>(m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2, m3Arg, m3_extraOffset, m4, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_a7794e32d2623609(const double *const * __restrict__ m0, size_t m0_extraOffset, const double *const * __restrict__ m1, size_t m1_extraOffset, const double * m2, double *const * __restrict__ m3, size_t m3_extraOffset, const double * m4, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (56 active) x 8 per block = block 16x8x1, 33792 B shared, occupancy grid
    // operands:
    //   m0 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m2 21×56(21×56) {0..21}×{0..56} none ordered
    //   m3 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m4 56×21(56×21) {0..56}×{0..21} none ordered runs-packed
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = m2[i,k] × t0[k,j]
    //   m3[i,j] += m4[i,k] × t1[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":56,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":4224}],"shared_bytes":33792,"shared_elements":4224,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[56,9]],"name":"m0","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"pointer_based","alias":"F","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"fMrT","bbox":[[0,0],[21,56]],"name":"m2","ordered":true,"parts":1,"shape":[21,56],"variant":false},{"addressing":"pointer_based","alias":"R","bbox":[[0,0],[56,9]],"name":"m3","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"none","alias":"rDivM","bbox":[[0,0],[56,21]],"name":"m4","ordered":true,"parts":1,"shape":[56,21],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[21,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[21,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[21,56]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[21,56]},{"addressing":"strided","bbox":[[0,0],[56,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,21]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[56,21]},{"addressing":"strided","bbox":[[0,0],[21,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[21,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[528 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const double *const __restrict__ glb_m2 = &m2[0];
      const double *const __restrict__ glb_m4 = &m4[0];
      double * __restrict__ s0 = &localShrMem0[0];
      double * __restrict__ s1 = &localShrMem0[0];
      double * __restrict__ s2 = &localShrMem0[0];
      double r0[36]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v11396_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v11396_flagSecondAhead < numElements0) ? v11396_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1][0 + m0_extraOffset];
      int32_t v11406_lead = threadIdx.x % 16;
      int32_t v11408_lead = v11406_lead + 48_i32;
      bool v11409_g = v11406_lead < 8;
      bool v11461_g = allowed_peel && v11409_g;
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v11431_i0 = 0; v11431_i0 < 3; ++v11431_i0) {
        int32_t v11433_lead = v11406_lead + (v11431_i0 * 16);
        #pragma unroll
        for (int32_t v11434_i1 = 0; v11434_i1 < 9; ++v11434_i1) {
          double v11437_data = allowed_peel ? (__ldcg(&peel_glb_m0[(v11433_lead + (v11434_i1 * 56))])) : (0.0);
          r0[(v11431_i0 + (v11434_i1 * 4))] = v11437_data;
        }
      }
      #pragma unroll
      for (int32_t v11440_i1 = 0; v11440_i1 < 9; ++v11440_i1) {
        double v11443_data = v11461_g ? (__ldcg(&peel_glb_m0[(v11408_lead + (v11440_i1 * 56))])) : (0.0);
        r0[(3 + (v11440_i1 * 4))] = v11443_data;
      }
      const double *const __restrict__ peel_glb_m1 = &m1[v3_batchId1][0 + m1_extraOffset];
      bool v11465_g = threadIdx.x < 1;
      // s0 = load{g>s}(peel_glb_m1[0, 1])
      #pragma unroll
      for (int32_t v11468_i = 0; v11468_i < 5; ++v11468_i) {
        int32_t v11470_a = threadIdx.x + (v11468_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v11470_a], &peel_glb_m1[v11470_a], 8);
        }
      }
      if (v11465_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &peel_glb_m1[0 + 0 + 1 * threadIdx.x + 80], 8);
        }
      }
      __pipeline_commit();
      uint32_t v11400_flagWord = flagWordFirst;
      uint32_t v11401_flagWordNext = flagWordSecond;
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const double *const __restrict__ wrap_glb_m0 = &m0[v15_batchId1][0 + m0_extraOffset];
        const double *const __restrict__ wrap_glb_m1 = &m1[v15_batchId1][0 + m1_extraOffset];
        size_t v11484_flagAheadAhead = v15_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v11484_flagAheadAhead < numElements0) ? v11484_flagAheadAhead : v15_batchId1)];
        const bool allowed = static_cast<bool>(v11400_flagWord);
        const bool allowed_next = static_cast<bool>(v11401_flagWordNext);
        if (allowed) {
          double *const __restrict__ glb_m3 = &m3[v12_batchId0][0 + m3_extraOffset];
          double r3[36]{};
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          double r1[36]{};
          // r1 = +(r0 * s0) + None
          // [(0, 56), (0, 9)] [(0, 9)]
          double v52_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v53_data = s0[0];
          double v58_data = s0[9];
          double v63_data = s0[18];
          double v68_data = s0[27];
          double v73_data = s0[36];
          double v78_data = s0[45];
          double v83_data = s0[54];
          double v88_data = s0[63];
          double v93_data = s0[72];
          double v97_data = r0[1];
          double v142_data = r0[2];
          double v187_data = r0[3];
          double v232_data = r0[4];
          double v233_data = s0[1];
          double v238_data = s0[10];
          double v243_data = s0[19];
          double v248_data = s0[28];
          double v253_data = s0[37];
          double v258_data = s0[46];
          double v263_data = s0[55];
          double v268_data = s0[64];
          double v273_data = s0[73];
          double v277_data = r0[5];
          double v322_data = r0[6];
          double v367_data = r0[7];
          double v412_data = r0[8];
          double v413_data = s0[2];
          double v418_data = s0[11];
          double v423_data = s0[20];
          double v428_data = s0[29];
          double v433_data = s0[38];
          double v438_data = s0[47];
          double v443_data = s0[56];
          double v448_data = s0[65];
          double v453_data = s0[74];
          double v457_data = r0[9];
          double v502_data = r0[10];
          double v547_data = r0[11];
          double v592_data = r0[12];
          double v593_data = s0[3];
          double v598_data = s0[12];
          double v603_data = s0[21];
          double v608_data = s0[30];
          double v613_data = s0[39];
          double v618_data = s0[48];
          double v623_data = s0[57];
          double v628_data = s0[66];
          double v633_data = s0[75];
          double v637_data = r0[13];
          double v682_data = r0[14];
          double v727_data = r0[15];
          double v772_data = r0[16];
          double v773_data = s0[4];
          double v778_data = s0[13];
          double v783_data = s0[22];
          double v788_data = s0[31];
          double v793_data = s0[40];
          double v798_data = s0[49];
          double v803_data = s0[58];
          double v808_data = s0[67];
          double v813_data = s0[76];
          double v817_data = r0[17];
          double v862_data = r0[18];
          double v907_data = r0[19];
          double v952_data = r0[20];
          double v953_data = s0[5];
          double v958_data = s0[14];
          double v963_data = s0[23];
          double v968_data = s0[32];
          double v973_data = s0[41];
          double v978_data = s0[50];
          double v983_data = s0[59];
          double v988_data = s0[68];
          double v993_data = s0[77];
          double v997_data = r0[21];
          double v1042_data = r0[22];
          double v1087_data = r0[23];
          double v1132_data = r0[24];
          double v1133_data = s0[6];
          double v1138_data = s0[15];
          double v1143_data = s0[24];
          double v1148_data = s0[33];
          double v1153_data = s0[42];
          double v1158_data = s0[51];
          double v1163_data = s0[60];
          double v1168_data = s0[69];
          double v1173_data = s0[78];
          double v1177_data = r0[25];
          double v1222_data = r0[26];
          double v1267_data = r0[27];
          double v1312_data = r0[28];
          double v1313_data = s0[7];
          double v1318_data = s0[16];
          double v1321_p = (((((((v52_data * v58_data) + (v232_data * v238_data)) + (v412_data * v418_data)) + (v592_data * v598_data)) + (v772_data * v778_data)) + (v952_data * v958_data)) + (v1132_data * v1138_data)) + (v1312_data * v1318_data);
          double v1323_data = s0[25];
          double v1326_p = (((((((v52_data * v63_data) + (v232_data * v243_data)) + (v412_data * v423_data)) + (v592_data * v603_data)) + (v772_data * v783_data)) + (v952_data * v963_data)) + (v1132_data * v1143_data)) + (v1312_data * v1323_data);
          double v1328_data = s0[34];
          double v1331_p = (((((((v52_data * v68_data) + (v232_data * v248_data)) + (v412_data * v428_data)) + (v592_data * v608_data)) + (v772_data * v788_data)) + (v952_data * v968_data)) + (v1132_data * v1148_data)) + (v1312_data * v1328_data);
          double v1333_data = s0[43];
          double v1336_p = (((((((v52_data * v73_data) + (v232_data * v253_data)) + (v412_data * v433_data)) + (v592_data * v613_data)) + (v772_data * v793_data)) + (v952_data * v973_data)) + (v1132_data * v1153_data)) + (v1312_data * v1333_data);
          double v1338_data = s0[52];
          double v1341_p = (((((((v52_data * v78_data) + (v232_data * v258_data)) + (v412_data * v438_data)) + (v592_data * v618_data)) + (v772_data * v798_data)) + (v952_data * v978_data)) + (v1132_data * v1158_data)) + (v1312_data * v1338_data);
          double v1343_data = s0[61];
          double v1346_p = (((((((v52_data * v83_data) + (v232_data * v263_data)) + (v412_data * v443_data)) + (v592_data * v623_data)) + (v772_data * v803_data)) + (v952_data * v983_data)) + (v1132_data * v1163_data)) + (v1312_data * v1343_data);
          double v1348_data = s0[70];
          double v1351_p = (((((((v52_data * v88_data) + (v232_data * v268_data)) + (v412_data * v448_data)) + (v592_data * v628_data)) + (v772_data * v808_data)) + (v952_data * v988_data)) + (v1132_data * v1168_data)) + (v1312_data * v1348_data);
          double v1353_data = s0[79];
          double v1356_p = (((((((v52_data * v93_data) + (v232_data * v273_data)) + (v412_data * v453_data)) + (v592_data * v633_data)) + (v772_data * v813_data)) + (v952_data * v993_data)) + (v1132_data * v1173_data)) + (v1312_data * v1353_data);
          double v1357_data = r0[29];
          double v1361_p = (((((((v97_data * v53_data) + (v277_data * v233_data)) + (v457_data * v413_data)) + (v637_data * v593_data)) + (v817_data * v773_data)) + (v997_data * v953_data)) + (v1177_data * v1133_data)) + (v1357_data * v1313_data);
          double v1366_p = (((((((v97_data * v58_data) + (v277_data * v238_data)) + (v457_data * v418_data)) + (v637_data * v598_data)) + (v817_data * v778_data)) + (v997_data * v958_data)) + (v1177_data * v1138_data)) + (v1357_data * v1318_data);
          double v1371_p = (((((((v97_data * v63_data) + (v277_data * v243_data)) + (v457_data * v423_data)) + (v637_data * v603_data)) + (v817_data * v783_data)) + (v997_data * v963_data)) + (v1177_data * v1143_data)) + (v1357_data * v1323_data);
          double v1376_p = (((((((v97_data * v68_data) + (v277_data * v248_data)) + (v457_data * v428_data)) + (v637_data * v608_data)) + (v817_data * v788_data)) + (v997_data * v968_data)) + (v1177_data * v1148_data)) + (v1357_data * v1328_data);
          double v1381_p = (((((((v97_data * v73_data) + (v277_data * v253_data)) + (v457_data * v433_data)) + (v637_data * v613_data)) + (v817_data * v793_data)) + (v997_data * v973_data)) + (v1177_data * v1153_data)) + (v1357_data * v1333_data);
          double v1386_p = (((((((v97_data * v78_data) + (v277_data * v258_data)) + (v457_data * v438_data)) + (v637_data * v618_data)) + (v817_data * v798_data)) + (v997_data * v978_data)) + (v1177_data * v1158_data)) + (v1357_data * v1338_data);
          double v1391_p = (((((((v97_data * v83_data) + (v277_data * v263_data)) + (v457_data * v443_data)) + (v637_data * v623_data)) + (v817_data * v803_data)) + (v997_data * v983_data)) + (v1177_data * v1163_data)) + (v1357_data * v1343_data);
          double v1396_p = (((((((v97_data * v88_data) + (v277_data * v268_data)) + (v457_data * v448_data)) + (v637_data * v628_data)) + (v817_data * v808_data)) + (v997_data * v988_data)) + (v1177_data * v1168_data)) + (v1357_data * v1348_data);
          double v1401_p = (((((((v97_data * v93_data) + (v277_data * v273_data)) + (v457_data * v453_data)) + (v637_data * v633_data)) + (v817_data * v813_data)) + (v997_data * v993_data)) + (v1177_data * v1173_data)) + (v1357_data * v1353_data);
          double v1402_data = r0[30];
          double v1406_p = (((((((v142_data * v53_data) + (v322_data * v233_data)) + (v502_data * v413_data)) + (v682_data * v593_data)) + (v862_data * v773_data)) + (v1042_data * v953_data)) + (v1222_data * v1133_data)) + (v1402_data * v1313_data);
          double v1411_p = (((((((v142_data * v58_data) + (v322_data * v238_data)) + (v502_data * v418_data)) + (v682_data * v598_data)) + (v862_data * v778_data)) + (v1042_data * v958_data)) + (v1222_data * v1138_data)) + (v1402_data * v1318_data);
          double v1416_p = (((((((v142_data * v63_data) + (v322_data * v243_data)) + (v502_data * v423_data)) + (v682_data * v603_data)) + (v862_data * v783_data)) + (v1042_data * v963_data)) + (v1222_data * v1143_data)) + (v1402_data * v1323_data);
          double v1421_p = (((((((v142_data * v68_data) + (v322_data * v248_data)) + (v502_data * v428_data)) + (v682_data * v608_data)) + (v862_data * v788_data)) + (v1042_data * v968_data)) + (v1222_data * v1148_data)) + (v1402_data * v1328_data);
          double v1426_p = (((((((v142_data * v73_data) + (v322_data * v253_data)) + (v502_data * v433_data)) + (v682_data * v613_data)) + (v862_data * v793_data)) + (v1042_data * v973_data)) + (v1222_data * v1153_data)) + (v1402_data * v1333_data);
          double v1431_p = (((((((v142_data * v78_data) + (v322_data * v258_data)) + (v502_data * v438_data)) + (v682_data * v618_data)) + (v862_data * v798_data)) + (v1042_data * v978_data)) + (v1222_data * v1158_data)) + (v1402_data * v1338_data);
          double v1436_p = (((((((v142_data * v83_data) + (v322_data * v263_data)) + (v502_data * v443_data)) + (v682_data * v623_data)) + (v862_data * v803_data)) + (v1042_data * v983_data)) + (v1222_data * v1163_data)) + (v1402_data * v1343_data);
          double v1441_p = (((((((v142_data * v88_data) + (v322_data * v268_data)) + (v502_data * v448_data)) + (v682_data * v628_data)) + (v862_data * v808_data)) + (v1042_data * v988_data)) + (v1222_data * v1168_data)) + (v1402_data * v1348_data);
          double v1446_p = (((((((v142_data * v93_data) + (v322_data * v273_data)) + (v502_data * v453_data)) + (v682_data * v633_data)) + (v862_data * v813_data)) + (v1042_data * v993_data)) + (v1222_data * v1173_data)) + (v1402_data * v1353_data);
          double v1447_data = r0[31];
          double v1451_p = (((((((v187_data * v53_data) + (v367_data * v233_data)) + (v547_data * v413_data)) + (v727_data * v593_data)) + (v907_data * v773_data)) + (v1087_data * v953_data)) + (v1267_data * v1133_data)) + (v1447_data * v1313_data);
          double v1456_p = (((((((v187_data * v58_data) + (v367_data * v238_data)) + (v547_data * v418_data)) + (v727_data * v598_data)) + (v907_data * v778_data)) + (v1087_data * v958_data)) + (v1267_data * v1138_data)) + (v1447_data * v1318_data);
          double v1461_p = (((((((v187_data * v63_data) + (v367_data * v243_data)) + (v547_data * v423_data)) + (v727_data * v603_data)) + (v907_data * v783_data)) + (v1087_data * v963_data)) + (v1267_data * v1143_data)) + (v1447_data * v1323_data);
          double v1466_p = (((((((v187_data * v68_data) + (v367_data * v248_data)) + (v547_data * v428_data)) + (v727_data * v608_data)) + (v907_data * v788_data)) + (v1087_data * v968_data)) + (v1267_data * v1148_data)) + (v1447_data * v1328_data);
          double v1471_p = (((((((v187_data * v73_data) + (v367_data * v253_data)) + (v547_data * v433_data)) + (v727_data * v613_data)) + (v907_data * v793_data)) + (v1087_data * v973_data)) + (v1267_data * v1153_data)) + (v1447_data * v1333_data);
          double v1476_p = (((((((v187_data * v78_data) + (v367_data * v258_data)) + (v547_data * v438_data)) + (v727_data * v618_data)) + (v907_data * v798_data)) + (v1087_data * v978_data)) + (v1267_data * v1158_data)) + (v1447_data * v1338_data);
          double v1481_p = (((((((v187_data * v83_data) + (v367_data * v263_data)) + (v547_data * v443_data)) + (v727_data * v623_data)) + (v907_data * v803_data)) + (v1087_data * v983_data)) + (v1267_data * v1163_data)) + (v1447_data * v1343_data);
          double v1486_p = (((((((v187_data * v88_data) + (v367_data * v268_data)) + (v547_data * v448_data)) + (v727_data * v628_data)) + (v907_data * v808_data)) + (v1087_data * v988_data)) + (v1267_data * v1168_data)) + (v1447_data * v1348_data);
          double v1491_p = (((((((v187_data * v93_data) + (v367_data * v273_data)) + (v547_data * v453_data)) + (v727_data * v633_data)) + (v907_data * v813_data)) + (v1087_data * v993_data)) + (v1267_data * v1173_data)) + (v1447_data * v1353_data);
          double v1492_data = r0[32];
          double v1493_data = s0[8];
          r1[0] = (((((((((v52_data * v53_data) + (v232_data * v233_data)) + (v412_data * v413_data)) + (v592_data * v593_data)) + (v772_data * v773_data)) + (v952_data * v953_data)) + (v1132_data * v1133_data)) + (v1312_data * v1313_data)) + (v1492_data * v1493_data));
          double v1498_data = s0[17];
          r1[4] = (v1321_p + (v1492_data * v1498_data));
          double v1503_data = s0[26];
          r1[8] = (v1326_p + (v1492_data * v1503_data));
          double v1508_data = s0[35];
          r1[12] = (v1331_p + (v1492_data * v1508_data));
          double v1513_data = s0[44];
          r1[16] = (v1336_p + (v1492_data * v1513_data));
          double v1518_data = s0[53];
          r1[20] = (v1341_p + (v1492_data * v1518_data));
          double v1523_data = s0[62];
          r1[24] = (v1346_p + (v1492_data * v1523_data));
          double v1528_data = s0[71];
          r1[28] = (v1351_p + (v1492_data * v1528_data));
          double v1533_data = s0[80];
          r1[32] = (v1356_p + (v1492_data * v1533_data));
          double v1537_data = r0[33];
          r1[1] = (v1361_p + (v1537_data * v1493_data));
          r1[5] = (v1366_p + (v1537_data * v1498_data));
          r1[9] = (v1371_p + (v1537_data * v1503_data));
          r1[13] = (v1376_p + (v1537_data * v1508_data));
          r1[17] = (v1381_p + (v1537_data * v1513_data));
          r1[21] = (v1386_p + (v1537_data * v1518_data));
          r1[25] = (v1391_p + (v1537_data * v1523_data));
          r1[29] = (v1396_p + (v1537_data * v1528_data));
          r1[33] = (v1401_p + (v1537_data * v1533_data));
          double v1582_data = r0[34];
          r1[2] = (v1406_p + (v1582_data * v1493_data));
          r1[6] = (v1411_p + (v1582_data * v1498_data));
          r1[10] = (v1416_p + (v1582_data * v1503_data));
          r1[14] = (v1421_p + (v1582_data * v1508_data));
          r1[18] = (v1426_p + (v1582_data * v1513_data));
          r1[22] = (v1431_p + (v1582_data * v1518_data));
          r1[26] = (v1436_p + (v1582_data * v1523_data));
          r1[30] = (v1441_p + (v1582_data * v1528_data));
          r1[34] = (v1446_p + (v1582_data * v1533_data));
          double v1627_data = r0[35];
          r1[3] = (v1451_p + (v1627_data * v1493_data));
          r1[7] = (v1456_p + (v1627_data * v1498_data));
          r1[11] = (v1461_p + (v1627_data * v1503_data));
          r1[15] = (v1466_p + (v1627_data * v1508_data));
          r1[19] = (v1471_p + (v1627_data * v1513_data));
          r1[23] = (v1476_p + (v1627_data * v1518_data));
          r1[27] = (v1481_p + (v1627_data * v1523_data));
          r1[31] = (v1486_p + (v1627_data * v1528_data));
          r1[35] = (v1491_p + (v1627_data * v1533_data));
          // r1 = 0 past (0, 56)
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v1673_i0 = 0; v1673_i0 < 3; ++v1673_i0) {
            int32_t v1679_lead = v11406_lead + (v1673_i0 * 16);
            #pragma unroll
            for (int32_t v1674_i1 = 0; v1674_i1 < 9; ++v1674_i1) {
              double v1677_data = r1[(v1673_i0 + (v1674_i1 * 4))];
              s1[(v1679_lead + (v1674_i1 * 56))] = v1677_data;
            }
          }
          if (v11409_g) {
            #pragma unroll
            for (int32_t v1683_i1 = 0; v1683_i1 < 9; ++v1683_i1) {
              double v1686_data = r1[(3 + (v1683_i1 * 4))];
              s1[(v11408_lead + (v1683_i1 * 56))] = v1686_data;
            }
          }
          double r2[18]{};
          // r2 = +(glb_m2 * s1) + None
          // [(0, 21), (0, 9)] [(0, 56)]
          int32_t v1692_a = v11406_lead * 2;
          tensorforge::VectorT<double, 2> v1694_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[v1692_a];
          double v1695_p = v1694_data[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v1696_data = s1[0];
          double v1702_data = s1[56];
          double v1708_data = s1[112];
          double v1714_data = s1[168];
          double v1720_data = s1[224];
          double v1726_data = s1[280];
          double v1732_data = s1[336];
          double v1738_data = s1[392];
          double v1744_data = s1[448];
          double v1749_p = v1694_data[1];
          tensorforge::VectorT<double, 2> v1803_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(32 + v1692_a)];
          double v1804_p = v1803_data[0];
          double v1805_data = s1[1];
          double v1811_data = s1[57];
          double v1817_data = s1[113];
          double v1823_data = s1[169];
          double v1829_data = s1[225];
          double v1835_data = s1[281];
          double v1841_data = s1[337];
          double v1847_data = s1[393];
          double v1853_data = s1[449];
          double v1858_p = v1803_data[1];
          tensorforge::VectorT<double, 2> v1912_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(64 + v1692_a)];
          double v1913_p = v1912_data[0];
          double v1914_data = s1[2];
          double v1920_data = s1[58];
          double v1926_data = s1[114];
          double v1932_data = s1[170];
          double v1938_data = s1[226];
          double v1944_data = s1[282];
          double v1950_data = s1[338];
          double v1956_data = s1[394];
          double v1962_data = s1[450];
          double v1967_p = v1912_data[1];
          tensorforge::VectorT<double, 2> v2021_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(96 + v1692_a)];
          double v2022_p = v2021_data[0];
          double v2023_data = s1[3];
          double v2029_data = s1[59];
          double v2035_data = s1[115];
          double v2041_data = s1[171];
          double v2047_data = s1[227];
          double v2053_data = s1[283];
          double v2059_data = s1[339];
          double v2065_data = s1[395];
          double v2071_data = s1[451];
          double v2076_p = v2021_data[1];
          tensorforge::VectorT<double, 2> v2130_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(128 + v1692_a)];
          double v2131_p = v2130_data[0];
          double v2132_data = s1[4];
          double v2138_data = s1[60];
          double v2144_data = s1[116];
          double v2150_data = s1[172];
          double v2156_data = s1[228];
          double v2162_data = s1[284];
          double v2168_data = s1[340];
          double v2174_data = s1[396];
          double v2180_data = s1[452];
          double v2185_p = v2130_data[1];
          tensorforge::VectorT<double, 2> v2239_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(160 + v1692_a)];
          double v2240_p = v2239_data[0];
          double v2241_data = s1[5];
          double v2247_data = s1[61];
          double v2253_data = s1[117];
          double v2259_data = s1[173];
          double v2265_data = s1[229];
          double v2271_data = s1[285];
          double v2277_data = s1[341];
          double v2283_data = s1[397];
          double v2289_data = s1[453];
          double v2294_p = v2239_data[1];
          tensorforge::VectorT<double, 2> v2348_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(192 + v1692_a)];
          double v2349_p = v2348_data[0];
          double v2350_data = s1[6];
          double v2356_data = s1[62];
          double v2362_data = s1[118];
          double v2368_data = s1[174];
          double v2374_data = s1[230];
          double v2380_data = s1[286];
          double v2386_data = s1[342];
          double v2392_data = s1[398];
          double v2398_data = s1[454];
          double v2403_p = v2348_data[1];
          tensorforge::VectorT<double, 2> v2457_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(224 + v1692_a)];
          double v2458_p = v2457_data[0];
          double v2459_data = s1[7];
          double v2465_data = s1[63];
          double v2471_data = s1[119];
          double v2477_data = s1[175];
          double v2483_data = s1[231];
          double v2489_data = s1[287];
          double v2495_data = s1[343];
          double v2501_data = s1[399];
          double v2507_data = s1[455];
          double v2512_p = v2457_data[1];
          tensorforge::VectorT<double, 2> v2566_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(256 + v1692_a)];
          double v2567_p = v2566_data[0];
          double v2568_data = s1[8];
          double v2574_data = s1[64];
          double v2580_data = s1[120];
          double v2586_data = s1[176];
          double v2592_data = s1[232];
          double v2598_data = s1[288];
          double v2604_data = s1[344];
          double v2610_data = s1[400];
          double v2616_data = s1[456];
          double v2621_p = v2566_data[1];
          tensorforge::VectorT<double, 2> v2675_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(288 + v1692_a)];
          double v2676_p = v2675_data[0];
          double v2677_data = s1[9];
          double v2683_data = s1[65];
          double v2689_data = s1[121];
          double v2695_data = s1[177];
          double v2701_data = s1[233];
          double v2707_data = s1[289];
          double v2713_data = s1[345];
          double v2719_data = s1[401];
          double v2725_data = s1[457];
          double v2730_p = v2675_data[1];
          tensorforge::VectorT<double, 2> v2784_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(320 + v1692_a)];
          double v2785_p = v2784_data[0];
          double v2786_data = s1[10];
          double v2792_data = s1[66];
          double v2798_data = s1[122];
          double v2804_data = s1[178];
          double v2810_data = s1[234];
          double v2816_data = s1[290];
          double v2822_data = s1[346];
          double v2828_data = s1[402];
          double v2834_data = s1[458];
          double v2839_p = v2784_data[1];
          tensorforge::VectorT<double, 2> v2893_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(352 + v1692_a)];
          double v2894_p = v2893_data[0];
          double v2895_data = s1[11];
          double v2901_data = s1[67];
          double v2907_data = s1[123];
          double v2913_data = s1[179];
          double v2919_data = s1[235];
          double v2925_data = s1[291];
          double v2931_data = s1[347];
          double v2937_data = s1[403];
          double v2943_data = s1[459];
          double v2948_p = v2893_data[1];
          tensorforge::VectorT<double, 2> v3002_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(384 + v1692_a)];
          double v3003_p = v3002_data[0];
          double v3004_data = s1[12];
          double v3010_data = s1[68];
          double v3016_data = s1[124];
          double v3022_data = s1[180];
          double v3028_data = s1[236];
          double v3034_data = s1[292];
          double v3040_data = s1[348];
          double v3046_data = s1[404];
          double v3052_data = s1[460];
          double v3057_p = v3002_data[1];
          tensorforge::VectorT<double, 2> v3111_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(416 + v1692_a)];
          double v3112_p = v3111_data[0];
          double v3113_data = s1[13];
          double v3119_data = s1[69];
          double v3125_data = s1[125];
          double v3131_data = s1[181];
          double v3137_data = s1[237];
          double v3143_data = s1[293];
          double v3149_data = s1[349];
          double v3155_data = s1[405];
          double v3161_data = s1[461];
          double v3166_p = v3111_data[1];
          tensorforge::VectorT<double, 2> v3220_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(448 + v1692_a)];
          double v3221_p = v3220_data[0];
          double v3222_data = s1[14];
          double v3228_data = s1[70];
          double v3234_data = s1[126];
          double v3240_data = s1[182];
          double v3246_data = s1[238];
          double v3252_data = s1[294];
          double v3258_data = s1[350];
          double v3264_data = s1[406];
          double v3270_data = s1[462];
          double v3275_p = v3220_data[1];
          tensorforge::VectorT<double, 2> v3329_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(480 + v1692_a)];
          double v3330_p = v3329_data[0];
          double v3331_data = s1[15];
          double v3337_data = s1[71];
          double v3343_data = s1[127];
          double v3349_data = s1[183];
          double v3355_data = s1[239];
          double v3361_data = s1[295];
          double v3367_data = s1[351];
          double v3373_data = s1[407];
          double v3379_data = s1[463];
          double v3384_p = v3329_data[1];
          tensorforge::VectorT<double, 2> v3438_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(512 + v1692_a)];
          double v3439_p = v3438_data[0];
          double v3440_data = s1[16];
          double v3446_data = s1[72];
          double v3452_data = s1[128];
          double v3458_data = s1[184];
          double v3464_data = s1[240];
          double v3470_data = s1[296];
          double v3476_data = s1[352];
          double v3482_data = s1[408];
          double v3488_data = s1[464];
          double v3493_p = v3438_data[1];
          tensorforge::VectorT<double, 2> v3547_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(544 + v1692_a)];
          double v3548_p = v3547_data[0];
          double v3549_data = s1[17];
          double v3555_data = s1[73];
          double v3561_data = s1[129];
          double v3567_data = s1[185];
          double v3573_data = s1[241];
          double v3579_data = s1[297];
          double v3585_data = s1[353];
          double v3591_data = s1[409];
          double v3597_data = s1[465];
          double v3602_p = v3547_data[1];
          tensorforge::VectorT<double, 2> v3656_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(576 + v1692_a)];
          double v3657_p = v3656_data[0];
          double v3658_data = s1[18];
          double v3664_data = s1[74];
          double v3670_data = s1[130];
          double v3676_data = s1[186];
          double v3682_data = s1[242];
          double v3688_data = s1[298];
          double v3694_data = s1[354];
          double v3700_data = s1[410];
          double v3706_data = s1[466];
          double v3711_p = v3656_data[1];
          tensorforge::VectorT<double, 2> v3765_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(608 + v1692_a)];
          double v3766_p = v3765_data[0];
          double v3767_data = s1[19];
          double v3773_data = s1[75];
          double v3779_data = s1[131];
          double v3785_data = s1[187];
          double v3791_data = s1[243];
          double v3797_data = s1[299];
          double v3803_data = s1[355];
          double v3809_data = s1[411];
          double v3815_data = s1[467];
          double v3820_p = v3765_data[1];
          tensorforge::VectorT<double, 2> v3874_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(640 + v1692_a)];
          double v3875_p = v3874_data[0];
          double v3876_data = s1[20];
          double v3882_data = s1[76];
          double v3888_data = s1[132];
          double v3894_data = s1[188];
          double v3900_data = s1[244];
          double v3906_data = s1[300];
          double v3912_data = s1[356];
          double v3918_data = s1[412];
          double v3924_data = s1[468];
          double v3929_p = v3874_data[1];
          tensorforge::VectorT<double, 2> v3983_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(672 + v1692_a)];
          double v3984_p = v3983_data[0];
          double v3985_data = s1[21];
          double v3991_data = s1[77];
          double v3997_data = s1[133];
          double v4003_data = s1[189];
          double v4009_data = s1[245];
          double v4015_data = s1[301];
          double v4021_data = s1[357];
          double v4027_data = s1[413];
          double v4033_data = s1[469];
          double v4038_p = v3983_data[1];
          tensorforge::VectorT<double, 2> v4092_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(704 + v1692_a)];
          double v4093_p = v4092_data[0];
          double v4094_data = s1[22];
          double v4100_data = s1[78];
          double v4106_data = s1[134];
          double v4112_data = s1[190];
          double v4118_data = s1[246];
          double v4124_data = s1[302];
          double v4130_data = s1[358];
          double v4136_data = s1[414];
          double v4142_data = s1[470];
          double v4147_p = v4092_data[1];
          tensorforge::VectorT<double, 2> v4201_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(736 + v1692_a)];
          double v4202_p = v4201_data[0];
          double v4203_data = s1[23];
          double v4209_data = s1[79];
          double v4215_data = s1[135];
          double v4221_data = s1[191];
          double v4227_data = s1[247];
          double v4233_data = s1[303];
          double v4239_data = s1[359];
          double v4245_data = s1[415];
          double v4251_data = s1[471];
          double v4256_p = v4201_data[1];
          tensorforge::VectorT<double, 2> v4310_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(768 + v1692_a)];
          double v4311_p = v4310_data[0];
          double v4312_data = s1[24];
          double v4318_data = s1[80];
          double v4324_data = s1[136];
          double v4330_data = s1[192];
          double v4336_data = s1[248];
          double v4342_data = s1[304];
          double v4348_data = s1[360];
          double v4354_data = s1[416];
          double v4360_data = s1[472];
          double v4365_p = v4310_data[1];
          tensorforge::VectorT<double, 2> v4419_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(800 + v1692_a)];
          double v4420_p = v4419_data[0];
          double v4421_data = s1[25];
          double v4427_data = s1[81];
          double v4433_data = s1[137];
          double v4439_data = s1[193];
          double v4445_data = s1[249];
          double v4451_data = s1[305];
          double v4457_data = s1[361];
          double v4463_data = s1[417];
          double v4469_data = s1[473];
          double v4474_p = v4419_data[1];
          tensorforge::VectorT<double, 2> v4528_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(832 + v1692_a)];
          double v4529_p = v4528_data[0];
          double v4530_data = s1[26];
          double v4536_data = s1[82];
          double v4542_data = s1[138];
          double v4548_data = s1[194];
          double v4554_data = s1[250];
          double v4560_data = s1[306];
          double v4566_data = s1[362];
          double v4572_data = s1[418];
          double v4578_data = s1[474];
          double v4583_p = v4528_data[1];
          tensorforge::VectorT<double, 2> v4637_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(864 + v1692_a)];
          double v4638_p = v4637_data[0];
          double v4639_data = s1[27];
          double v4645_data = s1[83];
          double v4651_data = s1[139];
          double v4657_data = s1[195];
          double v4663_data = s1[251];
          double v4669_data = s1[307];
          double v4675_data = s1[363];
          double v4681_data = s1[419];
          double v4687_data = s1[475];
          double v4692_p = v4637_data[1];
          tensorforge::VectorT<double, 2> v4746_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(896 + v1692_a)];
          double v4747_p = v4746_data[0];
          double v4748_data = s1[28];
          double v4754_data = s1[84];
          double v4760_data = s1[140];
          double v4766_data = s1[196];
          double v4772_data = s1[252];
          double v4778_data = s1[308];
          double v4784_data = s1[364];
          double v4790_data = s1[420];
          double v4796_data = s1[476];
          double v4801_p = v4746_data[1];
          tensorforge::VectorT<double, 2> v4855_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(928 + v1692_a)];
          double v4856_p = v4855_data[0];
          double v4857_data = s1[29];
          double v4863_data = s1[85];
          double v4869_data = s1[141];
          double v4875_data = s1[197];
          double v4881_data = s1[253];
          double v4887_data = s1[309];
          double v4893_data = s1[365];
          double v4899_data = s1[421];
          double v4905_data = s1[477];
          double v4910_p = v4855_data[1];
          tensorforge::VectorT<double, 2> v4964_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(960 + v1692_a)];
          double v4965_p = v4964_data[0];
          double v4966_data = s1[30];
          double v4972_data = s1[86];
          double v4978_data = s1[142];
          double v4984_data = s1[198];
          double v4990_data = s1[254];
          double v4996_data = s1[310];
          double v5002_data = s1[366];
          double v5008_data = s1[422];
          double v5014_data = s1[478];
          double v5019_p = v4964_data[1];
          tensorforge::VectorT<double, 2> v5073_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(992 + v1692_a)];
          double v5074_p = v5073_data[0];
          double v5075_data = s1[31];
          double v5081_data = s1[87];
          double v5087_data = s1[143];
          double v5093_data = s1[199];
          double v5099_data = s1[255];
          double v5105_data = s1[311];
          double v5111_data = s1[367];
          double v5117_data = s1[423];
          double v5123_data = s1[479];
          double v5128_p = v5073_data[1];
          tensorforge::VectorT<double, 2> v5182_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1024 + v1692_a)];
          double v5183_p = v5182_data[0];
          double v5184_data = s1[32];
          double v5190_data = s1[88];
          double v5196_data = s1[144];
          double v5202_data = s1[200];
          double v5208_data = s1[256];
          double v5214_data = s1[312];
          double v5220_data = s1[368];
          double v5226_data = s1[424];
          double v5232_data = s1[480];
          double v5237_p = v5182_data[1];
          tensorforge::VectorT<double, 2> v5291_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1056 + v1692_a)];
          double v5292_p = v5291_data[0];
          double v5293_data = s1[33];
          double v5299_data = s1[89];
          double v5305_data = s1[145];
          double v5311_data = s1[201];
          double v5317_data = s1[257];
          double v5323_data = s1[313];
          double v5329_data = s1[369];
          double v5335_data = s1[425];
          double v5341_data = s1[481];
          double v5346_p = v5291_data[1];
          tensorforge::VectorT<double, 2> v5400_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1088 + v1692_a)];
          double v5401_p = v5400_data[0];
          double v5402_data = s1[34];
          double v5408_data = s1[90];
          double v5414_data = s1[146];
          double v5420_data = s1[202];
          double v5426_data = s1[258];
          double v5432_data = s1[314];
          double v5438_data = s1[370];
          double v5444_data = s1[426];
          double v5450_data = s1[482];
          double v5455_p = v5400_data[1];
          tensorforge::VectorT<double, 2> v5509_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1120 + v1692_a)];
          double v5510_p = v5509_data[0];
          double v5511_data = s1[35];
          double v5517_data = s1[91];
          double v5523_data = s1[147];
          double v5529_data = s1[203];
          double v5535_data = s1[259];
          double v5541_data = s1[315];
          double v5547_data = s1[371];
          double v5553_data = s1[427];
          double v5559_data = s1[483];
          double v5564_p = v5509_data[1];
          tensorforge::VectorT<double, 2> v5618_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1152 + v1692_a)];
          double v5619_p = v5618_data[0];
          double v5620_data = s1[36];
          double v5626_data = s1[92];
          double v5632_data = s1[148];
          double v5638_data = s1[204];
          double v5644_data = s1[260];
          double v5650_data = s1[316];
          double v5656_data = s1[372];
          double v5662_data = s1[428];
          double v5668_data = s1[484];
          double v5673_p = v5618_data[1];
          tensorforge::VectorT<double, 2> v5727_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1184 + v1692_a)];
          double v5728_p = v5727_data[0];
          double v5729_data = s1[37];
          double v5735_data = s1[93];
          double v5741_data = s1[149];
          double v5747_data = s1[205];
          double v5753_data = s1[261];
          double v5759_data = s1[317];
          double v5765_data = s1[373];
          double v5771_data = s1[429];
          double v5777_data = s1[485];
          double v5782_p = v5727_data[1];
          tensorforge::VectorT<double, 2> v5836_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1216 + v1692_a)];
          double v5837_p = v5836_data[0];
          double v5838_data = s1[38];
          double v5844_data = s1[94];
          double v5850_data = s1[150];
          double v5856_data = s1[206];
          double v5862_data = s1[262];
          double v5868_data = s1[318];
          double v5874_data = s1[374];
          double v5880_data = s1[430];
          double v5886_data = s1[486];
          double v5891_p = v5836_data[1];
          tensorforge::VectorT<double, 2> v5945_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1248 + v1692_a)];
          double v5946_p = v5945_data[0];
          double v5947_data = s1[39];
          double v5953_data = s1[95];
          double v5959_data = s1[151];
          double v5965_data = s1[207];
          double v5971_data = s1[263];
          double v5977_data = s1[319];
          double v5983_data = s1[375];
          double v5989_data = s1[431];
          double v5995_data = s1[487];
          double v6000_p = v5945_data[1];
          tensorforge::VectorT<double, 2> v6054_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1280 + v1692_a)];
          double v6055_p = v6054_data[0];
          double v6056_data = s1[40];
          double v6062_data = s1[96];
          double v6068_data = s1[152];
          double v6074_data = s1[208];
          double v6080_data = s1[264];
          double v6086_data = s1[320];
          double v6092_data = s1[376];
          double v6098_data = s1[432];
          double v6104_data = s1[488];
          double v6109_p = v6054_data[1];
          tensorforge::VectorT<double, 2> v6163_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1312 + v1692_a)];
          double v6164_p = v6163_data[0];
          double v6165_data = s1[41];
          double v6171_data = s1[97];
          double v6177_data = s1[153];
          double v6183_data = s1[209];
          double v6189_data = s1[265];
          double v6195_data = s1[321];
          double v6201_data = s1[377];
          double v6207_data = s1[433];
          double v6213_data = s1[489];
          double v6218_p = v6163_data[1];
          tensorforge::VectorT<double, 2> v6272_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1344 + v1692_a)];
          double v6273_p = v6272_data[0];
          double v6274_data = s1[42];
          double v6280_data = s1[98];
          double v6286_data = s1[154];
          double v6292_data = s1[210];
          double v6298_data = s1[266];
          double v6304_data = s1[322];
          double v6310_data = s1[378];
          double v6316_data = s1[434];
          double v6322_data = s1[490];
          double v6327_p = v6272_data[1];
          tensorforge::VectorT<double, 2> v6381_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1376 + v1692_a)];
          double v6382_p = v6381_data[0];
          double v6383_data = s1[43];
          double v6389_data = s1[99];
          double v6395_data = s1[155];
          double v6401_data = s1[211];
          double v6407_data = s1[267];
          double v6413_data = s1[323];
          double v6419_data = s1[379];
          double v6425_data = s1[435];
          double v6431_data = s1[491];
          double v6436_p = v6381_data[1];
          tensorforge::VectorT<double, 2> v6490_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1408 + v1692_a)];
          double v6491_p = v6490_data[0];
          double v6492_data = s1[44];
          double v6498_data = s1[100];
          double v6504_data = s1[156];
          double v6510_data = s1[212];
          double v6516_data = s1[268];
          double v6522_data = s1[324];
          double v6528_data = s1[380];
          double v6534_data = s1[436];
          double v6540_data = s1[492];
          double v6545_p = v6490_data[1];
          tensorforge::VectorT<double, 2> v6599_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1440 + v1692_a)];
          double v6600_p = v6599_data[0];
          double v6601_data = s1[45];
          double v6607_data = s1[101];
          double v6613_data = s1[157];
          double v6619_data = s1[213];
          double v6625_data = s1[269];
          double v6631_data = s1[325];
          double v6637_data = s1[381];
          double v6643_data = s1[437];
          double v6649_data = s1[493];
          double v6654_p = v6599_data[1];
          tensorforge::VectorT<double, 2> v6708_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1472 + v1692_a)];
          double v6709_p = v6708_data[0];
          double v6710_data = s1[46];
          double v6716_data = s1[102];
          double v6722_data = s1[158];
          double v6728_data = s1[214];
          double v6734_data = s1[270];
          double v6740_data = s1[326];
          double v6746_data = s1[382];
          double v6752_data = s1[438];
          double v6758_data = s1[494];
          double v6763_p = v6708_data[1];
          tensorforge::VectorT<double, 2> v6817_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1504 + v1692_a)];
          double v6818_p = v6817_data[0];
          double v6819_data = s1[47];
          double v6825_data = s1[103];
          double v6831_data = s1[159];
          double v6837_data = s1[215];
          double v6843_data = s1[271];
          double v6849_data = s1[327];
          double v6855_data = s1[383];
          double v6861_data = s1[439];
          double v6867_data = s1[495];
          double v6872_p = v6817_data[1];
          tensorforge::VectorT<double, 2> v6926_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1536 + v1692_a)];
          double v6927_p = v6926_data[0];
          double v6928_data = s1[48];
          double v6934_data = s1[104];
          double v6940_data = s1[160];
          double v6946_data = s1[216];
          double v6952_data = s1[272];
          double v6958_data = s1[328];
          double v6964_data = s1[384];
          double v6970_data = s1[440];
          double v6976_data = s1[496];
          double v6981_p = v6926_data[1];
          tensorforge::VectorT<double, 2> v7035_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1568 + v1692_a)];
          double v7036_p = v7035_data[0];
          double v7037_data = s1[49];
          double v7043_data = s1[105];
          double v7049_data = s1[161];
          double v7055_data = s1[217];
          double v7061_data = s1[273];
          double v7067_data = s1[329];
          double v7073_data = s1[385];
          double v7079_data = s1[441];
          double v7085_data = s1[497];
          double v7090_p = v7035_data[1];
          tensorforge::VectorT<double, 2> v7144_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1600 + v1692_a)];
          double v7145_p = v7144_data[0];
          double v7146_data = s1[50];
          double v7152_data = s1[106];
          double v7158_data = s1[162];
          double v7164_data = s1[218];
          double v7170_data = s1[274];
          double v7176_data = s1[330];
          double v7182_data = s1[386];
          double v7188_data = s1[442];
          double v7194_data = s1[498];
          double v7199_p = v7144_data[1];
          tensorforge::VectorT<double, 2> v7253_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1632 + v1692_a)];
          double v7254_p = v7253_data[0];
          double v7255_data = s1[51];
          double v7261_data = s1[107];
          double v7267_data = s1[163];
          double v7273_data = s1[219];
          double v7279_data = s1[275];
          double v7285_data = s1[331];
          double v7291_data = s1[387];
          double v7297_data = s1[443];
          double v7303_data = s1[499];
          double v7308_p = v7253_data[1];
          tensorforge::VectorT<double, 2> v7362_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1664 + v1692_a)];
          double v7363_p = v7362_data[0];
          double v7364_data = s1[52];
          double v7370_data = s1[108];
          double v7376_data = s1[164];
          double v7382_data = s1[220];
          double v7388_data = s1[276];
          double v7394_data = s1[332];
          double v7400_data = s1[388];
          double v7406_data = s1[444];
          double v7412_data = s1[500];
          double v7417_p = v7362_data[1];
          tensorforge::VectorT<double, 2> v7471_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1696 + v1692_a)];
          double v7472_p = v7471_data[0];
          double v7473_data = s1[53];
          double v7479_data = s1[109];
          double v7485_data = s1[165];
          double v7491_data = s1[221];
          double v7497_data = s1[277];
          double v7503_data = s1[333];
          double v7509_data = s1[389];
          double v7515_data = s1[445];
          double v7521_data = s1[501];
          double v7526_p = v7471_data[1];
          tensorforge::VectorT<double, 2> v7580_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1728 + v1692_a)];
          double v7581_p = v7580_data[0];
          double v7582_data = s1[54];
          double v7588_data = s1[110];
          double v7591_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1702_data) + (v1804_p * v1811_data)) + (v1913_p * v1920_data)) + (v2022_p * v2029_data)) + (v2131_p * v2138_data)) + (v2240_p * v2247_data)) + (v2349_p * v2356_data)) + (v2458_p * v2465_data)) + (v2567_p * v2574_data)) + (v2676_p * v2683_data)) + (v2785_p * v2792_data)) + (v2894_p * v2901_data)) + (v3003_p * v3010_data)) + (v3112_p * v3119_data)) + (v3221_p * v3228_data)) + (v3330_p * v3337_data)) + (v3439_p * v3446_data)) + (v3548_p * v3555_data)) + (v3657_p * v3664_data)) + (v3766_p * v3773_data)) + (v3875_p * v3882_data)) + (v3984_p * v3991_data)) + (v4093_p * v4100_data)) + (v4202_p * v4209_data)) + (v4311_p * v4318_data)) + (v4420_p * v4427_data)) + (v4529_p * v4536_data)) + (v4638_p * v4645_data)) + (v4747_p * v4754_data)) + (v4856_p * v4863_data)) + (v4965_p * v4972_data)) + (v5074_p * v5081_data)) + (v5183_p * v5190_data)) + (v5292_p * v5299_data)) + (v5401_p * v5408_data)) + (v5510_p * v5517_data)) + (v5619_p * v5626_data)) + (v5728_p * v5735_data)) + (v5837_p * v5844_data)) + (v5946_p * v5953_data)) + (v6055_p * v6062_data)) + (v6164_p * v6171_data)) + (v6273_p * v6280_data)) + (v6382_p * v6389_data)) + (v6491_p * v6498_data)) + (v6600_p * v6607_data)) + (v6709_p * v6716_data)) + (v6818_p * v6825_data)) + (v6927_p * v6934_data)) + (v7036_p * v7043_data)) + (v7145_p * v7152_data)) + (v7254_p * v7261_data)) + (v7363_p * v7370_data)) + (v7472_p * v7479_data)) + (v7581_p * v7588_data);
          double v7594_data = s1[166];
          double v7597_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1708_data) + (v1804_p * v1817_data)) + (v1913_p * v1926_data)) + (v2022_p * v2035_data)) + (v2131_p * v2144_data)) + (v2240_p * v2253_data)) + (v2349_p * v2362_data)) + (v2458_p * v2471_data)) + (v2567_p * v2580_data)) + (v2676_p * v2689_data)) + (v2785_p * v2798_data)) + (v2894_p * v2907_data)) + (v3003_p * v3016_data)) + (v3112_p * v3125_data)) + (v3221_p * v3234_data)) + (v3330_p * v3343_data)) + (v3439_p * v3452_data)) + (v3548_p * v3561_data)) + (v3657_p * v3670_data)) + (v3766_p * v3779_data)) + (v3875_p * v3888_data)) + (v3984_p * v3997_data)) + (v4093_p * v4106_data)) + (v4202_p * v4215_data)) + (v4311_p * v4324_data)) + (v4420_p * v4433_data)) + (v4529_p * v4542_data)) + (v4638_p * v4651_data)) + (v4747_p * v4760_data)) + (v4856_p * v4869_data)) + (v4965_p * v4978_data)) + (v5074_p * v5087_data)) + (v5183_p * v5196_data)) + (v5292_p * v5305_data)) + (v5401_p * v5414_data)) + (v5510_p * v5523_data)) + (v5619_p * v5632_data)) + (v5728_p * v5741_data)) + (v5837_p * v5850_data)) + (v5946_p * v5959_data)) + (v6055_p * v6068_data)) + (v6164_p * v6177_data)) + (v6273_p * v6286_data)) + (v6382_p * v6395_data)) + (v6491_p * v6504_data)) + (v6600_p * v6613_data)) + (v6709_p * v6722_data)) + (v6818_p * v6831_data)) + (v6927_p * v6940_data)) + (v7036_p * v7049_data)) + (v7145_p * v7158_data)) + (v7254_p * v7267_data)) + (v7363_p * v7376_data)) + (v7472_p * v7485_data)) + (v7581_p * v7594_data);
          double v7600_data = s1[222];
          double v7603_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1714_data) + (v1804_p * v1823_data)) + (v1913_p * v1932_data)) + (v2022_p * v2041_data)) + (v2131_p * v2150_data)) + (v2240_p * v2259_data)) + (v2349_p * v2368_data)) + (v2458_p * v2477_data)) + (v2567_p * v2586_data)) + (v2676_p * v2695_data)) + (v2785_p * v2804_data)) + (v2894_p * v2913_data)) + (v3003_p * v3022_data)) + (v3112_p * v3131_data)) + (v3221_p * v3240_data)) + (v3330_p * v3349_data)) + (v3439_p * v3458_data)) + (v3548_p * v3567_data)) + (v3657_p * v3676_data)) + (v3766_p * v3785_data)) + (v3875_p * v3894_data)) + (v3984_p * v4003_data)) + (v4093_p * v4112_data)) + (v4202_p * v4221_data)) + (v4311_p * v4330_data)) + (v4420_p * v4439_data)) + (v4529_p * v4548_data)) + (v4638_p * v4657_data)) + (v4747_p * v4766_data)) + (v4856_p * v4875_data)) + (v4965_p * v4984_data)) + (v5074_p * v5093_data)) + (v5183_p * v5202_data)) + (v5292_p * v5311_data)) + (v5401_p * v5420_data)) + (v5510_p * v5529_data)) + (v5619_p * v5638_data)) + (v5728_p * v5747_data)) + (v5837_p * v5856_data)) + (v5946_p * v5965_data)) + (v6055_p * v6074_data)) + (v6164_p * v6183_data)) + (v6273_p * v6292_data)) + (v6382_p * v6401_data)) + (v6491_p * v6510_data)) + (v6600_p * v6619_data)) + (v6709_p * v6728_data)) + (v6818_p * v6837_data)) + (v6927_p * v6946_data)) + (v7036_p * v7055_data)) + (v7145_p * v7164_data)) + (v7254_p * v7273_data)) + (v7363_p * v7382_data)) + (v7472_p * v7491_data)) + (v7581_p * v7600_data);
          double v7606_data = s1[278];
          double v7609_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1720_data) + (v1804_p * v1829_data)) + (v1913_p * v1938_data)) + (v2022_p * v2047_data)) + (v2131_p * v2156_data)) + (v2240_p * v2265_data)) + (v2349_p * v2374_data)) + (v2458_p * v2483_data)) + (v2567_p * v2592_data)) + (v2676_p * v2701_data)) + (v2785_p * v2810_data)) + (v2894_p * v2919_data)) + (v3003_p * v3028_data)) + (v3112_p * v3137_data)) + (v3221_p * v3246_data)) + (v3330_p * v3355_data)) + (v3439_p * v3464_data)) + (v3548_p * v3573_data)) + (v3657_p * v3682_data)) + (v3766_p * v3791_data)) + (v3875_p * v3900_data)) + (v3984_p * v4009_data)) + (v4093_p * v4118_data)) + (v4202_p * v4227_data)) + (v4311_p * v4336_data)) + (v4420_p * v4445_data)) + (v4529_p * v4554_data)) + (v4638_p * v4663_data)) + (v4747_p * v4772_data)) + (v4856_p * v4881_data)) + (v4965_p * v4990_data)) + (v5074_p * v5099_data)) + (v5183_p * v5208_data)) + (v5292_p * v5317_data)) + (v5401_p * v5426_data)) + (v5510_p * v5535_data)) + (v5619_p * v5644_data)) + (v5728_p * v5753_data)) + (v5837_p * v5862_data)) + (v5946_p * v5971_data)) + (v6055_p * v6080_data)) + (v6164_p * v6189_data)) + (v6273_p * v6298_data)) + (v6382_p * v6407_data)) + (v6491_p * v6516_data)) + (v6600_p * v6625_data)) + (v6709_p * v6734_data)) + (v6818_p * v6843_data)) + (v6927_p * v6952_data)) + (v7036_p * v7061_data)) + (v7145_p * v7170_data)) + (v7254_p * v7279_data)) + (v7363_p * v7388_data)) + (v7472_p * v7497_data)) + (v7581_p * v7606_data);
          double v7612_data = s1[334];
          double v7615_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1726_data) + (v1804_p * v1835_data)) + (v1913_p * v1944_data)) + (v2022_p * v2053_data)) + (v2131_p * v2162_data)) + (v2240_p * v2271_data)) + (v2349_p * v2380_data)) + (v2458_p * v2489_data)) + (v2567_p * v2598_data)) + (v2676_p * v2707_data)) + (v2785_p * v2816_data)) + (v2894_p * v2925_data)) + (v3003_p * v3034_data)) + (v3112_p * v3143_data)) + (v3221_p * v3252_data)) + (v3330_p * v3361_data)) + (v3439_p * v3470_data)) + (v3548_p * v3579_data)) + (v3657_p * v3688_data)) + (v3766_p * v3797_data)) + (v3875_p * v3906_data)) + (v3984_p * v4015_data)) + (v4093_p * v4124_data)) + (v4202_p * v4233_data)) + (v4311_p * v4342_data)) + (v4420_p * v4451_data)) + (v4529_p * v4560_data)) + (v4638_p * v4669_data)) + (v4747_p * v4778_data)) + (v4856_p * v4887_data)) + (v4965_p * v4996_data)) + (v5074_p * v5105_data)) + (v5183_p * v5214_data)) + (v5292_p * v5323_data)) + (v5401_p * v5432_data)) + (v5510_p * v5541_data)) + (v5619_p * v5650_data)) + (v5728_p * v5759_data)) + (v5837_p * v5868_data)) + (v5946_p * v5977_data)) + (v6055_p * v6086_data)) + (v6164_p * v6195_data)) + (v6273_p * v6304_data)) + (v6382_p * v6413_data)) + (v6491_p * v6522_data)) + (v6600_p * v6631_data)) + (v6709_p * v6740_data)) + (v6818_p * v6849_data)) + (v6927_p * v6958_data)) + (v7036_p * v7067_data)) + (v7145_p * v7176_data)) + (v7254_p * v7285_data)) + (v7363_p * v7394_data)) + (v7472_p * v7503_data)) + (v7581_p * v7612_data);
          double v7618_data = s1[390];
          double v7621_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1732_data) + (v1804_p * v1841_data)) + (v1913_p * v1950_data)) + (v2022_p * v2059_data)) + (v2131_p * v2168_data)) + (v2240_p * v2277_data)) + (v2349_p * v2386_data)) + (v2458_p * v2495_data)) + (v2567_p * v2604_data)) + (v2676_p * v2713_data)) + (v2785_p * v2822_data)) + (v2894_p * v2931_data)) + (v3003_p * v3040_data)) + (v3112_p * v3149_data)) + (v3221_p * v3258_data)) + (v3330_p * v3367_data)) + (v3439_p * v3476_data)) + (v3548_p * v3585_data)) + (v3657_p * v3694_data)) + (v3766_p * v3803_data)) + (v3875_p * v3912_data)) + (v3984_p * v4021_data)) + (v4093_p * v4130_data)) + (v4202_p * v4239_data)) + (v4311_p * v4348_data)) + (v4420_p * v4457_data)) + (v4529_p * v4566_data)) + (v4638_p * v4675_data)) + (v4747_p * v4784_data)) + (v4856_p * v4893_data)) + (v4965_p * v5002_data)) + (v5074_p * v5111_data)) + (v5183_p * v5220_data)) + (v5292_p * v5329_data)) + (v5401_p * v5438_data)) + (v5510_p * v5547_data)) + (v5619_p * v5656_data)) + (v5728_p * v5765_data)) + (v5837_p * v5874_data)) + (v5946_p * v5983_data)) + (v6055_p * v6092_data)) + (v6164_p * v6201_data)) + (v6273_p * v6310_data)) + (v6382_p * v6419_data)) + (v6491_p * v6528_data)) + (v6600_p * v6637_data)) + (v6709_p * v6746_data)) + (v6818_p * v6855_data)) + (v6927_p * v6964_data)) + (v7036_p * v7073_data)) + (v7145_p * v7182_data)) + (v7254_p * v7291_data)) + (v7363_p * v7400_data)) + (v7472_p * v7509_data)) + (v7581_p * v7618_data);
          double v7624_data = s1[446];
          double v7627_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1738_data) + (v1804_p * v1847_data)) + (v1913_p * v1956_data)) + (v2022_p * v2065_data)) + (v2131_p * v2174_data)) + (v2240_p * v2283_data)) + (v2349_p * v2392_data)) + (v2458_p * v2501_data)) + (v2567_p * v2610_data)) + (v2676_p * v2719_data)) + (v2785_p * v2828_data)) + (v2894_p * v2937_data)) + (v3003_p * v3046_data)) + (v3112_p * v3155_data)) + (v3221_p * v3264_data)) + (v3330_p * v3373_data)) + (v3439_p * v3482_data)) + (v3548_p * v3591_data)) + (v3657_p * v3700_data)) + (v3766_p * v3809_data)) + (v3875_p * v3918_data)) + (v3984_p * v4027_data)) + (v4093_p * v4136_data)) + (v4202_p * v4245_data)) + (v4311_p * v4354_data)) + (v4420_p * v4463_data)) + (v4529_p * v4572_data)) + (v4638_p * v4681_data)) + (v4747_p * v4790_data)) + (v4856_p * v4899_data)) + (v4965_p * v5008_data)) + (v5074_p * v5117_data)) + (v5183_p * v5226_data)) + (v5292_p * v5335_data)) + (v5401_p * v5444_data)) + (v5510_p * v5553_data)) + (v5619_p * v5662_data)) + (v5728_p * v5771_data)) + (v5837_p * v5880_data)) + (v5946_p * v5989_data)) + (v6055_p * v6098_data)) + (v6164_p * v6207_data)) + (v6273_p * v6316_data)) + (v6382_p * v6425_data)) + (v6491_p * v6534_data)) + (v6600_p * v6643_data)) + (v6709_p * v6752_data)) + (v6818_p * v6861_data)) + (v6927_p * v6970_data)) + (v7036_p * v7079_data)) + (v7145_p * v7188_data)) + (v7254_p * v7297_data)) + (v7363_p * v7406_data)) + (v7472_p * v7515_data)) + (v7581_p * v7624_data);
          double v7630_data = s1[502];
          double v7633_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1744_data) + (v1804_p * v1853_data)) + (v1913_p * v1962_data)) + (v2022_p * v2071_data)) + (v2131_p * v2180_data)) + (v2240_p * v2289_data)) + (v2349_p * v2398_data)) + (v2458_p * v2507_data)) + (v2567_p * v2616_data)) + (v2676_p * v2725_data)) + (v2785_p * v2834_data)) + (v2894_p * v2943_data)) + (v3003_p * v3052_data)) + (v3112_p * v3161_data)) + (v3221_p * v3270_data)) + (v3330_p * v3379_data)) + (v3439_p * v3488_data)) + (v3548_p * v3597_data)) + (v3657_p * v3706_data)) + (v3766_p * v3815_data)) + (v3875_p * v3924_data)) + (v3984_p * v4033_data)) + (v4093_p * v4142_data)) + (v4202_p * v4251_data)) + (v4311_p * v4360_data)) + (v4420_p * v4469_data)) + (v4529_p * v4578_data)) + (v4638_p * v4687_data)) + (v4747_p * v4796_data)) + (v4856_p * v4905_data)) + (v4965_p * v5014_data)) + (v5074_p * v5123_data)) + (v5183_p * v5232_data)) + (v5292_p * v5341_data)) + (v5401_p * v5450_data)) + (v5510_p * v5559_data)) + (v5619_p * v5668_data)) + (v5728_p * v5777_data)) + (v5837_p * v5886_data)) + (v5946_p * v5995_data)) + (v6055_p * v6104_data)) + (v6164_p * v6213_data)) + (v6273_p * v6322_data)) + (v6382_p * v6431_data)) + (v6491_p * v6540_data)) + (v6600_p * v6649_data)) + (v6709_p * v6758_data)) + (v6818_p * v6867_data)) + (v6927_p * v6976_data)) + (v7036_p * v7085_data)) + (v7145_p * v7194_data)) + (v7254_p * v7303_data)) + (v7363_p * v7412_data)) + (v7472_p * v7521_data)) + (v7581_p * v7630_data);
          double v7635_p = v7580_data[1];
          double v7639_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1696_data) + (v1858_p * v1805_data)) + (v1967_p * v1914_data)) + (v2076_p * v2023_data)) + (v2185_p * v2132_data)) + (v2294_p * v2241_data)) + (v2403_p * v2350_data)) + (v2512_p * v2459_data)) + (v2621_p * v2568_data)) + (v2730_p * v2677_data)) + (v2839_p * v2786_data)) + (v2948_p * v2895_data)) + (v3057_p * v3004_data)) + (v3166_p * v3113_data)) + (v3275_p * v3222_data)) + (v3384_p * v3331_data)) + (v3493_p * v3440_data)) + (v3602_p * v3549_data)) + (v3711_p * v3658_data)) + (v3820_p * v3767_data)) + (v3929_p * v3876_data)) + (v4038_p * v3985_data)) + (v4147_p * v4094_data)) + (v4256_p * v4203_data)) + (v4365_p * v4312_data)) + (v4474_p * v4421_data)) + (v4583_p * v4530_data)) + (v4692_p * v4639_data)) + (v4801_p * v4748_data)) + (v4910_p * v4857_data)) + (v5019_p * v4966_data)) + (v5128_p * v5075_data)) + (v5237_p * v5184_data)) + (v5346_p * v5293_data)) + (v5455_p * v5402_data)) + (v5564_p * v5511_data)) + (v5673_p * v5620_data)) + (v5782_p * v5729_data)) + (v5891_p * v5838_data)) + (v6000_p * v5947_data)) + (v6109_p * v6056_data)) + (v6218_p * v6165_data)) + (v6327_p * v6274_data)) + (v6436_p * v6383_data)) + (v6545_p * v6492_data)) + (v6654_p * v6601_data)) + (v6763_p * v6710_data)) + (v6872_p * v6819_data)) + (v6981_p * v6928_data)) + (v7090_p * v7037_data)) + (v7199_p * v7146_data)) + (v7308_p * v7255_data)) + (v7417_p * v7364_data)) + (v7526_p * v7473_data)) + (v7635_p * v7582_data);
          double v7645_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1702_data) + (v1858_p * v1811_data)) + (v1967_p * v1920_data)) + (v2076_p * v2029_data)) + (v2185_p * v2138_data)) + (v2294_p * v2247_data)) + (v2403_p * v2356_data)) + (v2512_p * v2465_data)) + (v2621_p * v2574_data)) + (v2730_p * v2683_data)) + (v2839_p * v2792_data)) + (v2948_p * v2901_data)) + (v3057_p * v3010_data)) + (v3166_p * v3119_data)) + (v3275_p * v3228_data)) + (v3384_p * v3337_data)) + (v3493_p * v3446_data)) + (v3602_p * v3555_data)) + (v3711_p * v3664_data)) + (v3820_p * v3773_data)) + (v3929_p * v3882_data)) + (v4038_p * v3991_data)) + (v4147_p * v4100_data)) + (v4256_p * v4209_data)) + (v4365_p * v4318_data)) + (v4474_p * v4427_data)) + (v4583_p * v4536_data)) + (v4692_p * v4645_data)) + (v4801_p * v4754_data)) + (v4910_p * v4863_data)) + (v5019_p * v4972_data)) + (v5128_p * v5081_data)) + (v5237_p * v5190_data)) + (v5346_p * v5299_data)) + (v5455_p * v5408_data)) + (v5564_p * v5517_data)) + (v5673_p * v5626_data)) + (v5782_p * v5735_data)) + (v5891_p * v5844_data)) + (v6000_p * v5953_data)) + (v6109_p * v6062_data)) + (v6218_p * v6171_data)) + (v6327_p * v6280_data)) + (v6436_p * v6389_data)) + (v6545_p * v6498_data)) + (v6654_p * v6607_data)) + (v6763_p * v6716_data)) + (v6872_p * v6825_data)) + (v6981_p * v6934_data)) + (v7090_p * v7043_data)) + (v7199_p * v7152_data)) + (v7308_p * v7261_data)) + (v7417_p * v7370_data)) + (v7526_p * v7479_data)) + (v7635_p * v7588_data);
          double v7651_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1708_data) + (v1858_p * v1817_data)) + (v1967_p * v1926_data)) + (v2076_p * v2035_data)) + (v2185_p * v2144_data)) + (v2294_p * v2253_data)) + (v2403_p * v2362_data)) + (v2512_p * v2471_data)) + (v2621_p * v2580_data)) + (v2730_p * v2689_data)) + (v2839_p * v2798_data)) + (v2948_p * v2907_data)) + (v3057_p * v3016_data)) + (v3166_p * v3125_data)) + (v3275_p * v3234_data)) + (v3384_p * v3343_data)) + (v3493_p * v3452_data)) + (v3602_p * v3561_data)) + (v3711_p * v3670_data)) + (v3820_p * v3779_data)) + (v3929_p * v3888_data)) + (v4038_p * v3997_data)) + (v4147_p * v4106_data)) + (v4256_p * v4215_data)) + (v4365_p * v4324_data)) + (v4474_p * v4433_data)) + (v4583_p * v4542_data)) + (v4692_p * v4651_data)) + (v4801_p * v4760_data)) + (v4910_p * v4869_data)) + (v5019_p * v4978_data)) + (v5128_p * v5087_data)) + (v5237_p * v5196_data)) + (v5346_p * v5305_data)) + (v5455_p * v5414_data)) + (v5564_p * v5523_data)) + (v5673_p * v5632_data)) + (v5782_p * v5741_data)) + (v5891_p * v5850_data)) + (v6000_p * v5959_data)) + (v6109_p * v6068_data)) + (v6218_p * v6177_data)) + (v6327_p * v6286_data)) + (v6436_p * v6395_data)) + (v6545_p * v6504_data)) + (v6654_p * v6613_data)) + (v6763_p * v6722_data)) + (v6872_p * v6831_data)) + (v6981_p * v6940_data)) + (v7090_p * v7049_data)) + (v7199_p * v7158_data)) + (v7308_p * v7267_data)) + (v7417_p * v7376_data)) + (v7526_p * v7485_data)) + (v7635_p * v7594_data);
          double v7657_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1714_data) + (v1858_p * v1823_data)) + (v1967_p * v1932_data)) + (v2076_p * v2041_data)) + (v2185_p * v2150_data)) + (v2294_p * v2259_data)) + (v2403_p * v2368_data)) + (v2512_p * v2477_data)) + (v2621_p * v2586_data)) + (v2730_p * v2695_data)) + (v2839_p * v2804_data)) + (v2948_p * v2913_data)) + (v3057_p * v3022_data)) + (v3166_p * v3131_data)) + (v3275_p * v3240_data)) + (v3384_p * v3349_data)) + (v3493_p * v3458_data)) + (v3602_p * v3567_data)) + (v3711_p * v3676_data)) + (v3820_p * v3785_data)) + (v3929_p * v3894_data)) + (v4038_p * v4003_data)) + (v4147_p * v4112_data)) + (v4256_p * v4221_data)) + (v4365_p * v4330_data)) + (v4474_p * v4439_data)) + (v4583_p * v4548_data)) + (v4692_p * v4657_data)) + (v4801_p * v4766_data)) + (v4910_p * v4875_data)) + (v5019_p * v4984_data)) + (v5128_p * v5093_data)) + (v5237_p * v5202_data)) + (v5346_p * v5311_data)) + (v5455_p * v5420_data)) + (v5564_p * v5529_data)) + (v5673_p * v5638_data)) + (v5782_p * v5747_data)) + (v5891_p * v5856_data)) + (v6000_p * v5965_data)) + (v6109_p * v6074_data)) + (v6218_p * v6183_data)) + (v6327_p * v6292_data)) + (v6436_p * v6401_data)) + (v6545_p * v6510_data)) + (v6654_p * v6619_data)) + (v6763_p * v6728_data)) + (v6872_p * v6837_data)) + (v6981_p * v6946_data)) + (v7090_p * v7055_data)) + (v7199_p * v7164_data)) + (v7308_p * v7273_data)) + (v7417_p * v7382_data)) + (v7526_p * v7491_data)) + (v7635_p * v7600_data);
          double v7663_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1720_data) + (v1858_p * v1829_data)) + (v1967_p * v1938_data)) + (v2076_p * v2047_data)) + (v2185_p * v2156_data)) + (v2294_p * v2265_data)) + (v2403_p * v2374_data)) + (v2512_p * v2483_data)) + (v2621_p * v2592_data)) + (v2730_p * v2701_data)) + (v2839_p * v2810_data)) + (v2948_p * v2919_data)) + (v3057_p * v3028_data)) + (v3166_p * v3137_data)) + (v3275_p * v3246_data)) + (v3384_p * v3355_data)) + (v3493_p * v3464_data)) + (v3602_p * v3573_data)) + (v3711_p * v3682_data)) + (v3820_p * v3791_data)) + (v3929_p * v3900_data)) + (v4038_p * v4009_data)) + (v4147_p * v4118_data)) + (v4256_p * v4227_data)) + (v4365_p * v4336_data)) + (v4474_p * v4445_data)) + (v4583_p * v4554_data)) + (v4692_p * v4663_data)) + (v4801_p * v4772_data)) + (v4910_p * v4881_data)) + (v5019_p * v4990_data)) + (v5128_p * v5099_data)) + (v5237_p * v5208_data)) + (v5346_p * v5317_data)) + (v5455_p * v5426_data)) + (v5564_p * v5535_data)) + (v5673_p * v5644_data)) + (v5782_p * v5753_data)) + (v5891_p * v5862_data)) + (v6000_p * v5971_data)) + (v6109_p * v6080_data)) + (v6218_p * v6189_data)) + (v6327_p * v6298_data)) + (v6436_p * v6407_data)) + (v6545_p * v6516_data)) + (v6654_p * v6625_data)) + (v6763_p * v6734_data)) + (v6872_p * v6843_data)) + (v6981_p * v6952_data)) + (v7090_p * v7061_data)) + (v7199_p * v7170_data)) + (v7308_p * v7279_data)) + (v7417_p * v7388_data)) + (v7526_p * v7497_data)) + (v7635_p * v7606_data);
          double v7669_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1726_data) + (v1858_p * v1835_data)) + (v1967_p * v1944_data)) + (v2076_p * v2053_data)) + (v2185_p * v2162_data)) + (v2294_p * v2271_data)) + (v2403_p * v2380_data)) + (v2512_p * v2489_data)) + (v2621_p * v2598_data)) + (v2730_p * v2707_data)) + (v2839_p * v2816_data)) + (v2948_p * v2925_data)) + (v3057_p * v3034_data)) + (v3166_p * v3143_data)) + (v3275_p * v3252_data)) + (v3384_p * v3361_data)) + (v3493_p * v3470_data)) + (v3602_p * v3579_data)) + (v3711_p * v3688_data)) + (v3820_p * v3797_data)) + (v3929_p * v3906_data)) + (v4038_p * v4015_data)) + (v4147_p * v4124_data)) + (v4256_p * v4233_data)) + (v4365_p * v4342_data)) + (v4474_p * v4451_data)) + (v4583_p * v4560_data)) + (v4692_p * v4669_data)) + (v4801_p * v4778_data)) + (v4910_p * v4887_data)) + (v5019_p * v4996_data)) + (v5128_p * v5105_data)) + (v5237_p * v5214_data)) + (v5346_p * v5323_data)) + (v5455_p * v5432_data)) + (v5564_p * v5541_data)) + (v5673_p * v5650_data)) + (v5782_p * v5759_data)) + (v5891_p * v5868_data)) + (v6000_p * v5977_data)) + (v6109_p * v6086_data)) + (v6218_p * v6195_data)) + (v6327_p * v6304_data)) + (v6436_p * v6413_data)) + (v6545_p * v6522_data)) + (v6654_p * v6631_data)) + (v6763_p * v6740_data)) + (v6872_p * v6849_data)) + (v6981_p * v6958_data)) + (v7090_p * v7067_data)) + (v7199_p * v7176_data)) + (v7308_p * v7285_data)) + (v7417_p * v7394_data)) + (v7526_p * v7503_data)) + (v7635_p * v7612_data);
          double v7675_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1732_data) + (v1858_p * v1841_data)) + (v1967_p * v1950_data)) + (v2076_p * v2059_data)) + (v2185_p * v2168_data)) + (v2294_p * v2277_data)) + (v2403_p * v2386_data)) + (v2512_p * v2495_data)) + (v2621_p * v2604_data)) + (v2730_p * v2713_data)) + (v2839_p * v2822_data)) + (v2948_p * v2931_data)) + (v3057_p * v3040_data)) + (v3166_p * v3149_data)) + (v3275_p * v3258_data)) + (v3384_p * v3367_data)) + (v3493_p * v3476_data)) + (v3602_p * v3585_data)) + (v3711_p * v3694_data)) + (v3820_p * v3803_data)) + (v3929_p * v3912_data)) + (v4038_p * v4021_data)) + (v4147_p * v4130_data)) + (v4256_p * v4239_data)) + (v4365_p * v4348_data)) + (v4474_p * v4457_data)) + (v4583_p * v4566_data)) + (v4692_p * v4675_data)) + (v4801_p * v4784_data)) + (v4910_p * v4893_data)) + (v5019_p * v5002_data)) + (v5128_p * v5111_data)) + (v5237_p * v5220_data)) + (v5346_p * v5329_data)) + (v5455_p * v5438_data)) + (v5564_p * v5547_data)) + (v5673_p * v5656_data)) + (v5782_p * v5765_data)) + (v5891_p * v5874_data)) + (v6000_p * v5983_data)) + (v6109_p * v6092_data)) + (v6218_p * v6201_data)) + (v6327_p * v6310_data)) + (v6436_p * v6419_data)) + (v6545_p * v6528_data)) + (v6654_p * v6637_data)) + (v6763_p * v6746_data)) + (v6872_p * v6855_data)) + (v6981_p * v6964_data)) + (v7090_p * v7073_data)) + (v7199_p * v7182_data)) + (v7308_p * v7291_data)) + (v7417_p * v7400_data)) + (v7526_p * v7509_data)) + (v7635_p * v7618_data);
          double v7681_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1738_data) + (v1858_p * v1847_data)) + (v1967_p * v1956_data)) + (v2076_p * v2065_data)) + (v2185_p * v2174_data)) + (v2294_p * v2283_data)) + (v2403_p * v2392_data)) + (v2512_p * v2501_data)) + (v2621_p * v2610_data)) + (v2730_p * v2719_data)) + (v2839_p * v2828_data)) + (v2948_p * v2937_data)) + (v3057_p * v3046_data)) + (v3166_p * v3155_data)) + (v3275_p * v3264_data)) + (v3384_p * v3373_data)) + (v3493_p * v3482_data)) + (v3602_p * v3591_data)) + (v3711_p * v3700_data)) + (v3820_p * v3809_data)) + (v3929_p * v3918_data)) + (v4038_p * v4027_data)) + (v4147_p * v4136_data)) + (v4256_p * v4245_data)) + (v4365_p * v4354_data)) + (v4474_p * v4463_data)) + (v4583_p * v4572_data)) + (v4692_p * v4681_data)) + (v4801_p * v4790_data)) + (v4910_p * v4899_data)) + (v5019_p * v5008_data)) + (v5128_p * v5117_data)) + (v5237_p * v5226_data)) + (v5346_p * v5335_data)) + (v5455_p * v5444_data)) + (v5564_p * v5553_data)) + (v5673_p * v5662_data)) + (v5782_p * v5771_data)) + (v5891_p * v5880_data)) + (v6000_p * v5989_data)) + (v6109_p * v6098_data)) + (v6218_p * v6207_data)) + (v6327_p * v6316_data)) + (v6436_p * v6425_data)) + (v6545_p * v6534_data)) + (v6654_p * v6643_data)) + (v6763_p * v6752_data)) + (v6872_p * v6861_data)) + (v6981_p * v6970_data)) + (v7090_p * v7079_data)) + (v7199_p * v7188_data)) + (v7308_p * v7297_data)) + (v7417_p * v7406_data)) + (v7526_p * v7515_data)) + (v7635_p * v7624_data);
          double v7687_p = ((((((((((((((((((((((((((((((((((((((((((((((((((((((v1749_p * v1744_data) + (v1858_p * v1853_data)) + (v1967_p * v1962_data)) + (v2076_p * v2071_data)) + (v2185_p * v2180_data)) + (v2294_p * v2289_data)) + (v2403_p * v2398_data)) + (v2512_p * v2507_data)) + (v2621_p * v2616_data)) + (v2730_p * v2725_data)) + (v2839_p * v2834_data)) + (v2948_p * v2943_data)) + (v3057_p * v3052_data)) + (v3166_p * v3161_data)) + (v3275_p * v3270_data)) + (v3384_p * v3379_data)) + (v3493_p * v3488_data)) + (v3602_p * v3597_data)) + (v3711_p * v3706_data)) + (v3820_p * v3815_data)) + (v3929_p * v3924_data)) + (v4038_p * v4033_data)) + (v4147_p * v4142_data)) + (v4256_p * v4251_data)) + (v4365_p * v4360_data)) + (v4474_p * v4469_data)) + (v4583_p * v4578_data)) + (v4692_p * v4687_data)) + (v4801_p * v4796_data)) + (v4910_p * v4905_data)) + (v5019_p * v5014_data)) + (v5128_p * v5123_data)) + (v5237_p * v5232_data)) + (v5346_p * v5341_data)) + (v5455_p * v5450_data)) + (v5564_p * v5559_data)) + (v5673_p * v5668_data)) + (v5782_p * v5777_data)) + (v5891_p * v5886_data)) + (v6000_p * v5995_data)) + (v6109_p * v6104_data)) + (v6218_p * v6213_data)) + (v6327_p * v6322_data)) + (v6436_p * v6431_data)) + (v6545_p * v6540_data)) + (v6654_p * v6649_data)) + (v6763_p * v6758_data)) + (v6872_p * v6867_data)) + (v6981_p * v6976_data)) + (v7090_p * v7085_data)) + (v7199_p * v7194_data)) + (v7308_p * v7303_data)) + (v7417_p * v7412_data)) + (v7526_p * v7521_data)) + (v7635_p * v7630_data);
          tensorforge::VectorT<double, 2> v7689_data = *(tensorforge::VectorT<double, 2>*)&glb_m2[(1760 + v1692_a)];
          double v7690_p = v7689_data[0];
          double v7691_data = s1[55];
          r2[0] = ((((((((((((((((((((((((((((((((((((((((((((((((((((((((v1695_p * v1696_data) + (v1804_p * v1805_data)) + (v1913_p * v1914_data)) + (v2022_p * v2023_data)) + (v2131_p * v2132_data)) + (v2240_p * v2241_data)) + (v2349_p * v2350_data)) + (v2458_p * v2459_data)) + (v2567_p * v2568_data)) + (v2676_p * v2677_data)) + (v2785_p * v2786_data)) + (v2894_p * v2895_data)) + (v3003_p * v3004_data)) + (v3112_p * v3113_data)) + (v3221_p * v3222_data)) + (v3330_p * v3331_data)) + (v3439_p * v3440_data)) + (v3548_p * v3549_data)) + (v3657_p * v3658_data)) + (v3766_p * v3767_data)) + (v3875_p * v3876_data)) + (v3984_p * v3985_data)) + (v4093_p * v4094_data)) + (v4202_p * v4203_data)) + (v4311_p * v4312_data)) + (v4420_p * v4421_data)) + (v4529_p * v4530_data)) + (v4638_p * v4639_data)) + (v4747_p * v4748_data)) + (v4856_p * v4857_data)) + (v4965_p * v4966_data)) + (v5074_p * v5075_data)) + (v5183_p * v5184_data)) + (v5292_p * v5293_data)) + (v5401_p * v5402_data)) + (v5510_p * v5511_data)) + (v5619_p * v5620_data)) + (v5728_p * v5729_data)) + (v5837_p * v5838_data)) + (v5946_p * v5947_data)) + (v6055_p * v6056_data)) + (v6164_p * v6165_data)) + (v6273_p * v6274_data)) + (v6382_p * v6383_data)) + (v6491_p * v6492_data)) + (v6600_p * v6601_data)) + (v6709_p * v6710_data)) + (v6818_p * v6819_data)) + (v6927_p * v6928_data)) + (v7036_p * v7037_data)) + (v7145_p * v7146_data)) + (v7254_p * v7255_data)) + (v7363_p * v7364_data)) + (v7472_p * v7473_data)) + (v7581_p * v7582_data)) + (v7690_p * v7691_data));
          double v7697_data = s1[111];
          r2[2] = (v7591_p + (v7690_p * v7697_data));
          double v7703_data = s1[167];
          r2[4] = (v7597_p + (v7690_p * v7703_data));
          double v7709_data = s1[223];
          r2[6] = (v7603_p + (v7690_p * v7709_data));
          double v7715_data = s1[279];
          r2[8] = (v7609_p + (v7690_p * v7715_data));
          double v7721_data = s1[335];
          r2[10] = (v7615_p + (v7690_p * v7721_data));
          double v7727_data = s1[391];
          r2[12] = (v7621_p + (v7690_p * v7727_data));
          double v7733_data = s1[447];
          r2[14] = (v7627_p + (v7690_p * v7733_data));
          double v7739_data = s1[503];
          r2[16] = (v7633_p + (v7690_p * v7739_data));
          double v7744_p = v7689_data[1];
          r2[1] = (v7639_p + (v7744_p * v7691_data));
          r2[3] = (v7645_p + (v7744_p * v7697_data));
          r2[5] = (v7651_p + (v7744_p * v7703_data));
          r2[7] = (v7657_p + (v7744_p * v7709_data));
          r2[9] = (v7663_p + (v7744_p * v7715_data));
          r2[11] = (v7669_p + (v7744_p * v7721_data));
          r2[13] = (v7675_p + (v7744_p * v7727_data));
          r2[15] = (v7681_p + (v7744_p * v7733_data));
          r2[17] = (v7687_p + (v7744_p * v7739_data));
          // r2 = 0 past (0, 21)
          // s2 = store{r>s}(localShrMem0, r2);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v7798_i0 = 0; v7798_i0 < 1; ++v7798_i0) {
            int32_t v7804_lead = v11406_lead + (v7798_i0 * 16);
            #pragma unroll
            for (int32_t v7799_i1 = 0; v7799_i1 < 9; ++v7799_i1) {
              double v7802_data = r2[(v7798_i0 + (v7799_i1 * 2))];
              s2[(v7804_lead + (v7799_i1 * 21))] = v7802_data;
            }
          }
          if (v11406_lead < 5) {
            int32_t v7813_lead = v11406_lead + 16_i32;
            #pragma unroll
            for (int32_t v7808_i1 = 0; v7808_i1 < 9; ++v7808_i1) {
              double v7811_data = r2[(1 + (v7808_i1 * 2))];
              s2[(v7813_lead + (v7808_i1 * 21))] = v7811_data;
            }
          }
          double r4[36]{};
          // ir4 = +(glb_m4 * s2)
          // [(0, 56), (0, 9)] [(0, 21)]
          double ir4[36]{};
          double v7821_data = glb_m4[v11406_lead];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v7822_data = s2[0];
          double v7827_data = s2[21];
          double v7832_data = s2[42];
          double v7837_data = s2[63];
          double v7842_data = s2[84];
          double v7847_data = s2[105];
          double v7852_data = s2[126];
          double v7857_data = s2[147];
          double v7862_data = s2[168];
          int32_t v7867_lead = v11406_lead + 16_i32;
          double v7869_data = glb_m4[v7867_lead];
          int32_t v7915_lead = v11406_lead + 32_i32;
          double v7917_data = glb_m4[v7915_lead];
          double v7965_data = v11409_g ? (glb_m4[v11408_lead]) : (0.0);
          double v8012_data = glb_m4[(v11406_lead + 56)];
          double v8013_data = s2[1];
          double v8018_data = s2[22];
          double v8023_data = s2[43];
          double v8028_data = s2[64];
          double v8033_data = s2[85];
          double v8038_data = s2[106];
          double v8043_data = s2[127];
          double v8048_data = s2[148];
          double v8053_data = s2[169];
          double v8059_data = glb_m4[(v7867_lead + 56)];
          double v8106_data = glb_m4[(v7915_lead + 56)];
          double v8153_data = v11409_g ? (glb_m4[(v11408_lead + 56)]) : (0.0);
          double v8200_data = glb_m4[(v11406_lead + 112)];
          double v8201_data = s2[2];
          double v8206_data = s2[23];
          double v8211_data = s2[44];
          double v8216_data = s2[65];
          double v8221_data = s2[86];
          double v8226_data = s2[107];
          double v8231_data = s2[128];
          double v8236_data = s2[149];
          double v8241_data = s2[170];
          double v8247_data = glb_m4[(v7867_lead + 112)];
          double v8294_data = glb_m4[(v7915_lead + 112)];
          double v8341_data = v11409_g ? (glb_m4[(v11408_lead + 112)]) : (0.0);
          double v8388_data = glb_m4[(v11406_lead + 168)];
          double v8389_data = s2[3];
          double v8394_data = s2[24];
          double v8399_data = s2[45];
          double v8404_data = s2[66];
          double v8409_data = s2[87];
          double v8414_data = s2[108];
          double v8419_data = s2[129];
          double v8424_data = s2[150];
          double v8429_data = s2[171];
          double v8435_data = glb_m4[(v7867_lead + 168)];
          double v8482_data = glb_m4[(v7915_lead + 168)];
          double v8529_data = v11409_g ? (glb_m4[(v11408_lead + 168)]) : (0.0);
          double v8576_data = glb_m4[(v11406_lead + 224)];
          double v8577_data = s2[4];
          double v8582_data = s2[25];
          double v8587_data = s2[46];
          double v8592_data = s2[67];
          double v8597_data = s2[88];
          double v8602_data = s2[109];
          double v8607_data = s2[130];
          double v8612_data = s2[151];
          double v8617_data = s2[172];
          double v8623_data = glb_m4[(v7867_lead + 224)];
          double v8670_data = glb_m4[(v7915_lead + 224)];
          double v8717_data = v11409_g ? (glb_m4[(v11408_lead + 224)]) : (0.0);
          double v8764_data = glb_m4[(v11406_lead + 280)];
          double v8765_data = s2[5];
          double v8770_data = s2[26];
          double v8775_data = s2[47];
          double v8780_data = s2[68];
          double v8785_data = s2[89];
          double v8790_data = s2[110];
          double v8795_data = s2[131];
          double v8800_data = s2[152];
          double v8805_data = s2[173];
          double v8811_data = glb_m4[(v7867_lead + 280)];
          double v8858_data = glb_m4[(v7915_lead + 280)];
          double v8905_data = v11409_g ? (glb_m4[(v11408_lead + 280)]) : (0.0);
          double v8952_data = glb_m4[(v11406_lead + 336)];
          double v8953_data = s2[6];
          double v8958_data = s2[27];
          double v8963_data = s2[48];
          double v8968_data = s2[69];
          double v8973_data = s2[90];
          double v8978_data = s2[111];
          double v8983_data = s2[132];
          double v8988_data = s2[153];
          double v8993_data = s2[174];
          double v8999_data = glb_m4[(v7867_lead + 336)];
          double v9046_data = glb_m4[(v7915_lead + 336)];
          double v9093_data = v11409_g ? (glb_m4[(v11408_lead + 336)]) : (0.0);
          double v9140_data = glb_m4[(v11406_lead + 392)];
          double v9141_data = s2[7];
          double v9146_data = s2[28];
          double v9151_data = s2[49];
          double v9156_data = s2[70];
          double v9161_data = s2[91];
          double v9166_data = s2[112];
          double v9171_data = s2[133];
          double v9176_data = s2[154];
          double v9181_data = s2[175];
          double v9187_data = glb_m4[(v7867_lead + 392)];
          double v9234_data = glb_m4[(v7915_lead + 392)];
          double v9281_data = v11409_g ? (glb_m4[(v11408_lead + 392)]) : (0.0);
          double v9328_data = glb_m4[(v11406_lead + 448)];
          double v9329_data = s2[8];
          double v9334_data = s2[29];
          double v9339_data = s2[50];
          double v9344_data = s2[71];
          double v9349_data = s2[92];
          double v9354_data = s2[113];
          double v9359_data = s2[134];
          double v9364_data = s2[155];
          double v9369_data = s2[176];
          double v9375_data = glb_m4[(v7867_lead + 448)];
          double v9422_data = glb_m4[(v7915_lead + 448)];
          double v9469_data = v11409_g ? (glb_m4[(v11408_lead + 448)]) : (0.0);
          double v9516_data = glb_m4[(v11406_lead + 504)];
          double v9517_data = s2[9];
          double v9522_data = s2[30];
          double v9527_data = s2[51];
          double v9532_data = s2[72];
          double v9537_data = s2[93];
          double v9542_data = s2[114];
          double v9547_data = s2[135];
          double v9552_data = s2[156];
          double v9557_data = s2[177];
          double v9563_data = glb_m4[(v7867_lead + 504)];
          double v9610_data = glb_m4[(v7915_lead + 504)];
          double v9657_data = v11409_g ? (glb_m4[(v11408_lead + 504)]) : (0.0);
          double v9704_data = glb_m4[(v11406_lead + 560)];
          double v9705_data = s2[10];
          double v9710_data = s2[31];
          double v9713_p = ((((((((((v7821_data * v7827_data) + (v8012_data * v8018_data)) + (v8200_data * v8206_data)) + (v8388_data * v8394_data)) + (v8576_data * v8582_data)) + (v8764_data * v8770_data)) + (v8952_data * v8958_data)) + (v9140_data * v9146_data)) + (v9328_data * v9334_data)) + (v9516_data * v9522_data)) + (v9704_data * v9710_data);
          double v9715_data = s2[52];
          double v9718_p = ((((((((((v7821_data * v7832_data) + (v8012_data * v8023_data)) + (v8200_data * v8211_data)) + (v8388_data * v8399_data)) + (v8576_data * v8587_data)) + (v8764_data * v8775_data)) + (v8952_data * v8963_data)) + (v9140_data * v9151_data)) + (v9328_data * v9339_data)) + (v9516_data * v9527_data)) + (v9704_data * v9715_data);
          double v9720_data = s2[73];
          double v9723_p = ((((((((((v7821_data * v7837_data) + (v8012_data * v8028_data)) + (v8200_data * v8216_data)) + (v8388_data * v8404_data)) + (v8576_data * v8592_data)) + (v8764_data * v8780_data)) + (v8952_data * v8968_data)) + (v9140_data * v9156_data)) + (v9328_data * v9344_data)) + (v9516_data * v9532_data)) + (v9704_data * v9720_data);
          double v9725_data = s2[94];
          double v9728_p = ((((((((((v7821_data * v7842_data) + (v8012_data * v8033_data)) + (v8200_data * v8221_data)) + (v8388_data * v8409_data)) + (v8576_data * v8597_data)) + (v8764_data * v8785_data)) + (v8952_data * v8973_data)) + (v9140_data * v9161_data)) + (v9328_data * v9349_data)) + (v9516_data * v9537_data)) + (v9704_data * v9725_data);
          double v9730_data = s2[115];
          double v9733_p = ((((((((((v7821_data * v7847_data) + (v8012_data * v8038_data)) + (v8200_data * v8226_data)) + (v8388_data * v8414_data)) + (v8576_data * v8602_data)) + (v8764_data * v8790_data)) + (v8952_data * v8978_data)) + (v9140_data * v9166_data)) + (v9328_data * v9354_data)) + (v9516_data * v9542_data)) + (v9704_data * v9730_data);
          double v9735_data = s2[136];
          double v9738_p = ((((((((((v7821_data * v7852_data) + (v8012_data * v8043_data)) + (v8200_data * v8231_data)) + (v8388_data * v8419_data)) + (v8576_data * v8607_data)) + (v8764_data * v8795_data)) + (v8952_data * v8983_data)) + (v9140_data * v9171_data)) + (v9328_data * v9359_data)) + (v9516_data * v9547_data)) + (v9704_data * v9735_data);
          double v9740_data = s2[157];
          double v9743_p = ((((((((((v7821_data * v7857_data) + (v8012_data * v8048_data)) + (v8200_data * v8236_data)) + (v8388_data * v8424_data)) + (v8576_data * v8612_data)) + (v8764_data * v8800_data)) + (v8952_data * v8988_data)) + (v9140_data * v9176_data)) + (v9328_data * v9364_data)) + (v9516_data * v9552_data)) + (v9704_data * v9740_data);
          double v9745_data = s2[178];
          double v9748_p = ((((((((((v7821_data * v7862_data) + (v8012_data * v8053_data)) + (v8200_data * v8241_data)) + (v8388_data * v8429_data)) + (v8576_data * v8617_data)) + (v8764_data * v8805_data)) + (v8952_data * v8993_data)) + (v9140_data * v9181_data)) + (v9328_data * v9369_data)) + (v9516_data * v9557_data)) + (v9704_data * v9745_data);
          double v9751_data = glb_m4[(v7867_lead + 560)];
          double v9755_p = ((((((((((v7869_data * v7822_data) + (v8059_data * v8013_data)) + (v8247_data * v8201_data)) + (v8435_data * v8389_data)) + (v8623_data * v8577_data)) + (v8811_data * v8765_data)) + (v8999_data * v8953_data)) + (v9187_data * v9141_data)) + (v9375_data * v9329_data)) + (v9563_data * v9517_data)) + (v9751_data * v9705_data);
          double v9760_p = ((((((((((v7869_data * v7827_data) + (v8059_data * v8018_data)) + (v8247_data * v8206_data)) + (v8435_data * v8394_data)) + (v8623_data * v8582_data)) + (v8811_data * v8770_data)) + (v8999_data * v8958_data)) + (v9187_data * v9146_data)) + (v9375_data * v9334_data)) + (v9563_data * v9522_data)) + (v9751_data * v9710_data);
          double v9765_p = ((((((((((v7869_data * v7832_data) + (v8059_data * v8023_data)) + (v8247_data * v8211_data)) + (v8435_data * v8399_data)) + (v8623_data * v8587_data)) + (v8811_data * v8775_data)) + (v8999_data * v8963_data)) + (v9187_data * v9151_data)) + (v9375_data * v9339_data)) + (v9563_data * v9527_data)) + (v9751_data * v9715_data);
          double v9770_p = ((((((((((v7869_data * v7837_data) + (v8059_data * v8028_data)) + (v8247_data * v8216_data)) + (v8435_data * v8404_data)) + (v8623_data * v8592_data)) + (v8811_data * v8780_data)) + (v8999_data * v8968_data)) + (v9187_data * v9156_data)) + (v9375_data * v9344_data)) + (v9563_data * v9532_data)) + (v9751_data * v9720_data);
          double v9775_p = ((((((((((v7869_data * v7842_data) + (v8059_data * v8033_data)) + (v8247_data * v8221_data)) + (v8435_data * v8409_data)) + (v8623_data * v8597_data)) + (v8811_data * v8785_data)) + (v8999_data * v8973_data)) + (v9187_data * v9161_data)) + (v9375_data * v9349_data)) + (v9563_data * v9537_data)) + (v9751_data * v9725_data);
          double v9780_p = ((((((((((v7869_data * v7847_data) + (v8059_data * v8038_data)) + (v8247_data * v8226_data)) + (v8435_data * v8414_data)) + (v8623_data * v8602_data)) + (v8811_data * v8790_data)) + (v8999_data * v8978_data)) + (v9187_data * v9166_data)) + (v9375_data * v9354_data)) + (v9563_data * v9542_data)) + (v9751_data * v9730_data);
          double v9785_p = ((((((((((v7869_data * v7852_data) + (v8059_data * v8043_data)) + (v8247_data * v8231_data)) + (v8435_data * v8419_data)) + (v8623_data * v8607_data)) + (v8811_data * v8795_data)) + (v8999_data * v8983_data)) + (v9187_data * v9171_data)) + (v9375_data * v9359_data)) + (v9563_data * v9547_data)) + (v9751_data * v9735_data);
          double v9790_p = ((((((((((v7869_data * v7857_data) + (v8059_data * v8048_data)) + (v8247_data * v8236_data)) + (v8435_data * v8424_data)) + (v8623_data * v8612_data)) + (v8811_data * v8800_data)) + (v8999_data * v8988_data)) + (v9187_data * v9176_data)) + (v9375_data * v9364_data)) + (v9563_data * v9552_data)) + (v9751_data * v9740_data);
          double v9795_p = ((((((((((v7869_data * v7862_data) + (v8059_data * v8053_data)) + (v8247_data * v8241_data)) + (v8435_data * v8429_data)) + (v8623_data * v8617_data)) + (v8811_data * v8805_data)) + (v8999_data * v8993_data)) + (v9187_data * v9181_data)) + (v9375_data * v9369_data)) + (v9563_data * v9557_data)) + (v9751_data * v9745_data);
          double v9798_data = glb_m4[(v7915_lead + 560)];
          double v9802_p = ((((((((((v7917_data * v7822_data) + (v8106_data * v8013_data)) + (v8294_data * v8201_data)) + (v8482_data * v8389_data)) + (v8670_data * v8577_data)) + (v8858_data * v8765_data)) + (v9046_data * v8953_data)) + (v9234_data * v9141_data)) + (v9422_data * v9329_data)) + (v9610_data * v9517_data)) + (v9798_data * v9705_data);
          double v9807_p = ((((((((((v7917_data * v7827_data) + (v8106_data * v8018_data)) + (v8294_data * v8206_data)) + (v8482_data * v8394_data)) + (v8670_data * v8582_data)) + (v8858_data * v8770_data)) + (v9046_data * v8958_data)) + (v9234_data * v9146_data)) + (v9422_data * v9334_data)) + (v9610_data * v9522_data)) + (v9798_data * v9710_data);
          double v9812_p = ((((((((((v7917_data * v7832_data) + (v8106_data * v8023_data)) + (v8294_data * v8211_data)) + (v8482_data * v8399_data)) + (v8670_data * v8587_data)) + (v8858_data * v8775_data)) + (v9046_data * v8963_data)) + (v9234_data * v9151_data)) + (v9422_data * v9339_data)) + (v9610_data * v9527_data)) + (v9798_data * v9715_data);
          double v9817_p = ((((((((((v7917_data * v7837_data) + (v8106_data * v8028_data)) + (v8294_data * v8216_data)) + (v8482_data * v8404_data)) + (v8670_data * v8592_data)) + (v8858_data * v8780_data)) + (v9046_data * v8968_data)) + (v9234_data * v9156_data)) + (v9422_data * v9344_data)) + (v9610_data * v9532_data)) + (v9798_data * v9720_data);
          double v9822_p = ((((((((((v7917_data * v7842_data) + (v8106_data * v8033_data)) + (v8294_data * v8221_data)) + (v8482_data * v8409_data)) + (v8670_data * v8597_data)) + (v8858_data * v8785_data)) + (v9046_data * v8973_data)) + (v9234_data * v9161_data)) + (v9422_data * v9349_data)) + (v9610_data * v9537_data)) + (v9798_data * v9725_data);
          double v9827_p = ((((((((((v7917_data * v7847_data) + (v8106_data * v8038_data)) + (v8294_data * v8226_data)) + (v8482_data * v8414_data)) + (v8670_data * v8602_data)) + (v8858_data * v8790_data)) + (v9046_data * v8978_data)) + (v9234_data * v9166_data)) + (v9422_data * v9354_data)) + (v9610_data * v9542_data)) + (v9798_data * v9730_data);
          double v9832_p = ((((((((((v7917_data * v7852_data) + (v8106_data * v8043_data)) + (v8294_data * v8231_data)) + (v8482_data * v8419_data)) + (v8670_data * v8607_data)) + (v8858_data * v8795_data)) + (v9046_data * v8983_data)) + (v9234_data * v9171_data)) + (v9422_data * v9359_data)) + (v9610_data * v9547_data)) + (v9798_data * v9735_data);
          double v9837_p = ((((((((((v7917_data * v7857_data) + (v8106_data * v8048_data)) + (v8294_data * v8236_data)) + (v8482_data * v8424_data)) + (v8670_data * v8612_data)) + (v8858_data * v8800_data)) + (v9046_data * v8988_data)) + (v9234_data * v9176_data)) + (v9422_data * v9364_data)) + (v9610_data * v9552_data)) + (v9798_data * v9740_data);
          double v9842_p = ((((((((((v7917_data * v7862_data) + (v8106_data * v8053_data)) + (v8294_data * v8241_data)) + (v8482_data * v8429_data)) + (v8670_data * v8617_data)) + (v8858_data * v8805_data)) + (v9046_data * v8993_data)) + (v9234_data * v9181_data)) + (v9422_data * v9369_data)) + (v9610_data * v9557_data)) + (v9798_data * v9745_data);
          double v9845_data = v11409_g ? (glb_m4[(v11408_lead + 560)]) : (0.0);
          double v9849_p = ((((((((((v7965_data * v7822_data) + (v8153_data * v8013_data)) + (v8341_data * v8201_data)) + (v8529_data * v8389_data)) + (v8717_data * v8577_data)) + (v8905_data * v8765_data)) + (v9093_data * v8953_data)) + (v9281_data * v9141_data)) + (v9469_data * v9329_data)) + (v9657_data * v9517_data)) + (v9845_data * v9705_data);
          double v9854_p = ((((((((((v7965_data * v7827_data) + (v8153_data * v8018_data)) + (v8341_data * v8206_data)) + (v8529_data * v8394_data)) + (v8717_data * v8582_data)) + (v8905_data * v8770_data)) + (v9093_data * v8958_data)) + (v9281_data * v9146_data)) + (v9469_data * v9334_data)) + (v9657_data * v9522_data)) + (v9845_data * v9710_data);
          double v9859_p = ((((((((((v7965_data * v7832_data) + (v8153_data * v8023_data)) + (v8341_data * v8211_data)) + (v8529_data * v8399_data)) + (v8717_data * v8587_data)) + (v8905_data * v8775_data)) + (v9093_data * v8963_data)) + (v9281_data * v9151_data)) + (v9469_data * v9339_data)) + (v9657_data * v9527_data)) + (v9845_data * v9715_data);
          double v9864_p = ((((((((((v7965_data * v7837_data) + (v8153_data * v8028_data)) + (v8341_data * v8216_data)) + (v8529_data * v8404_data)) + (v8717_data * v8592_data)) + (v8905_data * v8780_data)) + (v9093_data * v8968_data)) + (v9281_data * v9156_data)) + (v9469_data * v9344_data)) + (v9657_data * v9532_data)) + (v9845_data * v9720_data);
          double v9869_p = ((((((((((v7965_data * v7842_data) + (v8153_data * v8033_data)) + (v8341_data * v8221_data)) + (v8529_data * v8409_data)) + (v8717_data * v8597_data)) + (v8905_data * v8785_data)) + (v9093_data * v8973_data)) + (v9281_data * v9161_data)) + (v9469_data * v9349_data)) + (v9657_data * v9537_data)) + (v9845_data * v9725_data);
          double v9874_p = ((((((((((v7965_data * v7847_data) + (v8153_data * v8038_data)) + (v8341_data * v8226_data)) + (v8529_data * v8414_data)) + (v8717_data * v8602_data)) + (v8905_data * v8790_data)) + (v9093_data * v8978_data)) + (v9281_data * v9166_data)) + (v9469_data * v9354_data)) + (v9657_data * v9542_data)) + (v9845_data * v9730_data);
          double v9879_p = ((((((((((v7965_data * v7852_data) + (v8153_data * v8043_data)) + (v8341_data * v8231_data)) + (v8529_data * v8419_data)) + (v8717_data * v8607_data)) + (v8905_data * v8795_data)) + (v9093_data * v8983_data)) + (v9281_data * v9171_data)) + (v9469_data * v9359_data)) + (v9657_data * v9547_data)) + (v9845_data * v9735_data);
          double v9884_p = ((((((((((v7965_data * v7857_data) + (v8153_data * v8048_data)) + (v8341_data * v8236_data)) + (v8529_data * v8424_data)) + (v8717_data * v8612_data)) + (v8905_data * v8800_data)) + (v9093_data * v8988_data)) + (v9281_data * v9176_data)) + (v9469_data * v9364_data)) + (v9657_data * v9552_data)) + (v9845_data * v9740_data);
          double v9889_p = ((((((((((v7965_data * v7862_data) + (v8153_data * v8053_data)) + (v8341_data * v8241_data)) + (v8529_data * v8429_data)) + (v8717_data * v8617_data)) + (v8905_data * v8805_data)) + (v9093_data * v8993_data)) + (v9281_data * v9181_data)) + (v9469_data * v9369_data)) + (v9657_data * v9557_data)) + (v9845_data * v9745_data);
          double v9892_data = glb_m4[(v11406_lead + 616)];
          double v9893_data = s2[11];
          ir4[0] = ((((((((((((v7821_data * v7822_data) + (v8012_data * v8013_data)) + (v8200_data * v8201_data)) + (v8388_data * v8389_data)) + (v8576_data * v8577_data)) + (v8764_data * v8765_data)) + (v8952_data * v8953_data)) + (v9140_data * v9141_data)) + (v9328_data * v9329_data)) + (v9516_data * v9517_data)) + (v9704_data * v9705_data)) + (v9892_data * v9893_data));
          double v9898_data = s2[32];
          ir4[4] = (v9713_p + (v9892_data * v9898_data));
          double v9903_data = s2[53];
          ir4[8] = (v9718_p + (v9892_data * v9903_data));
          double v9908_data = s2[74];
          ir4[12] = (v9723_p + (v9892_data * v9908_data));
          double v9913_data = s2[95];
          ir4[16] = (v9728_p + (v9892_data * v9913_data));
          double v9918_data = s2[116];
          ir4[20] = (v9733_p + (v9892_data * v9918_data));
          double v9923_data = s2[137];
          ir4[24] = (v9738_p + (v9892_data * v9923_data));
          double v9928_data = s2[158];
          ir4[28] = (v9743_p + (v9892_data * v9928_data));
          double v9933_data = s2[179];
          ir4[32] = (v9748_p + (v9892_data * v9933_data));
          double v9939_data = glb_m4[(v7867_lead + 616)];
          double v9986_data = glb_m4[(v7915_lead + 616)];
          double v10033_data = v11409_g ? (glb_m4[(v11408_lead + 616)]) : (0.0);
          double v10080_data = glb_m4[(v7867_lead + 656)];
          double v10081_data = s2[12];
          double v10086_data = s2[33];
          double v10091_data = s2[54];
          double v10096_data = s2[75];
          double v10101_data = s2[96];
          double v10106_data = s2[117];
          double v10111_data = s2[138];
          double v10116_data = s2[159];
          double v10121_data = s2[180];
          double v10127_data = glb_m4[(v7915_lead + 656)];
          double v10174_data = v11409_g ? (glb_m4[(v11408_lead + 656)]) : (0.0);
          double v10221_data = glb_m4[(v7867_lead + 696)];
          double v10222_data = s2[13];
          double v10227_data = s2[34];
          double v10232_data = s2[55];
          double v10237_data = s2[76];
          double v10242_data = s2[97];
          double v10247_data = s2[118];
          double v10252_data = s2[139];
          double v10257_data = s2[160];
          double v10262_data = s2[181];
          double v10268_data = glb_m4[(v7915_lead + 696)];
          double v10315_data = v11409_g ? (glb_m4[(v11408_lead + 696)]) : (0.0);
          double v10362_data = glb_m4[(v7867_lead + 736)];
          double v10363_data = s2[14];
          double v10368_data = s2[35];
          double v10373_data = s2[56];
          double v10378_data = s2[77];
          double v10383_data = s2[98];
          double v10388_data = s2[119];
          double v10393_data = s2[140];
          double v10398_data = s2[161];
          double v10403_data = s2[182];
          double v10409_data = glb_m4[(v7915_lead + 736)];
          double v10456_data = v11409_g ? (glb_m4[(v11408_lead + 736)]) : (0.0);
          double v10503_data = glb_m4[(v7867_lead + 776)];
          double v10504_data = s2[15];
          double v10509_data = s2[36];
          double v10514_data = s2[57];
          double v10519_data = s2[78];
          double v10524_data = s2[99];
          double v10529_data = s2[120];
          double v10534_data = s2[141];
          double v10539_data = s2[162];
          double v10544_data = s2[183];
          double v10550_data = glb_m4[(v7915_lead + 776)];
          double v10597_data = v11409_g ? (glb_m4[(v11408_lead + 776)]) : (0.0);
          double v10621_p = ((((v9869_p + (v10033_data * v9913_data)) + (v10174_data * v10101_data)) + (v10315_data * v10242_data)) + (v10456_data * v10383_data)) + (v10597_data * v10524_data);
          double v10626_p = ((((v9874_p + (v10033_data * v9918_data)) + (v10174_data * v10106_data)) + (v10315_data * v10247_data)) + (v10456_data * v10388_data)) + (v10597_data * v10529_data);
          double v10631_p = ((((v9879_p + (v10033_data * v9923_data)) + (v10174_data * v10111_data)) + (v10315_data * v10252_data)) + (v10456_data * v10393_data)) + (v10597_data * v10534_data);
          double v10636_p = ((((v9884_p + (v10033_data * v9928_data)) + (v10174_data * v10116_data)) + (v10315_data * v10257_data)) + (v10456_data * v10398_data)) + (v10597_data * v10539_data);
          double v10641_p = ((((v9889_p + (v10033_data * v9933_data)) + (v10174_data * v10121_data)) + (v10315_data * v10262_data)) + (v10456_data * v10403_data)) + (v10597_data * v10544_data);
          double v10644_data = glb_m4[(v7867_lead + 816)];
          double v10645_data = s2[16];
          double v10648_p = (((((v9755_p + (v9939_data * v9893_data)) + (v10080_data * v10081_data)) + (v10221_data * v10222_data)) + (v10362_data * v10363_data)) + (v10503_data * v10504_data)) + (v10644_data * v10645_data);
          double v10650_data = s2[37];
          double v10653_p = (((((v9760_p + (v9939_data * v9898_data)) + (v10080_data * v10086_data)) + (v10221_data * v10227_data)) + (v10362_data * v10368_data)) + (v10503_data * v10509_data)) + (v10644_data * v10650_data);
          double v10655_data = s2[58];
          double v10658_p = (((((v9765_p + (v9939_data * v9903_data)) + (v10080_data * v10091_data)) + (v10221_data * v10232_data)) + (v10362_data * v10373_data)) + (v10503_data * v10514_data)) + (v10644_data * v10655_data);
          double v10660_data = s2[79];
          double v10663_p = (((((v9770_p + (v9939_data * v9908_data)) + (v10080_data * v10096_data)) + (v10221_data * v10237_data)) + (v10362_data * v10378_data)) + (v10503_data * v10519_data)) + (v10644_data * v10660_data);
          double v10665_data = s2[100];
          double v10668_p = (((((v9775_p + (v9939_data * v9913_data)) + (v10080_data * v10101_data)) + (v10221_data * v10242_data)) + (v10362_data * v10383_data)) + (v10503_data * v10524_data)) + (v10644_data * v10665_data);
          double v10670_data = s2[121];
          double v10673_p = (((((v9780_p + (v9939_data * v9918_data)) + (v10080_data * v10106_data)) + (v10221_data * v10247_data)) + (v10362_data * v10388_data)) + (v10503_data * v10529_data)) + (v10644_data * v10670_data);
          double v10675_data = s2[142];
          double v10678_p = (((((v9785_p + (v9939_data * v9923_data)) + (v10080_data * v10111_data)) + (v10221_data * v10252_data)) + (v10362_data * v10393_data)) + (v10503_data * v10534_data)) + (v10644_data * v10675_data);
          double v10680_data = s2[163];
          double v10683_p = (((((v9790_p + (v9939_data * v9928_data)) + (v10080_data * v10116_data)) + (v10221_data * v10257_data)) + (v10362_data * v10398_data)) + (v10503_data * v10539_data)) + (v10644_data * v10680_data);
          double v10685_data = s2[184];
          double v10688_p = (((((v9795_p + (v9939_data * v9933_data)) + (v10080_data * v10121_data)) + (v10221_data * v10262_data)) + (v10362_data * v10403_data)) + (v10503_data * v10544_data)) + (v10644_data * v10685_data);
          double v10691_data = glb_m4[(v7915_lead + 816)];
          double v10695_p = (((((v9802_p + (v9986_data * v9893_data)) + (v10127_data * v10081_data)) + (v10268_data * v10222_data)) + (v10409_data * v10363_data)) + (v10550_data * v10504_data)) + (v10691_data * v10645_data);
          double v10700_p = (((((v9807_p + (v9986_data * v9898_data)) + (v10127_data * v10086_data)) + (v10268_data * v10227_data)) + (v10409_data * v10368_data)) + (v10550_data * v10509_data)) + (v10691_data * v10650_data);
          double v10705_p = (((((v9812_p + (v9986_data * v9903_data)) + (v10127_data * v10091_data)) + (v10268_data * v10232_data)) + (v10409_data * v10373_data)) + (v10550_data * v10514_data)) + (v10691_data * v10655_data);
          double v10710_p = (((((v9817_p + (v9986_data * v9908_data)) + (v10127_data * v10096_data)) + (v10268_data * v10237_data)) + (v10409_data * v10378_data)) + (v10550_data * v10519_data)) + (v10691_data * v10660_data);
          double v10715_p = (((((v9822_p + (v9986_data * v9913_data)) + (v10127_data * v10101_data)) + (v10268_data * v10242_data)) + (v10409_data * v10383_data)) + (v10550_data * v10524_data)) + (v10691_data * v10665_data);
          double v10720_p = (((((v9827_p + (v9986_data * v9918_data)) + (v10127_data * v10106_data)) + (v10268_data * v10247_data)) + (v10409_data * v10388_data)) + (v10550_data * v10529_data)) + (v10691_data * v10670_data);
          double v10725_p = (((((v9832_p + (v9986_data * v9923_data)) + (v10127_data * v10111_data)) + (v10268_data * v10252_data)) + (v10409_data * v10393_data)) + (v10550_data * v10534_data)) + (v10691_data * v10675_data);
          double v10730_p = (((((v9837_p + (v9986_data * v9928_data)) + (v10127_data * v10116_data)) + (v10268_data * v10257_data)) + (v10409_data * v10398_data)) + (v10550_data * v10539_data)) + (v10691_data * v10680_data);
          double v10735_p = (((((v9842_p + (v9986_data * v9933_data)) + (v10127_data * v10121_data)) + (v10268_data * v10262_data)) + (v10409_data * v10403_data)) + (v10550_data * v10544_data)) + (v10691_data * v10685_data);
          double v10738_data = v11409_g ? (glb_m4[(v11408_lead + 816)]) : (0.0);
          double v10742_p = (((((v9849_p + (v10033_data * v9893_data)) + (v10174_data * v10081_data)) + (v10315_data * v10222_data)) + (v10456_data * v10363_data)) + (v10597_data * v10504_data)) + (v10738_data * v10645_data);
          double v10747_p = (((((v9854_p + (v10033_data * v9898_data)) + (v10174_data * v10086_data)) + (v10315_data * v10227_data)) + (v10456_data * v10368_data)) + (v10597_data * v10509_data)) + (v10738_data * v10650_data);
          double v10752_p = (((((v9859_p + (v10033_data * v9903_data)) + (v10174_data * v10091_data)) + (v10315_data * v10232_data)) + (v10456_data * v10373_data)) + (v10597_data * v10514_data)) + (v10738_data * v10655_data);
          double v10757_p = (((((v9864_p + (v10033_data * v9908_data)) + (v10174_data * v10096_data)) + (v10315_data * v10237_data)) + (v10456_data * v10378_data)) + (v10597_data * v10519_data)) + (v10738_data * v10660_data);
          double v10760_p = v10738_data * v10665_data;
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v11348_i0 = 0; v11348_i0 < 3; ++v11348_i0) {
            int32_t v11351_lead = v11406_lead + (v11348_i0 * 16);
            #pragma unroll
            for (int32_t v11349_i1 = 0; v11349_i1 < 9; ++v11349_i1) {
              double v11354_data = glb_m3[(v11351_lead + (v11349_i1 * 56))];
              r3[(v11348_i0 + (v11349_i1 * 4))] = v11354_data;
            }
          }
          #pragma unroll
          for (int32_t v11357_i1 = 0; v11357_i1 < 9; ++v11357_i1) {
            double v11362_data = v11409_g ? (glb_m3[(v11408_lead + (v11357_i1 * 56))]) : (0.0);
            r3[(3 + (v11357_i1 * 4))] = v11362_data;
          }
          double v10785_data = glb_m4[(v7867_lead + 856)];
          double v10786_data = s2[17];
          double v10791_data = s2[38];
          double v10796_data = s2[59];
          double v10801_data = s2[80];
          double v10806_data = s2[101];
          double v10811_data = s2[122];
          double v10816_data = s2[143];
          double v10821_data = s2[164];
          double v10826_data = s2[185];
          double v10832_data = glb_m4[(v7915_lead + 856)];
          double v10879_data = v11409_g ? (glb_m4[(v11408_lead + 856)]) : (0.0);
          double v10926_data = glb_m4[(v7867_lead + 896)];
          double v10927_data = s2[18];
          double v10932_data = s2[39];
          double v10937_data = s2[60];
          double v10942_data = s2[81];
          double v10947_data = s2[102];
          double v10952_data = s2[123];
          double v10957_data = s2[144];
          double v10962_data = s2[165];
          double v10967_data = s2[186];
          double v10973_data = glb_m4[(v7915_lead + 896)];
          double v11020_data = v11409_g ? (glb_m4[(v11408_lead + 896)]) : (0.0);
          double v11067_data = glb_m4[(v7867_lead + 936)];
          double v11068_data = s2[19];
          double v11073_data = s2[40];
          double v11076_p = ((v10653_p + (v10785_data * v10791_data)) + (v10926_data * v10932_data)) + (v11067_data * v11073_data);
          double v11078_data = s2[61];
          double v11081_p = ((v10658_p + (v10785_data * v10796_data)) + (v10926_data * v10937_data)) + (v11067_data * v11078_data);
          double v11083_data = s2[82];
          double v11086_p = ((v10663_p + (v10785_data * v10801_data)) + (v10926_data * v10942_data)) + (v11067_data * v11083_data);
          double v11088_data = s2[103];
          double v11091_p = ((v10668_p + (v10785_data * v10806_data)) + (v10926_data * v10947_data)) + (v11067_data * v11088_data);
          double v11093_data = s2[124];
          double v11096_p = ((v10673_p + (v10785_data * v10811_data)) + (v10926_data * v10952_data)) + (v11067_data * v11093_data);
          double v11098_data = s2[145];
          double v11101_p = ((v10678_p + (v10785_data * v10816_data)) + (v10926_data * v10957_data)) + (v11067_data * v11098_data);
          double v11103_data = s2[166];
          double v11106_p = ((v10683_p + (v10785_data * v10821_data)) + (v10926_data * v10962_data)) + (v11067_data * v11103_data);
          double v11108_data = s2[187];
          double v11111_p = ((v10688_p + (v10785_data * v10826_data)) + (v10926_data * v10967_data)) + (v11067_data * v11108_data);
          double v11114_data = glb_m4[(v7915_lead + 936)];
          double v11118_p = ((v10695_p + (v10832_data * v10786_data)) + (v10973_data * v10927_data)) + (v11114_data * v11068_data);
          double v11123_p = ((v10700_p + (v10832_data * v10791_data)) + (v10973_data * v10932_data)) + (v11114_data * v11073_data);
          double v11128_p = ((v10705_p + (v10832_data * v10796_data)) + (v10973_data * v10937_data)) + (v11114_data * v11078_data);
          double v11133_p = ((v10710_p + (v10832_data * v10801_data)) + (v10973_data * v10942_data)) + (v11114_data * v11083_data);
          double v11138_p = ((v10715_p + (v10832_data * v10806_data)) + (v10973_data * v10947_data)) + (v11114_data * v11088_data);
          double v11143_p = ((v10720_p + (v10832_data * v10811_data)) + (v10973_data * v10952_data)) + (v11114_data * v11093_data);
          double v11148_p = ((v10725_p + (v10832_data * v10816_data)) + (v10973_data * v10957_data)) + (v11114_data * v11098_data);
          double v11153_p = ((v10730_p + (v10832_data * v10821_data)) + (v10973_data * v10962_data)) + (v11114_data * v11103_data);
          double v11158_p = ((v10735_p + (v10832_data * v10826_data)) + (v10973_data * v10967_data)) + (v11114_data * v11108_data);
          double v11161_data = v11409_g ? (glb_m4[(v11408_lead + 936)]) : (0.0);
          double v11165_p = ((v10742_p + (v10879_data * v10786_data)) + (v11020_data * v10927_data)) + (v11161_data * v11068_data);
          double v11170_p = ((v10747_p + (v10879_data * v10791_data)) + (v11020_data * v10932_data)) + (v11161_data * v11073_data);
          double v11175_p = ((v10752_p + (v10879_data * v10796_data)) + (v11020_data * v10937_data)) + (v11161_data * v11078_data);
          double v11180_p = ((v10757_p + (v10879_data * v10801_data)) + (v11020_data * v10942_data)) + (v11161_data * v11083_data);
          double v11185_p = (((v10621_p + v10760_p) + (v10879_data * v10806_data)) + (v11020_data * v10947_data)) + (v11161_data * v11088_data);
          double v11190_p = (((v10626_p + (v10738_data * v10670_data)) + (v10879_data * v10811_data)) + (v11020_data * v10952_data)) + (v11161_data * v11093_data);
          double v11195_p = (((v10631_p + (v10738_data * v10675_data)) + (v10879_data * v10816_data)) + (v11020_data * v10957_data)) + (v11161_data * v11098_data);
          double v11200_p = (((v10636_p + (v10738_data * v10680_data)) + (v10879_data * v10821_data)) + (v11020_data * v10962_data)) + (v11161_data * v11103_data);
          double v11205_p = (((v10641_p + (v10738_data * v10685_data)) + (v10879_data * v10826_data)) + (v11020_data * v10967_data)) + (v11161_data * v11108_data);
          double v11208_data = glb_m4[(v7867_lead + 976)];
          double v11209_data = s2[20];
          ir4[1] = ((((v10648_p + (v10785_data * v10786_data)) + (v10926_data * v10927_data)) + (v11067_data * v11068_data)) + (v11208_data * v11209_data));
          double v11214_data = s2[41];
          ir4[5] = (v11076_p + (v11208_data * v11214_data));
          double v11219_data = s2[62];
          ir4[9] = (v11081_p + (v11208_data * v11219_data));
          double v11224_data = s2[83];
          ir4[13] = (v11086_p + (v11208_data * v11224_data));
          double v11229_data = s2[104];
          ir4[17] = (v11091_p + (v11208_data * v11229_data));
          double v11234_data = s2[125];
          ir4[21] = (v11096_p + (v11208_data * v11234_data));
          double v11239_data = s2[146];
          ir4[25] = (v11101_p + (v11208_data * v11239_data));
          double v11244_data = s2[167];
          ir4[29] = (v11106_p + (v11208_data * v11244_data));
          double v11249_data = s2[188];
          ir4[33] = (v11111_p + (v11208_data * v11249_data));
          double v11255_data = glb_m4[(v7915_lead + 976)];
          ir4[2] = (v11118_p + (v11255_data * v11209_data));
          ir4[6] = (v11123_p + (v11255_data * v11214_data));
          ir4[10] = (v11128_p + (v11255_data * v11219_data));
          ir4[14] = (v11133_p + (v11255_data * v11224_data));
          ir4[18] = (v11138_p + (v11255_data * v11229_data));
          ir4[22] = (v11143_p + (v11255_data * v11234_data));
          ir4[26] = (v11148_p + (v11255_data * v11239_data));
          ir4[30] = (v11153_p + (v11255_data * v11244_data));
          ir4[34] = (v11158_p + (v11255_data * v11249_data));
          bool v11462_g = allowed_next && v11409_g;
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v11446_i0 = 0; v11446_i0 < 3; ++v11446_i0) {
            int32_t v11448_lead = v11406_lead + (v11446_i0 * 16);
            #pragma unroll
            for (int32_t v11449_i1 = 0; v11449_i1 < 9; ++v11449_i1) {
              double v11452_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v11448_lead + (v11449_i1 * 56))])) : (0.0);
              r0[(v11446_i0 + (v11449_i1 * 4))] = v11452_data;
            }
          }
          #pragma unroll
          for (int32_t v11455_i1 = 0; v11455_i1 < 9; ++v11455_i1) {
            double v11458_data = v11462_g ? (__ldcg(&wrap_glb_m0[(v11408_lead + (v11455_i1 * 56))])) : (0.0);
            r0[(3 + (v11455_i1 * 4))] = v11458_data;
          }
          double v11302_data = v11409_g ? (glb_m4[(v11408_lead + 976)]) : (0.0);
          ir4[3] = (v11165_p + (v11302_data * v11209_data));
          ir4[7] = (v11170_p + (v11302_data * v11214_data));
          ir4[11] = (v11175_p + (v11302_data * v11219_data));
          ir4[15] = (v11180_p + (v11302_data * v11224_data));
          ir4[19] = (v11185_p + (v11302_data * v11229_data));
          ir4[23] = (v11190_p + (v11302_data * v11234_data));
          ir4[27] = (v11195_p + (v11302_data * v11239_data));
          ir4[31] = (v11200_p + (v11302_data * v11244_data));
          ir4[35] = (v11205_p + (v11302_data * v11249_data));
          // r4 = ir4 + r3
          #pragma unroll
          for (int32_t v11365_n0 = 0; v11365_n0 < 3; ++v11365_n0) {
            #pragma unroll
            for (int32_t v11366_n1 = 0; v11366_n1 < 9; ++v11366_n1) {
              int32_t v11368_a = v11365_n0 + (v11366_n1 * 4);
              double v11369_data = ir4[v11368_a];
              double v11370_data = r3[v11368_a];
              r4[v11368_a] = (v11370_data + v11369_data);
            }
          }
          if (v11409_g) {
            #pragma unroll
            for (int32_t v11372_n1 = 0; v11372_n1 < 9; ++v11372_n1) {
              int32_t v11374_a = 3 + (v11372_n1 * 4);
              double v11375_data = ir4[v11374_a];
              double v11376_data = r3[v11374_a];
              r4[v11374_a] = (v11376_data + v11375_data);
            }
          }
          // glb_m3 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v11378_i0 = 0; v11378_i0 < 3; ++v11378_i0) {
            int32_t v11384_lead = v11406_lead + (v11378_i0 * 16);
            #pragma unroll
            for (int32_t v11379_i1 = 0; v11379_i1 < 9; ++v11379_i1) {
              double v11382_data = r4[(v11378_i0 + (v11379_i1 * 4))];
              glb_m3[(v11384_lead + (v11379_i1 * 56))] = v11382_data;
            }
          }
          if (v11409_g) {
            #pragma unroll
            for (int32_t v11387_i1 = 0; v11387_i1 < 9; ++v11387_i1) {
              double v11390_data = r4[(3 + (v11387_i1 * 4))];
              glb_m3[(v11408_lead + (v11387_i1 * 56))] = v11390_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          bool v11463_g = allowed_next && v11409_g;
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v11416_i0 = 0; v11416_i0 < 3; ++v11416_i0) {
            int32_t v11418_lead = v11406_lead + (v11416_i0 * 16);
            #pragma unroll
            for (int32_t v11419_i1 = 0; v11419_i1 < 9; ++v11419_i1) {
              double v11422_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v11418_lead + (v11419_i1 * 56))])) : (0.0);
              r0[(v11416_i0 + (v11419_i1 * 4))] = v11422_data;
            }
          }
          #pragma unroll
          for (int32_t v11425_i1 = 0; v11425_i1 < 9; ++v11425_i1) {
            double v11428_data = v11463_g ? (__ldcg(&wrap_glb_m0[(v11408_lead + (v11425_i1 * 56))])) : (0.0);
            r0[(3 + (v11425_i1 * 4))] = v11428_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m1[0, 1])
        #pragma unroll
        for (int32_t v11473_i = 0; v11473_i < 5; ++v11473_i) {
          int32_t v11475_a = threadIdx.x + (v11473_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v11475_a], &wrap_glb_m1[v11475_a], 8);
          }
        }
        if (v11465_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &wrap_glb_m1[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
        }
        __pipeline_commit();
        v11400_flagWord = v11401_flagWordNext;
        v11401_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

