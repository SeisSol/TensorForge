// === base name ===
kernel_59c099b0aa2e688d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_59c099b0aa2e688d = {{16, 8, 1}, 16, 55, 1, 8, 21504, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_59c099b0aa2e688d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_59c099b0aa2e688d(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_59c099b0aa2e688d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_59c099b0aa2e688d, block.x * block.y * block.z, 2688 * sizeof(double));
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
  config.sharedMemBytes = 2688 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_59c099b0aa2e688d(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_59c099b0aa2e688d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_59c099b0aa2e688d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const double *const * m0Arg = (const double *const *)m0;
  const double *const * m1Arg = (const double *const *)m1;
  double *const * m2Arg = (double *const *)m2;
  const double *const * m4Arg = (const double *const *)m4;
  const double *const * m6Arg = (const double *const *)m6;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_59c099b0aa2e688d<<<grid,block,config.sharedMemBytes,stream>>>(m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3, m4Arg, m4_extraOffset, m5, m6Arg, m6_extraOffset, m7, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_59c099b0aa2e688d(const double *const * __restrict__ m0, size_t m0_extraOffset, const double *const * __restrict__ m1, size_t m1_extraOffset, double *const * __restrict__ m2, size_t m2_extraOffset, const double * m3, const double *const * __restrict__ m4, size_t m4_extraOffset, const double * m5, const double *const * __restrict__ m6, size_t m6_extraOffset, const double * m7, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (55 active) x 8 per block = block 16x8x1, 21504 B shared, occupancy grid
    // operands:
    //   m0 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m2 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m3 56×56(56×35) {0..56}×{0..35} none ordered runs-packed
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m5 56×56(56×35) {0..56}×{0..35} none ordered runs-packed
    //   m6 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m7 56×56(56×35) {0..56}×{0..35} none ordered runs-packed
    // operations:
    //   t0[i,j]@{0..35}×{0..9} = m0[i,k]@{0..35}×{0..9} × m1[k,j]
    //   m2[i,j]@{1..54}×{0..9} += m3[i,k]@{1..54}×{0..35} × t0[k,j]@{0..35}×{0..9}
    //   t1[i,j]@{0..35}×{0..9} = m0[i,k]@{0..35}×{0..9} × m4[k,j]
    //   m2[i,j]@{1..55}×{0..9} += m5[i,k]@{1..55}×{0..35} × t1[k,j]@{0..35}×{0..9}
    //   t2[i,j]@{0..35}×{0..9} = m0[i,k]@{0..35}×{0..9} × m6[k,j]
    //   m2[i,j]@{1..56}×{0..9} += m7[i,k]@{1..56}×{0..35} × t2[k,j]@{0..35}×{0..9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":55,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2688}],"shared_bytes":21504,"shared_elements":2688,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[56,9]],"name":"m0","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"pointer_based","alias":"star0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[56,9]],"name":"m2","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"none","alias":"kDivM0","bbox":[[0,0],[56,35]],"name":"m3","ordered":true,"parts":1,"shape":[56,56],"variant":false},{"addressing":"pointer_based","alias":"star1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"kDivM1","bbox":[[0,0],[56,35]],"name":"m5","ordered":true,"parts":1,"shape":[56,56],"variant":false},{"addressing":"pointer_based","alias":"star2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"kDivM2","bbox":[[0,0],[56,35]],"name":"m7","ordered":true,"parts":1,"shape":[56,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[54,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[54,35]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[55,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[55,35]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[56,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[56,35]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[336 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const double *const __restrict__ glb_m3 = &m3[0];
      const double *const __restrict__ glb_m5 = &m5[0];
      const double *const __restrict__ glb_m7 = &m7[0];
      double * __restrict__ s0 = &localShrMem0[0];
      double * __restrict__ s1 = &localShrMem0[0];
      double * __restrict__ s2 = &localShrMem0[0];
      double * __restrict__ s3 = &localShrMem0[0];
      double * __restrict__ s4 = &localShrMem0[0];
      double * __restrict__ s5 = &localShrMem0[0];
      double r0[27]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v18970_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v18970_flagSecondAhead < numElements0) ? v18970_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1][0 + m0_extraOffset];
      int32_t v18980_lead = threadIdx.x % 16;
      bool v18981_g = v18980_lead < 3;
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v19003_i0 = 0; v19003_i0 < 2; ++v19003_i0) {
        int32_t v19005_lead = v18980_lead + (v19003_i0 * 16);
        #pragma unroll
        for (int32_t v19006_i1 = 0; v19006_i1 < 9; ++v19006_i1) {
          double v19009_data = allowed_peel ? (__ldcg(&peel_glb_m0[(v19005_lead + (v19006_i1 * 56))])) : (0.0);
          r0[(v19003_i0 + (v19006_i1 * 3))] = v19009_data;
        }
      }
      if (v18981_g) {
        int32_t v19013_lead = v18980_lead + 32_i32;
        #pragma unroll
        for (int32_t v19014_i1 = 0; v19014_i1 < 9; ++v19014_i1) {
          double v19017_data = allowed_peel ? (__ldcg(&peel_glb_m0[(v19013_lead + (v19014_i1 * 56))])) : (0.0);
          r0[(2 + (v19014_i1 * 3))] = v19017_data;
        }
      }
      const double *const __restrict__ peel_glb_m1 = &m1[v3_batchId1][0 + m1_extraOffset];
      bool v19038_g = threadIdx.x < 1;
      // s0 = load{g>s}(peel_glb_m1[0, 1])
      #pragma unroll
      for (int32_t v19041_i = 0; v19041_i < 5; ++v19041_i) {
        int32_t v19043_a = threadIdx.x + (v19041_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v19043_a], &peel_glb_m1[v19043_a], 8);
        }
      }
      if (v19038_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &peel_glb_m1[0 + 0 + 1 * threadIdx.x + 80], 8);
        }
      }
      __pipeline_commit();
      uint32_t v18974_flagWord = flagWordFirst;
      uint32_t v18975_flagWordNext = flagWordSecond;
      for (size_t v16_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v16_batchId0 < numElements0; v16_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v17_ahead1 = v16_batchId0 + (gridDim.x * blockDim.y);
        size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
        const double *const __restrict__ wrap_glb_m0 = &m0[v19_batchId1][0 + m0_extraOffset];
        const double *const __restrict__ wrap_glb_m1 = &m1[v19_batchId1][0 + m1_extraOffset];
        size_t v19057_flagAheadAhead = v19_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v19057_flagAheadAhead < numElements0) ? v19057_flagAheadAhead : v19_batchId1)];
        const bool allowed = static_cast<bool>(v18974_flagWord);
        const bool allowed_next = static_cast<bool>(v18975_flagWordNext);
        if (allowed) {
          double *const __restrict__ glb_m2 = &m2[v16_batchId0][0 + m2_extraOffset];
          const double *const __restrict__ glb_m4 = &m4[v16_batchId0][0 + m4_extraOffset];
          const double *const __restrict__ glb_m6 = &m6[v16_batchId0][0 + m6_extraOffset];
          double r2[36]{};
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          double r1[27]{};
          // r1 = +(r0 * s0) + None
          // [(0, 35), (0, 9)] [(0, 9)]
          double v58_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v59_data = s0[0];
          double v64_data = s0[9];
          double v69_data = s0[18];
          double v74_data = s0[27];
          double v79_data = s0[36];
          double v84_data = s0[45];
          double v89_data = s0[54];
          double v94_data = s0[63];
          double v99_data = s0[72];
          double v103_data = r0[1];
          double v148_data = r0[2];
          double v193_data = r0[3];
          double v194_data = s0[1];
          double v199_data = s0[10];
          double v204_data = s0[19];
          double v209_data = s0[28];
          double v214_data = s0[37];
          double v219_data = s0[46];
          double v224_data = s0[55];
          double v229_data = s0[64];
          double v234_data = s0[73];
          double v238_data = r0[4];
          double v283_data = r0[5];
          double v328_data = r0[6];
          double v329_data = s0[2];
          double v334_data = s0[11];
          double v339_data = s0[20];
          double v344_data = s0[29];
          double v349_data = s0[38];
          double v354_data = s0[47];
          double v359_data = s0[56];
          double v364_data = s0[65];
          double v369_data = s0[74];
          double v373_data = r0[7];
          double v418_data = r0[8];
          double v463_data = r0[9];
          double v464_data = s0[3];
          double v469_data = s0[12];
          double v474_data = s0[21];
          double v479_data = s0[30];
          double v484_data = s0[39];
          double v489_data = s0[48];
          double v494_data = s0[57];
          double v499_data = s0[66];
          double v504_data = s0[75];
          double v508_data = r0[10];
          double v553_data = r0[11];
          double v598_data = r0[12];
          double v599_data = s0[4];
          double v604_data = s0[13];
          double v609_data = s0[22];
          double v614_data = s0[31];
          double v619_data = s0[40];
          double v624_data = s0[49];
          double v629_data = s0[58];
          double v634_data = s0[67];
          double v639_data = s0[76];
          double v643_data = r0[13];
          double v688_data = r0[14];
          double v733_data = r0[15];
          double v734_data = s0[5];
          double v739_data = s0[14];
          double v744_data = s0[23];
          double v749_data = s0[32];
          double v754_data = s0[41];
          double v759_data = s0[50];
          double v764_data = s0[59];
          double v769_data = s0[68];
          double v774_data = s0[77];
          double v778_data = r0[16];
          double v823_data = r0[17];
          double v868_data = r0[18];
          double v869_data = s0[6];
          double v874_data = s0[15];
          double v879_data = s0[24];
          double v884_data = s0[33];
          double v889_data = s0[42];
          double v894_data = s0[51];
          double v899_data = s0[60];
          double v904_data = s0[69];
          double v909_data = s0[78];
          double v913_data = r0[19];
          double v958_data = r0[20];
          double v1003_data = r0[21];
          double v1004_data = s0[7];
          double v1009_data = s0[16];
          double v1012_p = (((((((v58_data * v64_data) + (v193_data * v199_data)) + (v328_data * v334_data)) + (v463_data * v469_data)) + (v598_data * v604_data)) + (v733_data * v739_data)) + (v868_data * v874_data)) + (v1003_data * v1009_data);
          double v1014_data = s0[25];
          double v1017_p = (((((((v58_data * v69_data) + (v193_data * v204_data)) + (v328_data * v339_data)) + (v463_data * v474_data)) + (v598_data * v609_data)) + (v733_data * v744_data)) + (v868_data * v879_data)) + (v1003_data * v1014_data);
          double v1019_data = s0[34];
          double v1022_p = (((((((v58_data * v74_data) + (v193_data * v209_data)) + (v328_data * v344_data)) + (v463_data * v479_data)) + (v598_data * v614_data)) + (v733_data * v749_data)) + (v868_data * v884_data)) + (v1003_data * v1019_data);
          double v1024_data = s0[43];
          double v1027_p = (((((((v58_data * v79_data) + (v193_data * v214_data)) + (v328_data * v349_data)) + (v463_data * v484_data)) + (v598_data * v619_data)) + (v733_data * v754_data)) + (v868_data * v889_data)) + (v1003_data * v1024_data);
          double v1029_data = s0[52];
          double v1032_p = (((((((v58_data * v84_data) + (v193_data * v219_data)) + (v328_data * v354_data)) + (v463_data * v489_data)) + (v598_data * v624_data)) + (v733_data * v759_data)) + (v868_data * v894_data)) + (v1003_data * v1029_data);
          double v1034_data = s0[61];
          double v1037_p = (((((((v58_data * v89_data) + (v193_data * v224_data)) + (v328_data * v359_data)) + (v463_data * v494_data)) + (v598_data * v629_data)) + (v733_data * v764_data)) + (v868_data * v899_data)) + (v1003_data * v1034_data);
          double v1039_data = s0[70];
          double v1042_p = (((((((v58_data * v94_data) + (v193_data * v229_data)) + (v328_data * v364_data)) + (v463_data * v499_data)) + (v598_data * v634_data)) + (v733_data * v769_data)) + (v868_data * v904_data)) + (v1003_data * v1039_data);
          double v1044_data = s0[79];
          double v1047_p = (((((((v58_data * v99_data) + (v193_data * v234_data)) + (v328_data * v369_data)) + (v463_data * v504_data)) + (v598_data * v639_data)) + (v733_data * v774_data)) + (v868_data * v909_data)) + (v1003_data * v1044_data);
          double v1048_data = r0[22];
          double v1052_p = (((((((v103_data * v59_data) + (v238_data * v194_data)) + (v373_data * v329_data)) + (v508_data * v464_data)) + (v643_data * v599_data)) + (v778_data * v734_data)) + (v913_data * v869_data)) + (v1048_data * v1004_data);
          double v1057_p = (((((((v103_data * v64_data) + (v238_data * v199_data)) + (v373_data * v334_data)) + (v508_data * v469_data)) + (v643_data * v604_data)) + (v778_data * v739_data)) + (v913_data * v874_data)) + (v1048_data * v1009_data);
          double v1062_p = (((((((v103_data * v69_data) + (v238_data * v204_data)) + (v373_data * v339_data)) + (v508_data * v474_data)) + (v643_data * v609_data)) + (v778_data * v744_data)) + (v913_data * v879_data)) + (v1048_data * v1014_data);
          double v1067_p = (((((((v103_data * v74_data) + (v238_data * v209_data)) + (v373_data * v344_data)) + (v508_data * v479_data)) + (v643_data * v614_data)) + (v778_data * v749_data)) + (v913_data * v884_data)) + (v1048_data * v1019_data);
          double v1072_p = (((((((v103_data * v79_data) + (v238_data * v214_data)) + (v373_data * v349_data)) + (v508_data * v484_data)) + (v643_data * v619_data)) + (v778_data * v754_data)) + (v913_data * v889_data)) + (v1048_data * v1024_data);
          double v1077_p = (((((((v103_data * v84_data) + (v238_data * v219_data)) + (v373_data * v354_data)) + (v508_data * v489_data)) + (v643_data * v624_data)) + (v778_data * v759_data)) + (v913_data * v894_data)) + (v1048_data * v1029_data);
          double v1082_p = (((((((v103_data * v89_data) + (v238_data * v224_data)) + (v373_data * v359_data)) + (v508_data * v494_data)) + (v643_data * v629_data)) + (v778_data * v764_data)) + (v913_data * v899_data)) + (v1048_data * v1034_data);
          double v1087_p = (((((((v103_data * v94_data) + (v238_data * v229_data)) + (v373_data * v364_data)) + (v508_data * v499_data)) + (v643_data * v634_data)) + (v778_data * v769_data)) + (v913_data * v904_data)) + (v1048_data * v1039_data);
          double v1092_p = (((((((v103_data * v99_data) + (v238_data * v234_data)) + (v373_data * v369_data)) + (v508_data * v504_data)) + (v643_data * v639_data)) + (v778_data * v774_data)) + (v913_data * v909_data)) + (v1048_data * v1044_data);
          double v1093_data = r0[23];
          double v1138_data = r0[24];
          double v1139_data = s0[8];
          r1[0] = (((((((((v58_data * v59_data) + (v193_data * v194_data)) + (v328_data * v329_data)) + (v463_data * v464_data)) + (v598_data * v599_data)) + (v733_data * v734_data)) + (v868_data * v869_data)) + (v1003_data * v1004_data)) + (v1138_data * v1139_data));
          double v1144_data = s0[17];
          r1[3] = (v1012_p + (v1138_data * v1144_data));
          double v1149_data = s0[26];
          r1[6] = (v1017_p + (v1138_data * v1149_data));
          double v1154_data = s0[35];
          r1[9] = (v1022_p + (v1138_data * v1154_data));
          double v1159_data = s0[44];
          r1[12] = (v1027_p + (v1138_data * v1159_data));
          double v1164_data = s0[53];
          r1[15] = (v1032_p + (v1138_data * v1164_data));
          double v1169_data = s0[62];
          r1[18] = (v1037_p + (v1138_data * v1169_data));
          double v1174_data = s0[71];
          r1[21] = (v1042_p + (v1138_data * v1174_data));
          double v1179_data = s0[80];
          r1[24] = (v1047_p + (v1138_data * v1179_data));
          double v1183_data = r0[25];
          r1[1] = (v1052_p + (v1183_data * v1139_data));
          r1[4] = (v1057_p + (v1183_data * v1144_data));
          r1[7] = (v1062_p + (v1183_data * v1149_data));
          r1[10] = (v1067_p + (v1183_data * v1154_data));
          r1[13] = (v1072_p + (v1183_data * v1159_data));
          r1[16] = (v1077_p + (v1183_data * v1164_data));
          r1[19] = (v1082_p + (v1183_data * v1169_data));
          r1[22] = (v1087_p + (v1183_data * v1174_data));
          r1[25] = (v1092_p + (v1183_data * v1179_data));
          double v1228_data = r0[26];
          // r1 = 0 past (0, 35)
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v1274_i0 = 0; v1274_i0 < 2; ++v1274_i0) {
            int32_t v1280_lead = v18980_lead + (v1274_i0 * 16);
            #pragma unroll
            for (int32_t v1275_i1 = 0; v1275_i1 < 9; ++v1275_i1) {
              double v1278_data = r1[(v1274_i0 + (v1275_i1 * 3))];
              s1[(v1280_lead + (v1275_i1 * 35))] = v1278_data;
            }
          }
          double r3[36]{};
          // ir3 = +(glb_m3 * s1)
          // [(1, 54), (0, 9)] [(0, 35)]
          double ir3[36]{};
          int32_t v1295_off = v18980_lead + -1;
          // r2 = load{g>r}(glb_m2);
          bool v5629_g = v18980_lead >= 1;
          if (v5629_g) {
            #pragma unroll
            for (int32_t v5630_i1 = 0; v5630_i1 < 9; ++v5630_i1) {
              double v5635_data = glb_m2[(v18980_lead + (v5630_i1 * 56))];
              r2[(v5630_i1 * 4)] = v5635_data;
            }
          }
          #pragma unroll
          for (int32_t v5638_i0 = 1; v5638_i0 < 3; ++v5638_i0) {
            int32_t v5641_lead = v18980_lead + (v5638_i0 * 16);
            #pragma unroll
            for (int32_t v5639_i1 = 0; v5639_i1 < 9; ++v5639_i1) {
              double v5644_data = glb_m2[(v5641_lead + (v5639_i1 * 56))];
              r2[(v5638_i0 + (v5639_i1 * 4))] = v5644_data;
            }
          }
          bool v5647_g = v18980_lead < 6;
          if (v5647_g) {
            int32_t v5650_lead = v18980_lead + 48_i32;
            #pragma unroll
            for (int32_t v5648_i1 = 0; v5648_i1 < 9; ++v5648_i1) {
              double v5653_data = glb_m2[(v5650_lead + (v5648_i1 * 56))];
              r2[(3 + (v5648_i1 * 4))] = v5653_data;
            }
          }
          double v1298_data = v5629_g ? (glb_m3[v1295_off]) : (0.0);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v1299_data = s1[0];
          double v1304_data = s1[35];
          double v1309_data = s1[70];
          double v1314_data = s1[105];
          double v1319_data = s1[140];
          double v1324_data = s1[175];
          double v1329_data = s1[210];
          double v1334_data = s1[245];
          double v1339_data = s1[280];
          int32_t v1344_lead = v18980_lead + 16_i32;
          int32_t v1345_off = v1344_lead + -1;
          double v1347_data = glb_m3[v1345_off];
          int32_t v1393_lead = v18980_lead + 32_i32;
          int32_t v1394_off = v1393_lead + -1;
          double v1396_data = glb_m3[v1394_off];
          int32_t v1442_lead = v18980_lead + 48_i32;
          int32_t v1443_off = v1442_lead + -1;
          double v1446_data = v5647_g ? (glb_m3[v1443_off]) : (0.0);
          double v1493_data = v5629_g ? (glb_m3[(v18980_lead + 52)]) : (0.0);
          double v1494_data = s1[1];
          double v1499_data = s1[36];
          double v1504_data = s1[71];
          double v1509_data = s1[106];
          double v1514_data = s1[141];
          double v1519_data = s1[176];
          double v1524_data = s1[211];
          double v1529_data = s1[246];
          double v1534_data = s1[281];
          double v1540_data = glb_m3[(v1344_lead + 52)];
          double v1587_data = glb_m3[(v1393_lead + 52)];
          double v1634_data = v5647_g ? (glb_m3[(v1442_lead + 52)]) : (0.0);
          double v1681_data = v5629_g ? (glb_m3[(v18980_lead + 105)]) : (0.0);
          double v1682_data = s1[2];
          double v1687_data = s1[37];
          double v1692_data = s1[72];
          double v1697_data = s1[107];
          double v1702_data = s1[142];
          double v1707_data = s1[177];
          double v1712_data = s1[212];
          double v1717_data = s1[247];
          double v1722_data = s1[282];
          double v1728_data = glb_m3[(v1344_lead + 105)];
          double v1775_data = glb_m3[(v1393_lead + 105)];
          double v1822_data = v5647_g ? (glb_m3[(v1442_lead + 105)]) : (0.0);
          double v1869_data = v5629_g ? (glb_m3[(v18980_lead + 158)]) : (0.0);
          double v1870_data = s1[3];
          double v1875_data = s1[38];
          double v1880_data = s1[73];
          double v1885_data = s1[108];
          double v1890_data = s1[143];
          double v1895_data = s1[178];
          double v1900_data = s1[213];
          double v1905_data = s1[248];
          double v1910_data = s1[283];
          double v1916_data = glb_m3[(v1344_lead + 158)];
          double v1963_data = glb_m3[(v1393_lead + 158)];
          double v2010_data = v5647_g ? (glb_m3[(v1442_lead + 158)]) : (0.0);
          double v2057_data = v5629_g ? (glb_m3[(v18980_lead + 211)]) : (0.0);
          double v2058_data = s1[4];
          double v2063_data = s1[39];
          double v2068_data = s1[74];
          double v2073_data = s1[109];
          double v2078_data = s1[144];
          double v2083_data = s1[179];
          double v2088_data = s1[214];
          double v2093_data = s1[249];
          double v2098_data = s1[284];
          double v2104_data = glb_m3[(v1344_lead + 211)];
          double v2151_data = glb_m3[(v1393_lead + 211)];
          double v2198_data = v5647_g ? (glb_m3[(v1442_lead + 211)]) : (0.0);
          double v2245_data = v5629_g ? (glb_m3[(v18980_lead + 264)]) : (0.0);
          double v2246_data = s1[5];
          double v2251_data = s1[40];
          double v2256_data = s1[75];
          double v2261_data = s1[110];
          double v2266_data = s1[145];
          double v2271_data = s1[180];
          double v2276_data = s1[215];
          double v2281_data = s1[250];
          double v2286_data = s1[285];
          double v2292_data = glb_m3[(v1344_lead + 264)];
          double v2339_data = glb_m3[(v1393_lead + 264)];
          double v2386_data = v5647_g ? (glb_m3[(v1442_lead + 264)]) : (0.0);
          double v2433_data = v5629_g ? (glb_m3[(v18980_lead + 317)]) : (0.0);
          double v2434_data = s1[6];
          double v2439_data = s1[41];
          double v2444_data = s1[76];
          double v2449_data = s1[111];
          double v2454_data = s1[146];
          double v2459_data = s1[181];
          double v2464_data = s1[216];
          double v2469_data = s1[251];
          double v2474_data = s1[286];
          double v2480_data = glb_m3[(v1344_lead + 317)];
          double v2527_data = glb_m3[(v1393_lead + 317)];
          double v2574_data = v5647_g ? (glb_m3[(v1442_lead + 317)]) : (0.0);
          double v2621_data = v5629_g ? (glb_m3[(v18980_lead + 370)]) : (0.0);
          double v2622_data = s1[7];
          double v2627_data = s1[42];
          double v2632_data = s1[77];
          double v2637_data = s1[112];
          double v2642_data = s1[147];
          double v2647_data = s1[182];
          double v2652_data = s1[217];
          double v2657_data = s1[252];
          double v2662_data = s1[287];
          double v2668_data = glb_m3[(v1344_lead + 370)];
          double v2715_data = glb_m3[(v1393_lead + 370)];
          double v2762_data = v5647_g ? (glb_m3[(v1442_lead + 370)]) : (0.0);
          bool v2809_g = v18980_lead >= 8;
          double v2810_data = v2809_g ? (glb_m3[(v18980_lead + 416)]) : (0.0);
          double v2811_data = s1[8];
          double v2816_data = s1[43];
          double v2821_data = s1[78];
          double v2826_data = s1[113];
          double v2831_data = s1[148];
          double v2836_data = s1[183];
          double v2841_data = s1[218];
          double v2846_data = s1[253];
          double v2851_data = s1[288];
          double v2857_data = glb_m3[(v1344_lead + 416)];
          double v2904_data = glb_m3[(v1393_lead + 416)];
          double v2951_data = v5647_g ? (glb_m3[(v1442_lead + 416)]) : (0.0);
          double v2998_data = v2809_g ? (glb_m3[(v18980_lead + 462)]) : (0.0);
          double v2999_data = s1[9];
          double v3004_data = s1[44];
          double v3009_data = s1[79];
          double v3014_data = s1[114];
          double v3019_data = s1[149];
          double v3024_data = s1[184];
          double v3029_data = s1[219];
          double v3034_data = s1[254];
          double v3039_data = s1[289];
          double v3045_data = glb_m3[(v1344_lead + 462)];
          double v3092_data = glb_m3[(v1393_lead + 462)];
          double v3139_data = v5647_g ? (glb_m3[(v1442_lead + 462)]) : (0.0);
          double v3186_data = v2809_g ? (glb_m3[(v18980_lead + 508)]) : (0.0);
          double v3187_data = s1[10];
          double v3192_data = s1[45];
          double v3195_p = ((((((((((v1298_data * v1304_data) + (v1493_data * v1499_data)) + (v1681_data * v1687_data)) + (v1869_data * v1875_data)) + (v2057_data * v2063_data)) + (v2245_data * v2251_data)) + (v2433_data * v2439_data)) + (v2621_data * v2627_data)) + (v2810_data * v2816_data)) + (v2998_data * v3004_data)) + (v3186_data * v3192_data);
          double v3197_data = s1[80];
          double v3200_p = ((((((((((v1298_data * v1309_data) + (v1493_data * v1504_data)) + (v1681_data * v1692_data)) + (v1869_data * v1880_data)) + (v2057_data * v2068_data)) + (v2245_data * v2256_data)) + (v2433_data * v2444_data)) + (v2621_data * v2632_data)) + (v2810_data * v2821_data)) + (v2998_data * v3009_data)) + (v3186_data * v3197_data);
          double v3202_data = s1[115];
          double v3205_p = ((((((((((v1298_data * v1314_data) + (v1493_data * v1509_data)) + (v1681_data * v1697_data)) + (v1869_data * v1885_data)) + (v2057_data * v2073_data)) + (v2245_data * v2261_data)) + (v2433_data * v2449_data)) + (v2621_data * v2637_data)) + (v2810_data * v2826_data)) + (v2998_data * v3014_data)) + (v3186_data * v3202_data);
          double v3207_data = s1[150];
          double v3210_p = ((((((((((v1298_data * v1319_data) + (v1493_data * v1514_data)) + (v1681_data * v1702_data)) + (v1869_data * v1890_data)) + (v2057_data * v2078_data)) + (v2245_data * v2266_data)) + (v2433_data * v2454_data)) + (v2621_data * v2642_data)) + (v2810_data * v2831_data)) + (v2998_data * v3019_data)) + (v3186_data * v3207_data);
          double v3212_data = s1[185];
          double v3215_p = ((((((((((v1298_data * v1324_data) + (v1493_data * v1519_data)) + (v1681_data * v1707_data)) + (v1869_data * v1895_data)) + (v2057_data * v2083_data)) + (v2245_data * v2271_data)) + (v2433_data * v2459_data)) + (v2621_data * v2647_data)) + (v2810_data * v2836_data)) + (v2998_data * v3024_data)) + (v3186_data * v3212_data);
          double v3217_data = s1[220];
          double v3220_p = ((((((((((v1298_data * v1329_data) + (v1493_data * v1524_data)) + (v1681_data * v1712_data)) + (v1869_data * v1900_data)) + (v2057_data * v2088_data)) + (v2245_data * v2276_data)) + (v2433_data * v2464_data)) + (v2621_data * v2652_data)) + (v2810_data * v2841_data)) + (v2998_data * v3029_data)) + (v3186_data * v3217_data);
          double v3222_data = s1[255];
          double v3225_p = ((((((((((v1298_data * v1334_data) + (v1493_data * v1529_data)) + (v1681_data * v1717_data)) + (v1869_data * v1905_data)) + (v2057_data * v2093_data)) + (v2245_data * v2281_data)) + (v2433_data * v2469_data)) + (v2621_data * v2657_data)) + (v2810_data * v2846_data)) + (v2998_data * v3034_data)) + (v3186_data * v3222_data);
          double v3227_data = s1[290];
          double v3230_p = ((((((((((v1298_data * v1339_data) + (v1493_data * v1534_data)) + (v1681_data * v1722_data)) + (v1869_data * v1910_data)) + (v2057_data * v2098_data)) + (v2245_data * v2286_data)) + (v2433_data * v2474_data)) + (v2621_data * v2662_data)) + (v2810_data * v2851_data)) + (v2998_data * v3039_data)) + (v3186_data * v3227_data);
          double v3233_data = glb_m3[(v1344_lead + 508)];
          double v3237_p = ((((((((((v1347_data * v1299_data) + (v1540_data * v1494_data)) + (v1728_data * v1682_data)) + (v1916_data * v1870_data)) + (v2104_data * v2058_data)) + (v2292_data * v2246_data)) + (v2480_data * v2434_data)) + (v2668_data * v2622_data)) + (v2857_data * v2811_data)) + (v3045_data * v2999_data)) + (v3233_data * v3187_data);
          double v3242_p = ((((((((((v1347_data * v1304_data) + (v1540_data * v1499_data)) + (v1728_data * v1687_data)) + (v1916_data * v1875_data)) + (v2104_data * v2063_data)) + (v2292_data * v2251_data)) + (v2480_data * v2439_data)) + (v2668_data * v2627_data)) + (v2857_data * v2816_data)) + (v3045_data * v3004_data)) + (v3233_data * v3192_data);
          double v3247_p = ((((((((((v1347_data * v1309_data) + (v1540_data * v1504_data)) + (v1728_data * v1692_data)) + (v1916_data * v1880_data)) + (v2104_data * v2068_data)) + (v2292_data * v2256_data)) + (v2480_data * v2444_data)) + (v2668_data * v2632_data)) + (v2857_data * v2821_data)) + (v3045_data * v3009_data)) + (v3233_data * v3197_data);
          double v3252_p = ((((((((((v1347_data * v1314_data) + (v1540_data * v1509_data)) + (v1728_data * v1697_data)) + (v1916_data * v1885_data)) + (v2104_data * v2073_data)) + (v2292_data * v2261_data)) + (v2480_data * v2449_data)) + (v2668_data * v2637_data)) + (v2857_data * v2826_data)) + (v3045_data * v3014_data)) + (v3233_data * v3202_data);
          double v3257_p = ((((((((((v1347_data * v1319_data) + (v1540_data * v1514_data)) + (v1728_data * v1702_data)) + (v1916_data * v1890_data)) + (v2104_data * v2078_data)) + (v2292_data * v2266_data)) + (v2480_data * v2454_data)) + (v2668_data * v2642_data)) + (v2857_data * v2831_data)) + (v3045_data * v3019_data)) + (v3233_data * v3207_data);
          double v3262_p = ((((((((((v1347_data * v1324_data) + (v1540_data * v1519_data)) + (v1728_data * v1707_data)) + (v1916_data * v1895_data)) + (v2104_data * v2083_data)) + (v2292_data * v2271_data)) + (v2480_data * v2459_data)) + (v2668_data * v2647_data)) + (v2857_data * v2836_data)) + (v3045_data * v3024_data)) + (v3233_data * v3212_data);
          double v3267_p = ((((((((((v1347_data * v1329_data) + (v1540_data * v1524_data)) + (v1728_data * v1712_data)) + (v1916_data * v1900_data)) + (v2104_data * v2088_data)) + (v2292_data * v2276_data)) + (v2480_data * v2464_data)) + (v2668_data * v2652_data)) + (v2857_data * v2841_data)) + (v3045_data * v3029_data)) + (v3233_data * v3217_data);
          double v3272_p = ((((((((((v1347_data * v1334_data) + (v1540_data * v1529_data)) + (v1728_data * v1717_data)) + (v1916_data * v1905_data)) + (v2104_data * v2093_data)) + (v2292_data * v2281_data)) + (v2480_data * v2469_data)) + (v2668_data * v2657_data)) + (v2857_data * v2846_data)) + (v3045_data * v3034_data)) + (v3233_data * v3222_data);
          double v3277_p = ((((((((((v1347_data * v1339_data) + (v1540_data * v1534_data)) + (v1728_data * v1722_data)) + (v1916_data * v1910_data)) + (v2104_data * v2098_data)) + (v2292_data * v2286_data)) + (v2480_data * v2474_data)) + (v2668_data * v2662_data)) + (v2857_data * v2851_data)) + (v3045_data * v3039_data)) + (v3233_data * v3227_data);
          double v3280_data = glb_m3[(v1393_lead + 508)];
          double v3284_p = ((((((((((v1396_data * v1299_data) + (v1587_data * v1494_data)) + (v1775_data * v1682_data)) + (v1963_data * v1870_data)) + (v2151_data * v2058_data)) + (v2339_data * v2246_data)) + (v2527_data * v2434_data)) + (v2715_data * v2622_data)) + (v2904_data * v2811_data)) + (v3092_data * v2999_data)) + (v3280_data * v3187_data);
          double v3289_p = ((((((((((v1396_data * v1304_data) + (v1587_data * v1499_data)) + (v1775_data * v1687_data)) + (v1963_data * v1875_data)) + (v2151_data * v2063_data)) + (v2339_data * v2251_data)) + (v2527_data * v2439_data)) + (v2715_data * v2627_data)) + (v2904_data * v2816_data)) + (v3092_data * v3004_data)) + (v3280_data * v3192_data);
          double v3294_p = ((((((((((v1396_data * v1309_data) + (v1587_data * v1504_data)) + (v1775_data * v1692_data)) + (v1963_data * v1880_data)) + (v2151_data * v2068_data)) + (v2339_data * v2256_data)) + (v2527_data * v2444_data)) + (v2715_data * v2632_data)) + (v2904_data * v2821_data)) + (v3092_data * v3009_data)) + (v3280_data * v3197_data);
          double v3299_p = ((((((((((v1396_data * v1314_data) + (v1587_data * v1509_data)) + (v1775_data * v1697_data)) + (v1963_data * v1885_data)) + (v2151_data * v2073_data)) + (v2339_data * v2261_data)) + (v2527_data * v2449_data)) + (v2715_data * v2637_data)) + (v2904_data * v2826_data)) + (v3092_data * v3014_data)) + (v3280_data * v3202_data);
          double v3304_p = ((((((((((v1396_data * v1319_data) + (v1587_data * v1514_data)) + (v1775_data * v1702_data)) + (v1963_data * v1890_data)) + (v2151_data * v2078_data)) + (v2339_data * v2266_data)) + (v2527_data * v2454_data)) + (v2715_data * v2642_data)) + (v2904_data * v2831_data)) + (v3092_data * v3019_data)) + (v3280_data * v3207_data);
          double v3309_p = ((((((((((v1396_data * v1324_data) + (v1587_data * v1519_data)) + (v1775_data * v1707_data)) + (v1963_data * v1895_data)) + (v2151_data * v2083_data)) + (v2339_data * v2271_data)) + (v2527_data * v2459_data)) + (v2715_data * v2647_data)) + (v2904_data * v2836_data)) + (v3092_data * v3024_data)) + (v3280_data * v3212_data);
          double v3314_p = ((((((((((v1396_data * v1329_data) + (v1587_data * v1524_data)) + (v1775_data * v1712_data)) + (v1963_data * v1900_data)) + (v2151_data * v2088_data)) + (v2339_data * v2276_data)) + (v2527_data * v2464_data)) + (v2715_data * v2652_data)) + (v2904_data * v2841_data)) + (v3092_data * v3029_data)) + (v3280_data * v3217_data);
          double v3319_p = ((((((((((v1396_data * v1334_data) + (v1587_data * v1529_data)) + (v1775_data * v1717_data)) + (v1963_data * v1905_data)) + (v2151_data * v2093_data)) + (v2339_data * v2281_data)) + (v2527_data * v2469_data)) + (v2715_data * v2657_data)) + (v2904_data * v2846_data)) + (v3092_data * v3034_data)) + (v3280_data * v3222_data);
          double v3324_p = ((((((((((v1396_data * v1339_data) + (v1587_data * v1534_data)) + (v1775_data * v1722_data)) + (v1963_data * v1910_data)) + (v2151_data * v2098_data)) + (v2339_data * v2286_data)) + (v2527_data * v2474_data)) + (v2715_data * v2662_data)) + (v2904_data * v2851_data)) + (v3092_data * v3039_data)) + (v3280_data * v3227_data);
          double v3327_data = v5647_g ? (glb_m3[(v1442_lead + 508)]) : (0.0);
          double v3331_p = ((((((((((v1446_data * v1299_data) + (v1634_data * v1494_data)) + (v1822_data * v1682_data)) + (v2010_data * v1870_data)) + (v2198_data * v2058_data)) + (v2386_data * v2246_data)) + (v2574_data * v2434_data)) + (v2762_data * v2622_data)) + (v2951_data * v2811_data)) + (v3139_data * v2999_data)) + (v3327_data * v3187_data);
          double v3336_p = ((((((((((v1446_data * v1304_data) + (v1634_data * v1499_data)) + (v1822_data * v1687_data)) + (v2010_data * v1875_data)) + (v2198_data * v2063_data)) + (v2386_data * v2251_data)) + (v2574_data * v2439_data)) + (v2762_data * v2627_data)) + (v2951_data * v2816_data)) + (v3139_data * v3004_data)) + (v3327_data * v3192_data);
          double v3341_p = ((((((((((v1446_data * v1309_data) + (v1634_data * v1504_data)) + (v1822_data * v1692_data)) + (v2010_data * v1880_data)) + (v2198_data * v2068_data)) + (v2386_data * v2256_data)) + (v2574_data * v2444_data)) + (v2762_data * v2632_data)) + (v2951_data * v2821_data)) + (v3139_data * v3009_data)) + (v3327_data * v3197_data);
          double v3346_p = ((((((((((v1446_data * v1314_data) + (v1634_data * v1509_data)) + (v1822_data * v1697_data)) + (v2010_data * v1885_data)) + (v2198_data * v2073_data)) + (v2386_data * v2261_data)) + (v2574_data * v2449_data)) + (v2762_data * v2637_data)) + (v2951_data * v2826_data)) + (v3139_data * v3014_data)) + (v3327_data * v3202_data);
          double v3351_p = ((((((((((v1446_data * v1319_data) + (v1634_data * v1514_data)) + (v1822_data * v1702_data)) + (v2010_data * v1890_data)) + (v2198_data * v2078_data)) + (v2386_data * v2266_data)) + (v2574_data * v2454_data)) + (v2762_data * v2642_data)) + (v2951_data * v2831_data)) + (v3139_data * v3019_data)) + (v3327_data * v3207_data);
          double v3356_p = ((((((((((v1446_data * v1324_data) + (v1634_data * v1519_data)) + (v1822_data * v1707_data)) + (v2010_data * v1895_data)) + (v2198_data * v2083_data)) + (v2386_data * v2271_data)) + (v2574_data * v2459_data)) + (v2762_data * v2647_data)) + (v2951_data * v2836_data)) + (v3139_data * v3024_data)) + (v3327_data * v3212_data);
          double v3361_p = ((((((((((v1446_data * v1329_data) + (v1634_data * v1524_data)) + (v1822_data * v1712_data)) + (v2010_data * v1900_data)) + (v2198_data * v2088_data)) + (v2386_data * v2276_data)) + (v2574_data * v2464_data)) + (v2762_data * v2652_data)) + (v2951_data * v2841_data)) + (v3139_data * v3029_data)) + (v3327_data * v3217_data);
          double v3366_p = ((((((((((v1446_data * v1334_data) + (v1634_data * v1529_data)) + (v1822_data * v1717_data)) + (v2010_data * v1905_data)) + (v2198_data * v2093_data)) + (v2386_data * v2281_data)) + (v2574_data * v2469_data)) + (v2762_data * v2657_data)) + (v2951_data * v2846_data)) + (v3139_data * v3034_data)) + (v3327_data * v3222_data);
          double v3371_p = ((((((((((v1446_data * v1339_data) + (v1634_data * v1534_data)) + (v1822_data * v1722_data)) + (v2010_data * v1910_data)) + (v2198_data * v2098_data)) + (v2386_data * v2286_data)) + (v2574_data * v2474_data)) + (v2762_data * v2662_data)) + (v2951_data * v2851_data)) + (v3139_data * v3039_data)) + (v3327_data * v3227_data);
          double v3374_data = v2809_g ? (glb_m3[(v18980_lead + 554)]) : (0.0);
          double v3375_data = s1[11];
          ir3[0] = ((((((((((((v1298_data * v1299_data) + (v1493_data * v1494_data)) + (v1681_data * v1682_data)) + (v1869_data * v1870_data)) + (v2057_data * v2058_data)) + (v2245_data * v2246_data)) + (v2433_data * v2434_data)) + (v2621_data * v2622_data)) + (v2810_data * v2811_data)) + (v2998_data * v2999_data)) + (v3186_data * v3187_data)) + (v3374_data * v3375_data));
          double v3380_data = s1[46];
          ir3[4] = (v3195_p + (v3374_data * v3380_data));
          double v3385_data = s1[81];
          ir3[8] = (v3200_p + (v3374_data * v3385_data));
          double v3390_data = s1[116];
          ir3[12] = (v3205_p + (v3374_data * v3390_data));
          double v3395_data = s1[151];
          ir3[16] = (v3210_p + (v3374_data * v3395_data));
          double v3400_data = s1[186];
          ir3[20] = (v3215_p + (v3374_data * v3400_data));
          double v3405_data = s1[221];
          ir3[24] = (v3220_p + (v3374_data * v3405_data));
          double v3410_data = s1[256];
          ir3[28] = (v3225_p + (v3374_data * v3410_data));
          double v3415_data = s1[291];
          ir3[32] = (v3230_p + (v3374_data * v3415_data));
          double v3421_data = glb_m3[(v1344_lead + 554)];
          double v3468_data = glb_m3[(v1393_lead + 554)];
          double v3515_data = v5647_g ? (glb_m3[(v1442_lead + 554)]) : (0.0);
          double v3562_data = glb_m3[(v1344_lead + 592)];
          double v3563_data = s1[12];
          double v3568_data = s1[47];
          double v3573_data = s1[82];
          double v3578_data = s1[117];
          double v3583_data = s1[152];
          double v3588_data = s1[187];
          double v3593_data = s1[222];
          double v3598_data = s1[257];
          double v3603_data = s1[292];
          double v3609_data = glb_m3[(v1393_lead + 592)];
          double v3656_data = v5647_g ? (glb_m3[(v1442_lead + 592)]) : (0.0);
          double v3703_data = glb_m3[(v1344_lead + 630)];
          double v3704_data = s1[13];
          double v3709_data = s1[48];
          double v3714_data = s1[83];
          double v3719_data = s1[118];
          double v3724_data = s1[153];
          double v3729_data = s1[188];
          double v3734_data = s1[223];
          double v3739_data = s1[258];
          double v3744_data = s1[293];
          double v3750_data = glb_m3[(v1393_lead + 630)];
          double v3797_data = v5647_g ? (glb_m3[(v1442_lead + 630)]) : (0.0);
          double v3844_data = glb_m3[(v1344_lead + 668)];
          double v3845_data = s1[14];
          double v3850_data = s1[49];
          double v3855_data = s1[84];
          double v3860_data = s1[119];
          double v3865_data = s1[154];
          double v3870_data = s1[189];
          double v3875_data = s1[224];
          double v3880_data = s1[259];
          double v3885_data = s1[294];
          double v3891_data = glb_m3[(v1393_lead + 668)];
          double v3938_data = v5647_g ? (glb_m3[(v1442_lead + 668)]) : (0.0);
          double v3985_data = glb_m3[(v1344_lead + 706)];
          double v3986_data = s1[15];
          double v3991_data = s1[50];
          double v3996_data = s1[85];
          double v4001_data = s1[120];
          double v4006_data = s1[155];
          double v4011_data = s1[190];
          double v4016_data = s1[225];
          double v4021_data = s1[260];
          double v4026_data = s1[295];
          double v4032_data = glb_m3[(v1393_lead + 706)];
          double v4079_data = v5647_g ? (glb_m3[(v1442_lead + 706)]) : (0.0);
          double v4126_data = v2809_g ? (glb_m3[(v1344_lead + 736)]) : (0.0);
          double v4127_data = s1[16];
          double v4132_data = s1[51];
          double v4137_data = s1[86];
          double v4142_data = s1[121];
          double v4147_data = s1[156];
          double v4152_data = s1[191];
          double v4157_data = s1[226];
          double v4162_data = s1[261];
          double v4167_data = s1[296];
          double v4173_data = glb_m3[(v1393_lead + 736)];
          double v4220_data = v5647_g ? (glb_m3[(v1442_lead + 736)]) : (0.0);
          double v4267_data = v2809_g ? (glb_m3[(v1344_lead + 766)]) : (0.0);
          double v4268_data = s1[17];
          double v4273_data = s1[52];
          double v4278_data = s1[87];
          double v4283_data = s1[122];
          double v4288_data = s1[157];
          double v4293_data = s1[192];
          double v4298_data = s1[227];
          double v4303_data = s1[262];
          double v4308_data = s1[297];
          double v4314_data = glb_m3[(v1393_lead + 766)];
          double v4361_data = v5647_g ? (glb_m3[(v1442_lead + 766)]) : (0.0);
          double v4408_data = v2809_g ? (glb_m3[(v1344_lead + 796)]) : (0.0);
          double v4409_data = s1[18];
          double v4414_data = s1[53];
          double v4417_p = (((((((v3242_p + (v3421_data * v3380_data)) + (v3562_data * v3568_data)) + (v3703_data * v3709_data)) + (v3844_data * v3850_data)) + (v3985_data * v3991_data)) + (v4126_data * v4132_data)) + (v4267_data * v4273_data)) + (v4408_data * v4414_data);
          double v4419_data = s1[88];
          double v4422_p = (((((((v3247_p + (v3421_data * v3385_data)) + (v3562_data * v3573_data)) + (v3703_data * v3714_data)) + (v3844_data * v3855_data)) + (v3985_data * v3996_data)) + (v4126_data * v4137_data)) + (v4267_data * v4278_data)) + (v4408_data * v4419_data);
          double v4424_data = s1[123];
          double v4427_p = (((((((v3252_p + (v3421_data * v3390_data)) + (v3562_data * v3578_data)) + (v3703_data * v3719_data)) + (v3844_data * v3860_data)) + (v3985_data * v4001_data)) + (v4126_data * v4142_data)) + (v4267_data * v4283_data)) + (v4408_data * v4424_data);
          double v4429_data = s1[158];
          double v4432_p = (((((((v3257_p + (v3421_data * v3395_data)) + (v3562_data * v3583_data)) + (v3703_data * v3724_data)) + (v3844_data * v3865_data)) + (v3985_data * v4006_data)) + (v4126_data * v4147_data)) + (v4267_data * v4288_data)) + (v4408_data * v4429_data);
          double v4434_data = s1[193];
          double v4437_p = (((((((v3262_p + (v3421_data * v3400_data)) + (v3562_data * v3588_data)) + (v3703_data * v3729_data)) + (v3844_data * v3870_data)) + (v3985_data * v4011_data)) + (v4126_data * v4152_data)) + (v4267_data * v4293_data)) + (v4408_data * v4434_data);
          double v4439_data = s1[228];
          double v4442_p = (((((((v3267_p + (v3421_data * v3405_data)) + (v3562_data * v3593_data)) + (v3703_data * v3734_data)) + (v3844_data * v3875_data)) + (v3985_data * v4016_data)) + (v4126_data * v4157_data)) + (v4267_data * v4298_data)) + (v4408_data * v4439_data);
          double v4444_data = s1[263];
          double v4447_p = (((((((v3272_p + (v3421_data * v3410_data)) + (v3562_data * v3598_data)) + (v3703_data * v3739_data)) + (v3844_data * v3880_data)) + (v3985_data * v4021_data)) + (v4126_data * v4162_data)) + (v4267_data * v4303_data)) + (v4408_data * v4444_data);
          double v4449_data = s1[298];
          double v4452_p = (((((((v3277_p + (v3421_data * v3415_data)) + (v3562_data * v3603_data)) + (v3703_data * v3744_data)) + (v3844_data * v3885_data)) + (v3985_data * v4026_data)) + (v4126_data * v4167_data)) + (v4267_data * v4308_data)) + (v4408_data * v4449_data);
          double v4455_data = glb_m3[(v1393_lead + 796)];
          double v4459_p = (((((((v3284_p + (v3468_data * v3375_data)) + (v3609_data * v3563_data)) + (v3750_data * v3704_data)) + (v3891_data * v3845_data)) + (v4032_data * v3986_data)) + (v4173_data * v4127_data)) + (v4314_data * v4268_data)) + (v4455_data * v4409_data);
          double v4464_p = (((((((v3289_p + (v3468_data * v3380_data)) + (v3609_data * v3568_data)) + (v3750_data * v3709_data)) + (v3891_data * v3850_data)) + (v4032_data * v3991_data)) + (v4173_data * v4132_data)) + (v4314_data * v4273_data)) + (v4455_data * v4414_data);
          double v4469_p = (((((((v3294_p + (v3468_data * v3385_data)) + (v3609_data * v3573_data)) + (v3750_data * v3714_data)) + (v3891_data * v3855_data)) + (v4032_data * v3996_data)) + (v4173_data * v4137_data)) + (v4314_data * v4278_data)) + (v4455_data * v4419_data);
          double v4474_p = (((((((v3299_p + (v3468_data * v3390_data)) + (v3609_data * v3578_data)) + (v3750_data * v3719_data)) + (v3891_data * v3860_data)) + (v4032_data * v4001_data)) + (v4173_data * v4142_data)) + (v4314_data * v4283_data)) + (v4455_data * v4424_data);
          double v4479_p = (((((((v3304_p + (v3468_data * v3395_data)) + (v3609_data * v3583_data)) + (v3750_data * v3724_data)) + (v3891_data * v3865_data)) + (v4032_data * v4006_data)) + (v4173_data * v4147_data)) + (v4314_data * v4288_data)) + (v4455_data * v4429_data);
          double v4484_p = (((((((v3309_p + (v3468_data * v3400_data)) + (v3609_data * v3588_data)) + (v3750_data * v3729_data)) + (v3891_data * v3870_data)) + (v4032_data * v4011_data)) + (v4173_data * v4152_data)) + (v4314_data * v4293_data)) + (v4455_data * v4434_data);
          double v4489_p = (((((((v3314_p + (v3468_data * v3405_data)) + (v3609_data * v3593_data)) + (v3750_data * v3734_data)) + (v3891_data * v3875_data)) + (v4032_data * v4016_data)) + (v4173_data * v4157_data)) + (v4314_data * v4298_data)) + (v4455_data * v4439_data);
          double v4494_p = (((((((v3319_p + (v3468_data * v3410_data)) + (v3609_data * v3598_data)) + (v3750_data * v3739_data)) + (v3891_data * v3880_data)) + (v4032_data * v4021_data)) + (v4173_data * v4162_data)) + (v4314_data * v4303_data)) + (v4455_data * v4444_data);
          double v4499_p = (((((((v3324_p + (v3468_data * v3415_data)) + (v3609_data * v3603_data)) + (v3750_data * v3744_data)) + (v3891_data * v3885_data)) + (v4032_data * v4026_data)) + (v4173_data * v4167_data)) + (v4314_data * v4308_data)) + (v4455_data * v4449_data);
          double v4502_data = v5647_g ? (glb_m3[(v1442_lead + 796)]) : (0.0);
          double v4506_p = (((((((v3331_p + (v3515_data * v3375_data)) + (v3656_data * v3563_data)) + (v3797_data * v3704_data)) + (v3938_data * v3845_data)) + (v4079_data * v3986_data)) + (v4220_data * v4127_data)) + (v4361_data * v4268_data)) + (v4502_data * v4409_data);
          double v4511_p = (((((((v3336_p + (v3515_data * v3380_data)) + (v3656_data * v3568_data)) + (v3797_data * v3709_data)) + (v3938_data * v3850_data)) + (v4079_data * v3991_data)) + (v4220_data * v4132_data)) + (v4361_data * v4273_data)) + (v4502_data * v4414_data);
          double v4516_p = (((((((v3341_p + (v3515_data * v3385_data)) + (v3656_data * v3573_data)) + (v3797_data * v3714_data)) + (v3938_data * v3855_data)) + (v4079_data * v3996_data)) + (v4220_data * v4137_data)) + (v4361_data * v4278_data)) + (v4502_data * v4419_data);
          double v4521_p = (((((((v3346_p + (v3515_data * v3390_data)) + (v3656_data * v3578_data)) + (v3797_data * v3719_data)) + (v3938_data * v3860_data)) + (v4079_data * v4001_data)) + (v4220_data * v4142_data)) + (v4361_data * v4283_data)) + (v4502_data * v4424_data);
          double v4526_p = (((((((v3351_p + (v3515_data * v3395_data)) + (v3656_data * v3583_data)) + (v3797_data * v3724_data)) + (v3938_data * v3865_data)) + (v4079_data * v4006_data)) + (v4220_data * v4147_data)) + (v4361_data * v4288_data)) + (v4502_data * v4429_data);
          double v4531_p = (((((((v3356_p + (v3515_data * v3400_data)) + (v3656_data * v3588_data)) + (v3797_data * v3729_data)) + (v3938_data * v3870_data)) + (v4079_data * v4011_data)) + (v4220_data * v4152_data)) + (v4361_data * v4293_data)) + (v4502_data * v4434_data);
          double v4536_p = (((((((v3361_p + (v3515_data * v3405_data)) + (v3656_data * v3593_data)) + (v3797_data * v3734_data)) + (v3938_data * v3875_data)) + (v4079_data * v4016_data)) + (v4220_data * v4157_data)) + (v4361_data * v4298_data)) + (v4502_data * v4439_data);
          double v4541_p = (((((((v3366_p + (v3515_data * v3410_data)) + (v3656_data * v3598_data)) + (v3797_data * v3739_data)) + (v3938_data * v3880_data)) + (v4079_data * v4021_data)) + (v4220_data * v4162_data)) + (v4361_data * v4303_data)) + (v4502_data * v4444_data);
          double v4546_p = (((((((v3371_p + (v3515_data * v3415_data)) + (v3656_data * v3603_data)) + (v3797_data * v3744_data)) + (v3938_data * v3885_data)) + (v4079_data * v4026_data)) + (v4220_data * v4167_data)) + (v4361_data * v4308_data)) + (v4502_data * v4449_data);
          double v4549_data = v2809_g ? (glb_m3[(v1344_lead + 826)]) : (0.0);
          double v4550_data = s1[19];
          ir3[1] = (((((((((v3237_p + (v3421_data * v3375_data)) + (v3562_data * v3563_data)) + (v3703_data * v3704_data)) + (v3844_data * v3845_data)) + (v3985_data * v3986_data)) + (v4126_data * v4127_data)) + (v4267_data * v4268_data)) + (v4408_data * v4409_data)) + (v4549_data * v4550_data));
          double v4555_data = s1[54];
          ir3[5] = (v4417_p + (v4549_data * v4555_data));
          double v4560_data = s1[89];
          ir3[9] = (v4422_p + (v4549_data * v4560_data));
          double v4565_data = s1[124];
          ir3[13] = (v4427_p + (v4549_data * v4565_data));
          double v4570_data = s1[159];
          ir3[17] = (v4432_p + (v4549_data * v4570_data));
          double v4575_data = s1[194];
          ir3[21] = (v4437_p + (v4549_data * v4575_data));
          double v4580_data = s1[229];
          ir3[25] = (v4442_p + (v4549_data * v4580_data));
          double v4585_data = s1[264];
          ir3[29] = (v4447_p + (v4549_data * v4585_data));
          double v4590_data = s1[299];
          ir3[33] = (v4452_p + (v4549_data * v4590_data));
          double v4596_data = glb_m3[(v1393_lead + 826)];
          double v4643_data = v5647_g ? (glb_m3[(v1442_lead + 826)]) : (0.0);
          double v4690_data = glb_m3[(v1393_lead + 848)];
          double v4691_data = s1[20];
          double v4696_data = s1[55];
          double v4701_data = s1[90];
          double v4706_data = s1[125];
          double v4711_data = s1[160];
          double v4716_data = s1[195];
          double v4721_data = s1[230];
          double v4726_data = s1[265];
          double v4731_data = s1[300];
          double v4737_data = v5647_g ? (glb_m3[(v1442_lead + 848)]) : (0.0);
          double v4784_data = glb_m3[(v1393_lead + 870)];
          double v4785_data = s1[21];
          double v4790_data = s1[56];
          double v4795_data = s1[91];
          double v4800_data = s1[126];
          double v4805_data = s1[161];
          double v4810_data = s1[196];
          double v4815_data = s1[231];
          double v4820_data = s1[266];
          double v4825_data = s1[301];
          double v4831_data = v5647_g ? (glb_m3[(v1442_lead + 870)]) : (0.0);
          double v4878_data = glb_m3[(v1393_lead + 892)];
          double v4879_data = s1[22];
          double v4884_data = s1[57];
          double v4889_data = s1[92];
          double v4894_data = s1[127];
          double v4899_data = s1[162];
          double v4904_data = s1[197];
          double v4909_data = s1[232];
          double v4914_data = s1[267];
          double v4919_data = s1[302];
          double v4925_data = v5647_g ? (glb_m3[(v1442_lead + 892)]) : (0.0);
          double v4972_data = glb_m3[(v1393_lead + 914)];
          double v4973_data = s1[23];
          double v4978_data = s1[58];
          double v4983_data = s1[93];
          double v4988_data = s1[128];
          double v4993_data = s1[163];
          double v4998_data = s1[198];
          double v5003_data = s1[233];
          double v5008_data = s1[268];
          double v5013_data = s1[303];
          double v5019_data = v5647_g ? (glb_m3[(v1442_lead + 914)]) : (0.0);
          double v5066_data = v2809_g ? (glb_m3[(v1393_lead + 928)]) : (0.0);
          double v5067_data = s1[24];
          double v5072_data = s1[59];
          double v5077_data = s1[94];
          double v5082_data = s1[129];
          double v5087_data = s1[164];
          double v5092_data = s1[199];
          double v5097_data = s1[234];
          double v5102_data = s1[269];
          double v5107_data = s1[304];
          double v5113_data = v5647_g ? (glb_m3[(v1442_lead + 928)]) : (0.0);
          double v5160_data = v2809_g ? (glb_m3[(v1393_lead + 942)]) : (0.0);
          double v5161_data = s1[25];
          double v5166_data = s1[60];
          double v5171_data = s1[95];
          double v5176_data = s1[130];
          double v5181_data = s1[165];
          double v5186_data = s1[200];
          double v5191_data = s1[235];
          double v5196_data = s1[270];
          double v5201_data = s1[305];
          double v5207_data = v5647_g ? (glb_m3[(v1442_lead + 942)]) : (0.0);
          double v5254_data = v2809_g ? (glb_m3[(v1393_lead + 956)]) : (0.0);
          double v5255_data = s1[26];
          double v5260_data = s1[61];
          double v5263_p = (((((((v4464_p + (v4596_data * v4555_data)) + (v4690_data * v4696_data)) + (v4784_data * v4790_data)) + (v4878_data * v4884_data)) + (v4972_data * v4978_data)) + (v5066_data * v5072_data)) + (v5160_data * v5166_data)) + (v5254_data * v5260_data);
          double v5265_data = s1[96];
          double v5268_p = (((((((v4469_p + (v4596_data * v4560_data)) + (v4690_data * v4701_data)) + (v4784_data * v4795_data)) + (v4878_data * v4889_data)) + (v4972_data * v4983_data)) + (v5066_data * v5077_data)) + (v5160_data * v5171_data)) + (v5254_data * v5265_data);
          double v5270_data = s1[131];
          double v5273_p = (((((((v4474_p + (v4596_data * v4565_data)) + (v4690_data * v4706_data)) + (v4784_data * v4800_data)) + (v4878_data * v4894_data)) + (v4972_data * v4988_data)) + (v5066_data * v5082_data)) + (v5160_data * v5176_data)) + (v5254_data * v5270_data);
          double v5275_data = s1[166];
          double v5278_p = (((((((v4479_p + (v4596_data * v4570_data)) + (v4690_data * v4711_data)) + (v4784_data * v4805_data)) + (v4878_data * v4899_data)) + (v4972_data * v4993_data)) + (v5066_data * v5087_data)) + (v5160_data * v5181_data)) + (v5254_data * v5275_data);
          double v5280_data = s1[201];
          double v5283_p = (((((((v4484_p + (v4596_data * v4575_data)) + (v4690_data * v4716_data)) + (v4784_data * v4810_data)) + (v4878_data * v4904_data)) + (v4972_data * v4998_data)) + (v5066_data * v5092_data)) + (v5160_data * v5186_data)) + (v5254_data * v5280_data);
          double v5285_data = s1[236];
          double v5288_p = (((((((v4489_p + (v4596_data * v4580_data)) + (v4690_data * v4721_data)) + (v4784_data * v4815_data)) + (v4878_data * v4909_data)) + (v4972_data * v5003_data)) + (v5066_data * v5097_data)) + (v5160_data * v5191_data)) + (v5254_data * v5285_data);
          double v5290_data = s1[271];
          double v5293_p = (((((((v4494_p + (v4596_data * v4585_data)) + (v4690_data * v4726_data)) + (v4784_data * v4820_data)) + (v4878_data * v4914_data)) + (v4972_data * v5008_data)) + (v5066_data * v5102_data)) + (v5160_data * v5196_data)) + (v5254_data * v5290_data);
          double v5295_data = s1[306];
          double v5298_p = (((((((v4499_p + (v4596_data * v4590_data)) + (v4690_data * v4731_data)) + (v4784_data * v4825_data)) + (v4878_data * v4919_data)) + (v4972_data * v5013_data)) + (v5066_data * v5107_data)) + (v5160_data * v5201_data)) + (v5254_data * v5295_data);
          double v5301_data = v5647_g ? (glb_m3[(v1442_lead + 956)]) : (0.0);
          double v5305_p = (((((((v4506_p + (v4643_data * v4550_data)) + (v4737_data * v4691_data)) + (v4831_data * v4785_data)) + (v4925_data * v4879_data)) + (v5019_data * v4973_data)) + (v5113_data * v5067_data)) + (v5207_data * v5161_data)) + (v5301_data * v5255_data);
          double v5310_p = (((((((v4511_p + (v4643_data * v4555_data)) + (v4737_data * v4696_data)) + (v4831_data * v4790_data)) + (v4925_data * v4884_data)) + (v5019_data * v4978_data)) + (v5113_data * v5072_data)) + (v5207_data * v5166_data)) + (v5301_data * v5260_data);
          double v5315_p = (((((((v4516_p + (v4643_data * v4560_data)) + (v4737_data * v4701_data)) + (v4831_data * v4795_data)) + (v4925_data * v4889_data)) + (v5019_data * v4983_data)) + (v5113_data * v5077_data)) + (v5207_data * v5171_data)) + (v5301_data * v5265_data);
          double v5320_p = (((((((v4521_p + (v4643_data * v4565_data)) + (v4737_data * v4706_data)) + (v4831_data * v4800_data)) + (v4925_data * v4894_data)) + (v5019_data * v4988_data)) + (v5113_data * v5082_data)) + (v5207_data * v5176_data)) + (v5301_data * v5270_data);
          double v5325_p = (((((((v4526_p + (v4643_data * v4570_data)) + (v4737_data * v4711_data)) + (v4831_data * v4805_data)) + (v4925_data * v4899_data)) + (v5019_data * v4993_data)) + (v5113_data * v5087_data)) + (v5207_data * v5181_data)) + (v5301_data * v5275_data);
          double v5330_p = (((((((v4531_p + (v4643_data * v4575_data)) + (v4737_data * v4716_data)) + (v4831_data * v4810_data)) + (v4925_data * v4904_data)) + (v5019_data * v4998_data)) + (v5113_data * v5092_data)) + (v5207_data * v5186_data)) + (v5301_data * v5280_data);
          double v5335_p = (((((((v4536_p + (v4643_data * v4580_data)) + (v4737_data * v4721_data)) + (v4831_data * v4815_data)) + (v4925_data * v4909_data)) + (v5019_data * v5003_data)) + (v5113_data * v5097_data)) + (v5207_data * v5191_data)) + (v5301_data * v5285_data);
          double v5340_p = (((((((v4541_p + (v4643_data * v4585_data)) + (v4737_data * v4726_data)) + (v4831_data * v4820_data)) + (v4925_data * v4914_data)) + (v5019_data * v5008_data)) + (v5113_data * v5102_data)) + (v5207_data * v5196_data)) + (v5301_data * v5290_data);
          double v5345_p = (((((((v4546_p + (v4643_data * v4590_data)) + (v4737_data * v4731_data)) + (v4831_data * v4825_data)) + (v4925_data * v4919_data)) + (v5019_data * v5013_data)) + (v5113_data * v5107_data)) + (v5207_data * v5201_data)) + (v5301_data * v5295_data);
          double v5348_data = v2809_g ? (glb_m3[(v1393_lead + 970)]) : (0.0);
          double v5349_data = s1[27];
          ir3[2] = (((((((((v4459_p + (v4596_data * v4550_data)) + (v4690_data * v4691_data)) + (v4784_data * v4785_data)) + (v4878_data * v4879_data)) + (v4972_data * v4973_data)) + (v5066_data * v5067_data)) + (v5160_data * v5161_data)) + (v5254_data * v5255_data)) + (v5348_data * v5349_data));
          double v5354_data = s1[62];
          ir3[6] = (v5263_p + (v5348_data * v5354_data));
          double v5359_data = s1[97];
          ir3[10] = (v5268_p + (v5348_data * v5359_data));
          double v5364_data = s1[132];
          ir3[14] = (v5273_p + (v5348_data * v5364_data));
          double v5369_data = s1[167];
          ir3[18] = (v5278_p + (v5348_data * v5369_data));
          double v5374_data = s1[202];
          ir3[22] = (v5283_p + (v5348_data * v5374_data));
          double v5379_data = s1[237];
          ir3[26] = (v5288_p + (v5348_data * v5379_data));
          double v5384_data = s1[272];
          ir3[30] = (v5293_p + (v5348_data * v5384_data));
          double v5389_data = s1[307];
          ir3[34] = (v5298_p + (v5348_data * v5389_data));
          double v5395_data = v5647_g ? (glb_m3[(v1442_lead + 970)]) : (0.0);
          double v5442_data = v5647_g ? (glb_m3[(v1442_lead + 976)]) : (0.0);
          double v5443_data = s1[28];
          double v5448_data = s1[63];
          double v5453_data = s1[98];
          double v5458_data = s1[133];
          double v5463_data = s1[168];
          double v5468_data = s1[203];
          double v5473_data = s1[238];
          double v5478_data = s1[273];
          double v5483_data = s1[308];
          double v5489_data = v5647_g ? (glb_m3[(v1442_lead + 982)]) : (0.0);
          double v5490_data = s1[29];
          double v5495_data = s1[64];
          double v5500_data = s1[99];
          double v5505_data = s1[134];
          double v5510_data = s1[169];
          double v5515_data = s1[204];
          double v5520_data = s1[239];
          double v5525_data = s1[274];
          double v5530_data = s1[309];
          double v5536_data = v5647_g ? (glb_m3[(v1442_lead + 988)]) : (0.0);
          double v5537_data = s1[30];
          double v5542_data = s1[65];
          double v5545_p = (((v5310_p + (v5395_data * v5354_data)) + (v5442_data * v5448_data)) + (v5489_data * v5495_data)) + (v5536_data * v5542_data);
          double v5547_data = s1[100];
          double v5550_p = (((v5315_p + (v5395_data * v5359_data)) + (v5442_data * v5453_data)) + (v5489_data * v5500_data)) + (v5536_data * v5547_data);
          double v5552_data = s1[135];
          double v5555_p = (((v5320_p + (v5395_data * v5364_data)) + (v5442_data * v5458_data)) + (v5489_data * v5505_data)) + (v5536_data * v5552_data);
          double v5557_data = s1[170];
          double v5560_p = (((v5325_p + (v5395_data * v5369_data)) + (v5442_data * v5463_data)) + (v5489_data * v5510_data)) + (v5536_data * v5557_data);
          double v5562_data = s1[205];
          double v5565_p = (((v5330_p + (v5395_data * v5374_data)) + (v5442_data * v5468_data)) + (v5489_data * v5515_data)) + (v5536_data * v5562_data);
          double v5567_data = s1[240];
          double v5570_p = (((v5335_p + (v5395_data * v5379_data)) + (v5442_data * v5473_data)) + (v5489_data * v5520_data)) + (v5536_data * v5567_data);
          double v5572_data = s1[275];
          double v5575_p = (((v5340_p + (v5395_data * v5384_data)) + (v5442_data * v5478_data)) + (v5489_data * v5525_data)) + (v5536_data * v5572_data);
          double v5577_data = s1[310];
          double v5580_p = (((v5345_p + (v5395_data * v5389_data)) + (v5442_data * v5483_data)) + (v5489_data * v5530_data)) + (v5536_data * v5577_data);
          double v5583_data = v5647_g ? (glb_m3[(v1442_lead + 994)]) : (0.0);
          double v5584_data = s1[31];
          ir3[3] = (((((v5305_p + (v5395_data * v5349_data)) + (v5442_data * v5443_data)) + (v5489_data * v5490_data)) + (v5536_data * v5537_data)) + (v5583_data * v5584_data));
          double v5589_data = s1[66];
          ir3[7] = (v5545_p + (v5583_data * v5589_data));
          double v5594_data = s1[101];
          ir3[11] = (v5550_p + (v5583_data * v5594_data));
          double v5599_data = s1[136];
          ir3[15] = (v5555_p + (v5583_data * v5599_data));
          double v5604_data = s1[171];
          ir3[19] = (v5560_p + (v5583_data * v5604_data));
          double v5609_data = s1[206];
          ir3[23] = (v5565_p + (v5583_data * v5609_data));
          double v5614_data = s1[241];
          ir3[27] = (v5570_p + (v5583_data * v5614_data));
          double v5619_data = s1[276];
          ir3[31] = (v5575_p + (v5583_data * v5619_data));
          double v5624_data = s1[311];
          ir3[35] = (v5580_p + (v5583_data * v5624_data));
          double r4[27]{};
          // s2 = load{g>s}(glb_m4[0, 1])
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v5718_i = 0; v5718_i < 5; ++v5718_i) {
            int32_t v5720_a = threadIdx.x + (v5718_i * 16);
            __pipeline_memcpy_async(&s2[v5720_a], &glb_m4[v5720_a], 8);
          }
          if (v19038_g) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 80], &glb_m4[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
          __pipeline_commit();
          // r4 = load{g>r}(glb_m0);
          // r3 = ir3 + r2
          if (v5629_g) {
            #pragma unroll
            for (int32_t v5656_n1 = 0; v5656_n1 < 9; ++v5656_n1) {
              int32_t v5657_a = v5656_n1 * 4;
              double v5659_data = ir3[v5657_a];
              double v5660_data = r2[v5657_a];
              r3[v5657_a] = (v5660_data + v5659_data);
            }
          }
          #pragma unroll
          for (int32_t v5662_n0 = 1; v5662_n0 < 3; ++v5662_n0) {
            #pragma unroll
            for (int32_t v5663_n1 = 0; v5663_n1 < 9; ++v5663_n1) {
              int32_t v5665_a = v5662_n0 + (v5663_n1 * 4);
              double v5666_data = ir3[v5665_a];
              double v5667_data = r2[v5665_a];
              r3[v5665_a] = (v5667_data + v5666_data);
            }
          }
          if (v5647_g) {
            #pragma unroll
            for (int32_t v5669_n1 = 0; v5669_n1 < 9; ++v5669_n1) {
              int32_t v5671_a = 3 + (v5669_n1 * 4);
              double v5672_data = ir3[v5671_a];
              double v5673_data = r2[v5671_a];
              r3[v5671_a] = (v5673_data + v5672_data);
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v5647_g) {
            #pragma unroll
            for (int32_t v5692_i1 = 0; v5692_i1 < 9; ++v5692_i1) {
              double v5695_data = r3[(3 + (v5692_i1 * 4))];
              glb_m2[(v1442_lead + (v5692_i1 * 56))] = v5695_data;
            }
          }
          double r6[36]{};
          // r6 = load{g>r}(glb_m2);
          bool v12016_g = v18980_lead < 7;
          // wait(s2 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          double r5[27]{};
          // r5 = +(r4 * s2) + None
          // [(0, 35), (0, 9)] [(0, 9)]
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v5725_data = s2[0];
          double v5730_data = s2[9];
          double v5735_data = s2[18];
          double v5740_data = s2[27];
          double v5745_data = s2[36];
          double v5750_data = s2[45];
          double v5755_data = s2[54];
          double v5760_data = s2[63];
          double v5765_data = s2[72];
          double v5860_data = s2[1];
          double v5865_data = s2[10];
          double v5870_data = s2[19];
          double v5875_data = s2[28];
          double v5880_data = s2[37];
          double v5885_data = s2[46];
          double v5890_data = s2[55];
          double v5895_data = s2[64];
          double v5900_data = s2[73];
          double v5995_data = s2[2];
          double v6000_data = s2[11];
          double v6005_data = s2[20];
          double v6010_data = s2[29];
          double v6015_data = s2[38];
          double v6020_data = s2[47];
          double v6025_data = s2[56];
          double v6030_data = s2[65];
          double v6035_data = s2[74];
          double v6130_data = s2[3];
          double v6135_data = s2[12];
          double v6140_data = s2[21];
          double v6145_data = s2[30];
          double v6150_data = s2[39];
          double v6155_data = s2[48];
          double v6160_data = s2[57];
          double v6165_data = s2[66];
          double v6170_data = s2[75];
          double v6265_data = s2[4];
          double v6270_data = s2[13];
          double v6275_data = s2[22];
          double v6280_data = s2[31];
          double v6285_data = s2[40];
          double v6290_data = s2[49];
          double v6295_data = s2[58];
          double v6300_data = s2[67];
          double v6305_data = s2[76];
          double v6400_data = s2[5];
          double v6405_data = s2[14];
          double v6410_data = s2[23];
          double v6415_data = s2[32];
          double v6420_data = s2[41];
          double v6425_data = s2[50];
          double v6430_data = s2[59];
          double v6435_data = s2[68];
          double v6440_data = s2[77];
          double v6535_data = s2[6];
          double v6540_data = s2[15];
          double v6545_data = s2[24];
          double v6550_data = s2[33];
          double v6555_data = s2[42];
          double v6560_data = s2[51];
          double v6565_data = s2[60];
          double v6570_data = s2[69];
          double v6575_data = s2[78];
          double v6670_data = s2[7];
          double v6675_data = s2[16];
          double v6678_p = (((((((v58_data * v5730_data) + (v193_data * v5865_data)) + (v328_data * v6000_data)) + (v463_data * v6135_data)) + (v598_data * v6270_data)) + (v733_data * v6405_data)) + (v868_data * v6540_data)) + (v1003_data * v6675_data);
          double v6680_data = s2[25];
          double v6683_p = (((((((v58_data * v5735_data) + (v193_data * v5870_data)) + (v328_data * v6005_data)) + (v463_data * v6140_data)) + (v598_data * v6275_data)) + (v733_data * v6410_data)) + (v868_data * v6545_data)) + (v1003_data * v6680_data);
          double v6685_data = s2[34];
          double v6688_p = (((((((v58_data * v5740_data) + (v193_data * v5875_data)) + (v328_data * v6010_data)) + (v463_data * v6145_data)) + (v598_data * v6280_data)) + (v733_data * v6415_data)) + (v868_data * v6550_data)) + (v1003_data * v6685_data);
          double v6690_data = s2[43];
          double v6693_p = (((((((v58_data * v5745_data) + (v193_data * v5880_data)) + (v328_data * v6015_data)) + (v463_data * v6150_data)) + (v598_data * v6285_data)) + (v733_data * v6420_data)) + (v868_data * v6555_data)) + (v1003_data * v6690_data);
          double v6695_data = s2[52];
          double v6698_p = (((((((v58_data * v5750_data) + (v193_data * v5885_data)) + (v328_data * v6020_data)) + (v463_data * v6155_data)) + (v598_data * v6290_data)) + (v733_data * v6425_data)) + (v868_data * v6560_data)) + (v1003_data * v6695_data);
          double v6700_data = s2[61];
          double v6703_p = (((((((v58_data * v5755_data) + (v193_data * v5890_data)) + (v328_data * v6025_data)) + (v463_data * v6160_data)) + (v598_data * v6295_data)) + (v733_data * v6430_data)) + (v868_data * v6565_data)) + (v1003_data * v6700_data);
          double v6705_data = s2[70];
          double v6708_p = (((((((v58_data * v5760_data) + (v193_data * v5895_data)) + (v328_data * v6030_data)) + (v463_data * v6165_data)) + (v598_data * v6300_data)) + (v733_data * v6435_data)) + (v868_data * v6570_data)) + (v1003_data * v6705_data);
          double v6710_data = s2[79];
          double v6713_p = (((((((v58_data * v5765_data) + (v193_data * v5900_data)) + (v328_data * v6035_data)) + (v463_data * v6170_data)) + (v598_data * v6305_data)) + (v733_data * v6440_data)) + (v868_data * v6575_data)) + (v1003_data * v6710_data);
          double v6718_p = (((((((v103_data * v5725_data) + (v238_data * v5860_data)) + (v373_data * v5995_data)) + (v508_data * v6130_data)) + (v643_data * v6265_data)) + (v778_data * v6400_data)) + (v913_data * v6535_data)) + (v1048_data * v6670_data);
          double v6723_p = (((((((v103_data * v5730_data) + (v238_data * v5865_data)) + (v373_data * v6000_data)) + (v508_data * v6135_data)) + (v643_data * v6270_data)) + (v778_data * v6405_data)) + (v913_data * v6540_data)) + (v1048_data * v6675_data);
          double v6728_p = (((((((v103_data * v5735_data) + (v238_data * v5870_data)) + (v373_data * v6005_data)) + (v508_data * v6140_data)) + (v643_data * v6275_data)) + (v778_data * v6410_data)) + (v913_data * v6545_data)) + (v1048_data * v6680_data);
          double v6733_p = (((((((v103_data * v5740_data) + (v238_data * v5875_data)) + (v373_data * v6010_data)) + (v508_data * v6145_data)) + (v643_data * v6280_data)) + (v778_data * v6415_data)) + (v913_data * v6550_data)) + (v1048_data * v6685_data);
          double v6738_p = (((((((v103_data * v5745_data) + (v238_data * v5880_data)) + (v373_data * v6015_data)) + (v508_data * v6150_data)) + (v643_data * v6285_data)) + (v778_data * v6420_data)) + (v913_data * v6555_data)) + (v1048_data * v6690_data);
          double v6743_p = (((((((v103_data * v5750_data) + (v238_data * v5885_data)) + (v373_data * v6020_data)) + (v508_data * v6155_data)) + (v643_data * v6290_data)) + (v778_data * v6425_data)) + (v913_data * v6560_data)) + (v1048_data * v6695_data);
          double v6748_p = (((((((v103_data * v5755_data) + (v238_data * v5890_data)) + (v373_data * v6025_data)) + (v508_data * v6160_data)) + (v643_data * v6295_data)) + (v778_data * v6430_data)) + (v913_data * v6565_data)) + (v1048_data * v6700_data);
          double v6753_p = (((((((v103_data * v5760_data) + (v238_data * v5895_data)) + (v373_data * v6030_data)) + (v508_data * v6165_data)) + (v643_data * v6300_data)) + (v778_data * v6435_data)) + (v913_data * v6570_data)) + (v1048_data * v6705_data);
          double v6758_p = (((((((v103_data * v5765_data) + (v238_data * v5900_data)) + (v373_data * v6035_data)) + (v508_data * v6170_data)) + (v643_data * v6305_data)) + (v778_data * v6440_data)) + (v913_data * v6575_data)) + (v1048_data * v6710_data);
          double v6763_p = (((((((v148_data * v5725_data) + (v283_data * v5860_data)) + (v418_data * v5995_data)) + (v553_data * v6130_data)) + (v688_data * v6265_data)) + (v823_data * v6400_data)) + (v958_data * v6535_data)) + (v1093_data * v6670_data);
          double v6768_p = (((((((v148_data * v5730_data) + (v283_data * v5865_data)) + (v418_data * v6000_data)) + (v553_data * v6135_data)) + (v688_data * v6270_data)) + (v823_data * v6405_data)) + (v958_data * v6540_data)) + (v1093_data * v6675_data);
          double v6773_p = (((((((v148_data * v5735_data) + (v283_data * v5870_data)) + (v418_data * v6005_data)) + (v553_data * v6140_data)) + (v688_data * v6275_data)) + (v823_data * v6410_data)) + (v958_data * v6545_data)) + (v1093_data * v6680_data);
          double v6778_p = (((((((v148_data * v5740_data) + (v283_data * v5875_data)) + (v418_data * v6010_data)) + (v553_data * v6145_data)) + (v688_data * v6280_data)) + (v823_data * v6415_data)) + (v958_data * v6550_data)) + (v1093_data * v6685_data);
          double v6783_p = (((((((v148_data * v5745_data) + (v283_data * v5880_data)) + (v418_data * v6015_data)) + (v553_data * v6150_data)) + (v688_data * v6285_data)) + (v823_data * v6420_data)) + (v958_data * v6555_data)) + (v1093_data * v6690_data);
          double v6788_p = (((((((v148_data * v5750_data) + (v283_data * v5885_data)) + (v418_data * v6020_data)) + (v553_data * v6155_data)) + (v688_data * v6290_data)) + (v823_data * v6425_data)) + (v958_data * v6560_data)) + (v1093_data * v6695_data);
          double v6793_p = (((((((v148_data * v5755_data) + (v283_data * v5890_data)) + (v418_data * v6025_data)) + (v553_data * v6160_data)) + (v688_data * v6295_data)) + (v823_data * v6430_data)) + (v958_data * v6565_data)) + (v1093_data * v6700_data);
          double v6798_p = (((((((v148_data * v5760_data) + (v283_data * v5895_data)) + (v418_data * v6030_data)) + (v553_data * v6165_data)) + (v688_data * v6300_data)) + (v823_data * v6435_data)) + (v958_data * v6570_data)) + (v1093_data * v6705_data);
          double v6803_p = (((((((v148_data * v5765_data) + (v283_data * v5900_data)) + (v418_data * v6035_data)) + (v553_data * v6170_data)) + (v688_data * v6305_data)) + (v823_data * v6440_data)) + (v958_data * v6575_data)) + (v1093_data * v6710_data);
          double v6805_data = s2[8];
          r5[0] = (((((((((v58_data * v5725_data) + (v193_data * v5860_data)) + (v328_data * v5995_data)) + (v463_data * v6130_data)) + (v598_data * v6265_data)) + (v733_data * v6400_data)) + (v868_data * v6535_data)) + (v1003_data * v6670_data)) + (v1138_data * v6805_data));
          double v6810_data = s2[17];
          r5[3] = (v6678_p + (v1138_data * v6810_data));
          double v6815_data = s2[26];
          r5[6] = (v6683_p + (v1138_data * v6815_data));
          double v6820_data = s2[35];
          r5[9] = (v6688_p + (v1138_data * v6820_data));
          double v6825_data = s2[44];
          r5[12] = (v6693_p + (v1138_data * v6825_data));
          double v6830_data = s2[53];
          r5[15] = (v6698_p + (v1138_data * v6830_data));
          double v6835_data = s2[62];
          r5[18] = (v6703_p + (v1138_data * v6835_data));
          double v6840_data = s2[71];
          r5[21] = (v6708_p + (v1138_data * v6840_data));
          double v6845_data = s2[80];
          r5[24] = (v6713_p + (v1138_data * v6845_data));
          r5[1] = (v6718_p + (v1183_data * v6805_data));
          r5[4] = (v6723_p + (v1183_data * v6810_data));
          r5[7] = (v6728_p + (v1183_data * v6815_data));
          r5[10] = (v6733_p + (v1183_data * v6820_data));
          r5[13] = (v6738_p + (v1183_data * v6825_data));
          r5[16] = (v6743_p + (v1183_data * v6830_data));
          r5[19] = (v6748_p + (v1183_data * v6835_data));
          r5[22] = (v6753_p + (v1183_data * v6840_data));
          r5[25] = (v6758_p + (v1183_data * v6845_data));
          r5[2] = (v6763_p + (v1228_data * v6805_data));
          r5[5] = (v6768_p + (v1228_data * v6810_data));
          r5[8] = (v6773_p + (v1228_data * v6815_data));
          r5[11] = (v6778_p + (v1228_data * v6820_data));
          r5[14] = (v6783_p + (v1228_data * v6825_data));
          r5[17] = (v6788_p + (v1228_data * v6830_data));
          r5[20] = (v6793_p + (v1228_data * v6835_data));
          r5[23] = (v6798_p + (v1228_data * v6840_data));
          r5[26] = (v6803_p + (v1228_data * v6845_data));
          // r5 = 0 past (0, 35)
          // s3 = store{r>s}(localShrMem0, r5);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v6940_i0 = 0; v6940_i0 < 2; ++v6940_i0) {
            int32_t v6946_lead = v18980_lead + (v6940_i0 * 16);
            #pragma unroll
            for (int32_t v6941_i1 = 0; v6941_i1 < 9; ++v6941_i1) {
              double v6944_data = r5[(v6940_i0 + (v6941_i1 * 3))];
              s3[(v6946_lead + (v6941_i1 * 35))] = v6944_data;
            }
          }
          if (v18981_g) {
            #pragma unroll
            for (int32_t v6949_i1 = 0; v6949_i1 < 9; ++v6949_i1) {
              double v6952_data = r5[(2 + (v6949_i1 * 3))];
              s3[(v1393_lead + (v6949_i1 * 35))] = v6952_data;
            }
          }
          double r7[36]{};
          // ir7 = +(glb_m5 * s3)
          // [(1, 55), (0, 9)] [(0, 35)]
          double ir7[36]{};
          double v6963_data = v5629_g ? (glb_m5[v1295_off]) : (0.0);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v6964_data = s3[0];
          double v6969_data = s3[35];
          double v6974_data = s3[70];
          double v6979_data = s3[105];
          double v6984_data = s3[140];
          double v6989_data = s3[175];
          double v6994_data = s3[210];
          double v6999_data = s3[245];
          double v7004_data = s3[280];
          double v7012_data = glb_m5[v1345_off];
          double v7061_data = glb_m5[v1394_off];
          double v7111_data = v12016_g ? (glb_m5[v1443_off]) : (0.0);
          double v7158_data = v5629_g ? (glb_m5[(v18980_lead + 53)]) : (0.0);
          double v7159_data = s3[1];
          double v7164_data = s3[36];
          double v7169_data = s3[71];
          double v7174_data = s3[106];
          double v7179_data = s3[141];
          double v7184_data = s3[176];
          double v7189_data = s3[211];
          double v7194_data = s3[246];
          double v7199_data = s3[281];
          double v7205_data = glb_m5[(v1344_lead + 53)];
          double v7252_data = glb_m5[(v1393_lead + 53)];
          double v7299_data = v12016_g ? (glb_m5[(v1442_lead + 53)]) : (0.0);
          double v7346_data = v5629_g ? (glb_m5[(v18980_lead + 107)]) : (0.0);
          double v7347_data = s3[2];
          double v7352_data = s3[37];
          double v7357_data = s3[72];
          double v7362_data = s3[107];
          double v7367_data = s3[142];
          double v7372_data = s3[177];
          double v7377_data = s3[212];
          double v7382_data = s3[247];
          double v7387_data = s3[282];
          double v7393_data = glb_m5[(v1344_lead + 107)];
          double v7440_data = glb_m5[(v1393_lead + 107)];
          double v7487_data = v12016_g ? (glb_m5[(v1442_lead + 107)]) : (0.0);
          double v7534_data = v5629_g ? (glb_m5[(v18980_lead + 161)]) : (0.0);
          double v7535_data = s3[3];
          double v7540_data = s3[38];
          double v7545_data = s3[73];
          double v7550_data = s3[108];
          double v7555_data = s3[143];
          double v7560_data = s3[178];
          double v7565_data = s3[213];
          double v7570_data = s3[248];
          double v7575_data = s3[283];
          double v7581_data = glb_m5[(v1344_lead + 161)];
          double v7628_data = glb_m5[(v1393_lead + 161)];
          double v7675_data = v12016_g ? (glb_m5[(v1442_lead + 161)]) : (0.0);
          double v7722_data = v5629_g ? (glb_m5[(v18980_lead + 215)]) : (0.0);
          double v7723_data = s3[4];
          double v7728_data = s3[39];
          double v7733_data = s3[74];
          double v7738_data = s3[109];
          double v7743_data = s3[144];
          double v7748_data = s3[179];
          double v7753_data = s3[214];
          double v7758_data = s3[249];
          double v7763_data = s3[284];
          double v7769_data = glb_m5[(v1344_lead + 215)];
          double v7816_data = glb_m5[(v1393_lead + 215)];
          double v7863_data = v12016_g ? (glb_m5[(v1442_lead + 215)]) : (0.0);
          double v7910_data = v5629_g ? (glb_m5[(v18980_lead + 269)]) : (0.0);
          double v7911_data = s3[5];
          double v7916_data = s3[40];
          double v7921_data = s3[75];
          double v7926_data = s3[110];
          double v7931_data = s3[145];
          double v7936_data = s3[180];
          double v7941_data = s3[215];
          double v7946_data = s3[250];
          double v7951_data = s3[285];
          double v7957_data = glb_m5[(v1344_lead + 269)];
          double v8004_data = glb_m5[(v1393_lead + 269)];
          double v8051_data = v12016_g ? (glb_m5[(v1442_lead + 269)]) : (0.0);
          double v8098_data = v5629_g ? (glb_m5[(v18980_lead + 323)]) : (0.0);
          double v8099_data = s3[6];
          double v8104_data = s3[41];
          double v8109_data = s3[76];
          double v8114_data = s3[111];
          double v8119_data = s3[146];
          double v8124_data = s3[181];
          double v8129_data = s3[216];
          double v8134_data = s3[251];
          double v8139_data = s3[286];
          double v8145_data = glb_m5[(v1344_lead + 323)];
          double v8192_data = glb_m5[(v1393_lead + 323)];
          double v8239_data = v12016_g ? (glb_m5[(v1442_lead + 323)]) : (0.0);
          double v8286_data = v5629_g ? (glb_m5[(v18980_lead + 377)]) : (0.0);
          double v8287_data = s3[7];
          double v8292_data = s3[42];
          double v8297_data = s3[77];
          double v8302_data = s3[112];
          double v8307_data = s3[147];
          double v8312_data = s3[182];
          double v8317_data = s3[217];
          double v8322_data = s3[252];
          double v8327_data = s3[287];
          double v8333_data = glb_m5[(v1344_lead + 377)];
          double v8380_data = glb_m5[(v1393_lead + 377)];
          double v8427_data = v12016_g ? (glb_m5[(v1442_lead + 377)]) : (0.0);
          double v8474_data = v5629_g ? (glb_m5[(v18980_lead + 431)]) : (0.0);
          double v8475_data = s3[8];
          double v8480_data = s3[43];
          double v8485_data = s3[78];
          double v8490_data = s3[113];
          double v8495_data = s3[148];
          double v8500_data = s3[183];
          double v8505_data = s3[218];
          double v8510_data = s3[253];
          double v8515_data = s3[288];
          double v8521_data = glb_m5[(v1344_lead + 431)];
          double v8568_data = glb_m5[(v1393_lead + 431)];
          double v8615_data = v12016_g ? (glb_m5[(v1442_lead + 431)]) : (0.0);
          double v8662_data = v5629_g ? (glb_m5[(v18980_lead + 485)]) : (0.0);
          double v8663_data = s3[9];
          double v8668_data = s3[44];
          double v8673_data = s3[79];
          double v8678_data = s3[114];
          double v8683_data = s3[149];
          double v8688_data = s3[184];
          double v8693_data = s3[219];
          double v8698_data = s3[254];
          double v8703_data = s3[289];
          double v8709_data = glb_m5[(v1344_lead + 485)];
          double v8756_data = glb_m5[(v1393_lead + 485)];
          double v8803_data = v12016_g ? (glb_m5[(v1442_lead + 485)]) : (0.0);
          double v8850_data = v5629_g ? (glb_m5[(v18980_lead + 539)]) : (0.0);
          double v8851_data = s3[10];
          double v8856_data = s3[45];
          double v8861_data = s3[80];
          double v8866_data = s3[115];
          double v8871_data = s3[150];
          double v8876_data = s3[185];
          double v8881_data = s3[220];
          double v8886_data = s3[255];
          double v8891_data = s3[290];
          double v8897_data = glb_m5[(v1344_lead + 539)];
          double v8944_data = glb_m5[(v1393_lead + 539)];
          double v8991_data = v12016_g ? (glb_m5[(v1442_lead + 539)]) : (0.0);
          double v9038_data = v5629_g ? (glb_m5[(v18980_lead + 593)]) : (0.0);
          double v9039_data = s3[11];
          double v9044_data = s3[46];
          double v9049_data = s3[81];
          double v9054_data = s3[116];
          double v9059_data = s3[151];
          double v9064_data = s3[186];
          double v9069_data = s3[221];
          double v9074_data = s3[256];
          double v9079_data = s3[291];
          double v9085_data = glb_m5[(v1344_lead + 593)];
          double v9132_data = glb_m5[(v1393_lead + 593)];
          double v9179_data = v12016_g ? (glb_m5[(v1442_lead + 593)]) : (0.0);
          double v9227_data = v2809_g ? (glb_m5[(v18980_lead + 640)]) : (0.0);
          double v9228_data = s3[12];
          double v9233_data = s3[47];
          double v9238_data = s3[82];
          double v9243_data = s3[117];
          double v9248_data = s3[152];
          double v9253_data = s3[187];
          double v9258_data = s3[222];
          double v9263_data = s3[257];
          double v9268_data = s3[292];
          double v9274_data = glb_m5[(v1344_lead + 640)];
          double v9321_data = glb_m5[(v1393_lead + 640)];
          double v9368_data = v12016_g ? (glb_m5[(v1442_lead + 640)]) : (0.0);
          double v9415_data = v2809_g ? (glb_m5[(v18980_lead + 687)]) : (0.0);
          double v9416_data = s3[13];
          double v9421_data = s3[48];
          double v9426_data = s3[83];
          double v9431_data = s3[118];
          double v9436_data = s3[153];
          double v9441_data = s3[188];
          double v9446_data = s3[223];
          double v9451_data = s3[258];
          double v9456_data = s3[293];
          double v9462_data = glb_m5[(v1344_lead + 687)];
          double v9509_data = glb_m5[(v1393_lead + 687)];
          double v9556_data = v12016_g ? (glb_m5[(v1442_lead + 687)]) : (0.0);
          double v9603_data = v2809_g ? (glb_m5[(v18980_lead + 734)]) : (0.0);
          double v9604_data = s3[14];
          double v9609_data = s3[49];
          double v9612_p = ((((((((((((((v6963_data * v6969_data) + (v7158_data * v7164_data)) + (v7346_data * v7352_data)) + (v7534_data * v7540_data)) + (v7722_data * v7728_data)) + (v7910_data * v7916_data)) + (v8098_data * v8104_data)) + (v8286_data * v8292_data)) + (v8474_data * v8480_data)) + (v8662_data * v8668_data)) + (v8850_data * v8856_data)) + (v9038_data * v9044_data)) + (v9227_data * v9233_data)) + (v9415_data * v9421_data)) + (v9603_data * v9609_data);
          double v9614_data = s3[84];
          double v9617_p = ((((((((((((((v6963_data * v6974_data) + (v7158_data * v7169_data)) + (v7346_data * v7357_data)) + (v7534_data * v7545_data)) + (v7722_data * v7733_data)) + (v7910_data * v7921_data)) + (v8098_data * v8109_data)) + (v8286_data * v8297_data)) + (v8474_data * v8485_data)) + (v8662_data * v8673_data)) + (v8850_data * v8861_data)) + (v9038_data * v9049_data)) + (v9227_data * v9238_data)) + (v9415_data * v9426_data)) + (v9603_data * v9614_data);
          double v9619_data = s3[119];
          double v9622_p = ((((((((((((((v6963_data * v6979_data) + (v7158_data * v7174_data)) + (v7346_data * v7362_data)) + (v7534_data * v7550_data)) + (v7722_data * v7738_data)) + (v7910_data * v7926_data)) + (v8098_data * v8114_data)) + (v8286_data * v8302_data)) + (v8474_data * v8490_data)) + (v8662_data * v8678_data)) + (v8850_data * v8866_data)) + (v9038_data * v9054_data)) + (v9227_data * v9243_data)) + (v9415_data * v9431_data)) + (v9603_data * v9619_data);
          double v9624_data = s3[154];
          double v9627_p = ((((((((((((((v6963_data * v6984_data) + (v7158_data * v7179_data)) + (v7346_data * v7367_data)) + (v7534_data * v7555_data)) + (v7722_data * v7743_data)) + (v7910_data * v7931_data)) + (v8098_data * v8119_data)) + (v8286_data * v8307_data)) + (v8474_data * v8495_data)) + (v8662_data * v8683_data)) + (v8850_data * v8871_data)) + (v9038_data * v9059_data)) + (v9227_data * v9248_data)) + (v9415_data * v9436_data)) + (v9603_data * v9624_data);
          double v9629_data = s3[189];
          double v9632_p = ((((((((((((((v6963_data * v6989_data) + (v7158_data * v7184_data)) + (v7346_data * v7372_data)) + (v7534_data * v7560_data)) + (v7722_data * v7748_data)) + (v7910_data * v7936_data)) + (v8098_data * v8124_data)) + (v8286_data * v8312_data)) + (v8474_data * v8500_data)) + (v8662_data * v8688_data)) + (v8850_data * v8876_data)) + (v9038_data * v9064_data)) + (v9227_data * v9253_data)) + (v9415_data * v9441_data)) + (v9603_data * v9629_data);
          double v9634_data = s3[224];
          double v9637_p = ((((((((((((((v6963_data * v6994_data) + (v7158_data * v7189_data)) + (v7346_data * v7377_data)) + (v7534_data * v7565_data)) + (v7722_data * v7753_data)) + (v7910_data * v7941_data)) + (v8098_data * v8129_data)) + (v8286_data * v8317_data)) + (v8474_data * v8505_data)) + (v8662_data * v8693_data)) + (v8850_data * v8881_data)) + (v9038_data * v9069_data)) + (v9227_data * v9258_data)) + (v9415_data * v9446_data)) + (v9603_data * v9634_data);
          double v9639_data = s3[259];
          double v9642_p = ((((((((((((((v6963_data * v6999_data) + (v7158_data * v7194_data)) + (v7346_data * v7382_data)) + (v7534_data * v7570_data)) + (v7722_data * v7758_data)) + (v7910_data * v7946_data)) + (v8098_data * v8134_data)) + (v8286_data * v8322_data)) + (v8474_data * v8510_data)) + (v8662_data * v8698_data)) + (v8850_data * v8886_data)) + (v9038_data * v9074_data)) + (v9227_data * v9263_data)) + (v9415_data * v9451_data)) + (v9603_data * v9639_data);
          double v9644_data = s3[294];
          double v9647_p = ((((((((((((((v6963_data * v7004_data) + (v7158_data * v7199_data)) + (v7346_data * v7387_data)) + (v7534_data * v7575_data)) + (v7722_data * v7763_data)) + (v7910_data * v7951_data)) + (v8098_data * v8139_data)) + (v8286_data * v8327_data)) + (v8474_data * v8515_data)) + (v8662_data * v8703_data)) + (v8850_data * v8891_data)) + (v9038_data * v9079_data)) + (v9227_data * v9268_data)) + (v9415_data * v9456_data)) + (v9603_data * v9644_data);
          double v9650_data = glb_m5[(v1344_lead + 734)];
          double v9654_p = ((((((((((((((v7012_data * v6964_data) + (v7205_data * v7159_data)) + (v7393_data * v7347_data)) + (v7581_data * v7535_data)) + (v7769_data * v7723_data)) + (v7957_data * v7911_data)) + (v8145_data * v8099_data)) + (v8333_data * v8287_data)) + (v8521_data * v8475_data)) + (v8709_data * v8663_data)) + (v8897_data * v8851_data)) + (v9085_data * v9039_data)) + (v9274_data * v9228_data)) + (v9462_data * v9416_data)) + (v9650_data * v9604_data);
          double v9659_p = ((((((((((((((v7012_data * v6969_data) + (v7205_data * v7164_data)) + (v7393_data * v7352_data)) + (v7581_data * v7540_data)) + (v7769_data * v7728_data)) + (v7957_data * v7916_data)) + (v8145_data * v8104_data)) + (v8333_data * v8292_data)) + (v8521_data * v8480_data)) + (v8709_data * v8668_data)) + (v8897_data * v8856_data)) + (v9085_data * v9044_data)) + (v9274_data * v9233_data)) + (v9462_data * v9421_data)) + (v9650_data * v9609_data);
          double v9664_p = ((((((((((((((v7012_data * v6974_data) + (v7205_data * v7169_data)) + (v7393_data * v7357_data)) + (v7581_data * v7545_data)) + (v7769_data * v7733_data)) + (v7957_data * v7921_data)) + (v8145_data * v8109_data)) + (v8333_data * v8297_data)) + (v8521_data * v8485_data)) + (v8709_data * v8673_data)) + (v8897_data * v8861_data)) + (v9085_data * v9049_data)) + (v9274_data * v9238_data)) + (v9462_data * v9426_data)) + (v9650_data * v9614_data);
          double v9669_p = ((((((((((((((v7012_data * v6979_data) + (v7205_data * v7174_data)) + (v7393_data * v7362_data)) + (v7581_data * v7550_data)) + (v7769_data * v7738_data)) + (v7957_data * v7926_data)) + (v8145_data * v8114_data)) + (v8333_data * v8302_data)) + (v8521_data * v8490_data)) + (v8709_data * v8678_data)) + (v8897_data * v8866_data)) + (v9085_data * v9054_data)) + (v9274_data * v9243_data)) + (v9462_data * v9431_data)) + (v9650_data * v9619_data);
          double v9674_p = ((((((((((((((v7012_data * v6984_data) + (v7205_data * v7179_data)) + (v7393_data * v7367_data)) + (v7581_data * v7555_data)) + (v7769_data * v7743_data)) + (v7957_data * v7931_data)) + (v8145_data * v8119_data)) + (v8333_data * v8307_data)) + (v8521_data * v8495_data)) + (v8709_data * v8683_data)) + (v8897_data * v8871_data)) + (v9085_data * v9059_data)) + (v9274_data * v9248_data)) + (v9462_data * v9436_data)) + (v9650_data * v9624_data);
          double v9679_p = ((((((((((((((v7012_data * v6989_data) + (v7205_data * v7184_data)) + (v7393_data * v7372_data)) + (v7581_data * v7560_data)) + (v7769_data * v7748_data)) + (v7957_data * v7936_data)) + (v8145_data * v8124_data)) + (v8333_data * v8312_data)) + (v8521_data * v8500_data)) + (v8709_data * v8688_data)) + (v8897_data * v8876_data)) + (v9085_data * v9064_data)) + (v9274_data * v9253_data)) + (v9462_data * v9441_data)) + (v9650_data * v9629_data);
          double v9684_p = ((((((((((((((v7012_data * v6994_data) + (v7205_data * v7189_data)) + (v7393_data * v7377_data)) + (v7581_data * v7565_data)) + (v7769_data * v7753_data)) + (v7957_data * v7941_data)) + (v8145_data * v8129_data)) + (v8333_data * v8317_data)) + (v8521_data * v8505_data)) + (v8709_data * v8693_data)) + (v8897_data * v8881_data)) + (v9085_data * v9069_data)) + (v9274_data * v9258_data)) + (v9462_data * v9446_data)) + (v9650_data * v9634_data);
          double v9689_p = ((((((((((((((v7012_data * v6999_data) + (v7205_data * v7194_data)) + (v7393_data * v7382_data)) + (v7581_data * v7570_data)) + (v7769_data * v7758_data)) + (v7957_data * v7946_data)) + (v8145_data * v8134_data)) + (v8333_data * v8322_data)) + (v8521_data * v8510_data)) + (v8709_data * v8698_data)) + (v8897_data * v8886_data)) + (v9085_data * v9074_data)) + (v9274_data * v9263_data)) + (v9462_data * v9451_data)) + (v9650_data * v9639_data);
          double v9694_p = ((((((((((((((v7012_data * v7004_data) + (v7205_data * v7199_data)) + (v7393_data * v7387_data)) + (v7581_data * v7575_data)) + (v7769_data * v7763_data)) + (v7957_data * v7951_data)) + (v8145_data * v8139_data)) + (v8333_data * v8327_data)) + (v8521_data * v8515_data)) + (v8709_data * v8703_data)) + (v8897_data * v8891_data)) + (v9085_data * v9079_data)) + (v9274_data * v9268_data)) + (v9462_data * v9456_data)) + (v9650_data * v9644_data);
          double v9697_data = glb_m5[(v1393_lead + 734)];
          double v9701_p = ((((((((((((((v7061_data * v6964_data) + (v7252_data * v7159_data)) + (v7440_data * v7347_data)) + (v7628_data * v7535_data)) + (v7816_data * v7723_data)) + (v8004_data * v7911_data)) + (v8192_data * v8099_data)) + (v8380_data * v8287_data)) + (v8568_data * v8475_data)) + (v8756_data * v8663_data)) + (v8944_data * v8851_data)) + (v9132_data * v9039_data)) + (v9321_data * v9228_data)) + (v9509_data * v9416_data)) + (v9697_data * v9604_data);
          double v9706_p = ((((((((((((((v7061_data * v6969_data) + (v7252_data * v7164_data)) + (v7440_data * v7352_data)) + (v7628_data * v7540_data)) + (v7816_data * v7728_data)) + (v8004_data * v7916_data)) + (v8192_data * v8104_data)) + (v8380_data * v8292_data)) + (v8568_data * v8480_data)) + (v8756_data * v8668_data)) + (v8944_data * v8856_data)) + (v9132_data * v9044_data)) + (v9321_data * v9233_data)) + (v9509_data * v9421_data)) + (v9697_data * v9609_data);
          double v9711_p = ((((((((((((((v7061_data * v6974_data) + (v7252_data * v7169_data)) + (v7440_data * v7357_data)) + (v7628_data * v7545_data)) + (v7816_data * v7733_data)) + (v8004_data * v7921_data)) + (v8192_data * v8109_data)) + (v8380_data * v8297_data)) + (v8568_data * v8485_data)) + (v8756_data * v8673_data)) + (v8944_data * v8861_data)) + (v9132_data * v9049_data)) + (v9321_data * v9238_data)) + (v9509_data * v9426_data)) + (v9697_data * v9614_data);
          double v9716_p = ((((((((((((((v7061_data * v6979_data) + (v7252_data * v7174_data)) + (v7440_data * v7362_data)) + (v7628_data * v7550_data)) + (v7816_data * v7738_data)) + (v8004_data * v7926_data)) + (v8192_data * v8114_data)) + (v8380_data * v8302_data)) + (v8568_data * v8490_data)) + (v8756_data * v8678_data)) + (v8944_data * v8866_data)) + (v9132_data * v9054_data)) + (v9321_data * v9243_data)) + (v9509_data * v9431_data)) + (v9697_data * v9619_data);
          double v9721_p = ((((((((((((((v7061_data * v6984_data) + (v7252_data * v7179_data)) + (v7440_data * v7367_data)) + (v7628_data * v7555_data)) + (v7816_data * v7743_data)) + (v8004_data * v7931_data)) + (v8192_data * v8119_data)) + (v8380_data * v8307_data)) + (v8568_data * v8495_data)) + (v8756_data * v8683_data)) + (v8944_data * v8871_data)) + (v9132_data * v9059_data)) + (v9321_data * v9248_data)) + (v9509_data * v9436_data)) + (v9697_data * v9624_data);
          double v9726_p = ((((((((((((((v7061_data * v6989_data) + (v7252_data * v7184_data)) + (v7440_data * v7372_data)) + (v7628_data * v7560_data)) + (v7816_data * v7748_data)) + (v8004_data * v7936_data)) + (v8192_data * v8124_data)) + (v8380_data * v8312_data)) + (v8568_data * v8500_data)) + (v8756_data * v8688_data)) + (v8944_data * v8876_data)) + (v9132_data * v9064_data)) + (v9321_data * v9253_data)) + (v9509_data * v9441_data)) + (v9697_data * v9629_data);
          double v9731_p = ((((((((((((((v7061_data * v6994_data) + (v7252_data * v7189_data)) + (v7440_data * v7377_data)) + (v7628_data * v7565_data)) + (v7816_data * v7753_data)) + (v8004_data * v7941_data)) + (v8192_data * v8129_data)) + (v8380_data * v8317_data)) + (v8568_data * v8505_data)) + (v8756_data * v8693_data)) + (v8944_data * v8881_data)) + (v9132_data * v9069_data)) + (v9321_data * v9258_data)) + (v9509_data * v9446_data)) + (v9697_data * v9634_data);
          double v9736_p = ((((((((((((((v7061_data * v6999_data) + (v7252_data * v7194_data)) + (v7440_data * v7382_data)) + (v7628_data * v7570_data)) + (v7816_data * v7758_data)) + (v8004_data * v7946_data)) + (v8192_data * v8134_data)) + (v8380_data * v8322_data)) + (v8568_data * v8510_data)) + (v8756_data * v8698_data)) + (v8944_data * v8886_data)) + (v9132_data * v9074_data)) + (v9321_data * v9263_data)) + (v9509_data * v9451_data)) + (v9697_data * v9639_data);
          double v9741_p = ((((((((((((((v7061_data * v7004_data) + (v7252_data * v7199_data)) + (v7440_data * v7387_data)) + (v7628_data * v7575_data)) + (v7816_data * v7763_data)) + (v8004_data * v7951_data)) + (v8192_data * v8139_data)) + (v8380_data * v8327_data)) + (v8568_data * v8515_data)) + (v8756_data * v8703_data)) + (v8944_data * v8891_data)) + (v9132_data * v9079_data)) + (v9321_data * v9268_data)) + (v9509_data * v9456_data)) + (v9697_data * v9644_data);
          double v9744_data = v12016_g ? (glb_m5[(v1442_lead + 734)]) : (0.0);
          double v9748_p = ((((((((((((((v7111_data * v6964_data) + (v7299_data * v7159_data)) + (v7487_data * v7347_data)) + (v7675_data * v7535_data)) + (v7863_data * v7723_data)) + (v8051_data * v7911_data)) + (v8239_data * v8099_data)) + (v8427_data * v8287_data)) + (v8615_data * v8475_data)) + (v8803_data * v8663_data)) + (v8991_data * v8851_data)) + (v9179_data * v9039_data)) + (v9368_data * v9228_data)) + (v9556_data * v9416_data)) + (v9744_data * v9604_data);
          double v9753_p = ((((((((((((((v7111_data * v6969_data) + (v7299_data * v7164_data)) + (v7487_data * v7352_data)) + (v7675_data * v7540_data)) + (v7863_data * v7728_data)) + (v8051_data * v7916_data)) + (v8239_data * v8104_data)) + (v8427_data * v8292_data)) + (v8615_data * v8480_data)) + (v8803_data * v8668_data)) + (v8991_data * v8856_data)) + (v9179_data * v9044_data)) + (v9368_data * v9233_data)) + (v9556_data * v9421_data)) + (v9744_data * v9609_data);
          double v9758_p = ((((((((((((((v7111_data * v6974_data) + (v7299_data * v7169_data)) + (v7487_data * v7357_data)) + (v7675_data * v7545_data)) + (v7863_data * v7733_data)) + (v8051_data * v7921_data)) + (v8239_data * v8109_data)) + (v8427_data * v8297_data)) + (v8615_data * v8485_data)) + (v8803_data * v8673_data)) + (v8991_data * v8861_data)) + (v9179_data * v9049_data)) + (v9368_data * v9238_data)) + (v9556_data * v9426_data)) + (v9744_data * v9614_data);
          double v9763_p = ((((((((((((((v7111_data * v6979_data) + (v7299_data * v7174_data)) + (v7487_data * v7362_data)) + (v7675_data * v7550_data)) + (v7863_data * v7738_data)) + (v8051_data * v7926_data)) + (v8239_data * v8114_data)) + (v8427_data * v8302_data)) + (v8615_data * v8490_data)) + (v8803_data * v8678_data)) + (v8991_data * v8866_data)) + (v9179_data * v9054_data)) + (v9368_data * v9243_data)) + (v9556_data * v9431_data)) + (v9744_data * v9619_data);
          double v9768_p = ((((((((((((((v7111_data * v6984_data) + (v7299_data * v7179_data)) + (v7487_data * v7367_data)) + (v7675_data * v7555_data)) + (v7863_data * v7743_data)) + (v8051_data * v7931_data)) + (v8239_data * v8119_data)) + (v8427_data * v8307_data)) + (v8615_data * v8495_data)) + (v8803_data * v8683_data)) + (v8991_data * v8871_data)) + (v9179_data * v9059_data)) + (v9368_data * v9248_data)) + (v9556_data * v9436_data)) + (v9744_data * v9624_data);
          double v9773_p = ((((((((((((((v7111_data * v6989_data) + (v7299_data * v7184_data)) + (v7487_data * v7372_data)) + (v7675_data * v7560_data)) + (v7863_data * v7748_data)) + (v8051_data * v7936_data)) + (v8239_data * v8124_data)) + (v8427_data * v8312_data)) + (v8615_data * v8500_data)) + (v8803_data * v8688_data)) + (v8991_data * v8876_data)) + (v9179_data * v9064_data)) + (v9368_data * v9253_data)) + (v9556_data * v9441_data)) + (v9744_data * v9629_data);
          double v9778_p = ((((((((((((((v7111_data * v6994_data) + (v7299_data * v7189_data)) + (v7487_data * v7377_data)) + (v7675_data * v7565_data)) + (v7863_data * v7753_data)) + (v8051_data * v7941_data)) + (v8239_data * v8129_data)) + (v8427_data * v8317_data)) + (v8615_data * v8505_data)) + (v8803_data * v8693_data)) + (v8991_data * v8881_data)) + (v9179_data * v9069_data)) + (v9368_data * v9258_data)) + (v9556_data * v9446_data)) + (v9744_data * v9634_data);
          double v9783_p = ((((((((((((((v7111_data * v6999_data) + (v7299_data * v7194_data)) + (v7487_data * v7382_data)) + (v7675_data * v7570_data)) + (v7863_data * v7758_data)) + (v8051_data * v7946_data)) + (v8239_data * v8134_data)) + (v8427_data * v8322_data)) + (v8615_data * v8510_data)) + (v8803_data * v8698_data)) + (v8991_data * v8886_data)) + (v9179_data * v9074_data)) + (v9368_data * v9263_data)) + (v9556_data * v9451_data)) + (v9744_data * v9639_data);
          double v9788_p = ((((((((((((((v7111_data * v7004_data) + (v7299_data * v7199_data)) + (v7487_data * v7387_data)) + (v7675_data * v7575_data)) + (v7863_data * v7763_data)) + (v8051_data * v7951_data)) + (v8239_data * v8139_data)) + (v8427_data * v8327_data)) + (v8615_data * v8515_data)) + (v8803_data * v8703_data)) + (v8991_data * v8891_data)) + (v9179_data * v9079_data)) + (v9368_data * v9268_data)) + (v9556_data * v9456_data)) + (v9744_data * v9644_data);
          double v9791_data = v2809_g ? (glb_m5[(v18980_lead + 781)]) : (0.0);
          double v9792_data = s3[15];
          ir7[0] = ((((((((((((((((v6963_data * v6964_data) + (v7158_data * v7159_data)) + (v7346_data * v7347_data)) + (v7534_data * v7535_data)) + (v7722_data * v7723_data)) + (v7910_data * v7911_data)) + (v8098_data * v8099_data)) + (v8286_data * v8287_data)) + (v8474_data * v8475_data)) + (v8662_data * v8663_data)) + (v8850_data * v8851_data)) + (v9038_data * v9039_data)) + (v9227_data * v9228_data)) + (v9415_data * v9416_data)) + (v9603_data * v9604_data)) + (v9791_data * v9792_data));
          double v9797_data = s3[50];
          ir7[4] = (v9612_p + (v9791_data * v9797_data));
          double v9802_data = s3[85];
          ir7[8] = (v9617_p + (v9791_data * v9802_data));
          double v9807_data = s3[120];
          ir7[12] = (v9622_p + (v9791_data * v9807_data));
          double v9812_data = s3[155];
          ir7[16] = (v9627_p + (v9791_data * v9812_data));
          double v9817_data = s3[190];
          ir7[20] = (v9632_p + (v9791_data * v9817_data));
          double v9822_data = s3[225];
          ir7[24] = (v9637_p + (v9791_data * v9822_data));
          double v9827_data = s3[260];
          ir7[28] = (v9642_p + (v9791_data * v9827_data));
          double v9832_data = s3[295];
          ir7[32] = (v9647_p + (v9791_data * v9832_data));
          double v9838_data = glb_m5[(v1344_lead + 781)];
          double v9885_data = glb_m5[(v1393_lead + 781)];
          double v9932_data = v12016_g ? (glb_m5[(v1442_lead + 781)]) : (0.0);
          double v9979_data = glb_m5[(v1344_lead + 820)];
          double v9980_data = s3[16];
          double v9985_data = s3[51];
          double v9990_data = s3[86];
          double v9995_data = s3[121];
          double v10000_data = s3[156];
          double v10005_data = s3[191];
          double v10010_data = s3[226];
          double v10015_data = s3[261];
          double v10020_data = s3[296];
          double v10026_data = glb_m5[(v1393_lead + 820)];
          double v10073_data = v12016_g ? (glb_m5[(v1442_lead + 820)]) : (0.0);
          double v10120_data = glb_m5[(v1344_lead + 859)];
          double v10121_data = s3[17];
          double v10126_data = s3[52];
          double v10131_data = s3[87];
          double v10136_data = s3[122];
          double v10141_data = s3[157];
          double v10146_data = s3[192];
          double v10151_data = s3[227];
          double v10156_data = s3[262];
          double v10161_data = s3[297];
          double v10167_data = glb_m5[(v1393_lead + 859)];
          double v10214_data = v12016_g ? (glb_m5[(v1442_lead + 859)]) : (0.0);
          double v10261_data = glb_m5[(v1344_lead + 898)];
          double v10262_data = s3[18];
          double v10267_data = s3[53];
          double v10272_data = s3[88];
          double v10277_data = s3[123];
          double v10282_data = s3[158];
          double v10287_data = s3[193];
          double v10292_data = s3[228];
          double v10297_data = s3[263];
          double v10302_data = s3[298];
          double v10308_data = glb_m5[(v1393_lead + 898)];
          double v10355_data = v12016_g ? (glb_m5[(v1442_lead + 898)]) : (0.0);
          double v10402_data = glb_m5[(v1344_lead + 937)];
          double v10403_data = s3[19];
          double v10408_data = s3[54];
          double v10413_data = s3[89];
          double v10418_data = s3[124];
          double v10423_data = s3[159];
          double v10428_data = s3[194];
          double v10433_data = s3[229];
          double v10438_data = s3[264];
          double v10443_data = s3[299];
          double v10449_data = glb_m5[(v1393_lead + 937)];
          double v10496_data = v12016_g ? (glb_m5[(v1442_lead + 937)]) : (0.0);
          int32_t v10541_off = v1344_lead + 968;
          double v10543_data = v2809_g ? (glb_m5[v10541_off]) : (0.0);
          double v10544_data = s3[20];
          double v10549_data = s3[55];
          double v10554_data = s3[90];
          double v10559_data = s3[125];
          double v10564_data = s3[160];
          double v10569_data = s3[195];
          double v10574_data = s3[230];
          double v10579_data = s3[265];
          double v10584_data = s3[300];
          int32_t v10588_off = v1393_lead + 968;
          double v10590_data = glb_m5[v10588_off];
          int32_t v10635_off = v1442_lead + 968;
          double v10637_data = v12016_g ? (glb_m5[v10635_off]) : (0.0);
          double v10684_data = v2809_g ? (glb_m5[(v1344_lead + 999)]) : (0.0);
          double v10685_data = s3[21];
          double v10690_data = s3[56];
          double v10695_data = s3[91];
          double v10700_data = s3[126];
          double v10705_data = s3[161];
          double v10710_data = s3[196];
          double v10715_data = s3[231];
          double v10720_data = s3[266];
          double v10725_data = s3[301];
          double v10731_data = glb_m5[(v1393_lead + 999)];
          double v10778_data = v12016_g ? (glb_m5[(v1442_lead + 999)]) : (0.0);
          double v10825_data = v2809_g ? (glb_m5[(v1344_lead + 1030)]) : (0.0);
          double v10826_data = s3[22];
          double v10831_data = s3[57];
          double v10834_p = (((((((v9659_p + (v9838_data * v9797_data)) + (v9979_data * v9985_data)) + (v10120_data * v10126_data)) + (v10261_data * v10267_data)) + (v10402_data * v10408_data)) + (v10543_data * v10549_data)) + (v10684_data * v10690_data)) + (v10825_data * v10831_data);
          double v10836_data = s3[92];
          double v10839_p = (((((((v9664_p + (v9838_data * v9802_data)) + (v9979_data * v9990_data)) + (v10120_data * v10131_data)) + (v10261_data * v10272_data)) + (v10402_data * v10413_data)) + (v10543_data * v10554_data)) + (v10684_data * v10695_data)) + (v10825_data * v10836_data);
          double v10841_data = s3[127];
          double v10844_p = (((((((v9669_p + (v9838_data * v9807_data)) + (v9979_data * v9995_data)) + (v10120_data * v10136_data)) + (v10261_data * v10277_data)) + (v10402_data * v10418_data)) + (v10543_data * v10559_data)) + (v10684_data * v10700_data)) + (v10825_data * v10841_data);
          double v10846_data = s3[162];
          double v10849_p = (((((((v9674_p + (v9838_data * v9812_data)) + (v9979_data * v10000_data)) + (v10120_data * v10141_data)) + (v10261_data * v10282_data)) + (v10402_data * v10423_data)) + (v10543_data * v10564_data)) + (v10684_data * v10705_data)) + (v10825_data * v10846_data);
          double v10851_data = s3[197];
          double v10854_p = (((((((v9679_p + (v9838_data * v9817_data)) + (v9979_data * v10005_data)) + (v10120_data * v10146_data)) + (v10261_data * v10287_data)) + (v10402_data * v10428_data)) + (v10543_data * v10569_data)) + (v10684_data * v10710_data)) + (v10825_data * v10851_data);
          double v10856_data = s3[232];
          double v10859_p = (((((((v9684_p + (v9838_data * v9822_data)) + (v9979_data * v10010_data)) + (v10120_data * v10151_data)) + (v10261_data * v10292_data)) + (v10402_data * v10433_data)) + (v10543_data * v10574_data)) + (v10684_data * v10715_data)) + (v10825_data * v10856_data);
          double v10861_data = s3[267];
          double v10864_p = (((((((v9689_p + (v9838_data * v9827_data)) + (v9979_data * v10015_data)) + (v10120_data * v10156_data)) + (v10261_data * v10297_data)) + (v10402_data * v10438_data)) + (v10543_data * v10579_data)) + (v10684_data * v10720_data)) + (v10825_data * v10861_data);
          double v10866_data = s3[302];
          double v10869_p = (((((((v9694_p + (v9838_data * v9832_data)) + (v9979_data * v10020_data)) + (v10120_data * v10161_data)) + (v10261_data * v10302_data)) + (v10402_data * v10443_data)) + (v10543_data * v10584_data)) + (v10684_data * v10725_data)) + (v10825_data * v10866_data);
          double v10872_data = glb_m5[(v1393_lead + 1030)];
          double v10876_p = (((((((v9701_p + (v9885_data * v9792_data)) + (v10026_data * v9980_data)) + (v10167_data * v10121_data)) + (v10308_data * v10262_data)) + (v10449_data * v10403_data)) + (v10590_data * v10544_data)) + (v10731_data * v10685_data)) + (v10872_data * v10826_data);
          double v10881_p = (((((((v9706_p + (v9885_data * v9797_data)) + (v10026_data * v9985_data)) + (v10167_data * v10126_data)) + (v10308_data * v10267_data)) + (v10449_data * v10408_data)) + (v10590_data * v10549_data)) + (v10731_data * v10690_data)) + (v10872_data * v10831_data);
          double v10886_p = (((((((v9711_p + (v9885_data * v9802_data)) + (v10026_data * v9990_data)) + (v10167_data * v10131_data)) + (v10308_data * v10272_data)) + (v10449_data * v10413_data)) + (v10590_data * v10554_data)) + (v10731_data * v10695_data)) + (v10872_data * v10836_data);
          double v10891_p = (((((((v9716_p + (v9885_data * v9807_data)) + (v10026_data * v9995_data)) + (v10167_data * v10136_data)) + (v10308_data * v10277_data)) + (v10449_data * v10418_data)) + (v10590_data * v10559_data)) + (v10731_data * v10700_data)) + (v10872_data * v10841_data);
          double v10896_p = (((((((v9721_p + (v9885_data * v9812_data)) + (v10026_data * v10000_data)) + (v10167_data * v10141_data)) + (v10308_data * v10282_data)) + (v10449_data * v10423_data)) + (v10590_data * v10564_data)) + (v10731_data * v10705_data)) + (v10872_data * v10846_data);
          double v10901_p = (((((((v9726_p + (v9885_data * v9817_data)) + (v10026_data * v10005_data)) + (v10167_data * v10146_data)) + (v10308_data * v10287_data)) + (v10449_data * v10428_data)) + (v10590_data * v10569_data)) + (v10731_data * v10710_data)) + (v10872_data * v10851_data);
          double v10906_p = (((((((v9731_p + (v9885_data * v9822_data)) + (v10026_data * v10010_data)) + (v10167_data * v10151_data)) + (v10308_data * v10292_data)) + (v10449_data * v10433_data)) + (v10590_data * v10574_data)) + (v10731_data * v10715_data)) + (v10872_data * v10856_data);
          double v10911_p = (((((((v9736_p + (v9885_data * v9827_data)) + (v10026_data * v10015_data)) + (v10167_data * v10156_data)) + (v10308_data * v10297_data)) + (v10449_data * v10438_data)) + (v10590_data * v10579_data)) + (v10731_data * v10720_data)) + (v10872_data * v10861_data);
          double v10916_p = (((((((v9741_p + (v9885_data * v9832_data)) + (v10026_data * v10020_data)) + (v10167_data * v10161_data)) + (v10308_data * v10302_data)) + (v10449_data * v10443_data)) + (v10590_data * v10584_data)) + (v10731_data * v10725_data)) + (v10872_data * v10866_data);
          double v10919_data = v12016_g ? (glb_m5[(v1442_lead + 1030)]) : (0.0);
          double v10923_p = (((((((v9748_p + (v9932_data * v9792_data)) + (v10073_data * v9980_data)) + (v10214_data * v10121_data)) + (v10355_data * v10262_data)) + (v10496_data * v10403_data)) + (v10637_data * v10544_data)) + (v10778_data * v10685_data)) + (v10919_data * v10826_data);
          double v10928_p = (((((((v9753_p + (v9932_data * v9797_data)) + (v10073_data * v9985_data)) + (v10214_data * v10126_data)) + (v10355_data * v10267_data)) + (v10496_data * v10408_data)) + (v10637_data * v10549_data)) + (v10778_data * v10690_data)) + (v10919_data * v10831_data);
          double v10933_p = (((((((v9758_p + (v9932_data * v9802_data)) + (v10073_data * v9990_data)) + (v10214_data * v10131_data)) + (v10355_data * v10272_data)) + (v10496_data * v10413_data)) + (v10637_data * v10554_data)) + (v10778_data * v10695_data)) + (v10919_data * v10836_data);
          double v10938_p = (((((((v9763_p + (v9932_data * v9807_data)) + (v10073_data * v9995_data)) + (v10214_data * v10136_data)) + (v10355_data * v10277_data)) + (v10496_data * v10418_data)) + (v10637_data * v10559_data)) + (v10778_data * v10700_data)) + (v10919_data * v10841_data);
          double v10943_p = (((((((v9768_p + (v9932_data * v9812_data)) + (v10073_data * v10000_data)) + (v10214_data * v10141_data)) + (v10355_data * v10282_data)) + (v10496_data * v10423_data)) + (v10637_data * v10564_data)) + (v10778_data * v10705_data)) + (v10919_data * v10846_data);
          double v10948_p = (((((((v9773_p + (v9932_data * v9817_data)) + (v10073_data * v10005_data)) + (v10214_data * v10146_data)) + (v10355_data * v10287_data)) + (v10496_data * v10428_data)) + (v10637_data * v10569_data)) + (v10778_data * v10710_data)) + (v10919_data * v10851_data);
          double v10953_p = (((((((v9778_p + (v9932_data * v9822_data)) + (v10073_data * v10010_data)) + (v10214_data * v10151_data)) + (v10355_data * v10292_data)) + (v10496_data * v10433_data)) + (v10637_data * v10574_data)) + (v10778_data * v10715_data)) + (v10919_data * v10856_data);
          double v10958_p = (((((((v9783_p + (v9932_data * v9827_data)) + (v10073_data * v10015_data)) + (v10214_data * v10156_data)) + (v10355_data * v10297_data)) + (v10496_data * v10438_data)) + (v10637_data * v10579_data)) + (v10778_data * v10720_data)) + (v10919_data * v10861_data);
          double v10963_p = (((((((v9788_p + (v9932_data * v9832_data)) + (v10073_data * v10020_data)) + (v10214_data * v10161_data)) + (v10355_data * v10302_data)) + (v10496_data * v10443_data)) + (v10637_data * v10584_data)) + (v10778_data * v10725_data)) + (v10919_data * v10866_data);
          double v10966_data = v2809_g ? (glb_m5[(v1344_lead + 1061)]) : (0.0);
          double v10967_data = s3[23];
          ir7[1] = (((((((((v9654_p + (v9838_data * v9792_data)) + (v9979_data * v9980_data)) + (v10120_data * v10121_data)) + (v10261_data * v10262_data)) + (v10402_data * v10403_data)) + (v10543_data * v10544_data)) + (v10684_data * v10685_data)) + (v10825_data * v10826_data)) + (v10966_data * v10967_data));
          double v10972_data = s3[58];
          ir7[5] = (v10834_p + (v10966_data * v10972_data));
          double v10977_data = s3[93];
          ir7[9] = (v10839_p + (v10966_data * v10977_data));
          double v10982_data = s3[128];
          ir7[13] = (v10844_p + (v10966_data * v10982_data));
          double v10987_data = s3[163];
          ir7[17] = (v10849_p + (v10966_data * v10987_data));
          double v10992_data = s3[198];
          ir7[21] = (v10854_p + (v10966_data * v10992_data));
          double v10997_data = s3[233];
          ir7[25] = (v10859_p + (v10966_data * v10997_data));
          double v11002_data = s3[268];
          ir7[29] = (v10864_p + (v10966_data * v11002_data));
          double v11007_data = s3[303];
          ir7[33] = (v10869_p + (v10966_data * v11007_data));
          double v11013_data = glb_m5[(v1393_lead + 1061)];
          double v11060_data = v12016_g ? (glb_m5[(v1442_lead + 1061)]) : (0.0);
          double v11107_data = glb_m5[(v1393_lead + 1084)];
          double v11108_data = s3[24];
          double v11113_data = s3[59];
          double v11118_data = s3[94];
          double v11123_data = s3[129];
          double v11128_data = s3[164];
          double v11133_data = s3[199];
          double v11138_data = s3[234];
          double v11143_data = s3[269];
          double v11148_data = s3[304];
          double v11154_data = v12016_g ? (glb_m5[(v1442_lead + 1084)]) : (0.0);
          double v11201_data = glb_m5[(v1393_lead + 1107)];
          double v11202_data = s3[25];
          double v11207_data = s3[60];
          double v11212_data = s3[95];
          double v11217_data = s3[130];
          double v11222_data = s3[165];
          double v11227_data = s3[200];
          double v11232_data = s3[235];
          double v11237_data = s3[270];
          double v11242_data = s3[305];
          double v11248_data = v12016_g ? (glb_m5[(v1442_lead + 1107)]) : (0.0);
          double v11295_data = glb_m5[(v1393_lead + 1130)];
          double v11296_data = s3[26];
          double v11301_data = s3[61];
          double v11306_data = s3[96];
          double v11311_data = s3[131];
          double v11316_data = s3[166];
          double v11321_data = s3[201];
          double v11326_data = s3[236];
          double v11331_data = s3[271];
          double v11336_data = s3[306];
          double v11342_data = v12016_g ? (glb_m5[(v1442_lead + 1130)]) : (0.0);
          double v11389_data = glb_m5[(v1393_lead + 1153)];
          double v11390_data = s3[27];
          double v11395_data = s3[62];
          double v11400_data = s3[97];
          double v11405_data = s3[132];
          double v11410_data = s3[167];
          double v11415_data = s3[202];
          double v11420_data = s3[237];
          double v11425_data = s3[272];
          double v11430_data = s3[307];
          double v11436_data = v12016_g ? (glb_m5[(v1442_lead + 1153)]) : (0.0);
          double v11483_data = v2809_g ? (glb_m5[(v1393_lead + 1168)]) : (0.0);
          double v11484_data = s3[28];
          double v11489_data = s3[63];
          double v11494_data = s3[98];
          double v11499_data = s3[133];
          double v11504_data = s3[168];
          double v11509_data = s3[203];
          double v11514_data = s3[238];
          double v11519_data = s3[273];
          double v11524_data = s3[308];
          double v11530_data = v12016_g ? (glb_m5[(v1442_lead + 1168)]) : (0.0);
          double v11559_p = (((((v10948_p + (v11060_data * v10992_data)) + (v11154_data * v11133_data)) + (v11248_data * v11227_data)) + (v11342_data * v11321_data)) + (v11436_data * v11415_data)) + (v11530_data * v11509_data);
          double v11564_p = (((((v10953_p + (v11060_data * v10997_data)) + (v11154_data * v11138_data)) + (v11248_data * v11232_data)) + (v11342_data * v11326_data)) + (v11436_data * v11420_data)) + (v11530_data * v11514_data);
          double v11569_p = (((((v10958_p + (v11060_data * v11002_data)) + (v11154_data * v11143_data)) + (v11248_data * v11237_data)) + (v11342_data * v11331_data)) + (v11436_data * v11425_data)) + (v11530_data * v11519_data);
          double v11574_p = (((((v10963_p + (v11060_data * v11007_data)) + (v11154_data * v11148_data)) + (v11248_data * v11242_data)) + (v11342_data * v11336_data)) + (v11436_data * v11430_data)) + (v11530_data * v11524_data);
          double v11577_data = v2809_g ? (glb_m5[(v1393_lead + 1183)]) : (0.0);
          double v11578_data = s3[29];
          double v11581_p = ((((((v10876_p + (v11013_data * v10967_data)) + (v11107_data * v11108_data)) + (v11201_data * v11202_data)) + (v11295_data * v11296_data)) + (v11389_data * v11390_data)) + (v11483_data * v11484_data)) + (v11577_data * v11578_data);
          double v11583_data = s3[64];
          double v11586_p = ((((((v10881_p + (v11013_data * v10972_data)) + (v11107_data * v11113_data)) + (v11201_data * v11207_data)) + (v11295_data * v11301_data)) + (v11389_data * v11395_data)) + (v11483_data * v11489_data)) + (v11577_data * v11583_data);
          double v11588_data = s3[99];
          double v11591_p = ((((((v10886_p + (v11013_data * v10977_data)) + (v11107_data * v11118_data)) + (v11201_data * v11212_data)) + (v11295_data * v11306_data)) + (v11389_data * v11400_data)) + (v11483_data * v11494_data)) + (v11577_data * v11588_data);
          double v11593_data = s3[134];
          double v11596_p = ((((((v10891_p + (v11013_data * v10982_data)) + (v11107_data * v11123_data)) + (v11201_data * v11217_data)) + (v11295_data * v11311_data)) + (v11389_data * v11405_data)) + (v11483_data * v11499_data)) + (v11577_data * v11593_data);
          double v11598_data = s3[169];
          double v11601_p = ((((((v10896_p + (v11013_data * v10987_data)) + (v11107_data * v11128_data)) + (v11201_data * v11222_data)) + (v11295_data * v11316_data)) + (v11389_data * v11410_data)) + (v11483_data * v11504_data)) + (v11577_data * v11598_data);
          double v11603_data = s3[204];
          double v11606_p = ((((((v10901_p + (v11013_data * v10992_data)) + (v11107_data * v11133_data)) + (v11201_data * v11227_data)) + (v11295_data * v11321_data)) + (v11389_data * v11415_data)) + (v11483_data * v11509_data)) + (v11577_data * v11603_data);
          double v11608_data = s3[239];
          double v11611_p = ((((((v10906_p + (v11013_data * v10997_data)) + (v11107_data * v11138_data)) + (v11201_data * v11232_data)) + (v11295_data * v11326_data)) + (v11389_data * v11420_data)) + (v11483_data * v11514_data)) + (v11577_data * v11608_data);
          double v11613_data = s3[274];
          double v11616_p = ((((((v10911_p + (v11013_data * v11002_data)) + (v11107_data * v11143_data)) + (v11201_data * v11237_data)) + (v11295_data * v11331_data)) + (v11389_data * v11425_data)) + (v11483_data * v11519_data)) + (v11577_data * v11613_data);
          double v11618_data = s3[309];
          double v11621_p = ((((((v10916_p + (v11013_data * v11007_data)) + (v11107_data * v11148_data)) + (v11201_data * v11242_data)) + (v11295_data * v11336_data)) + (v11389_data * v11430_data)) + (v11483_data * v11524_data)) + (v11577_data * v11618_data);
          double v11624_data = v12016_g ? (glb_m5[(v1442_lead + 1183)]) : (0.0);
          double v11628_p = ((((((v10923_p + (v11060_data * v10967_data)) + (v11154_data * v11108_data)) + (v11248_data * v11202_data)) + (v11342_data * v11296_data)) + (v11436_data * v11390_data)) + (v11530_data * v11484_data)) + (v11624_data * v11578_data);
          double v11633_p = ((((((v10928_p + (v11060_data * v10972_data)) + (v11154_data * v11113_data)) + (v11248_data * v11207_data)) + (v11342_data * v11301_data)) + (v11436_data * v11395_data)) + (v11530_data * v11489_data)) + (v11624_data * v11583_data);
          double v11638_p = ((((((v10933_p + (v11060_data * v10977_data)) + (v11154_data * v11118_data)) + (v11248_data * v11212_data)) + (v11342_data * v11306_data)) + (v11436_data * v11400_data)) + (v11530_data * v11494_data)) + (v11624_data * v11588_data);
          double v11643_p = ((((((v10938_p + (v11060_data * v10982_data)) + (v11154_data * v11123_data)) + (v11248_data * v11217_data)) + (v11342_data * v11311_data)) + (v11436_data * v11405_data)) + (v11530_data * v11499_data)) + (v11624_data * v11593_data);
          double v11648_p = ((((((v10943_p + (v11060_data * v10987_data)) + (v11154_data * v11128_data)) + (v11248_data * v11222_data)) + (v11342_data * v11316_data)) + (v11436_data * v11410_data)) + (v11530_data * v11504_data)) + (v11624_data * v11598_data);
          double v11651_p = v11624_data * v11603_data;
          if (v12016_g) {
            #pragma unroll
            for (int32_t v12017_i1 = 0; v12017_i1 < 9; ++v12017_i1) {
              double v12022_data = glb_m2[(v1442_lead + (v12017_i1 * 56))];
              r6[(3 + (v12017_i1 * 4))] = v12022_data;
            }
          }
          double v11671_data = v2809_g ? (glb_m5[(v1393_lead + 1198)]) : (0.0);
          double v11672_data = s3[30];
          double v11677_data = s3[65];
          double v11680_p = v11586_p + (v11671_data * v11677_data);
          double v11682_data = s3[100];
          double v11685_p = v11591_p + (v11671_data * v11682_data);
          double v11687_data = s3[135];
          double v11690_p = v11596_p + (v11671_data * v11687_data);
          double v11692_data = s3[170];
          double v11695_p = v11601_p + (v11671_data * v11692_data);
          double v11697_data = s3[205];
          double v11700_p = v11606_p + (v11671_data * v11697_data);
          double v11702_data = s3[240];
          double v11705_p = v11611_p + (v11671_data * v11702_data);
          double v11707_data = s3[275];
          double v11710_p = v11616_p + (v11671_data * v11707_data);
          double v11712_data = s3[310];
          double v11715_p = v11621_p + (v11671_data * v11712_data);
          double v11718_data = v12016_g ? (glb_m5[(v1442_lead + 1198)]) : (0.0);
          double v11722_p = v11628_p + (v11718_data * v11672_data);
          double v11727_p = v11633_p + (v11718_data * v11677_data);
          double v11732_p = v11638_p + (v11718_data * v11682_data);
          double v11737_p = v11643_p + (v11718_data * v11687_data);
          double v11742_p = v11648_p + (v11718_data * v11692_data);
          double v11747_p = (v11559_p + v11651_p) + (v11718_data * v11697_data);
          double v11752_p = (v11564_p + (v11624_data * v11608_data)) + (v11718_data * v11702_data);
          double v11757_p = (v11569_p + (v11624_data * v11613_data)) + (v11718_data * v11707_data);
          double v11762_p = (v11574_p + (v11624_data * v11618_data)) + (v11718_data * v11712_data);
          double v11765_data = v2809_g ? (glb_m5[(v1393_lead + 1213)]) : (0.0);
          double v11766_data = s3[31];
          ir7[2] = ((v11581_p + (v11671_data * v11672_data)) + (v11765_data * v11766_data));
          double v11771_data = s3[66];
          ir7[6] = (v11680_p + (v11765_data * v11771_data));
          double v11776_data = s3[101];
          ir7[10] = (v11685_p + (v11765_data * v11776_data));
          double v11781_data = s3[136];
          ir7[14] = (v11690_p + (v11765_data * v11781_data));
          double v11786_data = s3[171];
          ir7[18] = (v11695_p + (v11765_data * v11786_data));
          double v11791_data = s3[206];
          ir7[22] = (v11700_p + (v11765_data * v11791_data));
          double v11796_data = s3[241];
          ir7[26] = (v11705_p + (v11765_data * v11796_data));
          double v11801_data = s3[276];
          ir7[30] = (v11710_p + (v11765_data * v11801_data));
          double v11806_data = s3[311];
          ir7[34] = (v11715_p + (v11765_data * v11806_data));
          double v11812_data = v12016_g ? (glb_m5[(v1442_lead + 1213)]) : (0.0);
          double v11859_data = v12016_g ? (glb_m5[(v1442_lead + 1220)]) : (0.0);
          double v11860_data = s3[32];
          double v11865_data = s3[67];
          double v11870_data = s3[102];
          double v11875_data = s3[137];
          double v11880_data = s3[172];
          double v11885_data = s3[207];
          double v11890_data = s3[242];
          double v11895_data = s3[277];
          double v11900_data = s3[312];
          double v11906_data = v12016_g ? (glb_m5[(v1442_lead + 1227)]) : (0.0);
          double v11907_data = s3[33];
          double v11912_data = s3[68];
          double v11915_p = ((v11727_p + (v11812_data * v11771_data)) + (v11859_data * v11865_data)) + (v11906_data * v11912_data);
          double v11917_data = s3[103];
          double v11920_p = ((v11732_p + (v11812_data * v11776_data)) + (v11859_data * v11870_data)) + (v11906_data * v11917_data);
          double v11922_data = s3[138];
          double v11925_p = ((v11737_p + (v11812_data * v11781_data)) + (v11859_data * v11875_data)) + (v11906_data * v11922_data);
          double v11927_data = s3[173];
          double v11930_p = ((v11742_p + (v11812_data * v11786_data)) + (v11859_data * v11880_data)) + (v11906_data * v11927_data);
          double v11932_data = s3[208];
          double v11935_p = ((v11747_p + (v11812_data * v11791_data)) + (v11859_data * v11885_data)) + (v11906_data * v11932_data);
          double v11937_data = s3[243];
          double v11940_p = ((v11752_p + (v11812_data * v11796_data)) + (v11859_data * v11890_data)) + (v11906_data * v11937_data);
          double v11942_data = s3[278];
          double v11945_p = ((v11757_p + (v11812_data * v11801_data)) + (v11859_data * v11895_data)) + (v11906_data * v11942_data);
          double v11947_data = s3[313];
          double v11950_p = ((v11762_p + (v11812_data * v11806_data)) + (v11859_data * v11900_data)) + (v11906_data * v11947_data);
          double v11953_data = v12016_g ? (glb_m5[(v1442_lead + 1234)]) : (0.0);
          double v11954_data = s3[34];
          ir7[3] = ((((v11722_p + (v11812_data * v11766_data)) + (v11859_data * v11860_data)) + (v11906_data * v11907_data)) + (v11953_data * v11954_data));
          double v11959_data = s3[69];
          ir7[7] = (v11915_p + (v11953_data * v11959_data));
          double v11964_data = s3[104];
          ir7[11] = (v11920_p + (v11953_data * v11964_data));
          double v11969_data = s3[139];
          ir7[15] = (v11925_p + (v11953_data * v11969_data));
          double v11974_data = s3[174];
          ir7[19] = (v11930_p + (v11953_data * v11974_data));
          double v11979_data = s3[209];
          ir7[23] = (v11935_p + (v11953_data * v11979_data));
          double v11984_data = s3[244];
          ir7[27] = (v11940_p + (v11953_data * v11984_data));
          double v11989_data = s3[279];
          ir7[31] = (v11945_p + (v11953_data * v11989_data));
          double v11994_data = s3[314];
          ir7[35] = (v11950_p + (v11953_data * v11994_data));
          double r8[27]{};
          // s4 = load{g>s}(glb_m6[0, 1])
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v12087_i = 0; v12087_i < 5; ++v12087_i) {
            int32_t v12089_a = threadIdx.x + (v12087_i * 16);
            __pipeline_memcpy_async(&s4[v12089_a], &glb_m6[v12089_a], 8);
          }
          if (v19038_g) {
            __pipeline_memcpy_async(&s4[0 + 0 + 1 * threadIdx.x + 80], &glb_m6[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
          __pipeline_commit();
          // r8 = load{g>r}(glb_m0);
          // r7 = ir7 + r6
          if (v5629_g) {
            #pragma unroll
            for (int32_t v12025_n1 = 0; v12025_n1 < 9; ++v12025_n1) {
              int32_t v12026_a = v12025_n1 * 4;
              double v12028_data = ir7[v12026_a];
              double v12029_data = r3[v12026_a];
              r7[v12026_a] = (v12029_data + v12028_data);
            }
          }
          #pragma unroll
          for (int32_t v12031_n0 = 1; v12031_n0 < 3; ++v12031_n0) {
            #pragma unroll
            for (int32_t v12032_n1 = 0; v12032_n1 < 9; ++v12032_n1) {
              int32_t v12033_a = v12032_n1 * 4;
              int32_t v12034_a = v12031_n0 + v12033_a;
              double v12035_data = ir7[v12034_a];
              double v12036_data = r3[(v12031_n0 + v12033_a)];
              r7[v12034_a] = (v12036_data + v12035_data);
            }
          }
          if (v12016_g) {
            #pragma unroll
            for (int32_t v12038_n1 = 0; v12038_n1 < 9; ++v12038_n1) {
              int32_t v12040_a = 3 + (v12038_n1 * 4);
              double v12041_data = ir7[v12040_a];
              double v12042_data = r6[v12040_a];
              r7[v12040_a] = (v12042_data + v12041_data);
            }
          }
          // glb_m2 = store{r>g}(r7);
          if (v12016_g) {
            #pragma unroll
            for (int32_t v12061_i1 = 0; v12061_i1 < 9; ++v12061_i1) {
              double v12064_data = r7[(3 + (v12061_i1 * 4))];
              glb_m2[(v1442_lead + (v12061_i1 * 56))] = v12064_data;
            }
          }
          double r10[36]{};
          // r10 = load{g>r}(glb_m2);
          bool v18902_g = v18980_lead < 8;
          // wait(s4 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          double r9[27]{};
          // r9 = +(r8 * s4) + None
          // [(0, 35), (0, 9)] [(0, 9)]
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v12094_data = s4[0];
          double v12099_data = s4[9];
          double v12104_data = s4[18];
          double v12109_data = s4[27];
          double v12114_data = s4[36];
          double v12119_data = s4[45];
          double v12124_data = s4[54];
          double v12129_data = s4[63];
          double v12134_data = s4[72];
          double v12229_data = s4[1];
          double v12234_data = s4[10];
          double v12239_data = s4[19];
          double v12244_data = s4[28];
          double v12249_data = s4[37];
          double v12254_data = s4[46];
          double v12259_data = s4[55];
          double v12264_data = s4[64];
          double v12269_data = s4[73];
          double v12364_data = s4[2];
          double v12369_data = s4[11];
          double v12374_data = s4[20];
          double v12379_data = s4[29];
          double v12384_data = s4[38];
          double v12389_data = s4[47];
          double v12394_data = s4[56];
          double v12399_data = s4[65];
          double v12404_data = s4[74];
          double v12499_data = s4[3];
          double v12504_data = s4[12];
          double v12509_data = s4[21];
          double v12514_data = s4[30];
          double v12519_data = s4[39];
          double v12524_data = s4[48];
          double v12529_data = s4[57];
          double v12534_data = s4[66];
          double v12539_data = s4[75];
          double v12634_data = s4[4];
          double v12639_data = s4[13];
          double v12644_data = s4[22];
          double v12649_data = s4[31];
          double v12654_data = s4[40];
          double v12659_data = s4[49];
          double v12664_data = s4[58];
          double v12669_data = s4[67];
          double v12674_data = s4[76];
          double v12769_data = s4[5];
          double v12774_data = s4[14];
          double v12779_data = s4[23];
          double v12784_data = s4[32];
          double v12789_data = s4[41];
          double v12794_data = s4[50];
          double v12799_data = s4[59];
          double v12804_data = s4[68];
          double v12809_data = s4[77];
          double v12904_data = s4[6];
          double v12909_data = s4[15];
          double v12914_data = s4[24];
          double v12919_data = s4[33];
          double v12924_data = s4[42];
          double v12929_data = s4[51];
          double v12934_data = s4[60];
          double v12939_data = s4[69];
          double v12944_data = s4[78];
          double v13039_data = s4[7];
          double v13044_data = s4[16];
          double v13047_p = (((((((v58_data * v12099_data) + (v193_data * v12234_data)) + (v328_data * v12369_data)) + (v463_data * v12504_data)) + (v598_data * v12639_data)) + (v733_data * v12774_data)) + (v868_data * v12909_data)) + (v1003_data * v13044_data);
          double v13049_data = s4[25];
          double v13052_p = (((((((v58_data * v12104_data) + (v193_data * v12239_data)) + (v328_data * v12374_data)) + (v463_data * v12509_data)) + (v598_data * v12644_data)) + (v733_data * v12779_data)) + (v868_data * v12914_data)) + (v1003_data * v13049_data);
          double v13054_data = s4[34];
          double v13057_p = (((((((v58_data * v12109_data) + (v193_data * v12244_data)) + (v328_data * v12379_data)) + (v463_data * v12514_data)) + (v598_data * v12649_data)) + (v733_data * v12784_data)) + (v868_data * v12919_data)) + (v1003_data * v13054_data);
          double v13059_data = s4[43];
          double v13062_p = (((((((v58_data * v12114_data) + (v193_data * v12249_data)) + (v328_data * v12384_data)) + (v463_data * v12519_data)) + (v598_data * v12654_data)) + (v733_data * v12789_data)) + (v868_data * v12924_data)) + (v1003_data * v13059_data);
          double v13064_data = s4[52];
          double v13067_p = (((((((v58_data * v12119_data) + (v193_data * v12254_data)) + (v328_data * v12389_data)) + (v463_data * v12524_data)) + (v598_data * v12659_data)) + (v733_data * v12794_data)) + (v868_data * v12929_data)) + (v1003_data * v13064_data);
          double v13069_data = s4[61];
          double v13072_p = (((((((v58_data * v12124_data) + (v193_data * v12259_data)) + (v328_data * v12394_data)) + (v463_data * v12529_data)) + (v598_data * v12664_data)) + (v733_data * v12799_data)) + (v868_data * v12934_data)) + (v1003_data * v13069_data);
          double v13074_data = s4[70];
          double v13077_p = (((((((v58_data * v12129_data) + (v193_data * v12264_data)) + (v328_data * v12399_data)) + (v463_data * v12534_data)) + (v598_data * v12669_data)) + (v733_data * v12804_data)) + (v868_data * v12939_data)) + (v1003_data * v13074_data);
          double v13079_data = s4[79];
          double v13082_p = (((((((v58_data * v12134_data) + (v193_data * v12269_data)) + (v328_data * v12404_data)) + (v463_data * v12539_data)) + (v598_data * v12674_data)) + (v733_data * v12809_data)) + (v868_data * v12944_data)) + (v1003_data * v13079_data);
          double v13087_p = (((((((v103_data * v12094_data) + (v238_data * v12229_data)) + (v373_data * v12364_data)) + (v508_data * v12499_data)) + (v643_data * v12634_data)) + (v778_data * v12769_data)) + (v913_data * v12904_data)) + (v1048_data * v13039_data);
          double v13092_p = (((((((v103_data * v12099_data) + (v238_data * v12234_data)) + (v373_data * v12369_data)) + (v508_data * v12504_data)) + (v643_data * v12639_data)) + (v778_data * v12774_data)) + (v913_data * v12909_data)) + (v1048_data * v13044_data);
          double v13097_p = (((((((v103_data * v12104_data) + (v238_data * v12239_data)) + (v373_data * v12374_data)) + (v508_data * v12509_data)) + (v643_data * v12644_data)) + (v778_data * v12779_data)) + (v913_data * v12914_data)) + (v1048_data * v13049_data);
          double v13102_p = (((((((v103_data * v12109_data) + (v238_data * v12244_data)) + (v373_data * v12379_data)) + (v508_data * v12514_data)) + (v643_data * v12649_data)) + (v778_data * v12784_data)) + (v913_data * v12919_data)) + (v1048_data * v13054_data);
          double v13107_p = (((((((v103_data * v12114_data) + (v238_data * v12249_data)) + (v373_data * v12384_data)) + (v508_data * v12519_data)) + (v643_data * v12654_data)) + (v778_data * v12789_data)) + (v913_data * v12924_data)) + (v1048_data * v13059_data);
          double v13112_p = (((((((v103_data * v12119_data) + (v238_data * v12254_data)) + (v373_data * v12389_data)) + (v508_data * v12524_data)) + (v643_data * v12659_data)) + (v778_data * v12794_data)) + (v913_data * v12929_data)) + (v1048_data * v13064_data);
          double v13117_p = (((((((v103_data * v12124_data) + (v238_data * v12259_data)) + (v373_data * v12394_data)) + (v508_data * v12529_data)) + (v643_data * v12664_data)) + (v778_data * v12799_data)) + (v913_data * v12934_data)) + (v1048_data * v13069_data);
          double v13122_p = (((((((v103_data * v12129_data) + (v238_data * v12264_data)) + (v373_data * v12399_data)) + (v508_data * v12534_data)) + (v643_data * v12669_data)) + (v778_data * v12804_data)) + (v913_data * v12939_data)) + (v1048_data * v13074_data);
          double v13127_p = (((((((v103_data * v12134_data) + (v238_data * v12269_data)) + (v373_data * v12404_data)) + (v508_data * v12539_data)) + (v643_data * v12674_data)) + (v778_data * v12809_data)) + (v913_data * v12944_data)) + (v1048_data * v13079_data);
          double v13132_p = (((((((v148_data * v12094_data) + (v283_data * v12229_data)) + (v418_data * v12364_data)) + (v553_data * v12499_data)) + (v688_data * v12634_data)) + (v823_data * v12769_data)) + (v958_data * v12904_data)) + (v1093_data * v13039_data);
          double v13137_p = (((((((v148_data * v12099_data) + (v283_data * v12234_data)) + (v418_data * v12369_data)) + (v553_data * v12504_data)) + (v688_data * v12639_data)) + (v823_data * v12774_data)) + (v958_data * v12909_data)) + (v1093_data * v13044_data);
          double v13142_p = (((((((v148_data * v12104_data) + (v283_data * v12239_data)) + (v418_data * v12374_data)) + (v553_data * v12509_data)) + (v688_data * v12644_data)) + (v823_data * v12779_data)) + (v958_data * v12914_data)) + (v1093_data * v13049_data);
          double v13147_p = (((((((v148_data * v12109_data) + (v283_data * v12244_data)) + (v418_data * v12379_data)) + (v553_data * v12514_data)) + (v688_data * v12649_data)) + (v823_data * v12784_data)) + (v958_data * v12919_data)) + (v1093_data * v13054_data);
          double v13152_p = (((((((v148_data * v12114_data) + (v283_data * v12249_data)) + (v418_data * v12384_data)) + (v553_data * v12519_data)) + (v688_data * v12654_data)) + (v823_data * v12789_data)) + (v958_data * v12924_data)) + (v1093_data * v13059_data);
          double v13157_p = (((((((v148_data * v12119_data) + (v283_data * v12254_data)) + (v418_data * v12389_data)) + (v553_data * v12524_data)) + (v688_data * v12659_data)) + (v823_data * v12794_data)) + (v958_data * v12929_data)) + (v1093_data * v13064_data);
          double v13162_p = (((((((v148_data * v12124_data) + (v283_data * v12259_data)) + (v418_data * v12394_data)) + (v553_data * v12529_data)) + (v688_data * v12664_data)) + (v823_data * v12799_data)) + (v958_data * v12934_data)) + (v1093_data * v13069_data);
          double v13167_p = (((((((v148_data * v12129_data) + (v283_data * v12264_data)) + (v418_data * v12399_data)) + (v553_data * v12534_data)) + (v688_data * v12669_data)) + (v823_data * v12804_data)) + (v958_data * v12939_data)) + (v1093_data * v13074_data);
          double v13172_p = (((((((v148_data * v12134_data) + (v283_data * v12269_data)) + (v418_data * v12404_data)) + (v553_data * v12539_data)) + (v688_data * v12674_data)) + (v823_data * v12809_data)) + (v958_data * v12944_data)) + (v1093_data * v13079_data);
          double v13174_data = s4[8];
          r9[0] = (((((((((v58_data * v12094_data) + (v193_data * v12229_data)) + (v328_data * v12364_data)) + (v463_data * v12499_data)) + (v598_data * v12634_data)) + (v733_data * v12769_data)) + (v868_data * v12904_data)) + (v1003_data * v13039_data)) + (v1138_data * v13174_data));
          double v13179_data = s4[17];
          r9[3] = (v13047_p + (v1138_data * v13179_data));
          double v13184_data = s4[26];
          r9[6] = (v13052_p + (v1138_data * v13184_data));
          double v13189_data = s4[35];
          r9[9] = (v13057_p + (v1138_data * v13189_data));
          double v13194_data = s4[44];
          r9[12] = (v13062_p + (v1138_data * v13194_data));
          double v13199_data = s4[53];
          r9[15] = (v13067_p + (v1138_data * v13199_data));
          double v13204_data = s4[62];
          r9[18] = (v13072_p + (v1138_data * v13204_data));
          double v13209_data = s4[71];
          r9[21] = (v13077_p + (v1138_data * v13209_data));
          double v13214_data = s4[80];
          r9[24] = (v13082_p + (v1138_data * v13214_data));
          r9[1] = (v13087_p + (v1183_data * v13174_data));
          r9[4] = (v13092_p + (v1183_data * v13179_data));
          r9[7] = (v13097_p + (v1183_data * v13184_data));
          r9[10] = (v13102_p + (v1183_data * v13189_data));
          r9[13] = (v13107_p + (v1183_data * v13194_data));
          r9[16] = (v13112_p + (v1183_data * v13199_data));
          r9[19] = (v13117_p + (v1183_data * v13204_data));
          r9[22] = (v13122_p + (v1183_data * v13209_data));
          r9[25] = (v13127_p + (v1183_data * v13214_data));
          r9[2] = (v13132_p + (v1228_data * v13174_data));
          r9[5] = (v13137_p + (v1228_data * v13179_data));
          r9[8] = (v13142_p + (v1228_data * v13184_data));
          r9[11] = (v13147_p + (v1228_data * v13189_data));
          r9[14] = (v13152_p + (v1228_data * v13194_data));
          r9[17] = (v13157_p + (v1228_data * v13199_data));
          r9[20] = (v13162_p + (v1228_data * v13204_data));
          r9[23] = (v13167_p + (v1228_data * v13209_data));
          r9[26] = (v13172_p + (v1228_data * v13214_data));
          // r9 = 0 past (0, 35)
          // s5 = store{r>s}(localShrMem0, r9);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v13309_i0 = 0; v13309_i0 < 2; ++v13309_i0) {
            int32_t v13315_lead = v18980_lead + (v13309_i0 * 16);
            #pragma unroll
            for (int32_t v13310_i1 = 0; v13310_i1 < 9; ++v13310_i1) {
              double v13313_data = r9[(v13309_i0 + (v13310_i1 * 3))];
              s5[(v13315_lead + (v13310_i1 * 35))] = v13313_data;
            }
          }
          if (v18981_g) {
            #pragma unroll
            for (int32_t v13318_i1 = 0; v13318_i1 < 9; ++v13318_i1) {
              double v13321_data = r9[(2 + (v13318_i1 * 3))];
              s5[(v1393_lead + (v13318_i1 * 35))] = v13321_data;
            }
          }
          double r11[36]{};
          // ir11 = +(glb_m7 * s5)
          // [(1, 56), (0, 9)] [(0, 35)]
          double ir11[36]{};
          double v13332_data = v5629_g ? (glb_m7[v1295_off]) : (0.0);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v13333_data = s5[0];
          double v13338_data = s5[35];
          double v13343_data = s5[70];
          double v13348_data = s5[105];
          double v13353_data = s5[140];
          double v13358_data = s5[175];
          double v13363_data = s5[210];
          double v13368_data = s5[245];
          double v13373_data = s5[280];
          double v13381_data = glb_m7[v1345_off];
          double v13430_data = glb_m7[v1394_off];
          double v13480_data = v18902_g ? (glb_m7[v1443_off]) : (0.0);
          double v13527_data = v5629_g ? (glb_m7[(v18980_lead + 54)]) : (0.0);
          double v13528_data = s5[1];
          double v13533_data = s5[36];
          double v13538_data = s5[71];
          double v13543_data = s5[106];
          double v13548_data = s5[141];
          double v13553_data = s5[176];
          double v13558_data = s5[211];
          double v13563_data = s5[246];
          double v13568_data = s5[281];
          double v13574_data = glb_m7[(v1344_lead + 54)];
          double v13621_data = glb_m7[(v1393_lead + 54)];
          double v13668_data = v18902_g ? (glb_m7[(v1442_lead + 54)]) : (0.0);
          double v13715_data = v5629_g ? (glb_m7[(v18980_lead + 109)]) : (0.0);
          double v13716_data = s5[2];
          double v13721_data = s5[37];
          double v13726_data = s5[72];
          double v13731_data = s5[107];
          double v13736_data = s5[142];
          double v13741_data = s5[177];
          double v13746_data = s5[212];
          double v13751_data = s5[247];
          double v13756_data = s5[282];
          double v13762_data = glb_m7[(v1344_lead + 109)];
          double v13809_data = glb_m7[(v1393_lead + 109)];
          double v13856_data = v18902_g ? (glb_m7[(v1442_lead + 109)]) : (0.0);
          double v13903_data = v5629_g ? (glb_m7[(v18980_lead + 164)]) : (0.0);
          double v13904_data = s5[3];
          double v13909_data = s5[38];
          double v13914_data = s5[73];
          double v13919_data = s5[108];
          double v13924_data = s5[143];
          double v13929_data = s5[178];
          double v13934_data = s5[213];
          double v13939_data = s5[248];
          double v13944_data = s5[283];
          double v13950_data = glb_m7[(v1344_lead + 164)];
          double v13997_data = glb_m7[(v1393_lead + 164)];
          double v14044_data = v18902_g ? (glb_m7[(v1442_lead + 164)]) : (0.0);
          double v14091_data = v5629_g ? (glb_m7[(v18980_lead + 219)]) : (0.0);
          double v14092_data = s5[4];
          double v14097_data = s5[39];
          double v14102_data = s5[74];
          double v14107_data = s5[109];
          double v14112_data = s5[144];
          double v14117_data = s5[179];
          double v14122_data = s5[214];
          double v14127_data = s5[249];
          double v14132_data = s5[284];
          double v14138_data = glb_m7[(v1344_lead + 219)];
          double v14185_data = glb_m7[(v1393_lead + 219)];
          double v14232_data = v18902_g ? (glb_m7[(v1442_lead + 219)]) : (0.0);
          double v14279_data = v5629_g ? (glb_m7[(v18980_lead + 274)]) : (0.0);
          double v14280_data = s5[5];
          double v14285_data = s5[40];
          double v14290_data = s5[75];
          double v14295_data = s5[110];
          double v14300_data = s5[145];
          double v14305_data = s5[180];
          double v14310_data = s5[215];
          double v14315_data = s5[250];
          double v14320_data = s5[285];
          double v14326_data = glb_m7[(v1344_lead + 274)];
          double v14373_data = glb_m7[(v1393_lead + 274)];
          double v14420_data = v18902_g ? (glb_m7[(v1442_lead + 274)]) : (0.0);
          double v14467_data = v5629_g ? (glb_m7[(v18980_lead + 329)]) : (0.0);
          double v14468_data = s5[6];
          double v14473_data = s5[41];
          double v14478_data = s5[76];
          double v14483_data = s5[111];
          double v14488_data = s5[146];
          double v14493_data = s5[181];
          double v14498_data = s5[216];
          double v14503_data = s5[251];
          double v14508_data = s5[286];
          double v14514_data = glb_m7[(v1344_lead + 329)];
          double v14561_data = glb_m7[(v1393_lead + 329)];
          double v14608_data = v18902_g ? (glb_m7[(v1442_lead + 329)]) : (0.0);
          double v14655_data = v5629_g ? (glb_m7[(v18980_lead + 384)]) : (0.0);
          double v14656_data = s5[7];
          double v14661_data = s5[42];
          double v14666_data = s5[77];
          double v14671_data = s5[112];
          double v14676_data = s5[147];
          double v14681_data = s5[182];
          double v14686_data = s5[217];
          double v14691_data = s5[252];
          double v14696_data = s5[287];
          double v14702_data = glb_m7[(v1344_lead + 384)];
          double v14749_data = glb_m7[(v1393_lead + 384)];
          double v14796_data = v18902_g ? (glb_m7[(v1442_lead + 384)]) : (0.0);
          double v14843_data = v5629_g ? (glb_m7[(v18980_lead + 439)]) : (0.0);
          double v14844_data = s5[8];
          double v14849_data = s5[43];
          double v14854_data = s5[78];
          double v14859_data = s5[113];
          double v14864_data = s5[148];
          double v14869_data = s5[183];
          double v14874_data = s5[218];
          double v14879_data = s5[253];
          double v14884_data = s5[288];
          double v14890_data = glb_m7[(v1344_lead + 439)];
          double v14937_data = glb_m7[(v1393_lead + 439)];
          double v14984_data = v18902_g ? (glb_m7[(v1442_lead + 439)]) : (0.0);
          double v15031_data = v5629_g ? (glb_m7[(v18980_lead + 494)]) : (0.0);
          double v15032_data = s5[9];
          double v15037_data = s5[44];
          double v15042_data = s5[79];
          double v15047_data = s5[114];
          double v15052_data = s5[149];
          double v15057_data = s5[184];
          double v15062_data = s5[219];
          double v15067_data = s5[254];
          double v15072_data = s5[289];
          double v15078_data = glb_m7[(v1344_lead + 494)];
          double v15125_data = glb_m7[(v1393_lead + 494)];
          double v15172_data = v18902_g ? (glb_m7[(v1442_lead + 494)]) : (0.0);
          double v15219_data = v5629_g ? (glb_m7[(v18980_lead + 549)]) : (0.0);
          double v15220_data = s5[10];
          double v15225_data = s5[45];
          double v15230_data = s5[80];
          double v15235_data = s5[115];
          double v15240_data = s5[150];
          double v15245_data = s5[185];
          double v15250_data = s5[220];
          double v15255_data = s5[255];
          double v15260_data = s5[290];
          double v15266_data = glb_m7[(v1344_lead + 549)];
          double v15313_data = glb_m7[(v1393_lead + 549)];
          double v15360_data = v18902_g ? (glb_m7[(v1442_lead + 549)]) : (0.0);
          double v15407_data = v5629_g ? (glb_m7[(v18980_lead + 604)]) : (0.0);
          double v15408_data = s5[11];
          double v15413_data = s5[46];
          double v15418_data = s5[81];
          double v15423_data = s5[116];
          double v15428_data = s5[151];
          double v15433_data = s5[186];
          double v15438_data = s5[221];
          double v15443_data = s5[256];
          double v15448_data = s5[291];
          double v15454_data = glb_m7[(v1344_lead + 604)];
          double v15501_data = glb_m7[(v1393_lead + 604)];
          double v15548_data = v18902_g ? (glb_m7[(v1442_lead + 604)]) : (0.0);
          double v15595_data = v5629_g ? (glb_m7[(v18980_lead + 659)]) : (0.0);
          double v15596_data = s5[12];
          double v15601_data = s5[47];
          double v15606_data = s5[82];
          double v15611_data = s5[117];
          double v15616_data = s5[152];
          double v15621_data = s5[187];
          double v15626_data = s5[222];
          double v15631_data = s5[257];
          double v15636_data = s5[292];
          double v15642_data = glb_m7[(v1344_lead + 659)];
          double v15689_data = glb_m7[(v1393_lead + 659)];
          double v15736_data = v18902_g ? (glb_m7[(v1442_lead + 659)]) : (0.0);
          double v15783_data = v5629_g ? (glb_m7[(v18980_lead + 714)]) : (0.0);
          double v15784_data = s5[13];
          double v15789_data = s5[48];
          double v15794_data = s5[83];
          double v15799_data = s5[118];
          double v15804_data = s5[153];
          double v15809_data = s5[188];
          double v15814_data = s5[223];
          double v15819_data = s5[258];
          double v15824_data = s5[293];
          double v15830_data = glb_m7[(v1344_lead + 714)];
          double v15877_data = glb_m7[(v1393_lead + 714)];
          double v15924_data = v18902_g ? (glb_m7[(v1442_lead + 714)]) : (0.0);
          double v15971_data = v5629_g ? (glb_m7[(v18980_lead + 769)]) : (0.0);
          double v15972_data = s5[14];
          double v15977_data = s5[49];
          double v15982_data = s5[84];
          double v15987_data = s5[119];
          double v15992_data = s5[154];
          double v15997_data = s5[189];
          double v16002_data = s5[224];
          double v16007_data = s5[259];
          double v16012_data = s5[294];
          double v16018_data = glb_m7[(v1344_lead + 769)];
          double v16065_data = glb_m7[(v1393_lead + 769)];
          double v16112_data = v18902_g ? (glb_m7[(v1442_lead + 769)]) : (0.0);
          double v16159_data = v5629_g ? (glb_m7[(v18980_lead + 824)]) : (0.0);
          double v16160_data = s5[15];
          double v16165_data = s5[50];
          double v16170_data = s5[85];
          double v16175_data = s5[120];
          double v16180_data = s5[155];
          double v16185_data = s5[190];
          double v16190_data = s5[225];
          double v16195_data = s5[260];
          double v16200_data = s5[295];
          double v16206_data = glb_m7[(v1344_lead + 824)];
          double v16253_data = glb_m7[(v1393_lead + 824)];
          double v16300_data = v18902_g ? (glb_m7[(v1442_lead + 824)]) : (0.0);
          double v16348_data = v2809_g ? (glb_m7[(v18980_lead + 872)]) : (0.0);
          double v16349_data = s5[16];
          double v16354_data = s5[51];
          double v16359_data = s5[86];
          double v16364_data = s5[121];
          double v16369_data = s5[156];
          double v16374_data = s5[191];
          double v16379_data = s5[226];
          double v16384_data = s5[261];
          double v16389_data = s5[296];
          double v16395_data = glb_m7[(v1344_lead + 872)];
          double v16442_data = glb_m7[(v1393_lead + 872)];
          double v16489_data = v18902_g ? (glb_m7[(v1442_lead + 872)]) : (0.0);
          double v16536_data = v2809_g ? (glb_m7[(v18980_lead + 920)]) : (0.0);
          double v16537_data = s5[17];
          double v16542_data = s5[52];
          double v16547_data = s5[87];
          double v16552_data = s5[122];
          double v16557_data = s5[157];
          double v16562_data = s5[192];
          double v16567_data = s5[227];
          double v16572_data = s5[262];
          double v16577_data = s5[297];
          double v16583_data = glb_m7[(v1344_lead + 920)];
          double v16630_data = glb_m7[(v1393_lead + 920)];
          double v16677_data = v18902_g ? (glb_m7[(v1442_lead + 920)]) : (0.0);
          double v16724_data = v2809_g ? (glb_m7[(v18980_lead + 968)]) : (0.0);
          double v16725_data = s5[18];
          double v16730_data = s5[53];
          double v16733_p = ((((((((((((((((((v13332_data * v13338_data) + (v13527_data * v13533_data)) + (v13715_data * v13721_data)) + (v13903_data * v13909_data)) + (v14091_data * v14097_data)) + (v14279_data * v14285_data)) + (v14467_data * v14473_data)) + (v14655_data * v14661_data)) + (v14843_data * v14849_data)) + (v15031_data * v15037_data)) + (v15219_data * v15225_data)) + (v15407_data * v15413_data)) + (v15595_data * v15601_data)) + (v15783_data * v15789_data)) + (v15971_data * v15977_data)) + (v16159_data * v16165_data)) + (v16348_data * v16354_data)) + (v16536_data * v16542_data)) + (v16724_data * v16730_data);
          double v16735_data = s5[88];
          double v16738_p = ((((((((((((((((((v13332_data * v13343_data) + (v13527_data * v13538_data)) + (v13715_data * v13726_data)) + (v13903_data * v13914_data)) + (v14091_data * v14102_data)) + (v14279_data * v14290_data)) + (v14467_data * v14478_data)) + (v14655_data * v14666_data)) + (v14843_data * v14854_data)) + (v15031_data * v15042_data)) + (v15219_data * v15230_data)) + (v15407_data * v15418_data)) + (v15595_data * v15606_data)) + (v15783_data * v15794_data)) + (v15971_data * v15982_data)) + (v16159_data * v16170_data)) + (v16348_data * v16359_data)) + (v16536_data * v16547_data)) + (v16724_data * v16735_data);
          double v16740_data = s5[123];
          double v16743_p = ((((((((((((((((((v13332_data * v13348_data) + (v13527_data * v13543_data)) + (v13715_data * v13731_data)) + (v13903_data * v13919_data)) + (v14091_data * v14107_data)) + (v14279_data * v14295_data)) + (v14467_data * v14483_data)) + (v14655_data * v14671_data)) + (v14843_data * v14859_data)) + (v15031_data * v15047_data)) + (v15219_data * v15235_data)) + (v15407_data * v15423_data)) + (v15595_data * v15611_data)) + (v15783_data * v15799_data)) + (v15971_data * v15987_data)) + (v16159_data * v16175_data)) + (v16348_data * v16364_data)) + (v16536_data * v16552_data)) + (v16724_data * v16740_data);
          double v16745_data = s5[158];
          double v16748_p = ((((((((((((((((((v13332_data * v13353_data) + (v13527_data * v13548_data)) + (v13715_data * v13736_data)) + (v13903_data * v13924_data)) + (v14091_data * v14112_data)) + (v14279_data * v14300_data)) + (v14467_data * v14488_data)) + (v14655_data * v14676_data)) + (v14843_data * v14864_data)) + (v15031_data * v15052_data)) + (v15219_data * v15240_data)) + (v15407_data * v15428_data)) + (v15595_data * v15616_data)) + (v15783_data * v15804_data)) + (v15971_data * v15992_data)) + (v16159_data * v16180_data)) + (v16348_data * v16369_data)) + (v16536_data * v16557_data)) + (v16724_data * v16745_data);
          double v16750_data = s5[193];
          double v16753_p = ((((((((((((((((((v13332_data * v13358_data) + (v13527_data * v13553_data)) + (v13715_data * v13741_data)) + (v13903_data * v13929_data)) + (v14091_data * v14117_data)) + (v14279_data * v14305_data)) + (v14467_data * v14493_data)) + (v14655_data * v14681_data)) + (v14843_data * v14869_data)) + (v15031_data * v15057_data)) + (v15219_data * v15245_data)) + (v15407_data * v15433_data)) + (v15595_data * v15621_data)) + (v15783_data * v15809_data)) + (v15971_data * v15997_data)) + (v16159_data * v16185_data)) + (v16348_data * v16374_data)) + (v16536_data * v16562_data)) + (v16724_data * v16750_data);
          double v16755_data = s5[228];
          double v16758_p = ((((((((((((((((((v13332_data * v13363_data) + (v13527_data * v13558_data)) + (v13715_data * v13746_data)) + (v13903_data * v13934_data)) + (v14091_data * v14122_data)) + (v14279_data * v14310_data)) + (v14467_data * v14498_data)) + (v14655_data * v14686_data)) + (v14843_data * v14874_data)) + (v15031_data * v15062_data)) + (v15219_data * v15250_data)) + (v15407_data * v15438_data)) + (v15595_data * v15626_data)) + (v15783_data * v15814_data)) + (v15971_data * v16002_data)) + (v16159_data * v16190_data)) + (v16348_data * v16379_data)) + (v16536_data * v16567_data)) + (v16724_data * v16755_data);
          double v16760_data = s5[263];
          double v16763_p = ((((((((((((((((((v13332_data * v13368_data) + (v13527_data * v13563_data)) + (v13715_data * v13751_data)) + (v13903_data * v13939_data)) + (v14091_data * v14127_data)) + (v14279_data * v14315_data)) + (v14467_data * v14503_data)) + (v14655_data * v14691_data)) + (v14843_data * v14879_data)) + (v15031_data * v15067_data)) + (v15219_data * v15255_data)) + (v15407_data * v15443_data)) + (v15595_data * v15631_data)) + (v15783_data * v15819_data)) + (v15971_data * v16007_data)) + (v16159_data * v16195_data)) + (v16348_data * v16384_data)) + (v16536_data * v16572_data)) + (v16724_data * v16760_data);
          double v16765_data = s5[298];
          double v16768_p = ((((((((((((((((((v13332_data * v13373_data) + (v13527_data * v13568_data)) + (v13715_data * v13756_data)) + (v13903_data * v13944_data)) + (v14091_data * v14132_data)) + (v14279_data * v14320_data)) + (v14467_data * v14508_data)) + (v14655_data * v14696_data)) + (v14843_data * v14884_data)) + (v15031_data * v15072_data)) + (v15219_data * v15260_data)) + (v15407_data * v15448_data)) + (v15595_data * v15636_data)) + (v15783_data * v15824_data)) + (v15971_data * v16012_data)) + (v16159_data * v16200_data)) + (v16348_data * v16389_data)) + (v16536_data * v16577_data)) + (v16724_data * v16765_data);
          double v16771_data = glb_m7[v10541_off];
          double v16775_p = ((((((((((((((((((v13381_data * v13333_data) + (v13574_data * v13528_data)) + (v13762_data * v13716_data)) + (v13950_data * v13904_data)) + (v14138_data * v14092_data)) + (v14326_data * v14280_data)) + (v14514_data * v14468_data)) + (v14702_data * v14656_data)) + (v14890_data * v14844_data)) + (v15078_data * v15032_data)) + (v15266_data * v15220_data)) + (v15454_data * v15408_data)) + (v15642_data * v15596_data)) + (v15830_data * v15784_data)) + (v16018_data * v15972_data)) + (v16206_data * v16160_data)) + (v16395_data * v16349_data)) + (v16583_data * v16537_data)) + (v16771_data * v16725_data);
          double v16780_p = ((((((((((((((((((v13381_data * v13338_data) + (v13574_data * v13533_data)) + (v13762_data * v13721_data)) + (v13950_data * v13909_data)) + (v14138_data * v14097_data)) + (v14326_data * v14285_data)) + (v14514_data * v14473_data)) + (v14702_data * v14661_data)) + (v14890_data * v14849_data)) + (v15078_data * v15037_data)) + (v15266_data * v15225_data)) + (v15454_data * v15413_data)) + (v15642_data * v15601_data)) + (v15830_data * v15789_data)) + (v16018_data * v15977_data)) + (v16206_data * v16165_data)) + (v16395_data * v16354_data)) + (v16583_data * v16542_data)) + (v16771_data * v16730_data);
          double v16785_p = ((((((((((((((((((v13381_data * v13343_data) + (v13574_data * v13538_data)) + (v13762_data * v13726_data)) + (v13950_data * v13914_data)) + (v14138_data * v14102_data)) + (v14326_data * v14290_data)) + (v14514_data * v14478_data)) + (v14702_data * v14666_data)) + (v14890_data * v14854_data)) + (v15078_data * v15042_data)) + (v15266_data * v15230_data)) + (v15454_data * v15418_data)) + (v15642_data * v15606_data)) + (v15830_data * v15794_data)) + (v16018_data * v15982_data)) + (v16206_data * v16170_data)) + (v16395_data * v16359_data)) + (v16583_data * v16547_data)) + (v16771_data * v16735_data);
          double v16790_p = ((((((((((((((((((v13381_data * v13348_data) + (v13574_data * v13543_data)) + (v13762_data * v13731_data)) + (v13950_data * v13919_data)) + (v14138_data * v14107_data)) + (v14326_data * v14295_data)) + (v14514_data * v14483_data)) + (v14702_data * v14671_data)) + (v14890_data * v14859_data)) + (v15078_data * v15047_data)) + (v15266_data * v15235_data)) + (v15454_data * v15423_data)) + (v15642_data * v15611_data)) + (v15830_data * v15799_data)) + (v16018_data * v15987_data)) + (v16206_data * v16175_data)) + (v16395_data * v16364_data)) + (v16583_data * v16552_data)) + (v16771_data * v16740_data);
          double v16795_p = ((((((((((((((((((v13381_data * v13353_data) + (v13574_data * v13548_data)) + (v13762_data * v13736_data)) + (v13950_data * v13924_data)) + (v14138_data * v14112_data)) + (v14326_data * v14300_data)) + (v14514_data * v14488_data)) + (v14702_data * v14676_data)) + (v14890_data * v14864_data)) + (v15078_data * v15052_data)) + (v15266_data * v15240_data)) + (v15454_data * v15428_data)) + (v15642_data * v15616_data)) + (v15830_data * v15804_data)) + (v16018_data * v15992_data)) + (v16206_data * v16180_data)) + (v16395_data * v16369_data)) + (v16583_data * v16557_data)) + (v16771_data * v16745_data);
          double v16800_p = ((((((((((((((((((v13381_data * v13358_data) + (v13574_data * v13553_data)) + (v13762_data * v13741_data)) + (v13950_data * v13929_data)) + (v14138_data * v14117_data)) + (v14326_data * v14305_data)) + (v14514_data * v14493_data)) + (v14702_data * v14681_data)) + (v14890_data * v14869_data)) + (v15078_data * v15057_data)) + (v15266_data * v15245_data)) + (v15454_data * v15433_data)) + (v15642_data * v15621_data)) + (v15830_data * v15809_data)) + (v16018_data * v15997_data)) + (v16206_data * v16185_data)) + (v16395_data * v16374_data)) + (v16583_data * v16562_data)) + (v16771_data * v16750_data);
          double v16805_p = ((((((((((((((((((v13381_data * v13363_data) + (v13574_data * v13558_data)) + (v13762_data * v13746_data)) + (v13950_data * v13934_data)) + (v14138_data * v14122_data)) + (v14326_data * v14310_data)) + (v14514_data * v14498_data)) + (v14702_data * v14686_data)) + (v14890_data * v14874_data)) + (v15078_data * v15062_data)) + (v15266_data * v15250_data)) + (v15454_data * v15438_data)) + (v15642_data * v15626_data)) + (v15830_data * v15814_data)) + (v16018_data * v16002_data)) + (v16206_data * v16190_data)) + (v16395_data * v16379_data)) + (v16583_data * v16567_data)) + (v16771_data * v16755_data);
          double v16810_p = ((((((((((((((((((v13381_data * v13368_data) + (v13574_data * v13563_data)) + (v13762_data * v13751_data)) + (v13950_data * v13939_data)) + (v14138_data * v14127_data)) + (v14326_data * v14315_data)) + (v14514_data * v14503_data)) + (v14702_data * v14691_data)) + (v14890_data * v14879_data)) + (v15078_data * v15067_data)) + (v15266_data * v15255_data)) + (v15454_data * v15443_data)) + (v15642_data * v15631_data)) + (v15830_data * v15819_data)) + (v16018_data * v16007_data)) + (v16206_data * v16195_data)) + (v16395_data * v16384_data)) + (v16583_data * v16572_data)) + (v16771_data * v16760_data);
          double v16815_p = ((((((((((((((((((v13381_data * v13373_data) + (v13574_data * v13568_data)) + (v13762_data * v13756_data)) + (v13950_data * v13944_data)) + (v14138_data * v14132_data)) + (v14326_data * v14320_data)) + (v14514_data * v14508_data)) + (v14702_data * v14696_data)) + (v14890_data * v14884_data)) + (v15078_data * v15072_data)) + (v15266_data * v15260_data)) + (v15454_data * v15448_data)) + (v15642_data * v15636_data)) + (v15830_data * v15824_data)) + (v16018_data * v16012_data)) + (v16206_data * v16200_data)) + (v16395_data * v16389_data)) + (v16583_data * v16577_data)) + (v16771_data * v16765_data);
          double v16818_data = glb_m7[v10588_off];
          double v16822_p = ((((((((((((((((((v13430_data * v13333_data) + (v13621_data * v13528_data)) + (v13809_data * v13716_data)) + (v13997_data * v13904_data)) + (v14185_data * v14092_data)) + (v14373_data * v14280_data)) + (v14561_data * v14468_data)) + (v14749_data * v14656_data)) + (v14937_data * v14844_data)) + (v15125_data * v15032_data)) + (v15313_data * v15220_data)) + (v15501_data * v15408_data)) + (v15689_data * v15596_data)) + (v15877_data * v15784_data)) + (v16065_data * v15972_data)) + (v16253_data * v16160_data)) + (v16442_data * v16349_data)) + (v16630_data * v16537_data)) + (v16818_data * v16725_data);
          double v16827_p = ((((((((((((((((((v13430_data * v13338_data) + (v13621_data * v13533_data)) + (v13809_data * v13721_data)) + (v13997_data * v13909_data)) + (v14185_data * v14097_data)) + (v14373_data * v14285_data)) + (v14561_data * v14473_data)) + (v14749_data * v14661_data)) + (v14937_data * v14849_data)) + (v15125_data * v15037_data)) + (v15313_data * v15225_data)) + (v15501_data * v15413_data)) + (v15689_data * v15601_data)) + (v15877_data * v15789_data)) + (v16065_data * v15977_data)) + (v16253_data * v16165_data)) + (v16442_data * v16354_data)) + (v16630_data * v16542_data)) + (v16818_data * v16730_data);
          double v16832_p = ((((((((((((((((((v13430_data * v13343_data) + (v13621_data * v13538_data)) + (v13809_data * v13726_data)) + (v13997_data * v13914_data)) + (v14185_data * v14102_data)) + (v14373_data * v14290_data)) + (v14561_data * v14478_data)) + (v14749_data * v14666_data)) + (v14937_data * v14854_data)) + (v15125_data * v15042_data)) + (v15313_data * v15230_data)) + (v15501_data * v15418_data)) + (v15689_data * v15606_data)) + (v15877_data * v15794_data)) + (v16065_data * v15982_data)) + (v16253_data * v16170_data)) + (v16442_data * v16359_data)) + (v16630_data * v16547_data)) + (v16818_data * v16735_data);
          double v16837_p = ((((((((((((((((((v13430_data * v13348_data) + (v13621_data * v13543_data)) + (v13809_data * v13731_data)) + (v13997_data * v13919_data)) + (v14185_data * v14107_data)) + (v14373_data * v14295_data)) + (v14561_data * v14483_data)) + (v14749_data * v14671_data)) + (v14937_data * v14859_data)) + (v15125_data * v15047_data)) + (v15313_data * v15235_data)) + (v15501_data * v15423_data)) + (v15689_data * v15611_data)) + (v15877_data * v15799_data)) + (v16065_data * v15987_data)) + (v16253_data * v16175_data)) + (v16442_data * v16364_data)) + (v16630_data * v16552_data)) + (v16818_data * v16740_data);
          double v16842_p = ((((((((((((((((((v13430_data * v13353_data) + (v13621_data * v13548_data)) + (v13809_data * v13736_data)) + (v13997_data * v13924_data)) + (v14185_data * v14112_data)) + (v14373_data * v14300_data)) + (v14561_data * v14488_data)) + (v14749_data * v14676_data)) + (v14937_data * v14864_data)) + (v15125_data * v15052_data)) + (v15313_data * v15240_data)) + (v15501_data * v15428_data)) + (v15689_data * v15616_data)) + (v15877_data * v15804_data)) + (v16065_data * v15992_data)) + (v16253_data * v16180_data)) + (v16442_data * v16369_data)) + (v16630_data * v16557_data)) + (v16818_data * v16745_data);
          double v16847_p = ((((((((((((((((((v13430_data * v13358_data) + (v13621_data * v13553_data)) + (v13809_data * v13741_data)) + (v13997_data * v13929_data)) + (v14185_data * v14117_data)) + (v14373_data * v14305_data)) + (v14561_data * v14493_data)) + (v14749_data * v14681_data)) + (v14937_data * v14869_data)) + (v15125_data * v15057_data)) + (v15313_data * v15245_data)) + (v15501_data * v15433_data)) + (v15689_data * v15621_data)) + (v15877_data * v15809_data)) + (v16065_data * v15997_data)) + (v16253_data * v16185_data)) + (v16442_data * v16374_data)) + (v16630_data * v16562_data)) + (v16818_data * v16750_data);
          double v16852_p = ((((((((((((((((((v13430_data * v13363_data) + (v13621_data * v13558_data)) + (v13809_data * v13746_data)) + (v13997_data * v13934_data)) + (v14185_data * v14122_data)) + (v14373_data * v14310_data)) + (v14561_data * v14498_data)) + (v14749_data * v14686_data)) + (v14937_data * v14874_data)) + (v15125_data * v15062_data)) + (v15313_data * v15250_data)) + (v15501_data * v15438_data)) + (v15689_data * v15626_data)) + (v15877_data * v15814_data)) + (v16065_data * v16002_data)) + (v16253_data * v16190_data)) + (v16442_data * v16379_data)) + (v16630_data * v16567_data)) + (v16818_data * v16755_data);
          double v16857_p = ((((((((((((((((((v13430_data * v13368_data) + (v13621_data * v13563_data)) + (v13809_data * v13751_data)) + (v13997_data * v13939_data)) + (v14185_data * v14127_data)) + (v14373_data * v14315_data)) + (v14561_data * v14503_data)) + (v14749_data * v14691_data)) + (v14937_data * v14879_data)) + (v15125_data * v15067_data)) + (v15313_data * v15255_data)) + (v15501_data * v15443_data)) + (v15689_data * v15631_data)) + (v15877_data * v15819_data)) + (v16065_data * v16007_data)) + (v16253_data * v16195_data)) + (v16442_data * v16384_data)) + (v16630_data * v16572_data)) + (v16818_data * v16760_data);
          double v16862_p = ((((((((((((((((((v13430_data * v13373_data) + (v13621_data * v13568_data)) + (v13809_data * v13756_data)) + (v13997_data * v13944_data)) + (v14185_data * v14132_data)) + (v14373_data * v14320_data)) + (v14561_data * v14508_data)) + (v14749_data * v14696_data)) + (v14937_data * v14884_data)) + (v15125_data * v15072_data)) + (v15313_data * v15260_data)) + (v15501_data * v15448_data)) + (v15689_data * v15636_data)) + (v15877_data * v15824_data)) + (v16065_data * v16012_data)) + (v16253_data * v16200_data)) + (v16442_data * v16389_data)) + (v16630_data * v16577_data)) + (v16818_data * v16765_data);
          double v16865_data = v18902_g ? (glb_m7[v10635_off]) : (0.0);
          double v16869_p = ((((((((((((((((((v13480_data * v13333_data) + (v13668_data * v13528_data)) + (v13856_data * v13716_data)) + (v14044_data * v13904_data)) + (v14232_data * v14092_data)) + (v14420_data * v14280_data)) + (v14608_data * v14468_data)) + (v14796_data * v14656_data)) + (v14984_data * v14844_data)) + (v15172_data * v15032_data)) + (v15360_data * v15220_data)) + (v15548_data * v15408_data)) + (v15736_data * v15596_data)) + (v15924_data * v15784_data)) + (v16112_data * v15972_data)) + (v16300_data * v16160_data)) + (v16489_data * v16349_data)) + (v16677_data * v16537_data)) + (v16865_data * v16725_data);
          double v16874_p = ((((((((((((((((((v13480_data * v13338_data) + (v13668_data * v13533_data)) + (v13856_data * v13721_data)) + (v14044_data * v13909_data)) + (v14232_data * v14097_data)) + (v14420_data * v14285_data)) + (v14608_data * v14473_data)) + (v14796_data * v14661_data)) + (v14984_data * v14849_data)) + (v15172_data * v15037_data)) + (v15360_data * v15225_data)) + (v15548_data * v15413_data)) + (v15736_data * v15601_data)) + (v15924_data * v15789_data)) + (v16112_data * v15977_data)) + (v16300_data * v16165_data)) + (v16489_data * v16354_data)) + (v16677_data * v16542_data)) + (v16865_data * v16730_data);
          double v16879_p = ((((((((((((((((((v13480_data * v13343_data) + (v13668_data * v13538_data)) + (v13856_data * v13726_data)) + (v14044_data * v13914_data)) + (v14232_data * v14102_data)) + (v14420_data * v14290_data)) + (v14608_data * v14478_data)) + (v14796_data * v14666_data)) + (v14984_data * v14854_data)) + (v15172_data * v15042_data)) + (v15360_data * v15230_data)) + (v15548_data * v15418_data)) + (v15736_data * v15606_data)) + (v15924_data * v15794_data)) + (v16112_data * v15982_data)) + (v16300_data * v16170_data)) + (v16489_data * v16359_data)) + (v16677_data * v16547_data)) + (v16865_data * v16735_data);
          double v16884_p = ((((((((((((((((((v13480_data * v13348_data) + (v13668_data * v13543_data)) + (v13856_data * v13731_data)) + (v14044_data * v13919_data)) + (v14232_data * v14107_data)) + (v14420_data * v14295_data)) + (v14608_data * v14483_data)) + (v14796_data * v14671_data)) + (v14984_data * v14859_data)) + (v15172_data * v15047_data)) + (v15360_data * v15235_data)) + (v15548_data * v15423_data)) + (v15736_data * v15611_data)) + (v15924_data * v15799_data)) + (v16112_data * v15987_data)) + (v16300_data * v16175_data)) + (v16489_data * v16364_data)) + (v16677_data * v16552_data)) + (v16865_data * v16740_data);
          double v16889_p = ((((((((((((((((((v13480_data * v13353_data) + (v13668_data * v13548_data)) + (v13856_data * v13736_data)) + (v14044_data * v13924_data)) + (v14232_data * v14112_data)) + (v14420_data * v14300_data)) + (v14608_data * v14488_data)) + (v14796_data * v14676_data)) + (v14984_data * v14864_data)) + (v15172_data * v15052_data)) + (v15360_data * v15240_data)) + (v15548_data * v15428_data)) + (v15736_data * v15616_data)) + (v15924_data * v15804_data)) + (v16112_data * v15992_data)) + (v16300_data * v16180_data)) + (v16489_data * v16369_data)) + (v16677_data * v16557_data)) + (v16865_data * v16745_data);
          double v16894_p = ((((((((((((((((((v13480_data * v13358_data) + (v13668_data * v13553_data)) + (v13856_data * v13741_data)) + (v14044_data * v13929_data)) + (v14232_data * v14117_data)) + (v14420_data * v14305_data)) + (v14608_data * v14493_data)) + (v14796_data * v14681_data)) + (v14984_data * v14869_data)) + (v15172_data * v15057_data)) + (v15360_data * v15245_data)) + (v15548_data * v15433_data)) + (v15736_data * v15621_data)) + (v15924_data * v15809_data)) + (v16112_data * v15997_data)) + (v16300_data * v16185_data)) + (v16489_data * v16374_data)) + (v16677_data * v16562_data)) + (v16865_data * v16750_data);
          double v16899_p = ((((((((((((((((((v13480_data * v13363_data) + (v13668_data * v13558_data)) + (v13856_data * v13746_data)) + (v14044_data * v13934_data)) + (v14232_data * v14122_data)) + (v14420_data * v14310_data)) + (v14608_data * v14498_data)) + (v14796_data * v14686_data)) + (v14984_data * v14874_data)) + (v15172_data * v15062_data)) + (v15360_data * v15250_data)) + (v15548_data * v15438_data)) + (v15736_data * v15626_data)) + (v15924_data * v15814_data)) + (v16112_data * v16002_data)) + (v16300_data * v16190_data)) + (v16489_data * v16379_data)) + (v16677_data * v16567_data)) + (v16865_data * v16755_data);
          double v16904_p = ((((((((((((((((((v13480_data * v13368_data) + (v13668_data * v13563_data)) + (v13856_data * v13751_data)) + (v14044_data * v13939_data)) + (v14232_data * v14127_data)) + (v14420_data * v14315_data)) + (v14608_data * v14503_data)) + (v14796_data * v14691_data)) + (v14984_data * v14879_data)) + (v15172_data * v15067_data)) + (v15360_data * v15255_data)) + (v15548_data * v15443_data)) + (v15736_data * v15631_data)) + (v15924_data * v15819_data)) + (v16112_data * v16007_data)) + (v16300_data * v16195_data)) + (v16489_data * v16384_data)) + (v16677_data * v16572_data)) + (v16865_data * v16760_data);
          double v16909_p = ((((((((((((((((((v13480_data * v13373_data) + (v13668_data * v13568_data)) + (v13856_data * v13756_data)) + (v14044_data * v13944_data)) + (v14232_data * v14132_data)) + (v14420_data * v14320_data)) + (v14608_data * v14508_data)) + (v14796_data * v14696_data)) + (v14984_data * v14884_data)) + (v15172_data * v15072_data)) + (v15360_data * v15260_data)) + (v15548_data * v15448_data)) + (v15736_data * v15636_data)) + (v15924_data * v15824_data)) + (v16112_data * v16012_data)) + (v16300_data * v16200_data)) + (v16489_data * v16389_data)) + (v16677_data * v16577_data)) + (v16865_data * v16765_data);
          double v16912_data = v2809_g ? (glb_m7[(v18980_lead + 1016)]) : (0.0);
          double v16913_data = s5[19];
          ir11[0] = ((((((((((((((((((((v13332_data * v13333_data) + (v13527_data * v13528_data)) + (v13715_data * v13716_data)) + (v13903_data * v13904_data)) + (v14091_data * v14092_data)) + (v14279_data * v14280_data)) + (v14467_data * v14468_data)) + (v14655_data * v14656_data)) + (v14843_data * v14844_data)) + (v15031_data * v15032_data)) + (v15219_data * v15220_data)) + (v15407_data * v15408_data)) + (v15595_data * v15596_data)) + (v15783_data * v15784_data)) + (v15971_data * v15972_data)) + (v16159_data * v16160_data)) + (v16348_data * v16349_data)) + (v16536_data * v16537_data)) + (v16724_data * v16725_data)) + (v16912_data * v16913_data));
          double v16918_data = s5[54];
          ir11[4] = (v16733_p + (v16912_data * v16918_data));
          double v16923_data = s5[89];
          ir11[8] = (v16738_p + (v16912_data * v16923_data));
          double v16928_data = s5[124];
          ir11[12] = (v16743_p + (v16912_data * v16928_data));
          double v16933_data = s5[159];
          ir11[16] = (v16748_p + (v16912_data * v16933_data));
          double v16938_data = s5[194];
          ir11[20] = (v16753_p + (v16912_data * v16938_data));
          double v16943_data = s5[229];
          ir11[24] = (v16758_p + (v16912_data * v16943_data));
          double v16948_data = s5[264];
          ir11[28] = (v16763_p + (v16912_data * v16948_data));
          double v16953_data = s5[299];
          ir11[32] = (v16768_p + (v16912_data * v16953_data));
          double v16959_data = glb_m7[(v1344_lead + 1016)];
          double v17006_data = glb_m7[(v1393_lead + 1016)];
          double v17053_data = v18902_g ? (glb_m7[(v1442_lead + 1016)]) : (0.0);
          double v17100_data = glb_m7[(v1344_lead + 1056)];
          double v17101_data = s5[20];
          double v17106_data = s5[55];
          double v17111_data = s5[90];
          double v17116_data = s5[125];
          double v17121_data = s5[160];
          double v17126_data = s5[195];
          double v17131_data = s5[230];
          double v17136_data = s5[265];
          double v17141_data = s5[300];
          double v17147_data = glb_m7[(v1393_lead + 1056)];
          double v17194_data = v18902_g ? (glb_m7[(v1442_lead + 1056)]) : (0.0);
          double v17241_data = glb_m7[(v1344_lead + 1096)];
          double v17242_data = s5[21];
          double v17247_data = s5[56];
          double v17252_data = s5[91];
          double v17257_data = s5[126];
          double v17262_data = s5[161];
          double v17267_data = s5[196];
          double v17272_data = s5[231];
          double v17277_data = s5[266];
          double v17282_data = s5[301];
          double v17288_data = glb_m7[(v1393_lead + 1096)];
          double v17335_data = v18902_g ? (glb_m7[(v1442_lead + 1096)]) : (0.0);
          double v17382_data = glb_m7[(v1344_lead + 1136)];
          double v17383_data = s5[22];
          double v17388_data = s5[57];
          double v17393_data = s5[92];
          double v17398_data = s5[127];
          double v17403_data = s5[162];
          double v17408_data = s5[197];
          double v17413_data = s5[232];
          double v17418_data = s5[267];
          double v17423_data = s5[302];
          double v17429_data = glb_m7[(v1393_lead + 1136)];
          double v17476_data = v18902_g ? (glb_m7[(v1442_lead + 1136)]) : (0.0);
          double v17523_data = glb_m7[(v1344_lead + 1176)];
          double v17524_data = s5[23];
          double v17529_data = s5[58];
          double v17534_data = s5[93];
          double v17539_data = s5[128];
          double v17544_data = s5[163];
          double v17549_data = s5[198];
          double v17554_data = s5[233];
          double v17559_data = s5[268];
          double v17564_data = s5[303];
          double v17570_data = glb_m7[(v1393_lead + 1176)];
          double v17617_data = v18902_g ? (glb_m7[(v1442_lead + 1176)]) : (0.0);
          double v17664_data = v2809_g ? (glb_m7[(v1344_lead + 1208)]) : (0.0);
          double v17665_data = s5[24];
          double v17670_data = s5[59];
          double v17675_data = s5[94];
          double v17680_data = s5[129];
          double v17685_data = s5[164];
          double v17690_data = s5[199];
          double v17695_data = s5[234];
          double v17700_data = s5[269];
          double v17705_data = s5[304];
          double v17711_data = glb_m7[(v1393_lead + 1208)];
          double v17758_data = v18902_g ? (glb_m7[(v1442_lead + 1208)]) : (0.0);
          double v17805_data = v2809_g ? (glb_m7[(v1344_lead + 1240)]) : (0.0);
          double v17806_data = s5[25];
          double v17811_data = s5[60];
          double v17816_data = s5[95];
          double v17821_data = s5[130];
          double v17826_data = s5[165];
          double v17831_data = s5[200];
          double v17836_data = s5[235];
          double v17841_data = s5[270];
          double v17846_data = s5[305];
          double v17852_data = glb_m7[(v1393_lead + 1240)];
          double v17899_data = v18902_g ? (glb_m7[(v1442_lead + 1240)]) : (0.0);
          double v17946_data = v2809_g ? (glb_m7[(v1344_lead + 1272)]) : (0.0);
          double v17947_data = s5[26];
          double v17952_data = s5[61];
          double v17955_p = (((((((v16780_p + (v16959_data * v16918_data)) + (v17100_data * v17106_data)) + (v17241_data * v17247_data)) + (v17382_data * v17388_data)) + (v17523_data * v17529_data)) + (v17664_data * v17670_data)) + (v17805_data * v17811_data)) + (v17946_data * v17952_data);
          double v17957_data = s5[96];
          double v17960_p = (((((((v16785_p + (v16959_data * v16923_data)) + (v17100_data * v17111_data)) + (v17241_data * v17252_data)) + (v17382_data * v17393_data)) + (v17523_data * v17534_data)) + (v17664_data * v17675_data)) + (v17805_data * v17816_data)) + (v17946_data * v17957_data);
          double v17962_data = s5[131];
          double v17965_p = (((((((v16790_p + (v16959_data * v16928_data)) + (v17100_data * v17116_data)) + (v17241_data * v17257_data)) + (v17382_data * v17398_data)) + (v17523_data * v17539_data)) + (v17664_data * v17680_data)) + (v17805_data * v17821_data)) + (v17946_data * v17962_data);
          double v17967_data = s5[166];
          double v17970_p = (((((((v16795_p + (v16959_data * v16933_data)) + (v17100_data * v17121_data)) + (v17241_data * v17262_data)) + (v17382_data * v17403_data)) + (v17523_data * v17544_data)) + (v17664_data * v17685_data)) + (v17805_data * v17826_data)) + (v17946_data * v17967_data);
          double v17972_data = s5[201];
          double v17975_p = (((((((v16800_p + (v16959_data * v16938_data)) + (v17100_data * v17126_data)) + (v17241_data * v17267_data)) + (v17382_data * v17408_data)) + (v17523_data * v17549_data)) + (v17664_data * v17690_data)) + (v17805_data * v17831_data)) + (v17946_data * v17972_data);
          double v17977_data = s5[236];
          double v17980_p = (((((((v16805_p + (v16959_data * v16943_data)) + (v17100_data * v17131_data)) + (v17241_data * v17272_data)) + (v17382_data * v17413_data)) + (v17523_data * v17554_data)) + (v17664_data * v17695_data)) + (v17805_data * v17836_data)) + (v17946_data * v17977_data);
          double v17982_data = s5[271];
          double v17985_p = (((((((v16810_p + (v16959_data * v16948_data)) + (v17100_data * v17136_data)) + (v17241_data * v17277_data)) + (v17382_data * v17418_data)) + (v17523_data * v17559_data)) + (v17664_data * v17700_data)) + (v17805_data * v17841_data)) + (v17946_data * v17982_data);
          double v17987_data = s5[306];
          double v17990_p = (((((((v16815_p + (v16959_data * v16953_data)) + (v17100_data * v17141_data)) + (v17241_data * v17282_data)) + (v17382_data * v17423_data)) + (v17523_data * v17564_data)) + (v17664_data * v17705_data)) + (v17805_data * v17846_data)) + (v17946_data * v17987_data);
          double v17993_data = glb_m7[(v1393_lead + 1272)];
          double v17997_p = (((((((v16822_p + (v17006_data * v16913_data)) + (v17147_data * v17101_data)) + (v17288_data * v17242_data)) + (v17429_data * v17383_data)) + (v17570_data * v17524_data)) + (v17711_data * v17665_data)) + (v17852_data * v17806_data)) + (v17993_data * v17947_data);
          double v18002_p = (((((((v16827_p + (v17006_data * v16918_data)) + (v17147_data * v17106_data)) + (v17288_data * v17247_data)) + (v17429_data * v17388_data)) + (v17570_data * v17529_data)) + (v17711_data * v17670_data)) + (v17852_data * v17811_data)) + (v17993_data * v17952_data);
          double v18007_p = (((((((v16832_p + (v17006_data * v16923_data)) + (v17147_data * v17111_data)) + (v17288_data * v17252_data)) + (v17429_data * v17393_data)) + (v17570_data * v17534_data)) + (v17711_data * v17675_data)) + (v17852_data * v17816_data)) + (v17993_data * v17957_data);
          double v18012_p = (((((((v16837_p + (v17006_data * v16928_data)) + (v17147_data * v17116_data)) + (v17288_data * v17257_data)) + (v17429_data * v17398_data)) + (v17570_data * v17539_data)) + (v17711_data * v17680_data)) + (v17852_data * v17821_data)) + (v17993_data * v17962_data);
          double v18017_p = (((((((v16842_p + (v17006_data * v16933_data)) + (v17147_data * v17121_data)) + (v17288_data * v17262_data)) + (v17429_data * v17403_data)) + (v17570_data * v17544_data)) + (v17711_data * v17685_data)) + (v17852_data * v17826_data)) + (v17993_data * v17967_data);
          double v18022_p = (((((((v16847_p + (v17006_data * v16938_data)) + (v17147_data * v17126_data)) + (v17288_data * v17267_data)) + (v17429_data * v17408_data)) + (v17570_data * v17549_data)) + (v17711_data * v17690_data)) + (v17852_data * v17831_data)) + (v17993_data * v17972_data);
          double v18027_p = (((((((v16852_p + (v17006_data * v16943_data)) + (v17147_data * v17131_data)) + (v17288_data * v17272_data)) + (v17429_data * v17413_data)) + (v17570_data * v17554_data)) + (v17711_data * v17695_data)) + (v17852_data * v17836_data)) + (v17993_data * v17977_data);
          double v18032_p = (((((((v16857_p + (v17006_data * v16948_data)) + (v17147_data * v17136_data)) + (v17288_data * v17277_data)) + (v17429_data * v17418_data)) + (v17570_data * v17559_data)) + (v17711_data * v17700_data)) + (v17852_data * v17841_data)) + (v17993_data * v17982_data);
          double v18037_p = (((((((v16862_p + (v17006_data * v16953_data)) + (v17147_data * v17141_data)) + (v17288_data * v17282_data)) + (v17429_data * v17423_data)) + (v17570_data * v17564_data)) + (v17711_data * v17705_data)) + (v17852_data * v17846_data)) + (v17993_data * v17987_data);
          double v18040_data = v18902_g ? (glb_m7[(v1442_lead + 1272)]) : (0.0);
          double v18044_p = (((((((v16869_p + (v17053_data * v16913_data)) + (v17194_data * v17101_data)) + (v17335_data * v17242_data)) + (v17476_data * v17383_data)) + (v17617_data * v17524_data)) + (v17758_data * v17665_data)) + (v17899_data * v17806_data)) + (v18040_data * v17947_data);
          double v18049_p = (((((((v16874_p + (v17053_data * v16918_data)) + (v17194_data * v17106_data)) + (v17335_data * v17247_data)) + (v17476_data * v17388_data)) + (v17617_data * v17529_data)) + (v17758_data * v17670_data)) + (v17899_data * v17811_data)) + (v18040_data * v17952_data);
          double v18054_p = (((((((v16879_p + (v17053_data * v16923_data)) + (v17194_data * v17111_data)) + (v17335_data * v17252_data)) + (v17476_data * v17393_data)) + (v17617_data * v17534_data)) + (v17758_data * v17675_data)) + (v17899_data * v17816_data)) + (v18040_data * v17957_data);
          double v18059_p = (((((((v16884_p + (v17053_data * v16928_data)) + (v17194_data * v17116_data)) + (v17335_data * v17257_data)) + (v17476_data * v17398_data)) + (v17617_data * v17539_data)) + (v17758_data * v17680_data)) + (v17899_data * v17821_data)) + (v18040_data * v17962_data);
          double v18064_p = (((((((v16889_p + (v17053_data * v16933_data)) + (v17194_data * v17121_data)) + (v17335_data * v17262_data)) + (v17476_data * v17403_data)) + (v17617_data * v17544_data)) + (v17758_data * v17685_data)) + (v17899_data * v17826_data)) + (v18040_data * v17967_data);
          double v18069_p = (((((((v16894_p + (v17053_data * v16938_data)) + (v17194_data * v17126_data)) + (v17335_data * v17267_data)) + (v17476_data * v17408_data)) + (v17617_data * v17549_data)) + (v17758_data * v17690_data)) + (v17899_data * v17831_data)) + (v18040_data * v17972_data);
          double v18074_p = (((((((v16899_p + (v17053_data * v16943_data)) + (v17194_data * v17131_data)) + (v17335_data * v17272_data)) + (v17476_data * v17413_data)) + (v17617_data * v17554_data)) + (v17758_data * v17695_data)) + (v17899_data * v17836_data)) + (v18040_data * v17977_data);
          double v18079_p = (((((((v16904_p + (v17053_data * v16948_data)) + (v17194_data * v17136_data)) + (v17335_data * v17277_data)) + (v17476_data * v17418_data)) + (v17617_data * v17559_data)) + (v17758_data * v17700_data)) + (v17899_data * v17841_data)) + (v18040_data * v17982_data);
          double v18084_p = (((((((v16909_p + (v17053_data * v16953_data)) + (v17194_data * v17141_data)) + (v17335_data * v17282_data)) + (v17476_data * v17423_data)) + (v17617_data * v17564_data)) + (v17758_data * v17705_data)) + (v17899_data * v17846_data)) + (v18040_data * v17987_data);
          double v18087_data = v2809_g ? (glb_m7[(v1344_lead + 1304)]) : (0.0);
          double v18088_data = s5[27];
          ir11[1] = (((((((((v16775_p + (v16959_data * v16913_data)) + (v17100_data * v17101_data)) + (v17241_data * v17242_data)) + (v17382_data * v17383_data)) + (v17523_data * v17524_data)) + (v17664_data * v17665_data)) + (v17805_data * v17806_data)) + (v17946_data * v17947_data)) + (v18087_data * v18088_data));
          double v18093_data = s5[62];
          ir11[5] = (v17955_p + (v18087_data * v18093_data));
          double v18098_data = s5[97];
          ir11[9] = (v17960_p + (v18087_data * v18098_data));
          double v18103_data = s5[132];
          ir11[13] = (v17965_p + (v18087_data * v18103_data));
          double v18108_data = s5[167];
          ir11[17] = (v17970_p + (v18087_data * v18108_data));
          double v18113_data = s5[202];
          ir11[21] = (v17975_p + (v18087_data * v18113_data));
          double v18118_data = s5[237];
          ir11[25] = (v17980_p + (v18087_data * v18118_data));
          double v18123_data = s5[272];
          ir11[29] = (v17985_p + (v18087_data * v18123_data));
          double v18128_data = s5[307];
          ir11[33] = (v17990_p + (v18087_data * v18128_data));
          double v18134_data = glb_m7[(v1393_lead + 1304)];
          double v18181_data = v18902_g ? (glb_m7[(v1442_lead + 1304)]) : (0.0);
          double v18228_data = glb_m7[(v1393_lead + 1328)];
          double v18229_data = s5[28];
          double v18232_p = (v17997_p + (v18134_data * v18088_data)) + (v18228_data * v18229_data);
          double v18234_data = s5[63];
          double v18237_p = (v18002_p + (v18134_data * v18093_data)) + (v18228_data * v18234_data);
          double v18239_data = s5[98];
          double v18242_p = (v18007_p + (v18134_data * v18098_data)) + (v18228_data * v18239_data);
          double v18244_data = s5[133];
          double v18247_p = (v18012_p + (v18134_data * v18103_data)) + (v18228_data * v18244_data);
          double v18249_data = s5[168];
          double v18252_p = (v18017_p + (v18134_data * v18108_data)) + (v18228_data * v18249_data);
          double v18254_data = s5[203];
          double v18257_p = (v18022_p + (v18134_data * v18113_data)) + (v18228_data * v18254_data);
          double v18259_data = s5[238];
          double v18262_p = (v18027_p + (v18134_data * v18118_data)) + (v18228_data * v18259_data);
          double v18264_data = s5[273];
          double v18267_p = (v18032_p + (v18134_data * v18123_data)) + (v18228_data * v18264_data);
          double v18269_data = s5[308];
          double v18272_p = (v18037_p + (v18134_data * v18128_data)) + (v18228_data * v18269_data);
          double v18275_data = v18902_g ? (glb_m7[(v1442_lead + 1328)]) : (0.0);
          double v18279_p = (v18044_p + (v18181_data * v18088_data)) + (v18275_data * v18229_data);
          double v18284_p = (v18049_p + (v18181_data * v18093_data)) + (v18275_data * v18234_data);
          double v18289_p = (v18054_p + (v18181_data * v18098_data)) + (v18275_data * v18239_data);
          double v18294_p = (v18059_p + (v18181_data * v18103_data)) + (v18275_data * v18244_data);
          double v18299_p = (v18064_p + (v18181_data * v18108_data)) + (v18275_data * v18249_data);
          double v18304_p = (v18069_p + (v18181_data * v18113_data)) + (v18275_data * v18254_data);
          double v18309_p = (v18074_p + (v18181_data * v18118_data)) + (v18275_data * v18259_data);
          double v18314_p = (v18079_p + (v18181_data * v18123_data)) + (v18275_data * v18264_data);
          double v18319_p = (v18084_p + (v18181_data * v18128_data)) + (v18275_data * v18269_data);
          if (v18902_g) {
            #pragma unroll
            for (int32_t v18903_i1 = 0; v18903_i1 < 9; ++v18903_i1) {
              double v18908_data = glb_m2[(v1442_lead + (v18903_i1 * 56))];
              r10[(3 + (v18903_i1 * 4))] = v18908_data;
            }
          }
          double v18322_data = glb_m7[(v1393_lead + 1352)];
          double v18323_data = s5[29];
          double v18328_data = s5[64];
          double v18333_data = s5[99];
          double v18338_data = s5[134];
          double v18343_data = s5[169];
          double v18348_data = s5[204];
          double v18353_data = s5[239];
          double v18358_data = s5[274];
          double v18363_data = s5[309];
          double v18369_data = v18902_g ? (glb_m7[(v1442_lead + 1352)]) : (0.0);
          double v18416_data = glb_m7[(v1393_lead + 1376)];
          double v18417_data = s5[30];
          double v18422_data = s5[65];
          double v18427_data = s5[100];
          double v18432_data = s5[135];
          double v18437_data = s5[170];
          double v18442_data = s5[205];
          double v18447_data = s5[240];
          double v18452_data = s5[275];
          double v18457_data = s5[310];
          double v18463_data = v18902_g ? (glb_m7[(v1442_lead + 1376)]) : (0.0);
          double v18510_data = glb_m7[(v1393_lead + 1400)];
          double v18511_data = s5[31];
          double v18516_data = s5[66];
          double v18521_data = s5[101];
          double v18526_data = s5[136];
          double v18531_data = s5[171];
          double v18536_data = s5[206];
          double v18541_data = s5[241];
          double v18546_data = s5[276];
          double v18551_data = s5[311];
          double v18557_data = v18902_g ? (glb_m7[(v1442_lead + 1400)]) : (0.0);
          double v18604_data = v2809_g ? (glb_m7[(v1393_lead + 1416)]) : (0.0);
          double v18605_data = s5[32];
          double v18610_data = s5[67];
          double v18615_data = s5[102];
          double v18620_data = s5[137];
          double v18625_data = s5[172];
          double v18630_data = s5[207];
          double v18635_data = s5[242];
          double v18640_data = s5[277];
          double v18645_data = s5[312];
          double v18651_data = v18902_g ? (glb_m7[(v1442_lead + 1416)]) : (0.0);
          double v18698_data = v2809_g ? (glb_m7[(v1393_lead + 1432)]) : (0.0);
          double v18699_data = s5[33];
          double v18704_data = s5[68];
          double v18707_p = ((((v18237_p + (v18322_data * v18328_data)) + (v18416_data * v18422_data)) + (v18510_data * v18516_data)) + (v18604_data * v18610_data)) + (v18698_data * v18704_data);
          double v18709_data = s5[103];
          double v18712_p = ((((v18242_p + (v18322_data * v18333_data)) + (v18416_data * v18427_data)) + (v18510_data * v18521_data)) + (v18604_data * v18615_data)) + (v18698_data * v18709_data);
          double v18714_data = s5[138];
          double v18717_p = ((((v18247_p + (v18322_data * v18338_data)) + (v18416_data * v18432_data)) + (v18510_data * v18526_data)) + (v18604_data * v18620_data)) + (v18698_data * v18714_data);
          double v18719_data = s5[173];
          double v18722_p = ((((v18252_p + (v18322_data * v18343_data)) + (v18416_data * v18437_data)) + (v18510_data * v18531_data)) + (v18604_data * v18625_data)) + (v18698_data * v18719_data);
          double v18724_data = s5[208];
          double v18727_p = ((((v18257_p + (v18322_data * v18348_data)) + (v18416_data * v18442_data)) + (v18510_data * v18536_data)) + (v18604_data * v18630_data)) + (v18698_data * v18724_data);
          double v18729_data = s5[243];
          double v18732_p = ((((v18262_p + (v18322_data * v18353_data)) + (v18416_data * v18447_data)) + (v18510_data * v18541_data)) + (v18604_data * v18635_data)) + (v18698_data * v18729_data);
          double v18734_data = s5[278];
          double v18737_p = ((((v18267_p + (v18322_data * v18358_data)) + (v18416_data * v18452_data)) + (v18510_data * v18546_data)) + (v18604_data * v18640_data)) + (v18698_data * v18734_data);
          double v18739_data = s5[313];
          double v18742_p = ((((v18272_p + (v18322_data * v18363_data)) + (v18416_data * v18457_data)) + (v18510_data * v18551_data)) + (v18604_data * v18645_data)) + (v18698_data * v18739_data);
          double v18745_data = v18902_g ? (glb_m7[(v1442_lead + 1432)]) : (0.0);
          double v18749_p = ((((v18279_p + (v18369_data * v18323_data)) + (v18463_data * v18417_data)) + (v18557_data * v18511_data)) + (v18651_data * v18605_data)) + (v18745_data * v18699_data);
          double v18754_p = ((((v18284_p + (v18369_data * v18328_data)) + (v18463_data * v18422_data)) + (v18557_data * v18516_data)) + (v18651_data * v18610_data)) + (v18745_data * v18704_data);
          double v18759_p = ((((v18289_p + (v18369_data * v18333_data)) + (v18463_data * v18427_data)) + (v18557_data * v18521_data)) + (v18651_data * v18615_data)) + (v18745_data * v18709_data);
          double v18764_p = ((((v18294_p + (v18369_data * v18338_data)) + (v18463_data * v18432_data)) + (v18557_data * v18526_data)) + (v18651_data * v18620_data)) + (v18745_data * v18714_data);
          double v18769_p = ((((v18299_p + (v18369_data * v18343_data)) + (v18463_data * v18437_data)) + (v18557_data * v18531_data)) + (v18651_data * v18625_data)) + (v18745_data * v18719_data);
          double v18774_p = ((((v18304_p + (v18369_data * v18348_data)) + (v18463_data * v18442_data)) + (v18557_data * v18536_data)) + (v18651_data * v18630_data)) + (v18745_data * v18724_data);
          double v18779_p = ((((v18309_p + (v18369_data * v18353_data)) + (v18463_data * v18447_data)) + (v18557_data * v18541_data)) + (v18651_data * v18635_data)) + (v18745_data * v18729_data);
          double v18784_p = ((((v18314_p + (v18369_data * v18358_data)) + (v18463_data * v18452_data)) + (v18557_data * v18546_data)) + (v18651_data * v18640_data)) + (v18745_data * v18734_data);
          double v18789_p = ((((v18319_p + (v18369_data * v18363_data)) + (v18463_data * v18457_data)) + (v18557_data * v18551_data)) + (v18651_data * v18645_data)) + (v18745_data * v18739_data);
          double v18792_data = v2809_g ? (glb_m7[(v1393_lead + 1448)]) : (0.0);
          double v18793_data = s5[34];
          ir11[2] = ((((((v18232_p + (v18322_data * v18323_data)) + (v18416_data * v18417_data)) + (v18510_data * v18511_data)) + (v18604_data * v18605_data)) + (v18698_data * v18699_data)) + (v18792_data * v18793_data));
          double v18798_data = s5[69];
          ir11[6] = (v18707_p + (v18792_data * v18798_data));
          double v18803_data = s5[104];
          ir11[10] = (v18712_p + (v18792_data * v18803_data));
          double v18808_data = s5[139];
          ir11[14] = (v18717_p + (v18792_data * v18808_data));
          double v18813_data = s5[174];
          ir11[18] = (v18722_p + (v18792_data * v18813_data));
          double v18818_data = s5[209];
          double v18821_p = v18727_p + (v18792_data * v18818_data);
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v19020_i0 = 0; v19020_i0 < 2; ++v19020_i0) {
            int32_t v19022_lead = v18980_lead + (v19020_i0 * 16);
            #pragma unroll
            for (int32_t v19023_i1 = 0; v19023_i1 < 9; ++v19023_i1) {
              double v19026_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v19022_lead + (v19023_i1 * 56))])) : (0.0);
              r0[(v19020_i0 + (v19023_i1 * 3))] = v19026_data;
            }
          }
          if (v18981_g) {
            #pragma unroll
            for (int32_t v19031_i1 = 0; v19031_i1 < 9; ++v19031_i1) {
              double v19034_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v1393_lead + (v19031_i1 * 56))])) : (0.0);
              r0[(2 + (v19031_i1 * 3))] = v19034_data;
            }
          }
          ir11[22] = v18821_p;
          double v18823_data = s5[244];
          ir11[26] = (v18732_p + (v18792_data * v18823_data));
          double v18828_data = s5[279];
          ir11[30] = (v18737_p + (v18792_data * v18828_data));
          double v18833_data = s5[314];
          ir11[34] = (v18742_p + (v18792_data * v18833_data));
          double v18839_data = v18902_g ? (glb_m7[(v1442_lead + 1448)]) : (0.0);
          ir11[3] = (v18749_p + (v18839_data * v18793_data));
          ir11[7] = (v18754_p + (v18839_data * v18798_data));
          ir11[11] = (v18759_p + (v18839_data * v18803_data));
          ir11[15] = (v18764_p + (v18839_data * v18808_data));
          ir11[19] = (v18769_p + (v18839_data * v18813_data));
          ir11[23] = (v18774_p + (v18839_data * v18818_data));
          ir11[27] = (v18779_p + (v18839_data * v18823_data));
          ir11[31] = (v18784_p + (v18839_data * v18828_data));
          ir11[35] = (v18789_p + (v18839_data * v18833_data));
          // r11 = ir11 + r10
          if (v5629_g) {
            #pragma unroll
            for (int32_t v18911_n1 = 0; v18911_n1 < 9; ++v18911_n1) {
              int32_t v18912_a = v18911_n1 * 4;
              double v18914_data = ir11[v18912_a];
              double v18915_data = r7[v18912_a];
              r11[v18912_a] = (v18915_data + v18914_data);
            }
          }
          #pragma unroll
          for (int32_t v18917_n0 = 1; v18917_n0 < 3; ++v18917_n0) {
            #pragma unroll
            for (int32_t v18918_n1 = 0; v18918_n1 < 9; ++v18918_n1) {
              int32_t v18919_a = v18918_n1 * 4;
              int32_t v18920_a = v18917_n0 + v18919_a;
              double v18921_data = ir11[v18920_a];
              double v18922_data = r7[(v18917_n0 + v18919_a)];
              r11[v18920_a] = (v18922_data + v18921_data);
            }
          }
          if (v18902_g) {
            #pragma unroll
            for (int32_t v18924_n1 = 0; v18924_n1 < 9; ++v18924_n1) {
              int32_t v18926_a = 3 + (v18924_n1 * 4);
              double v18927_data = ir11[v18926_a];
              double v18928_data = r10[v18926_a];
              r11[v18926_a] = (v18928_data + v18927_data);
            }
          }
          // glb_m2 = store{r>g}(r11);
          if (v5629_g) {
            #pragma unroll
            for (int32_t v18930_i1 = 0; v18930_i1 < 9; ++v18930_i1) {
              double v18933_data = r11[(v18930_i1 * 4)];
              glb_m2[(v18980_lead + (v18930_i1 * 56))] = v18933_data;
            }
          }
          #pragma unroll
          for (int32_t v18938_i0 = 1; v18938_i0 < 3; ++v18938_i0) {
            int32_t v18944_lead = v18980_lead + (v18938_i0 * 16);
            #pragma unroll
            for (int32_t v18939_i1 = 0; v18939_i1 < 9; ++v18939_i1) {
              double v18942_data = r11[(v18938_i0 + (v18939_i1 * 4))];
              glb_m2[(v18944_lead + (v18939_i1 * 56))] = v18942_data;
            }
          }
          if (v18902_g) {
            #pragma unroll
            for (int32_t v18947_i1 = 0; v18947_i1 < 9; ++v18947_i1) {
              double v18950_data = r11[(3 + (v18947_i1 * 4))];
              glb_m2[(v1442_lead + (v18947_i1 * 56))] = v18950_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v18986_i0 = 0; v18986_i0 < 2; ++v18986_i0) {
            int32_t v18988_lead = v18980_lead + (v18986_i0 * 16);
            #pragma unroll
            for (int32_t v18989_i1 = 0; v18989_i1 < 9; ++v18989_i1) {
              double v18992_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v18988_lead + (v18989_i1 * 56))])) : (0.0);
              r0[(v18986_i0 + (v18989_i1 * 3))] = v18992_data;
            }
          }
          if (v18981_g) {
            int32_t v18996_lead = v18980_lead + 32_i32;
            #pragma unroll
            for (int32_t v18997_i1 = 0; v18997_i1 < 9; ++v18997_i1) {
              double v19000_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v18996_lead + (v18997_i1 * 56))])) : (0.0);
              r0[(2 + (v18997_i1 * 3))] = v19000_data;
            }
          }
        }
        // s0 = load{g>s}(wrap_glb_m1[0, 1])
        #pragma unroll
        for (int32_t v19046_i = 0; v19046_i < 5; ++v19046_i) {
          int32_t v19048_a = threadIdx.x + (v19046_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v19048_a], &wrap_glb_m1[v19048_a], 8);
          }
        }
        if (v19038_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &wrap_glb_m1[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
        }
        __pipeline_commit();
        v18974_flagWord = v18975_flagWordNext;
        v18975_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

