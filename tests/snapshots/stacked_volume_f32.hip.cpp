// === base name ===
kernel_5fa4229a4c0ba53a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5fa4229a4c0ba53a = {{16, 16, 1}, 16, 55, 1, 16, 45312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5fa4229a4c0ba53a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5fa4229a4c0ba53a(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, const float * m3, const float ** m4, size_t m4_extraOffset, const float * m5, const float ** m6, size_t m6_extraOffset, const float * m7, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5fa4229a4c0ba53a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5fa4229a4c0ba53a, block.x * block.y * block.z, 11328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (11328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5fa4229a4c0ba53a, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (11328 * sizeof(float)));
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
  config.sharedMemBytes = 11328 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5fa4229a4c0ba53a(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, const float * m3, const float ** m4, size_t m4_extraOffset, const float * m5, const float ** m6, size_t m6_extraOffset, const float * m7, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5fa4229a4c0ba53a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5fa4229a4c0ba53a), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  const float *const * m0Arg = (const float *const *)m0;
  const float *const * m1Arg = (const float *const *)m1;
  float *const * m2Arg = (float *const *)m2;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m3;
  const float *const * m4Arg = (const float *const *)m4;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m5;
  const float *const * m6Arg = (const float *const *)m6;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m7;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5fa4229a4c0ba53a, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, m5Arg, m6Arg, m6_extraOffset, m7Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5fa4229a4c0ba53a(const float *const * __restrict__ m0, size_t m0_extraOffset, const float *const * __restrict__ m1, size_t m1_extraOffset, float *const * __restrict__ m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, const float *const * __restrict__ m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m5, const float *const * __restrict__ m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m7, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (55 active) x 16 per block = block 16x16x1, 45312 B shared, occupancy grid
    // operands:
    //   m0 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m2 56×9(56×9) {0..56}×{0..9} pointer_based
    //   m3 56×56(56×35) {0..56}×{0..35} none
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m5 56×56(56×35) {0..56}×{0..35} none
    //   m6 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m7 56×56(56×35) {0..56}×{0..35} none
    // operations:
    //   t0[i,j]@{0..35}×{0..9} = m0[i,k]@{0..35}×{0..9} × m1[k,j]
    //   m2[i,j]@{1..54}×{0..9} += m3[i,k]@{1..54}×{0..35} × t0[k,j]@{0..35}×{0..9}
    //   t1[i,j]@{0..35}×{0..9} = m0[i,k]@{0..35}×{0..9} × m4[k,j]
    //   m2[i,j]@{1..55}×{0..9} += m5[i,k]@{1..55}×{0..35} × t1[k,j]@{0..35}×{0..9}
    //   t2[i,j]@{0..35}×{0..9} = m0[i,k]@{0..35}×{0..9} × m6[k,j]
    //   m2[i,j]@{1..56}×{0..9} += m7[i,k]@{1..56}×{0..35} × t2[k,j]@{0..35}×{0..9}
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":55,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":11328}],"shared_bytes":45312,"shared_elements":11328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[56,9]],"name":"m0","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"pointer_based","alias":"star0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[56,9]],"name":"m2","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"none","alias":"kDivM0","bbox":[[0,0],[56,35]],"name":"m3","ordered":false,"parts":1,"shape":[56,56],"variant":false},{"addressing":"pointer_based","alias":"star1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"kDivM1","bbox":[[0,0],[56,35]],"name":"m5","ordered":false,"parts":1,"shape":[56,56],"variant":false},{"addressing":"pointer_based","alias":"star2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"kDivM2","bbox":[[0,0],[56,35]],"name":"m7","ordered":false,"parts":1,"shape":[56,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[54,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[54,35]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[55,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[55,35]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[56,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[56,35]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[336 * threadIdx.y + 5952];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[0];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      tensorforge::VectorT<float, 4> v2436_ld = *(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m3[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<float, 4>*)&glb_m3[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v2436_ld;
      tensorforge::VectorT<float, 2> v2437_ld = *(tensorforge::VectorRelaxedT<float, 2>*)&ptr_glb_m3[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
      *(tensorforge::VectorT<float, 2>*)&glb_m3[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v2437_ld;
      float v2438_ld = ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536];
      glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536] = v2438_ld;
      bool v2443_g = (threadIdx.x + (threadIdx.y * 16)) < 168;
      if (v2443_g) {
        float v2444_ld = ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1792];
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1792] = v2444_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m5[0];
      float * __restrict__ glb_m5 = &totalShrMem[1984];
      // glb_m5 = load{g>s}(ptr_glb_m5[0, 1])
      tensorforge::VectorT<float, 4> v2447_ld = *(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m5[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<float, 4>*)&glb_m5[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v2447_ld;
      tensorforge::VectorT<float, 2> v2448_ld = *(tensorforge::VectorRelaxedT<float, 2>*)&ptr_glb_m5[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
      *(tensorforge::VectorT<float, 2>*)&glb_m5[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v2448_ld;
      float v2449_ld = ptr_glb_m5[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536];
      glb_m5[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536] = v2449_ld;
      if (v2443_g) {
        float v2450_ld = ptr_glb_m5[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1792];
        glb_m5[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1792] = v2450_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m7[0];
      float * __restrict__ glb_m7 = &totalShrMem[3968];
      // glb_m7 = load{g>s}(ptr_glb_m7[0, 1])
      tensorforge::VectorT<float, 4> v2453_ld = *(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m7[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<float, 4>*)&glb_m7[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v2453_ld;
      tensorforge::VectorT<float, 2> v2454_ld = *(tensorforge::VectorRelaxedT<float, 2>*)&ptr_glb_m7[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
      *(tensorforge::VectorT<float, 2>*)&glb_m7[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v2454_ld;
      float v2455_ld = ptr_glb_m7[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536];
      glb_m7[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536] = v2455_ld;
      if (v2443_g) {
        float v2456_ld = ptr_glb_m7[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1792];
        glb_m7[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1792] = v2456_ld;
      }
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      size_t v2461_batchIdLane0 = threadIdx.y % 4;
      size_t v2464_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v2483_lead = threadIdx.x % 16;
      bool v2493_g = v2483_lead < 3;
      bool v2508_g = v2483_lead < 9;
      int32_t v2724_m = threadIdx.y % 4;
      int32_t v2736_sh = (threadIdx.x % 4) * 35;
      int32_t v2738_sh = ((threadIdx.x / 4) - v2724_m) * 336;
      int32_t v2739_sh = (v2724_m + v2736_sh) + v2738_sh;
      int32_t v2743_a = v2739_sh + 140;
      int32_t v2745_r = v2724_m * 56;
      int32_t v2748_a = v2483_lead + v2745_r;
      int32_t v2754_lead = v2483_lead + 16_i32;
      int32_t v2755_a = v2754_lead + v2745_r;
      int32_t v2761_lead = v2483_lead + 32_i32;
      int32_t v2762_a = v2761_lead + v2745_r;
      int32_t v2768_lead = v2483_lead + 48_i32;
      int32_t v2769_a = v2768_lead + v2745_r;
      int32_t v2774_a = v2739_sh + 4;
      int32_t v2777_a = v2774_a + 140;
      int32_t v2779_a = v2748_a + 224;
      int32_t v2783_a = v2755_a + 224;
      int32_t v2787_a = v2762_a + 224;
      int32_t v2791_a = v2769_a + 224;
      int32_t v2795_a = v2739_sh + 8;
      int32_t v2798_a = v2795_a + 140;
      int32_t v2800_a = v2748_a + 448;
      int32_t v2804_a = v2755_a + 448;
      int32_t v2808_a = v2762_a + 448;
      int32_t v2812_a = v2769_a + 448;
      int32_t v2816_a = v2739_sh + 12;
      int32_t v2819_a = v2816_a + 140;
      int32_t v2821_a = v2755_a + 672;
      int32_t v2825_a = v2762_a + 672;
      int32_t v2829_a = v2769_a + 672;
      int32_t v2833_a = v2739_sh + 16;
      int32_t v2836_a = v2833_a + 140;
      int32_t v2838_a = v2755_a + 896;
      int32_t v2842_a = v2762_a + 896;
      int32_t v2846_a = v2769_a + 896;
      int32_t v2850_a = v2739_sh + 20;
      int32_t v2853_a = v2850_a + 140;
      int32_t v2855_a = v2762_a + 1120;
      int32_t v2859_a = v2769_a + 1120;
      int32_t v2863_a = v2739_sh + 24;
      int32_t v2866_a = v2863_a + 140;
      int32_t v2868_a = v2762_a + 1344;
      int32_t v2872_a = v2769_a + 1344;
      int32_t v2876_a = v2739_sh + 28;
      int32_t v2879_a = v2876_a + 140;
      int32_t v2881_a = v2769_a + 1568;
      int32_t v2933_a = v2483_lead + 56;
      int32_t v2935_a = v2754_lead + 56;
      int32_t v2937_a = v2761_lead + 56;
      int32_t v2939_a = v2768_lead + 56;
      int32_t v2941_a = v2483_lead + 112;
      int32_t v2943_a = v2754_lead + 112;
      int32_t v2945_a = v2761_lead + 112;
      int32_t v2947_a = v2768_lead + 112;
      int32_t v2949_a = v2483_lead + 168;
      int32_t v2951_a = v2754_lead + 168;
      int32_t v2953_a = v2761_lead + 168;
      int32_t v2955_a = v2768_lead + 168;
      int32_t v2957_a = v2483_lead + 224;
      int32_t v2959_a = v2754_lead + 224;
      int32_t v2961_a = v2761_lead + 224;
      int32_t v2963_a = v2768_lead + 224;
      int32_t v2965_a = v2483_lead + 280;
      int32_t v2967_a = v2754_lead + 280;
      int32_t v2969_a = v2761_lead + 280;
      int32_t v2971_a = v2768_lead + 280;
      int32_t v2973_a = v2483_lead + 336;
      int32_t v2975_a = v2754_lead + 336;
      int32_t v2977_a = v2761_lead + 336;
      int32_t v2979_a = v2768_lead + 336;
      int32_t v2981_a = v2483_lead + 392;
      int32_t v2983_a = v2754_lead + 392;
      int32_t v2985_a = v2761_lead + 392;
      int32_t v2987_a = v2768_lead + 392;
      int32_t v2989_a = v2483_lead + 448;
      int32_t v2991_a = v2754_lead + 448;
      int32_t v2993_a = v2761_lead + 448;
      int32_t v2995_a = v2768_lead + 448;
      int32_t v2997_a = v2483_lead + 504;
      int32_t v2999_a = v2754_lead + 504;
      int32_t v3001_a = v2761_lead + 504;
      int32_t v3003_a = v2768_lead + 504;
      int32_t v3005_a = v2483_lead + 560;
      int32_t v3007_a = v2754_lead + 560;
      int32_t v3009_a = v2761_lead + 560;
      int32_t v3011_a = v2768_lead + 560;
      int32_t v3013_a = v2483_lead + 616;
      int32_t v3015_a = v2754_lead + 616;
      int32_t v3017_a = v2761_lead + 616;
      int32_t v3019_a = v2768_lead + 616;
      int32_t v3021_a = v2754_lead + 672;
      int32_t v3023_a = v2761_lead + 672;
      int32_t v3025_a = v2768_lead + 672;
      int32_t v3027_a = v2754_lead + 728;
      int32_t v3029_a = v2761_lead + 728;
      int32_t v3031_a = v2768_lead + 728;
      int32_t v3033_a = v2754_lead + 784;
      int32_t v3035_a = v2761_lead + 784;
      int32_t v3037_a = v2768_lead + 784;
      int32_t v3039_a = v2754_lead + 840;
      int32_t v3041_a = v2761_lead + 840;
      int32_t v3043_a = v2768_lead + 840;
      int32_t v3045_a = v2754_lead + 896;
      int32_t v3047_a = v2761_lead + 896;
      int32_t v3049_a = v2768_lead + 896;
      int32_t v3051_a = v2754_lead + 952;
      int32_t v3053_a = v2761_lead + 952;
      int32_t v3055_a = v2768_lead + 952;
      int32_t v3057_a = v2754_lead + 1008;
      int32_t v3059_a = v2761_lead + 1008;
      int32_t v3061_a = v2768_lead + 1008;
      int32_t v3063_a = v2754_lead + 1064;
      int32_t v3065_a = v2761_lead + 1064;
      int32_t v3067_a = v2768_lead + 1064;
      int32_t v3069_a = v2761_lead + 1120;
      int32_t v3071_a = v2768_lead + 1120;
      int32_t v3073_a = v2761_lead + 1176;
      int32_t v3075_a = v2768_lead + 1176;
      int32_t v3077_a = v2761_lead + 1232;
      int32_t v3079_a = v2768_lead + 1232;
      int32_t v3081_a = v2761_lead + 1288;
      int32_t v3083_a = v2768_lead + 1288;
      int32_t v3085_a = v2761_lead + 1344;
      int32_t v3087_a = v2768_lead + 1344;
      int32_t v3089_a = v2761_lead + 1400;
      int32_t v3091_a = v2768_lead + 1400;
      int32_t v3093_a = v2761_lead + 1456;
      int32_t v3095_a = v2768_lead + 1456;
      int32_t v3097_a = v2761_lead + 1512;
      int32_t v3099_a = v2768_lead + 1512;
      int32_t v3101_a = v2768_lead + 1568;
      int32_t v3103_a = v2768_lead + 1624;
      int32_t v3105_a = v2768_lead + 1680;
      int32_t v3107_a = v2768_lead + 1736;
      bool v3117_g = v2483_lead >= 1;
      bool v3135_g = v2483_lead < 6;
      int32_t v3525_a = v2748_a + 672;
      int32_t v3563_a = v2755_a + 1120;
      int32_t v3593_a = v2762_a + 1568;
      int32_t v3607_a = ((((min((max((v2724_m + 32), 0)), 34)) - 32) + v2736_sh) + v2738_sh) + 32;
      bool v3610_g = v2724_m == 2;
      bool v3612_g = v2724_m == 1;
      bool v3614_g = v2724_m == 0;
      int32_t v3616_a = v3607_a + 140;
      int32_t v3621_a = v2768_lead + 1792;
      int32_t v3623_a = v2768_lead + 1848;
      int32_t v3625_a = v2768_lead + 1904;
      int32_t v3768_a = v2483_lead + 672;
      int32_t v3776_a = v2483_lead + 728;
      int32_t v3784_a = v2483_lead + 784;
      int32_t v3792_a = v2483_lead + 840;
      int32_t v3824_a = v2754_lead + 1120;
      int32_t v3830_a = v2754_lead + 1176;
      int32_t v3836_a = v2754_lead + 1232;
      int32_t v3842_a = v2754_lead + 1288;
      int32_t v3864_a = v2761_lead + 1568;
      int32_t v3868_a = v2761_lead + 1624;
      int32_t v3872_a = v2761_lead + 1680;
      int32_t v3876_a = v2761_lead + 1736;
      bool v3911_g = v2483_lead < 7;
      int32_t v4322_a = v2748_a + 896;
      int32_t v4360_a = v2755_a + 1344;
      int32_t v4405_a = v2761_lead + 1792;
      int32_t v4407_a = v2761_lead + 1848;
      int32_t v4409_a = v2761_lead + 1904;
      int32_t v4595_a = v2483_lead + 896;
      int32_t v4603_a = v2483_lead + 952;
      int32_t v4611_a = v2483_lead + 1008;
      int32_t v4619_a = v2483_lead + 1064;
      int32_t v4651_a = v2754_lead + 1344;
      int32_t v4657_a = v2754_lead + 1400;
      int32_t v4663_a = v2754_lead + 1456;
      int32_t v4669_a = v2754_lead + 1512;
      bool v4728_g = v2483_lead < 8;
      float r0[27]{};
      size_t v4798_row = v2464_batchIdGroup0First + v2461_batchIdLane0;
      const bool peel_batchIdActive0 = v4798_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v4798_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[(peel_batchIdActive0 ? v4798_row : v2464_batchIdGroup0First)][0 + m0_extraOffset];
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v4806_i0 = 0; v4806_i0 < 2; ++v4806_i0) {
        int32_t v4808_lead = v2483_lead + (v4806_i0 * 16);
        #pragma unroll
        for (int32_t v4809_i1 = 0; v4809_i1 < 9; ++v4809_i1) {
          float v4812_data = __builtin_nontemporal_load(&peel_glb_m0[(v4808_lead + (v4809_i1 * 56))]);
          r0[(v4806_i0 + (v4809_i1 * 3))] = v4812_data;
        }
      }
      if (v2493_g) {
        #pragma unroll
        for (int32_t v4815_i1 = 0; v4815_i1 < 9; ++v4815_i1) {
          float v4818_data = __builtin_nontemporal_load(&peel_glb_m0[(v2761_lead + (v4815_i1 * 56))]);
          r0[(2 + (v4815_i1 * 3))] = v4818_data;
        }
      }
      for (size_t v2465_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v2465_batchIdGroup0 < numElements0; v2465_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v4795_groupNextAhead = v2465_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v4797_groupNext = (v4795_groupNextAhead < numElements0) ? v4795_groupNextAhead : v2465_batchIdGroup0;
        size_t v4802_row = v4797_groupNext + v2461_batchIdLane0;
        const bool wrap_batchIdActive0 = v4802_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v4802_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[(wrap_batchIdActive0 ? v4802_row : v4797_groupNext)][0 + m0_extraOffset];
        size_t v2466_row = v2465_batchIdGroup0 + v2461_batchIdLane0;
        const bool batchIdActive0 = v2466_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v2466_row]));
        size_t v2468_batchId0 = batchIdActive0 ? v2466_row : v2465_batchIdGroup0;
        size_t v2469_ahead1 = v2468_batchId0 + (gridDim.x * blockDim.y);
        size_t v2471_batchId1 = (v2469_ahead1 < numElements0) ? v2469_ahead1 : v2468_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v2468_batchId0][0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v2468_batchId0][0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v2468_batchId0][0 + m4_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v2468_batchId0][0 + m6_extraOffset];
        float r1[9]{};
        // r1 = load{g>r}(glb_m1);
        #pragma unroll
        for (int32_t v2503_i1 = 0; v2503_i1 < 9; ++v2503_i1) {
          float v2509_data = v2508_g ? (__builtin_nontemporal_load(&glb_m1[(v2483_lead + (v2503_i1 * 9))])) : (0.0f);
          r1[v2503_i1] = v2509_data;
        }
        float r2[27]{};
        // r2 = +(r0 * r1) + None
        // [(0, 35), (0, 9)] [(0, 9)]
        float v2512_data = r1[0];
        float v2513_data = r1[1];
        float v2514_data = r1[2];
        float v2515_data = r1[3];
        float v2516_tp{};
        float v2517_tp{};
        float v2518_tp{};
        float v2519_tp{};
        tensorforge::transpose4x4b32Fused(v2516_tp, v2517_tp, v2518_tp, v2519_tp, v2512_data, v2513_data, v2514_data, v2515_data);
        tensorforge::VectorT<float, 4> v2520_acc{};
        float v2521_data = r0[0];
        float v2522_data = r0[3];
        float v2523_data = r0[6];
        float v2524_data = r0[9];
        tensorforge::VectorT<float, 4> v2525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2521_data, v2520_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2517_tp, v2522_data, v2525_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2518_tp, v2523_data, v2526_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2519_tp, v2524_data, v2527_acc, 2, 0, 0);
        float v2529_data = r0[12];
        float v2530_data = r0[15];
        float v2531_data = r0[18];
        float v2532_data = r0[21];
        tensorforge::VectorT<float, 4> v2533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2529_data, v2528_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2517_tp, v2530_data, v2533_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2518_tp, v2531_data, v2534_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2519_tp, v2532_data, v2535_acc, 2, 1, 0);
        float v2537_data = r0[24];
        tensorforge::VectorT<float, 4> v2539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2537_data, v2536_acc, 2, 2, 0);
        r2[0] = (v2539_acc[0]);
        r2[3] = (v2539_acc[1]);
        r2[6] = (v2539_acc[2]);
        r2[9] = (v2539_acc[3]);
        tensorforge::VectorT<float, 4> v2544_acc{};
        float v2545_data = r0[1];
        float v2546_data = r0[4];
        float v2547_data = r0[7];
        float v2548_data = r0[10];
        tensorforge::VectorT<float, 4> v2549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2545_data, v2544_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2517_tp, v2546_data, v2549_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2518_tp, v2547_data, v2550_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2519_tp, v2548_data, v2551_acc, 2, 0, 0);
        float v2553_data = r0[13];
        float v2554_data = r0[16];
        float v2555_data = r0[19];
        float v2556_data = r0[22];
        tensorforge::VectorT<float, 4> v2557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2553_data, v2552_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2517_tp, v2554_data, v2557_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2518_tp, v2555_data, v2558_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2519_tp, v2556_data, v2559_acc, 2, 1, 0);
        float v2561_data = r0[25];
        tensorforge::VectorT<float, 4> v2563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2561_data, v2560_acc, 2, 2, 0);
        r2[1] = (v2563_acc[0]);
        r2[4] = (v2563_acc[1]);
        r2[7] = (v2563_acc[2]);
        r2[10] = (v2563_acc[3]);
        tensorforge::VectorT<float, 4> v2568_acc{};
        float v2569_data = r0[2];
        float v2570_data = r0[5];
        float v2571_data = r0[8];
        float v2572_data = r0[11];
        tensorforge::VectorT<float, 4> v2573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2569_data, v2568_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2517_tp, v2570_data, v2573_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2518_tp, v2571_data, v2574_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2519_tp, v2572_data, v2575_acc, 2, 0, 0);
        float v2577_data = r0[14];
        float v2578_data = r0[17];
        float v2579_data = r0[20];
        float v2580_data = r0[23];
        tensorforge::VectorT<float, 4> v2581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2577_data, v2576_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2517_tp, v2578_data, v2581_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2518_tp, v2579_data, v2582_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2519_tp, v2580_data, v2583_acc, 2, 1, 0);
        float v2585_data = r0[26];
        tensorforge::VectorT<float, 4> v2587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2516_tp, v2585_data, v2584_acc, 2, 2, 0);
        r2[2] = (v2587_acc[0]);
        r2[5] = (v2587_acc[1]);
        r2[8] = (v2587_acc[2]);
        r2[11] = (v2587_acc[3]);
        float v2592_data = r1[4];
        float v2593_data = r1[5];
        float v2594_data = r1[6];
        float v2595_data = r1[7];
        float v2596_tp{};
        float v2597_tp{};
        float v2598_tp{};
        float v2599_tp{};
        tensorforge::transpose4x4b32Fused(v2596_tp, v2597_tp, v2598_tp, v2599_tp, v2592_data, v2593_data, v2594_data, v2595_data);
        tensorforge::VectorT<float, 4> v2600_acc{};
        tensorforge::VectorT<float, 4> v2605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2521_data, v2600_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2597_tp, v2522_data, v2605_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2607_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2598_tp, v2523_data, v2606_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2599_tp, v2524_data, v2607_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2529_data, v2608_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2597_tp, v2530_data, v2613_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2598_tp, v2531_data, v2614_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2599_tp, v2532_data, v2615_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2537_data, v2616_acc, 2, 2, 0);
        r2[12] = (v2619_acc[0]);
        r2[15] = (v2619_acc[1]);
        r2[18] = (v2619_acc[2]);
        r2[21] = (v2619_acc[3]);
        tensorforge::VectorT<float, 4> v2624_acc{};
        tensorforge::VectorT<float, 4> v2629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2545_data, v2624_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2597_tp, v2546_data, v2629_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2631_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2598_tp, v2547_data, v2630_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2599_tp, v2548_data, v2631_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2553_data, v2632_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2597_tp, v2554_data, v2637_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2598_tp, v2555_data, v2638_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2599_tp, v2556_data, v2639_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2643_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2561_data, v2640_acc, 2, 2, 0);
        r2[13] = (v2643_acc[0]);
        r2[16] = (v2643_acc[1]);
        r2[19] = (v2643_acc[2]);
        r2[22] = (v2643_acc[3]);
        tensorforge::VectorT<float, 4> v2648_acc{};
        tensorforge::VectorT<float, 4> v2653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2569_data, v2648_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2597_tp, v2570_data, v2653_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2598_tp, v2571_data, v2654_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2599_tp, v2572_data, v2655_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v2661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2577_data, v2656_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2597_tp, v2578_data, v2661_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2598_tp, v2579_data, v2662_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2599_tp, v2580_data, v2663_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v2667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v2596_tp, v2585_data, v2664_acc, 2, 2, 0);
        r2[14] = (v2667_acc[0]);
        r2[17] = (v2667_acc[1]);
        r2[20] = (v2667_acc[2]);
        r2[23] = (v2667_acc[3]);
        float v2699_acc{};
        float v2700_acc{};
        float v2701_acc{};
        float v2702_data = r1[8];
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %3, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %30 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v2699_acc), "+&v"(v2700_acc), "+&v"(v2701_acc)
        : "v"(v2702_data), "v"(v2521_data), "v"(v2545_data), "v"(v2569_data), "v"(v2522_data), "v"(v2546_data), "v"(v2570_data), "v"(v2523_data), "v"(v2547_data), "v"(v2571_data), "v"(v2524_data), "v"(v2548_data), "v"(v2572_data), "v"(v2529_data), "v"(v2553_data), "v"(v2577_data), "v"(v2530_data), "v"(v2554_data), "v"(v2578_data), "v"(v2531_data), "v"(v2555_data), "v"(v2579_data), "v"(v2532_data), "v"(v2556_data), "v"(v2580_data), "v"(v2537_data), "v"(v2561_data), "v"(v2585_data)
        );
        r2[24] = v2699_acc;
        r2[25] = v2700_acc;
        r2[26] = v2701_acc;
        // r2 = 0 past (0, 35)
        // s0 = store{r>s}(localShrMem0, r2);
        #pragma unroll
        for (int32_t v2704_i0 = 0; v2704_i0 < 2; ++v2704_i0) {
          int32_t v2710_lead = v2483_lead + (v2704_i0 * 16);
          #pragma unroll
          for (int32_t v2705_i1 = 0; v2705_i1 < 9; ++v2705_i1) {
            float v2708_data = r2[(v2704_i0 + (v2705_i1 * 3))];
            s0[(v2710_lead + (v2705_i1 * 35))] = v2708_data;
          }
        }
        if (v2493_g) {
          #pragma unroll
          for (int32_t v2713_i1 = 0; v2713_i1 < 9; ++v2713_i1) {
            float v2716_data = r2[(2 + (v2713_i1 * 3))];
            s0[(v2761_lead + (v2713_i1 * 35))] = v2716_data;
          }
        }
        float r4[36]{};
        // ir4 = +(glb_m3 * s0)
        // [(1, 54), (0, 9)] [(0, 35)]
        float ir4[36]{};
        tensorforge::VectorT<float, 4> v2728_acc{};
        tensorforge::VectorT<float, 4> v2729_acc{};
        tensorforge::VectorT<float, 4> v2730_acc{};
        tensorforge::VectorT<float, 4> v2731_acc{};
        tensorforge::VectorT<float, 4> v2732_acc{};
        tensorforge::VectorT<float, 4> v2733_acc{};
        tensorforge::VectorT<float, 4> v2734_acc{};
        tensorforge::VectorT<float, 4> v2735_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        float v2742_data = s0[v2739_sh];
        float v2744_data = s0[v2743_a];
        float v2750_data = glb_m3[v2748_a];
        tensorforge::VectorT<float, 4> v2751_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2742_data, v2750_data, v2728_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2752_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2744_data, v2750_data, v2729_acc, 0, 0, 0);
        float v2757_data = glb_m3[v2755_a];
        tensorforge::VectorT<float, 4> v2758_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2742_data, v2757_data, v2730_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2759_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2744_data, v2757_data, v2731_acc, 0, 0, 0);
        float v2764_data = glb_m3[v2762_a];
        tensorforge::VectorT<float, 4> v2765_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2742_data, v2764_data, v2732_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2766_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2744_data, v2764_data, v2733_acc, 0, 0, 0);
        float v2771_data = glb_m3[v2769_a];
        tensorforge::VectorT<float, 4> v2772_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2742_data, v2771_data, v2734_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2773_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2744_data, v2771_data, v2735_acc, 0, 0, 0);
        float v2776_data = s0[v2774_a];
        float v2778_data = s0[v2777_a];
        float v2780_data = glb_m3[v2779_a];
        tensorforge::VectorT<float, 4> v2781_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2776_data, v2780_data, v2751_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2782_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2778_data, v2780_data, v2752_acc, 0, 0, 0);
        float v2784_data = glb_m3[v2783_a];
        tensorforge::VectorT<float, 4> v2785_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2776_data, v2784_data, v2758_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2786_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2778_data, v2784_data, v2759_acc, 0, 0, 0);
        float v2788_data = glb_m3[v2787_a];
        tensorforge::VectorT<float, 4> v2789_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2776_data, v2788_data, v2765_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2790_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2778_data, v2788_data, v2766_acc, 0, 0, 0);
        float v2792_data = glb_m3[v2791_a];
        tensorforge::VectorT<float, 4> v2793_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2776_data, v2792_data, v2772_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2794_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2778_data, v2792_data, v2773_acc, 0, 0, 0);
        float v2797_data = s0[v2795_a];
        float v2799_data = s0[v2798_a];
        float v2801_data = glb_m3[v2800_a];
        tensorforge::VectorT<float, 4> v2802_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2797_data, v2801_data, v2781_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2803_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2799_data, v2801_data, v2782_acc, 0, 0, 0);
        float v2805_data = glb_m3[v2804_a];
        tensorforge::VectorT<float, 4> v2806_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2797_data, v2805_data, v2785_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2807_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2799_data, v2805_data, v2786_acc, 0, 0, 0);
        float v2809_data = glb_m3[v2808_a];
        tensorforge::VectorT<float, 4> v2810_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2797_data, v2809_data, v2789_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2811_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2799_data, v2809_data, v2790_acc, 0, 0, 0);
        float v2813_data = glb_m3[v2812_a];
        tensorforge::VectorT<float, 4> v2814_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2797_data, v2813_data, v2793_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2815_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2799_data, v2813_data, v2794_acc, 0, 0, 0);
        float v2818_data = s0[v2816_a];
        float v2820_data = s0[v2819_a];
        float v2822_data = glb_m3[v2821_a];
        tensorforge::VectorT<float, 4> v2823_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2818_data, v2822_data, v2806_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2824_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2820_data, v2822_data, v2807_acc, 0, 0, 0);
        float v2826_data = glb_m3[v2825_a];
        tensorforge::VectorT<float, 4> v2827_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2818_data, v2826_data, v2810_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2828_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2820_data, v2826_data, v2811_acc, 0, 0, 0);
        float v2830_data = glb_m3[v2829_a];
        tensorforge::VectorT<float, 4> v2831_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2818_data, v2830_data, v2814_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2832_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2820_data, v2830_data, v2815_acc, 0, 0, 0);
        float v2835_data = s0[v2833_a];
        float v2837_data = s0[v2836_a];
        float v2839_data = glb_m3[v2838_a];
        tensorforge::VectorT<float, 4> v2840_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2835_data, v2839_data, v2823_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2841_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2837_data, v2839_data, v2824_acc, 0, 0, 0);
        float v2843_data = glb_m3[v2842_a];
        tensorforge::VectorT<float, 4> v2844_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2835_data, v2843_data, v2827_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2845_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2837_data, v2843_data, v2828_acc, 0, 0, 0);
        float v2847_data = glb_m3[v2846_a];
        tensorforge::VectorT<float, 4> v2848_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2835_data, v2847_data, v2831_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2849_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2837_data, v2847_data, v2832_acc, 0, 0, 0);
        float v2852_data = s0[v2850_a];
        float v2854_data = s0[v2853_a];
        float v2856_data = glb_m3[v2855_a];
        tensorforge::VectorT<float, 4> v2857_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2852_data, v2856_data, v2844_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2858_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2854_data, v2856_data, v2845_acc, 0, 0, 0);
        float v2860_data = glb_m3[v2859_a];
        tensorforge::VectorT<float, 4> v2861_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2852_data, v2860_data, v2848_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2862_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2854_data, v2860_data, v2849_acc, 0, 0, 0);
        float v2865_data = s0[v2863_a];
        float v2867_data = s0[v2866_a];
        float v2869_data = glb_m3[v2868_a];
        tensorforge::VectorT<float, 4> v2870_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2865_data, v2869_data, v2857_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2871_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2867_data, v2869_data, v2858_acc, 0, 0, 0);
        float v2873_data = glb_m3[v2872_a];
        tensorforge::VectorT<float, 4> v2874_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2865_data, v2873_data, v2861_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2875_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2867_data, v2873_data, v2862_acc, 0, 0, 0);
        float v2878_data = s0[v2876_a];
        float v2880_data = s0[v2879_a];
        float v2882_data = glb_m3[v2881_a];
        tensorforge::VectorT<float, 4> v2883_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2878_data, v2882_data, v2874_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v2884_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v2880_data, v2882_data, v2875_acc, 0, 0, 0);
        ir4[0] = (v2802_acc[0]);
        ir4[4] = (v2802_acc[1]);
        ir4[8] = (v2802_acc[2]);
        ir4[12] = (v2802_acc[3]);
        ir4[16] = (v2803_acc[0]);
        ir4[20] = (v2803_acc[1]);
        ir4[24] = (v2803_acc[2]);
        ir4[28] = (v2803_acc[3]);
        ir4[1] = (v2840_acc[0]);
        ir4[5] = (v2840_acc[1]);
        ir4[9] = (v2840_acc[2]);
        ir4[13] = (v2840_acc[3]);
        ir4[17] = (v2841_acc[0]);
        ir4[21] = (v2841_acc[1]);
        ir4[25] = (v2841_acc[2]);
        ir4[29] = (v2841_acc[3]);
        ir4[2] = (v2870_acc[0]);
        ir4[6] = (v2870_acc[1]);
        ir4[10] = (v2870_acc[2]);
        ir4[14] = (v2870_acc[3]);
        ir4[18] = (v2871_acc[0]);
        ir4[22] = (v2871_acc[1]);
        ir4[26] = (v2871_acc[2]);
        ir4[30] = (v2871_acc[3]);
        ir4[3] = (v2883_acc[0]);
        ir4[7] = (v2883_acc[1]);
        ir4[11] = (v2883_acc[2]);
        ir4[15] = (v2883_acc[3]);
        ir4[19] = (v2884_acc[0]);
        ir4[23] = (v2884_acc[1]);
        ir4[27] = (v2884_acc[2]);
        ir4[31] = (v2884_acc[3]);
        float v2920_data = glb_m3[v2483_lead];
        float v2924_data = glb_m3[v2754_lead];
        float v2928_data = glb_m3[v2761_lead];
        float v2932_data = glb_m3[v2768_lead];
        float v2934_data = glb_m3[v2933_a];
        float v2936_data = glb_m3[v2935_a];
        float v2938_data = glb_m3[v2937_a];
        float v2940_data = glb_m3[v2939_a];
        float v2942_data = glb_m3[v2941_a];
        float v2944_data = glb_m3[v2943_a];
        float v2946_data = glb_m3[v2945_a];
        float v2948_data = glb_m3[v2947_a];
        float v2950_data = glb_m3[v2949_a];
        float v2952_data = glb_m3[v2951_a];
        float v2954_data = glb_m3[v2953_a];
        float v2956_data = glb_m3[v2955_a];
        float v2958_data = glb_m3[v2957_a];
        float v2960_data = glb_m3[v2959_a];
        float v2962_data = glb_m3[v2961_a];
        float v2964_data = glb_m3[v2963_a];
        float v2966_data = glb_m3[v2965_a];
        float v2968_data = glb_m3[v2967_a];
        float v2970_data = glb_m3[v2969_a];
        float v2972_data = glb_m3[v2971_a];
        float v2974_data = glb_m3[v2973_a];
        float v2976_data = glb_m3[v2975_a];
        float v2978_data = glb_m3[v2977_a];
        float v2980_data = glb_m3[v2979_a];
        float v2982_data = glb_m3[v2981_a];
        float v2984_data = glb_m3[v2983_a];
        float v2986_data = glb_m3[v2985_a];
        float v2988_data = glb_m3[v2987_a];
        float v2990_data = glb_m3[v2989_a];
        float v2992_data = glb_m3[v2991_a];
        float v2994_data = glb_m3[v2993_a];
        float v2996_data = glb_m3[v2995_a];
        float v2998_data = glb_m3[v2997_a];
        float v3000_data = glb_m3[v2999_a];
        float v3002_data = glb_m3[v3001_a];
        float v3004_data = glb_m3[v3003_a];
        float v3006_data = glb_m3[v3005_a];
        float v3008_data = glb_m3[v3007_a];
        float v3010_data = glb_m3[v3009_a];
        float v3012_data = glb_m3[v3011_a];
        float v3014_data = glb_m3[v3013_a];
        float v3016_data = glb_m3[v3015_a];
        float v3018_data = glb_m3[v3017_a];
        float v3020_data = glb_m3[v3019_a];
        float v3022_data = glb_m3[v3021_a];
        float v3024_data = glb_m3[v3023_a];
        float v3026_data = glb_m3[v3025_a];
        float v3028_data = glb_m3[v3027_a];
        float v3030_data = glb_m3[v3029_a];
        float v3032_data = glb_m3[v3031_a];
        float v3034_data = glb_m3[v3033_a];
        float v3036_data = glb_m3[v3035_a];
        float v3038_data = glb_m3[v3037_a];
        float v3040_data = glb_m3[v3039_a];
        float v3042_data = glb_m3[v3041_a];
        float v3044_data = glb_m3[v3043_a];
        float v3046_data = glb_m3[v3045_a];
        float v3048_data = glb_m3[v3047_a];
        float v3050_data = glb_m3[v3049_a];
        float v3052_data = glb_m3[v3051_a];
        float v3054_data = glb_m3[v3053_a];
        float v3056_data = glb_m3[v3055_a];
        float v3058_data = glb_m3[v3057_a];
        float v3060_data = glb_m3[v3059_a];
        float v3062_data = glb_m3[v3061_a];
        float v3064_data = glb_m3[v3063_a];
        float v3066_data = glb_m3[v3065_a];
        float v3068_data = glb_m3[v3067_a];
        float v3070_data = glb_m3[v3069_a];
        float v3072_data = glb_m3[v3071_a];
        float v3074_data = glb_m3[v3073_a];
        float v3076_data = glb_m3[v3075_a];
        float v3078_data = glb_m3[v3077_a];
        float v3080_data = glb_m3[v3079_a];
        float v3082_data = glb_m3[v3081_a];
        float v3084_data = glb_m3[v3083_a];
        float v3086_data = glb_m3[v3085_a];
        float v3088_data = glb_m3[v3087_a];
        float v3090_data = glb_m3[v3089_a];
        float v3092_data = glb_m3[v3091_a];
        float v3094_data = glb_m3[v3093_a];
        float v3096_data = glb_m3[v3095_a];
        float v3098_data = glb_m3[v3097_a];
        float v3100_data = glb_m3[v3099_a];
        float v3102_data = glb_m3[v3101_a];
        float v3104_data = glb_m3[v3103_a];
        float v3106_data = glb_m3[v3105_a];
        float v3108_data = glb_m3[v3107_a];
        float v3109_acc{};
        float v3110_acc{};
        float v3111_acc{};
        float v3112_acc{};
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %22 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %23 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %25 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %26 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %29 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %36 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3109_acc), "+&v"(v3110_acc), "+&v"(v3111_acc), "+&v"(v3112_acc)
        : "v"(v2699_acc), "v"(v2920_data), "v"(v2924_data), "v"(v2928_data), "v"(v2932_data), "v"(v2934_data), "v"(v2936_data), "v"(v2938_data), "v"(v2940_data), "v"(v2942_data), "v"(v2944_data), "v"(v2946_data), "v"(v2948_data), "v"(v2950_data), "v"(v2952_data), "v"(v2954_data), "v"(v2956_data), "v"(v2958_data), "v"(v2960_data), "v"(v2962_data), "v"(v2964_data), "v"(v2966_data), "v"(v2968_data), "v"(v2970_data), "v"(v2972_data), "v"(v2974_data), "v"(v2976_data), "v"(v2978_data), "v"(v2980_data), "v"(v2982_data), "v"(v2984_data), "v"(v2986_data), "v"(v2988_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %21 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %22 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %23 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %24 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %25 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %26 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %27 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %28 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %29 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %33, %34 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %33, %35 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %33, %36 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %33, %37 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3109_acc), "+&v"(v3110_acc), "+&v"(v3111_acc), "+&v"(v3112_acc)
        : "v"(v2699_acc), "v"(v2990_data), "v"(v2992_data), "v"(v2994_data), "v"(v2996_data), "v"(v2998_data), "v"(v3000_data), "v"(v3002_data), "v"(v3004_data), "v"(v3006_data), "v"(v3008_data), "v"(v3010_data), "v"(v3012_data), "v"(v3014_data), "v"(v3016_data), "v"(v3018_data), "v"(v3020_data), "v"(v3022_data), "v"(v3024_data), "v"(v3026_data), "v"(v3028_data), "v"(v3030_data), "v"(v3032_data), "v"(v3034_data), "v"(v3036_data), "v"(v3038_data), "v"(v3040_data), "v"(v3042_data), "v"(v3044_data), "v"(v2700_acc), "v"(v3046_data), "v"(v3048_data), "v"(v3050_data), "v"(v3052_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %3, %4 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %6 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %13 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %15 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %22 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %23 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %24 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %25 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %26 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %27 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %28 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %29 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %30 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %31 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3111_acc), "+&v"(v3112_acc), "+&v"(v3110_acc)
        : "v"(v2700_acc), "v"(v3054_data), "v"(v3056_data), "v"(v3058_data), "v"(v3060_data), "v"(v3062_data), "v"(v3064_data), "v"(v3066_data), "v"(v3068_data), "v"(v3070_data), "v"(v3072_data), "v"(v3074_data), "v"(v3076_data), "v"(v3078_data), "v"(v3080_data), "v"(v3082_data), "v"(v3084_data), "v"(v3086_data), "v"(v3088_data), "v"(v3090_data), "v"(v3092_data), "v"(v3094_data), "v"(v3096_data), "v"(v3098_data), "v"(v3100_data), "v"(v3102_data), "v"(v3104_data), "v"(v3106_data), "v"(v3108_data)
        );
        ir4[32] = v3109_acc;
        ir4[33] = v3110_acc;
        ir4[34] = v3111_acc;
        ir4[35] = v3112_acc;
        float r3[36]{};
        // r3 = load{g>r}(glb_m2);
        float r5[27]{};
        // r5 = load{g>r}(glb_m0);
        // r4 = ir4 + r3
        // glb_m2 = store{r>g}(r4);
        if (v3135_g) {
          #pragma unroll
          for (int32_t v3180_i1 = 0; v3180_i1 < 9; ++v3180_i1) {
            float v3183_data = ir4[((v3180_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[(v2768_lead + (v3180_i1 * 56))], v3183_data);
            }
          }
        }
        float r6[9]{};
        // r6 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v3207_i1 = 0; v3207_i1 < 9; ++v3207_i1) {
          float v3213_data = v2508_g ? (__builtin_nontemporal_load(&glb_m4[(v2483_lead + (v3207_i1 * 9))])) : (0.0f);
          r6[v3207_i1] = v3213_data;
        }
        float r7[27]{};
        // r7 = +(r5 * r6) + None
        // [(0, 35), (0, 9)] [(0, 9)]
        float v3216_data = r6[0];
        float v3217_data = r6[1];
        float v3218_data = r6[2];
        float v3219_data = r6[3];
        float v3220_tp{};
        float v3221_tp{};
        float v3222_tp{};
        float v3223_tp{};
        tensorforge::transpose4x4b32Fused(v3220_tp, v3221_tp, v3222_tp, v3223_tp, v3216_data, v3217_data, v3218_data, v3219_data);
        tensorforge::VectorT<float, 4> v3224_acc{};
        tensorforge::VectorT<float, 4> v3229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2521_data, v3224_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3221_tp, v2522_data, v3229_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3222_tp, v2523_data, v3230_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3223_tp, v2524_data, v3231_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2529_data, v3232_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3221_tp, v2530_data, v3237_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3222_tp, v2531_data, v3238_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3223_tp, v2532_data, v3239_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2537_data, v3240_acc, 2, 2, 0);
        r7[0] = (v3243_acc[0]);
        r7[3] = (v3243_acc[1]);
        r7[6] = (v3243_acc[2]);
        r7[9] = (v3243_acc[3]);
        tensorforge::VectorT<float, 4> v3248_acc{};
        tensorforge::VectorT<float, 4> v3253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2545_data, v3248_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3221_tp, v2546_data, v3253_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3222_tp, v2547_data, v3254_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3223_tp, v2548_data, v3255_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2553_data, v3256_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3221_tp, v2554_data, v3261_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3222_tp, v2555_data, v3262_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3223_tp, v2556_data, v3263_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2561_data, v3264_acc, 2, 2, 0);
        r7[1] = (v3267_acc[0]);
        r7[4] = (v3267_acc[1]);
        r7[7] = (v3267_acc[2]);
        r7[10] = (v3267_acc[3]);
        tensorforge::VectorT<float, 4> v3272_acc{};
        tensorforge::VectorT<float, 4> v3277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2569_data, v3272_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3221_tp, v2570_data, v3277_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3222_tp, v2571_data, v3278_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3223_tp, v2572_data, v3279_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2577_data, v3280_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3221_tp, v2578_data, v3285_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3222_tp, v2579_data, v3286_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3223_tp, v2580_data, v3287_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3220_tp, v2585_data, v3288_acc, 2, 2, 0);
        r7[2] = (v3291_acc[0]);
        r7[5] = (v3291_acc[1]);
        r7[8] = (v3291_acc[2]);
        r7[11] = (v3291_acc[3]);
        float v3296_data = r6[4];
        float v3297_data = r6[5];
        float v3298_data = r6[6];
        float v3299_data = r6[7];
        float v3300_tp{};
        float v3301_tp{};
        float v3302_tp{};
        float v3303_tp{};
        tensorforge::transpose4x4b32Fused(v3300_tp, v3301_tp, v3302_tp, v3303_tp, v3296_data, v3297_data, v3298_data, v3299_data);
        tensorforge::VectorT<float, 4> v3304_acc{};
        tensorforge::VectorT<float, 4> v3309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2521_data, v3304_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3301_tp, v2522_data, v3309_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3302_tp, v2523_data, v3310_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3303_tp, v2524_data, v3311_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2529_data, v3312_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3301_tp, v2530_data, v3317_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3302_tp, v2531_data, v3318_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3303_tp, v2532_data, v3319_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2537_data, v3320_acc, 2, 2, 0);
        r7[12] = (v3323_acc[0]);
        r7[15] = (v3323_acc[1]);
        r7[18] = (v3323_acc[2]);
        r7[21] = (v3323_acc[3]);
        tensorforge::VectorT<float, 4> v3328_acc{};
        tensorforge::VectorT<float, 4> v3333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2545_data, v3328_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3301_tp, v2546_data, v3333_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3302_tp, v2547_data, v3334_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3303_tp, v2548_data, v3335_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2553_data, v3336_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3301_tp, v2554_data, v3341_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3302_tp, v2555_data, v3342_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3303_tp, v2556_data, v3343_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2561_data, v3344_acc, 2, 2, 0);
        r7[13] = (v3347_acc[0]);
        r7[16] = (v3347_acc[1]);
        r7[19] = (v3347_acc[2]);
        r7[22] = (v3347_acc[3]);
        tensorforge::VectorT<float, 4> v3352_acc{};
        tensorforge::VectorT<float, 4> v3357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2569_data, v3352_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3301_tp, v2570_data, v3357_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3302_tp, v2571_data, v3358_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3303_tp, v2572_data, v3359_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v3365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2577_data, v3360_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3301_tp, v2578_data, v3365_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3302_tp, v2579_data, v3366_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3303_tp, v2580_data, v3367_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v3371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3300_tp, v2585_data, v3368_acc, 2, 2, 0);
        r7[14] = (v3371_acc[0]);
        r7[17] = (v3371_acc[1]);
        r7[20] = (v3371_acc[2]);
        r7[23] = (v3371_acc[3]);
        float v3403_acc{};
        float v3404_acc{};
        float v3405_acc{};
        float v3406_data = r6[8];
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %3, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %30 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3403_acc), "+&v"(v3404_acc), "+&v"(v3405_acc)
        : "v"(v3406_data), "v"(v2521_data), "v"(v2545_data), "v"(v2569_data), "v"(v2522_data), "v"(v2546_data), "v"(v2570_data), "v"(v2523_data), "v"(v2547_data), "v"(v2571_data), "v"(v2524_data), "v"(v2548_data), "v"(v2572_data), "v"(v2529_data), "v"(v2553_data), "v"(v2577_data), "v"(v2530_data), "v"(v2554_data), "v"(v2578_data), "v"(v2531_data), "v"(v2555_data), "v"(v2579_data), "v"(v2532_data), "v"(v2556_data), "v"(v2580_data), "v"(v2537_data), "v"(v2561_data), "v"(v2585_data)
        );
        r7[24] = v3403_acc;
        r7[25] = v3404_acc;
        r7[26] = v3405_acc;
        // r7 = 0 past (0, 35)
        // s1 = store{r>s}(localShrMem0, r7);
        #pragma unroll
        for (int32_t v3408_i0 = 0; v3408_i0 < 2; ++v3408_i0) {
          int32_t v3414_lead = v2483_lead + (v3408_i0 * 16);
          #pragma unroll
          for (int32_t v3409_i1 = 0; v3409_i1 < 9; ++v3409_i1) {
            float v3412_data = r7[(v3408_i0 + (v3409_i1 * 3))];
            s1[(v3414_lead + (v3409_i1 * 35))] = v3412_data;
          }
        }
        if (v2493_g) {
          #pragma unroll
          for (int32_t v3417_i1 = 0; v3417_i1 < 9; ++v3417_i1) {
            float v3420_data = r7[(2 + (v3417_i1 * 3))];
            s1[(v2761_lead + (v3417_i1 * 35))] = v3420_data;
          }
        }
        float r9[36]{};
        // ir9 = +(glb_m5 * s1)
        // [(1, 55), (0, 9)] [(0, 35)]
        float ir9[36]{};
        tensorforge::VectorT<float, 4> v3432_acc{};
        tensorforge::VectorT<float, 4> v3433_acc{};
        tensorforge::VectorT<float, 4> v3434_acc{};
        tensorforge::VectorT<float, 4> v3435_acc{};
        tensorforge::VectorT<float, 4> v3436_acc{};
        tensorforge::VectorT<float, 4> v3437_acc{};
        tensorforge::VectorT<float, 4> v3438_acc{};
        tensorforge::VectorT<float, 4> v3439_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        float v3446_data = s1[v2739_sh];
        float v3448_data = s1[v2743_a];
        float v3454_data = glb_m5[v2748_a];
        tensorforge::VectorT<float, 4> v3455_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3446_data, v3454_data, v3432_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3456_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3448_data, v3454_data, v3433_acc, 0, 0, 0);
        float v3461_data = glb_m5[v2755_a];
        tensorforge::VectorT<float, 4> v3462_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3446_data, v3461_data, v3434_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3463_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3448_data, v3461_data, v3435_acc, 0, 0, 0);
        float v3468_data = glb_m5[v2762_a];
        tensorforge::VectorT<float, 4> v3469_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3446_data, v3468_data, v3436_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3470_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3448_data, v3468_data, v3437_acc, 0, 0, 0);
        float v3475_data = glb_m5[v2769_a];
        tensorforge::VectorT<float, 4> v3476_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3446_data, v3475_data, v3438_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3477_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3448_data, v3475_data, v3439_acc, 0, 0, 0);
        float v3480_data = s1[v2774_a];
        float v3482_data = s1[v2777_a];
        float v3484_data = glb_m5[v2779_a];
        tensorforge::VectorT<float, 4> v3485_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3480_data, v3484_data, v3455_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3486_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3482_data, v3484_data, v3456_acc, 0, 0, 0);
        float v3488_data = glb_m5[v2783_a];
        tensorforge::VectorT<float, 4> v3489_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3480_data, v3488_data, v3462_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3490_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3482_data, v3488_data, v3463_acc, 0, 0, 0);
        float v3492_data = glb_m5[v2787_a];
        tensorforge::VectorT<float, 4> v3493_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3480_data, v3492_data, v3469_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3494_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3482_data, v3492_data, v3470_acc, 0, 0, 0);
        float v3496_data = glb_m5[v2791_a];
        tensorforge::VectorT<float, 4> v3497_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3480_data, v3496_data, v3476_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3498_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3482_data, v3496_data, v3477_acc, 0, 0, 0);
        float v3501_data = s1[v2795_a];
        float v3503_data = s1[v2798_a];
        float v3505_data = glb_m5[v2800_a];
        tensorforge::VectorT<float, 4> v3506_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3501_data, v3505_data, v3485_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3507_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3503_data, v3505_data, v3486_acc, 0, 0, 0);
        float v3509_data = glb_m5[v2804_a];
        tensorforge::VectorT<float, 4> v3510_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3501_data, v3509_data, v3489_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3511_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3503_data, v3509_data, v3490_acc, 0, 0, 0);
        float v3513_data = glb_m5[v2808_a];
        tensorforge::VectorT<float, 4> v3514_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3501_data, v3513_data, v3493_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3515_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3503_data, v3513_data, v3494_acc, 0, 0, 0);
        float v3517_data = glb_m5[v2812_a];
        tensorforge::VectorT<float, 4> v3518_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3501_data, v3517_data, v3497_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3519_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3503_data, v3517_data, v3498_acc, 0, 0, 0);
        float v3522_data = s1[v2816_a];
        float v3524_data = s1[v2819_a];
        float v3526_data = glb_m5[v3525_a];
        tensorforge::VectorT<float, 4> v3527_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3522_data, v3526_data, v3506_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3528_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3524_data, v3526_data, v3507_acc, 0, 0, 0);
        float v3530_data = glb_m5[v2821_a];
        tensorforge::VectorT<float, 4> v3531_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3522_data, v3530_data, v3510_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3532_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3524_data, v3530_data, v3511_acc, 0, 0, 0);
        float v3534_data = glb_m5[v2825_a];
        tensorforge::VectorT<float, 4> v3535_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3522_data, v3534_data, v3514_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3536_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3524_data, v3534_data, v3515_acc, 0, 0, 0);
        float v3538_data = glb_m5[v2829_a];
        tensorforge::VectorT<float, 4> v3539_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3522_data, v3538_data, v3518_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3540_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3524_data, v3538_data, v3519_acc, 0, 0, 0);
        float v3543_data = s1[v2833_a];
        float v3545_data = s1[v2836_a];
        float v3547_data = glb_m5[v2838_a];
        tensorforge::VectorT<float, 4> v3548_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3543_data, v3547_data, v3531_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3549_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3545_data, v3547_data, v3532_acc, 0, 0, 0);
        float v3551_data = glb_m5[v2842_a];
        tensorforge::VectorT<float, 4> v3552_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3543_data, v3551_data, v3535_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3553_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3545_data, v3551_data, v3536_acc, 0, 0, 0);
        float v3555_data = glb_m5[v2846_a];
        tensorforge::VectorT<float, 4> v3556_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3543_data, v3555_data, v3539_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3557_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3545_data, v3555_data, v3540_acc, 0, 0, 0);
        float v3560_data = s1[v2850_a];
        float v3562_data = s1[v2853_a];
        float v3564_data = glb_m5[v3563_a];
        tensorforge::VectorT<float, 4> v3565_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3560_data, v3564_data, v3548_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3566_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3562_data, v3564_data, v3549_acc, 0, 0, 0);
        float v3568_data = glb_m5[v2855_a];
        tensorforge::VectorT<float, 4> v3569_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3560_data, v3568_data, v3552_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3570_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3562_data, v3568_data, v3553_acc, 0, 0, 0);
        float v3572_data = glb_m5[v2859_a];
        tensorforge::VectorT<float, 4> v3573_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3560_data, v3572_data, v3556_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3574_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3562_data, v3572_data, v3557_acc, 0, 0, 0);
        float v3577_data = s1[v2863_a];
        float v3579_data = s1[v2866_a];
        float v3581_data = glb_m5[v2868_a];
        tensorforge::VectorT<float, 4> v3582_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3577_data, v3581_data, v3569_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3583_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3579_data, v3581_data, v3570_acc, 0, 0, 0);
        float v3585_data = glb_m5[v2872_a];
        tensorforge::VectorT<float, 4> v3586_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3577_data, v3585_data, v3573_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3587_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3579_data, v3585_data, v3574_acc, 0, 0, 0);
        float v3590_data = s1[v2876_a];
        float v3592_data = s1[v2879_a];
        float v3594_data = glb_m5[v3593_a];
        tensorforge::VectorT<float, 4> v3595_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3590_data, v3594_data, v3582_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3596_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3592_data, v3594_data, v3583_acc, 0, 0, 0);
        float v3598_data = glb_m5[v2881_a];
        tensorforge::VectorT<float, 4> v3599_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3590_data, v3598_data, v3586_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3600_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v3592_data, v3598_data, v3587_acc, 0, 0, 0);
        float v3609_data = s1[v3607_a];
        float v3617_data = s1[v3616_a];
        float v3622_data = glb_m5[v3621_a];
        float v3624_data = glb_m5[v3623_a];
        float v3626_data = glb_m5[v3625_a];
        float v3629_ks = v3614_g ? v3622_data : (v3612_g ? v3624_data : (v3610_g ? v3626_data : 0.0f));
        tensorforge::VectorT<float, 4> v3630_acc = __builtin_amdgcn_mfma_f32_16x16x4f32((v3614_g ? v3609_data : (v3612_g ? v3609_data : (v3610_g ? v3609_data : 0.0f))), v3629_ks, v3599_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v3631_acc = __builtin_amdgcn_mfma_f32_16x16x4f32((v3614_g ? v3617_data : (v3612_g ? v3617_data : (v3610_g ? v3617_data : 0.0f))), v3629_ks, v3600_acc, 0, 0, 0);
        ir9[0] = (v3527_acc[0]);
        ir9[4] = (v3527_acc[1]);
        ir9[8] = (v3527_acc[2]);
        ir9[12] = (v3527_acc[3]);
        ir9[16] = (v3528_acc[0]);
        ir9[20] = (v3528_acc[1]);
        ir9[24] = (v3528_acc[2]);
        ir9[28] = (v3528_acc[3]);
        ir9[1] = (v3565_acc[0]);
        ir9[5] = (v3565_acc[1]);
        ir9[9] = (v3565_acc[2]);
        ir9[13] = (v3565_acc[3]);
        ir9[17] = (v3566_acc[0]);
        ir9[21] = (v3566_acc[1]);
        ir9[25] = (v3566_acc[2]);
        ir9[29] = (v3566_acc[3]);
        ir9[2] = (v3595_acc[0]);
        ir9[6] = (v3595_acc[1]);
        ir9[10] = (v3595_acc[2]);
        ir9[14] = (v3595_acc[3]);
        ir9[18] = (v3596_acc[0]);
        ir9[22] = (v3596_acc[1]);
        ir9[26] = (v3596_acc[2]);
        ir9[30] = (v3596_acc[3]);
        ir9[3] = (v3630_acc[0]);
        ir9[7] = (v3630_acc[1]);
        ir9[11] = (v3630_acc[2]);
        ir9[15] = (v3630_acc[3]);
        ir9[19] = (v3631_acc[0]);
        ir9[23] = (v3631_acc[1]);
        ir9[27] = (v3631_acc[2]);
        ir9[31] = (v3631_acc[3]);
        float v3667_data = glb_m5[v2483_lead];
        float v3671_data = glb_m5[v2754_lead];
        float v3675_data = glb_m5[v2761_lead];
        float v3679_data = glb_m5[v2768_lead];
        float v3681_data = glb_m5[v2933_a];
        float v3683_data = glb_m5[v2935_a];
        float v3685_data = glb_m5[v2937_a];
        float v3687_data = glb_m5[v2939_a];
        float v3689_data = glb_m5[v2941_a];
        float v3691_data = glb_m5[v2943_a];
        float v3693_data = glb_m5[v2945_a];
        float v3695_data = glb_m5[v2947_a];
        float v3697_data = glb_m5[v2949_a];
        float v3699_data = glb_m5[v2951_a];
        float v3701_data = glb_m5[v2953_a];
        float v3703_data = glb_m5[v2955_a];
        float v3705_data = glb_m5[v2957_a];
        float v3707_data = glb_m5[v2959_a];
        float v3709_data = glb_m5[v2961_a];
        float v3711_data = glb_m5[v2963_a];
        float v3713_data = glb_m5[v2965_a];
        float v3715_data = glb_m5[v2967_a];
        float v3717_data = glb_m5[v2969_a];
        float v3719_data = glb_m5[v2971_a];
        float v3721_data = glb_m5[v2973_a];
        float v3723_data = glb_m5[v2975_a];
        float v3725_data = glb_m5[v2977_a];
        float v3727_data = glb_m5[v2979_a];
        float v3729_data = glb_m5[v2981_a];
        float v3731_data = glb_m5[v2983_a];
        float v3733_data = glb_m5[v2985_a];
        float v3735_data = glb_m5[v2987_a];
        float v3737_data = glb_m5[v2989_a];
        float v3739_data = glb_m5[v2991_a];
        float v3741_data = glb_m5[v2993_a];
        float v3743_data = glb_m5[v2995_a];
        float v3745_data = glb_m5[v2997_a];
        float v3747_data = glb_m5[v2999_a];
        float v3749_data = glb_m5[v3001_a];
        float v3751_data = glb_m5[v3003_a];
        float v3753_data = glb_m5[v3005_a];
        float v3755_data = glb_m5[v3007_a];
        float v3757_data = glb_m5[v3009_a];
        float v3759_data = glb_m5[v3011_a];
        float v3761_data = glb_m5[v3013_a];
        float v3763_data = glb_m5[v3015_a];
        float v3765_data = glb_m5[v3017_a];
        float v3767_data = glb_m5[v3019_a];
        float v3769_data = glb_m5[v3768_a];
        float v3771_data = glb_m5[v3021_a];
        float v3773_data = glb_m5[v3023_a];
        float v3775_data = glb_m5[v3025_a];
        float v3777_data = glb_m5[v3776_a];
        float v3779_data = glb_m5[v3027_a];
        float v3781_data = glb_m5[v3029_a];
        float v3783_data = glb_m5[v3031_a];
        float v3785_data = glb_m5[v3784_a];
        float v3787_data = glb_m5[v3033_a];
        float v3789_data = glb_m5[v3035_a];
        float v3791_data = glb_m5[v3037_a];
        float v3793_data = glb_m5[v3792_a];
        float v3795_data = glb_m5[v3039_a];
        float v3797_data = glb_m5[v3041_a];
        float v3799_data = glb_m5[v3043_a];
        float v3801_data = glb_m5[v3045_a];
        float v3803_data = glb_m5[v3047_a];
        float v3805_data = glb_m5[v3049_a];
        float v3807_data = glb_m5[v3051_a];
        float v3809_data = glb_m5[v3053_a];
        float v3811_data = glb_m5[v3055_a];
        float v3813_data = glb_m5[v3057_a];
        float v3815_data = glb_m5[v3059_a];
        float v3817_data = glb_m5[v3061_a];
        float v3819_data = glb_m5[v3063_a];
        float v3821_data = glb_m5[v3065_a];
        float v3823_data = glb_m5[v3067_a];
        float v3825_data = glb_m5[v3824_a];
        float v3827_data = glb_m5[v3069_a];
        float v3829_data = glb_m5[v3071_a];
        float v3831_data = glb_m5[v3830_a];
        float v3833_data = glb_m5[v3073_a];
        float v3835_data = glb_m5[v3075_a];
        float v3837_data = glb_m5[v3836_a];
        float v3839_data = glb_m5[v3077_a];
        float v3841_data = glb_m5[v3079_a];
        float v3843_data = glb_m5[v3842_a];
        float v3845_data = glb_m5[v3081_a];
        float v3847_data = glb_m5[v3083_a];
        float v3849_data = glb_m5[v3085_a];
        float v3851_data = glb_m5[v3087_a];
        float v3853_data = glb_m5[v3089_a];
        float v3855_data = glb_m5[v3091_a];
        float v3857_data = glb_m5[v3093_a];
        float v3859_data = glb_m5[v3095_a];
        float v3861_data = glb_m5[v3097_a];
        float v3863_data = glb_m5[v3099_a];
        float v3865_data = glb_m5[v3864_a];
        float v3867_data = glb_m5[v3101_a];
        float v3869_data = glb_m5[v3868_a];
        float v3871_data = glb_m5[v3103_a];
        float v3873_data = glb_m5[v3872_a];
        float v3875_data = glb_m5[v3105_a];
        float v3877_data = glb_m5[v3876_a];
        float v3879_data = glb_m5[v3107_a];
        float v3886_acc{};
        float v3887_acc{};
        float v3888_acc{};
        float v3889_acc{};
        float v3892_data = r7[26];
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %22 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %23 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %25 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %26 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %29 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %36 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3886_acc), "+&v"(v3887_acc), "+&v"(v3888_acc), "+&v"(v3889_acc)
        : "v"(v3403_acc), "v"(v3667_data), "v"(v3671_data), "v"(v3675_data), "v"(v3679_data), "v"(v3681_data), "v"(v3683_data), "v"(v3685_data), "v"(v3687_data), "v"(v3689_data), "v"(v3691_data), "v"(v3693_data), "v"(v3695_data), "v"(v3697_data), "v"(v3699_data), "v"(v3701_data), "v"(v3703_data), "v"(v3705_data), "v"(v3707_data), "v"(v3709_data), "v"(v3711_data), "v"(v3713_data), "v"(v3715_data), "v"(v3717_data), "v"(v3719_data), "v"(v3721_data), "v"(v3723_data), "v"(v3725_data), "v"(v3727_data), "v"(v3729_data), "v"(v3731_data), "v"(v3733_data), "v"(v3735_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %21 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %22 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %23 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %24 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %25 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %26 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %27 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %28 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %29 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %35 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %36 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3886_acc), "+&v"(v3887_acc), "+&v"(v3888_acc), "+&v"(v3889_acc)
        : "v"(v3403_acc), "v"(v3737_data), "v"(v3739_data), "v"(v3741_data), "v"(v3743_data), "v"(v3745_data), "v"(v3747_data), "v"(v3749_data), "v"(v3751_data), "v"(v3753_data), "v"(v3755_data), "v"(v3757_data), "v"(v3759_data), "v"(v3761_data), "v"(v3763_data), "v"(v3765_data), "v"(v3767_data), "v"(v3769_data), "v"(v3771_data), "v"(v3773_data), "v"(v3775_data), "v"(v3777_data), "v"(v3779_data), "v"(v3781_data), "v"(v3783_data), "v"(v3785_data), "v"(v3787_data), "v"(v3789_data), "v"(v3791_data), "v"(v3793_data), "v"(v3795_data), "v"(v3797_data), "v"(v3799_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %3, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %30 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %31 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %32 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %33 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %34 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %35 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3887_acc), "+&v"(v3888_acc), "+&v"(v3889_acc)
        : "v"(v3404_acc), "v"(v3801_data), "v"(v3803_data), "v"(v3805_data), "v"(v3807_data), "v"(v3809_data), "v"(v3811_data), "v"(v3813_data), "v"(v3815_data), "v"(v3817_data), "v"(v3819_data), "v"(v3821_data), "v"(v3823_data), "v"(v3825_data), "v"(v3827_data), "v"(v3829_data), "v"(v3831_data), "v"(v3833_data), "v"(v3835_data), "v"(v3837_data), "v"(v3839_data), "v"(v3841_data), "v"(v3843_data), "v"(v3845_data), "v"(v3847_data), "v"(v3849_data), "v"(v3851_data), "v"(v3853_data), "v"(v3855_data), "v"(v3857_data), "v"(v3859_data), "v"(v3861_data), "v"(v3863_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %2, %3 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %2, %4 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %2, %5 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %2, %6 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %2, %7 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %2, %8 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %2, %9 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %2, %10 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %11, %13 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %11, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v3888_acc), "+&v"(v3889_acc)
        : "v"(v3404_acc), "v"(v3865_data), "v"(v3867_data), "v"(v3869_data), "v"(v3871_data), "v"(v3873_data), "v"(v3875_data), "v"(v3877_data), "v"(v3879_data), "v"(v3892_data), "v"(v3622_data), "v"(v3624_data), "v"(v3626_data)
        );
        ir9[32] = v3886_acc;
        ir9[33] = v3887_acc;
        ir9[34] = v3888_acc;
        ir9[35] = v3889_acc;
        float r8[36]{};
        // r8 = load{g>r}(glb_m2);
        float r10[27]{};
        // r10 = load{g>r}(glb_m0);
        // r9 = ir9 + r8
        if (v3117_g) {
          #pragma unroll
          for (int32_t v3920_n1 = 0; v3920_n1 < 9; ++v3920_n1) {
            int32_t v3921_a = v3920_n1 * 4;
            float v3923_data = ir9[v3921_a];
            float v3924_data = ir4[v3921_a];
            r9[v3921_a] = (v3924_data + v3923_data);
          }
        }
        #pragma unroll
        for (int32_t v3926_n0 = 1; v3926_n0 < 3; ++v3926_n0) {
          #pragma unroll
          for (int32_t v3927_n1 = 0; v3927_n1 < 9; ++v3927_n1) {
            int32_t v3928_a = v3927_n1 * 4;
            int32_t v3929_a = v3926_n0 + v3928_a;
            float v3930_data = ir9[v3929_a];
            float v3931_data = ir4[(v3926_n0 + v3928_a)];
            r9[v3929_a] = (v3931_data + v3930_data);
          }
        }
        // glb_m2 = store{r>g}(r9);
        if (v3911_g) {
          #pragma unroll
          for (int32_t v3956_i1 = 0; v3956_i1 < 9; ++v3956_i1) {
            float v3959_data = ir9[((v3956_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[(v2768_lead + (v3956_i1 * 56))], v3959_data);
            }
          }
        }
        float r11[9]{};
        // r11 = load{g>r}(glb_m6);
        #pragma unroll
        for (int32_t v3983_i1 = 0; v3983_i1 < 9; ++v3983_i1) {
          float v3989_data = v2508_g ? (__builtin_nontemporal_load(&glb_m6[(v2483_lead + (v3983_i1 * 9))])) : (0.0f);
          r11[v3983_i1] = v3989_data;
        }
        float r12[27]{};
        // r12 = +(r10 * r11) + None
        // [(0, 35), (0, 9)] [(0, 9)]
        float v3992_data = r11[0];
        float v3993_data = r11[1];
        float v3994_data = r11[2];
        float v3995_data = r11[3];
        float v3996_tp{};
        float v3997_tp{};
        float v3998_tp{};
        float v3999_tp{};
        tensorforge::transpose4x4b32Fused(v3996_tp, v3997_tp, v3998_tp, v3999_tp, v3992_data, v3993_data, v3994_data, v3995_data);
        tensorforge::VectorT<float, 4> v4000_acc{};
        tensorforge::VectorT<float, 4> v4005_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2521_data, v4000_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4006_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3997_tp, v2522_data, v4005_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4007_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3998_tp, v2523_data, v4006_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4008_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3999_tp, v2524_data, v4007_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4013_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2529_data, v4008_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4014_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3997_tp, v2530_data, v4013_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3998_tp, v2531_data, v4014_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4016_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3999_tp, v2532_data, v4015_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4019_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2537_data, v4016_acc, 2, 2, 0);
        r12[0] = (v4019_acc[0]);
        r12[3] = (v4019_acc[1]);
        r12[6] = (v4019_acc[2]);
        r12[9] = (v4019_acc[3]);
        tensorforge::VectorT<float, 4> v4024_acc{};
        tensorforge::VectorT<float, 4> v4029_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2545_data, v4024_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4030_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3997_tp, v2546_data, v4029_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4031_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3998_tp, v2547_data, v4030_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4032_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3999_tp, v2548_data, v4031_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4037_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2553_data, v4032_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4038_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3997_tp, v2554_data, v4037_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4039_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3998_tp, v2555_data, v4038_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4040_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3999_tp, v2556_data, v4039_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4043_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2561_data, v4040_acc, 2, 2, 0);
        r12[1] = (v4043_acc[0]);
        r12[4] = (v4043_acc[1]);
        r12[7] = (v4043_acc[2]);
        r12[10] = (v4043_acc[3]);
        tensorforge::VectorT<float, 4> v4048_acc{};
        tensorforge::VectorT<float, 4> v4053_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2569_data, v4048_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3997_tp, v2570_data, v4053_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4055_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3998_tp, v2571_data, v4054_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4056_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3999_tp, v2572_data, v4055_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4061_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2577_data, v4056_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4062_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3997_tp, v2578_data, v4061_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4063_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3998_tp, v2579_data, v4062_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4064_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3999_tp, v2580_data, v4063_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4067_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v3996_tp, v2585_data, v4064_acc, 2, 2, 0);
        r12[2] = (v4067_acc[0]);
        r12[5] = (v4067_acc[1]);
        r12[8] = (v4067_acc[2]);
        r12[11] = (v4067_acc[3]);
        float v4072_data = r11[4];
        float v4073_data = r11[5];
        float v4074_data = r11[6];
        float v4075_data = r11[7];
        float v4076_tp{};
        float v4077_tp{};
        float v4078_tp{};
        float v4079_tp{};
        tensorforge::transpose4x4b32Fused(v4076_tp, v4077_tp, v4078_tp, v4079_tp, v4072_data, v4073_data, v4074_data, v4075_data);
        tensorforge::VectorT<float, 4> v4080_acc{};
        tensorforge::VectorT<float, 4> v4085_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2521_data, v4080_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4086_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4077_tp, v2522_data, v4085_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4087_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4078_tp, v2523_data, v4086_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4088_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4079_tp, v2524_data, v4087_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4093_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2529_data, v4088_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4094_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4077_tp, v2530_data, v4093_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4095_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4078_tp, v2531_data, v4094_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4096_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4079_tp, v2532_data, v4095_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4099_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2537_data, v4096_acc, 2, 2, 0);
        r12[12] = (v4099_acc[0]);
        r12[15] = (v4099_acc[1]);
        r12[18] = (v4099_acc[2]);
        r12[21] = (v4099_acc[3]);
        tensorforge::VectorT<float, 4> v4104_acc{};
        tensorforge::VectorT<float, 4> v4109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2545_data, v4104_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4077_tp, v2546_data, v4109_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4078_tp, v2547_data, v4110_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4079_tp, v2548_data, v4111_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2553_data, v4112_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4077_tp, v2554_data, v4117_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4078_tp, v2555_data, v4118_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4079_tp, v2556_data, v4119_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2561_data, v4120_acc, 2, 2, 0);
        r12[13] = (v4123_acc[0]);
        r12[16] = (v4123_acc[1]);
        r12[19] = (v4123_acc[2]);
        r12[22] = (v4123_acc[3]);
        tensorforge::VectorT<float, 4> v4128_acc{};
        tensorforge::VectorT<float, 4> v4133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2569_data, v4128_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4077_tp, v2570_data, v4133_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4078_tp, v2571_data, v4134_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4079_tp, v2572_data, v4135_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v4141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2577_data, v4136_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4077_tp, v2578_data, v4141_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4078_tp, v2579_data, v4142_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4079_tp, v2580_data, v4143_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v4147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4076_tp, v2585_data, v4144_acc, 2, 2, 0);
        r12[14] = (v4147_acc[0]);
        r12[17] = (v4147_acc[1]);
        r12[20] = (v4147_acc[2]);
        r12[23] = (v4147_acc[3]);
        float v4179_acc{};
        float v4180_acc{};
        float v4181_acc{};
        float v4182_data = r11[8];
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %3, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %30 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4179_acc), "+&v"(v4180_acc), "+&v"(v4181_acc)
        : "v"(v4182_data), "v"(v2521_data), "v"(v2545_data), "v"(v2569_data), "v"(v2522_data), "v"(v2546_data), "v"(v2570_data), "v"(v2523_data), "v"(v2547_data), "v"(v2571_data), "v"(v2524_data), "v"(v2548_data), "v"(v2572_data), "v"(v2529_data), "v"(v2553_data), "v"(v2577_data), "v"(v2530_data), "v"(v2554_data), "v"(v2578_data), "v"(v2531_data), "v"(v2555_data), "v"(v2579_data), "v"(v2532_data), "v"(v2556_data), "v"(v2580_data), "v"(v2537_data), "v"(v2561_data), "v"(v2585_data)
        );
        r12[24] = v4179_acc;
        r12[25] = v4180_acc;
        r12[26] = v4181_acc;
        // r12 = 0 past (0, 35)
        // s2 = store{r>s}(localShrMem0, r12);
        #pragma unroll
        for (int32_t v4184_i0 = 0; v4184_i0 < 2; ++v4184_i0) {
          int32_t v4190_lead = v2483_lead + (v4184_i0 * 16);
          #pragma unroll
          for (int32_t v4185_i1 = 0; v4185_i1 < 9; ++v4185_i1) {
            float v4188_data = r12[(v4184_i0 + (v4185_i1 * 3))];
            s2[(v4190_lead + (v4185_i1 * 35))] = v4188_data;
          }
        }
        if (v2493_g) {
          #pragma unroll
          for (int32_t v4193_i1 = 0; v4193_i1 < 9; ++v4193_i1) {
            float v4196_data = r12[(2 + (v4193_i1 * 3))];
            s2[(v2761_lead + (v4193_i1 * 35))] = v4196_data;
          }
        }
        float r14[36]{};
        // ir14 = +(glb_m7 * s2)
        // [(1, 56), (0, 9)] [(0, 35)]
        float ir14[36]{};
        tensorforge::VectorT<float, 4> v4208_acc{};
        tensorforge::VectorT<float, 4> v4209_acc{};
        tensorforge::VectorT<float, 4> v4210_acc{};
        tensorforge::VectorT<float, 4> v4211_acc{};
        tensorforge::VectorT<float, 4> v4212_acc{};
        tensorforge::VectorT<float, 4> v4213_acc{};
        tensorforge::VectorT<float, 4> v4214_acc{};
        tensorforge::VectorT<float, 4> v4215_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        float v4222_data = s2[v2739_sh];
        float v4224_data = s2[v2743_a];
        float v4230_data = glb_m7[v2748_a];
        tensorforge::VectorT<float, 4> v4231_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4222_data, v4230_data, v4208_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4232_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4224_data, v4230_data, v4209_acc, 0, 0, 0);
        float v4237_data = glb_m7[v2755_a];
        tensorforge::VectorT<float, 4> v4238_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4222_data, v4237_data, v4210_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4239_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4224_data, v4237_data, v4211_acc, 0, 0, 0);
        float v4244_data = glb_m7[v2762_a];
        tensorforge::VectorT<float, 4> v4245_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4222_data, v4244_data, v4212_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4246_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4224_data, v4244_data, v4213_acc, 0, 0, 0);
        float v4251_data = glb_m7[v2769_a];
        tensorforge::VectorT<float, 4> v4252_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4222_data, v4251_data, v4214_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4253_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4224_data, v4251_data, v4215_acc, 0, 0, 0);
        float v4256_data = s2[v2774_a];
        float v4258_data = s2[v2777_a];
        float v4260_data = glb_m7[v2779_a];
        tensorforge::VectorT<float, 4> v4261_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4256_data, v4260_data, v4231_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4262_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4258_data, v4260_data, v4232_acc, 0, 0, 0);
        float v4264_data = glb_m7[v2783_a];
        tensorforge::VectorT<float, 4> v4265_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4256_data, v4264_data, v4238_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4266_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4258_data, v4264_data, v4239_acc, 0, 0, 0);
        float v4268_data = glb_m7[v2787_a];
        tensorforge::VectorT<float, 4> v4269_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4256_data, v4268_data, v4245_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4270_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4258_data, v4268_data, v4246_acc, 0, 0, 0);
        float v4272_data = glb_m7[v2791_a];
        tensorforge::VectorT<float, 4> v4273_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4256_data, v4272_data, v4252_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4274_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4258_data, v4272_data, v4253_acc, 0, 0, 0);
        float v4277_data = s2[v2795_a];
        float v4279_data = s2[v2798_a];
        float v4281_data = glb_m7[v2800_a];
        tensorforge::VectorT<float, 4> v4282_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4277_data, v4281_data, v4261_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4283_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4279_data, v4281_data, v4262_acc, 0, 0, 0);
        float v4285_data = glb_m7[v2804_a];
        tensorforge::VectorT<float, 4> v4286_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4277_data, v4285_data, v4265_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4287_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4279_data, v4285_data, v4266_acc, 0, 0, 0);
        float v4289_data = glb_m7[v2808_a];
        tensorforge::VectorT<float, 4> v4290_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4277_data, v4289_data, v4269_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4291_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4279_data, v4289_data, v4270_acc, 0, 0, 0);
        float v4293_data = glb_m7[v2812_a];
        tensorforge::VectorT<float, 4> v4294_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4277_data, v4293_data, v4273_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4295_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4279_data, v4293_data, v4274_acc, 0, 0, 0);
        float v4298_data = s2[v2816_a];
        float v4300_data = s2[v2819_a];
        float v4302_data = glb_m7[v3525_a];
        tensorforge::VectorT<float, 4> v4303_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4298_data, v4302_data, v4282_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4304_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4300_data, v4302_data, v4283_acc, 0, 0, 0);
        float v4306_data = glb_m7[v2821_a];
        tensorforge::VectorT<float, 4> v4307_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4298_data, v4306_data, v4286_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4308_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4300_data, v4306_data, v4287_acc, 0, 0, 0);
        float v4310_data = glb_m7[v2825_a];
        tensorforge::VectorT<float, 4> v4311_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4298_data, v4310_data, v4290_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4312_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4300_data, v4310_data, v4291_acc, 0, 0, 0);
        float v4314_data = glb_m7[v2829_a];
        tensorforge::VectorT<float, 4> v4315_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4298_data, v4314_data, v4294_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4316_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4300_data, v4314_data, v4295_acc, 0, 0, 0);
        float v4319_data = s2[v2833_a];
        float v4321_data = s2[v2836_a];
        float v4323_data = glb_m7[v4322_a];
        tensorforge::VectorT<float, 4> v4324_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4319_data, v4323_data, v4303_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4325_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4321_data, v4323_data, v4304_acc, 0, 0, 0);
        float v4327_data = glb_m7[v2838_a];
        tensorforge::VectorT<float, 4> v4328_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4319_data, v4327_data, v4307_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4329_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4321_data, v4327_data, v4308_acc, 0, 0, 0);
        float v4331_data = glb_m7[v2842_a];
        tensorforge::VectorT<float, 4> v4332_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4319_data, v4331_data, v4311_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4333_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4321_data, v4331_data, v4312_acc, 0, 0, 0);
        float v4335_data = glb_m7[v2846_a];
        tensorforge::VectorT<float, 4> v4336_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4319_data, v4335_data, v4315_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4337_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4321_data, v4335_data, v4316_acc, 0, 0, 0);
        float v4340_data = s2[v2850_a];
        float v4342_data = s2[v2853_a];
        float v4344_data = glb_m7[v3563_a];
        tensorforge::VectorT<float, 4> v4345_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4340_data, v4344_data, v4328_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4346_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4342_data, v4344_data, v4329_acc, 0, 0, 0);
        float v4348_data = glb_m7[v2855_a];
        tensorforge::VectorT<float, 4> v4349_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4340_data, v4348_data, v4332_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4350_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4342_data, v4348_data, v4333_acc, 0, 0, 0);
        float v4352_data = glb_m7[v2859_a];
        tensorforge::VectorT<float, 4> v4353_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4340_data, v4352_data, v4336_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4354_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4342_data, v4352_data, v4337_acc, 0, 0, 0);
        float v4357_data = s2[v2863_a];
        float v4359_data = s2[v2866_a];
        float v4361_data = glb_m7[v4360_a];
        tensorforge::VectorT<float, 4> v4362_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4357_data, v4361_data, v4345_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4363_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4359_data, v4361_data, v4346_acc, 0, 0, 0);
        float v4365_data = glb_m7[v2868_a];
        tensorforge::VectorT<float, 4> v4366_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4357_data, v4365_data, v4349_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4367_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4359_data, v4365_data, v4350_acc, 0, 0, 0);
        float v4369_data = glb_m7[v2872_a];
        tensorforge::VectorT<float, 4> v4370_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4357_data, v4369_data, v4353_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4371_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4359_data, v4369_data, v4354_acc, 0, 0, 0);
        float v4374_data = s2[v2876_a];
        float v4376_data = s2[v2879_a];
        float v4378_data = glb_m7[v3593_a];
        tensorforge::VectorT<float, 4> v4379_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4374_data, v4378_data, v4366_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4380_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4376_data, v4378_data, v4367_acc, 0, 0, 0);
        float v4382_data = glb_m7[v2881_a];
        tensorforge::VectorT<float, 4> v4383_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4374_data, v4382_data, v4370_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4384_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4376_data, v4382_data, v4371_acc, 0, 0, 0);
        float v4393_data = s2[v3607_a];
        float v4399_ks = v3614_g ? v4393_data : (v3612_g ? v4393_data : (v3610_g ? v4393_data : 0.0f));
        float v4401_data = s2[v3616_a];
        float v4404_ks = v3614_g ? v4401_data : (v3612_g ? v4401_data : (v3610_g ? v4401_data : 0.0f));
        float v4406_data = glb_m7[v4405_a];
        float v4408_data = glb_m7[v4407_a];
        float v4410_data = glb_m7[v4409_a];
        float v4413_ks = v3614_g ? v4406_data : (v3612_g ? v4408_data : (v3610_g ? v4410_data : 0.0f));
        tensorforge::VectorT<float, 4> v4414_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4399_ks, v4413_ks, v4379_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4415_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4404_ks, v4413_ks, v4380_acc, 0, 0, 0);
        float v4417_data = glb_m7[v3621_a];
        float v4419_data = glb_m7[v3623_a];
        float v4421_data = glb_m7[v3625_a];
        float v4424_ks = v3614_g ? v4417_data : (v3612_g ? v4419_data : (v3610_g ? v4421_data : 0.0f));
        tensorforge::VectorT<float, 4> v4425_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4399_ks, v4424_ks, v4383_acc, 0, 0, 0);
        tensorforge::VectorT<float, 4> v4426_acc = __builtin_amdgcn_mfma_f32_16x16x4f32(v4404_ks, v4424_ks, v4384_acc, 0, 0, 0);
        ir14[0] = (v4324_acc[0]);
        ir14[4] = (v4324_acc[1]);
        ir14[8] = (v4324_acc[2]);
        ir14[12] = (v4324_acc[3]);
        ir14[16] = (v4325_acc[0]);
        ir14[20] = (v4325_acc[1]);
        ir14[24] = (v4325_acc[2]);
        ir14[28] = (v4325_acc[3]);
        ir14[1] = (v4362_acc[0]);
        ir14[5] = (v4362_acc[1]);
        ir14[9] = (v4362_acc[2]);
        ir14[13] = (v4362_acc[3]);
        ir14[17] = (v4363_acc[0]);
        ir14[21] = (v4363_acc[1]);
        ir14[25] = (v4363_acc[2]);
        ir14[29] = (v4363_acc[3]);
        ir14[2] = (v4414_acc[0]);
        ir14[6] = (v4414_acc[1]);
        ir14[10] = (v4414_acc[2]);
        ir14[14] = (v4414_acc[3]);
        ir14[18] = (v4415_acc[0]);
        ir14[22] = (v4415_acc[1]);
        ir14[26] = (v4415_acc[2]);
        ir14[30] = (v4415_acc[3]);
        ir14[3] = (v4425_acc[0]);
        ir14[7] = (v4425_acc[1]);
        ir14[11] = (v4425_acc[2]);
        ir14[15] = (v4425_acc[3]);
        ir14[19] = (v4426_acc[0]);
        ir14[23] = (v4426_acc[1]);
        ir14[27] = (v4426_acc[2]);
        ir14[31] = (v4426_acc[3]);
        float v4462_data = glb_m7[v2483_lead];
        float v4466_data = glb_m7[v2754_lead];
        float v4470_data = glb_m7[v2761_lead];
        float v4474_data = glb_m7[v2768_lead];
        float v4476_data = glb_m7[v2933_a];
        float v4478_data = glb_m7[v2935_a];
        float v4480_data = glb_m7[v2937_a];
        float v4482_data = glb_m7[v2939_a];
        float v4484_data = glb_m7[v2941_a];
        float v4486_data = glb_m7[v2943_a];
        float v4488_data = glb_m7[v2945_a];
        float v4490_data = glb_m7[v2947_a];
        float v4492_data = glb_m7[v2949_a];
        float v4494_data = glb_m7[v2951_a];
        float v4496_data = glb_m7[v2953_a];
        float v4498_data = glb_m7[v2955_a];
        float v4500_data = glb_m7[v2957_a];
        float v4502_data = glb_m7[v2959_a];
        float v4504_data = glb_m7[v2961_a];
        float v4506_data = glb_m7[v2963_a];
        float v4508_data = glb_m7[v2965_a];
        float v4510_data = glb_m7[v2967_a];
        float v4512_data = glb_m7[v2969_a];
        float v4514_data = glb_m7[v2971_a];
        float v4516_data = glb_m7[v2973_a];
        float v4518_data = glb_m7[v2975_a];
        float v4520_data = glb_m7[v2977_a];
        float v4522_data = glb_m7[v2979_a];
        float v4524_data = glb_m7[v2981_a];
        float v4526_data = glb_m7[v2983_a];
        float v4528_data = glb_m7[v2985_a];
        float v4530_data = glb_m7[v2987_a];
        float v4532_data = glb_m7[v2989_a];
        float v4534_data = glb_m7[v2991_a];
        float v4536_data = glb_m7[v2993_a];
        float v4538_data = glb_m7[v2995_a];
        float v4540_data = glb_m7[v2997_a];
        float v4542_data = glb_m7[v2999_a];
        float v4544_data = glb_m7[v3001_a];
        float v4546_data = glb_m7[v3003_a];
        float v4548_data = glb_m7[v3005_a];
        float v4550_data = glb_m7[v3007_a];
        float v4552_data = glb_m7[v3009_a];
        float v4554_data = glb_m7[v3011_a];
        float v4556_data = glb_m7[v3013_a];
        float v4558_data = glb_m7[v3015_a];
        float v4560_data = glb_m7[v3017_a];
        float v4562_data = glb_m7[v3019_a];
        float v4564_data = glb_m7[v3768_a];
        float v4566_data = glb_m7[v3021_a];
        float v4568_data = glb_m7[v3023_a];
        float v4570_data = glb_m7[v3025_a];
        float v4572_data = glb_m7[v3776_a];
        float v4574_data = glb_m7[v3027_a];
        float v4576_data = glb_m7[v3029_a];
        float v4578_data = glb_m7[v3031_a];
        float v4580_data = glb_m7[v3784_a];
        float v4582_data = glb_m7[v3033_a];
        float v4584_data = glb_m7[v3035_a];
        float v4586_data = glb_m7[v3037_a];
        float v4588_data = glb_m7[v3792_a];
        float v4590_data = glb_m7[v3039_a];
        float v4592_data = glb_m7[v3041_a];
        float v4594_data = glb_m7[v3043_a];
        float v4596_data = glb_m7[v4595_a];
        float v4598_data = glb_m7[v3045_a];
        float v4600_data = glb_m7[v3047_a];
        float v4602_data = glb_m7[v3049_a];
        float v4604_data = glb_m7[v4603_a];
        float v4606_data = glb_m7[v3051_a];
        float v4608_data = glb_m7[v3053_a];
        float v4610_data = glb_m7[v3055_a];
        float v4612_data = glb_m7[v4611_a];
        float v4614_data = glb_m7[v3057_a];
        float v4616_data = glb_m7[v3059_a];
        float v4618_data = glb_m7[v3061_a];
        float v4620_data = glb_m7[v4619_a];
        float v4622_data = glb_m7[v3063_a];
        float v4624_data = glb_m7[v3065_a];
        float v4626_data = glb_m7[v3067_a];
        float v4628_data = glb_m7[v3824_a];
        float v4630_data = glb_m7[v3069_a];
        float v4632_data = glb_m7[v3071_a];
        float v4634_data = glb_m7[v3830_a];
        float v4636_data = glb_m7[v3073_a];
        float v4638_data = glb_m7[v3075_a];
        float v4640_data = glb_m7[v3836_a];
        float v4642_data = glb_m7[v3077_a];
        float v4644_data = glb_m7[v3079_a];
        float v4646_data = glb_m7[v3842_a];
        float v4648_data = glb_m7[v3081_a];
        float v4650_data = glb_m7[v3083_a];
        float v4652_data = glb_m7[v4651_a];
        float v4654_data = glb_m7[v3085_a];
        float v4656_data = glb_m7[v3087_a];
        float v4658_data = glb_m7[v4657_a];
        float v4660_data = glb_m7[v3089_a];
        float v4662_data = glb_m7[v3091_a];
        float v4664_data = glb_m7[v4663_a];
        float v4666_data = glb_m7[v3093_a];
        float v4668_data = glb_m7[v3095_a];
        float v4670_data = glb_m7[v4669_a];
        float v4672_data = glb_m7[v3097_a];
        float v4674_data = glb_m7[v3099_a];
        float v4676_data = glb_m7[v3864_a];
        float v4678_data = glb_m7[v3101_a];
        float v4680_data = glb_m7[v3868_a];
        float v4682_data = glb_m7[v3103_a];
        float v4684_data = glb_m7[v3872_a];
        float v4686_data = glb_m7[v3105_a];
        float v4688_data = glb_m7[v3876_a];
        float v4690_data = glb_m7[v3107_a];
        float v4703_acc{};
        float v4704_acc{};
        float v4705_acc{};
        float v4706_acc{};
        float v4709_data = r12[26];
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %22 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %23 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %25 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %26 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %29 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %36 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4703_acc), "+&v"(v4704_acc), "+&v"(v4705_acc), "+&v"(v4706_acc)
        : "v"(v4179_acc), "v"(v4462_data), "v"(v4466_data), "v"(v4470_data), "v"(v4474_data), "v"(v4476_data), "v"(v4478_data), "v"(v4480_data), "v"(v4482_data), "v"(v4484_data), "v"(v4486_data), "v"(v4488_data), "v"(v4490_data), "v"(v4492_data), "v"(v4494_data), "v"(v4496_data), "v"(v4498_data), "v"(v4500_data), "v"(v4502_data), "v"(v4504_data), "v"(v4506_data), "v"(v4508_data), "v"(v4510_data), "v"(v4512_data), "v"(v4514_data), "v"(v4516_data), "v"(v4518_data), "v"(v4520_data), "v"(v4522_data), "v"(v4524_data), "v"(v4526_data), "v"(v4528_data), "v"(v4530_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %21 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %22 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %23 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %24 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %25 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %26 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %27 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %28 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %29 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %35 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %36 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4703_acc), "+&v"(v4704_acc), "+&v"(v4705_acc), "+&v"(v4706_acc)
        : "v"(v4179_acc), "v"(v4532_data), "v"(v4534_data), "v"(v4536_data), "v"(v4538_data), "v"(v4540_data), "v"(v4542_data), "v"(v4544_data), "v"(v4546_data), "v"(v4548_data), "v"(v4550_data), "v"(v4552_data), "v"(v4554_data), "v"(v4556_data), "v"(v4558_data), "v"(v4560_data), "v"(v4562_data), "v"(v4564_data), "v"(v4566_data), "v"(v4568_data), "v"(v4570_data), "v"(v4572_data), "v"(v4574_data), "v"(v4576_data), "v"(v4578_data), "v"(v4580_data), "v"(v4582_data), "v"(v4584_data), "v"(v4586_data), "v"(v4588_data), "v"(v4590_data), "v"(v4592_data), "v"(v4594_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %4, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %16 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %4, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %19 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %20 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %22 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %23 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %24 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %25 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %26 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %27 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %28 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %29 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %30 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %31 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %32 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %33 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %4, %34 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %3, %4, %35 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %4, %36 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4703_acc), "+&v"(v4704_acc), "+&v"(v4705_acc), "+&v"(v4706_acc)
        : "v"(v4180_acc), "v"(v4596_data), "v"(v4598_data), "v"(v4600_data), "v"(v4602_data), "v"(v4604_data), "v"(v4606_data), "v"(v4608_data), "v"(v4610_data), "v"(v4612_data), "v"(v4614_data), "v"(v4616_data), "v"(v4618_data), "v"(v4620_data), "v"(v4622_data), "v"(v4624_data), "v"(v4626_data), "v"(v4628_data), "v"(v4630_data), "v"(v4632_data), "v"(v4634_data), "v"(v4636_data), "v"(v4638_data), "v"(v4640_data), "v"(v4642_data), "v"(v4644_data), "v"(v4646_data), "v"(v4648_data), "v"(v4650_data), "v"(v4652_data), "v"(v4654_data), "v"(v4656_data), "v"(v4658_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %3, %4 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %5 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %6 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %7 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %8 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %2, %3, %9 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %10 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %11 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %12 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %13 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %14 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %15 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %16 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %17 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %3, %18 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %3, %19 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %20, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %20, %22 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %20, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %20, %24 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %20, %25 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %1, %20, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4705_acc), "+&v"(v4706_acc), "+&v"(v4704_acc)
        : "v"(v4180_acc), "v"(v4660_data), "v"(v4662_data), "v"(v4664_data), "v"(v4666_data), "v"(v4668_data), "v"(v4670_data), "v"(v4672_data), "v"(v4674_data), "v"(v4676_data), "v"(v4678_data), "v"(v4680_data), "v"(v4682_data), "v"(v4684_data), "v"(v4686_data), "v"(v4688_data), "v"(v4690_data), "v"(v4709_data), "v"(v4406_data), "v"(v4417_data), "v"(v4408_data), "v"(v4419_data), "v"(v4410_data), "v"(v4421_data)
        );
        ir14[32] = v4703_acc;
        ir14[33] = v4704_acc;
        ir14[34] = v4705_acc;
        ir14[35] = v4706_acc;
        float r13[36]{};
        // r13 = load{g>r}(glb_m2);
        // r14 = ir14 + r13
        // r0 = load{g>r}(wrap_glb_m0);
        #pragma unroll
        for (int32_t v4821_i0 = 0; v4821_i0 < 2; ++v4821_i0) {
          int32_t v4823_lead = v2483_lead + (v4821_i0 * 16);
          #pragma unroll
          for (int32_t v4824_i1 = 0; v4824_i1 < 9; ++v4824_i1) {
            float v4827_data = __builtin_nontemporal_load(&wrap_glb_m0[(v4823_lead + (v4824_i1 * 56))]);
            r0[(v4821_i0 + (v4824_i1 * 3))] = v4827_data;
          }
        }
        if (v2493_g) {
          #pragma unroll
          for (int32_t v4830_i1 = 0; v4830_i1 < 9; ++v4830_i1) {
            float v4833_data = __builtin_nontemporal_load(&wrap_glb_m0[(v2761_lead + (v4830_i1 * 56))]);
            r0[(2 + (v4830_i1 * 3))] = v4833_data;
          }
        }
        if (v3117_g) {
          #pragma unroll
          for (int32_t v4737_n1 = 0; v4737_n1 < 9; ++v4737_n1) {
            int32_t v4738_a = v4737_n1 * 4;
            float v4740_data = ir14[v4738_a];
            float v4741_data = r9[v4738_a];
            r14[v4738_a] = (v4741_data + v4740_data);
          }
        }
        #pragma unroll
        for (int32_t v4743_n0 = 1; v4743_n0 < 3; ++v4743_n0) {
          #pragma unroll
          for (int32_t v4744_n1 = 0; v4744_n1 < 9; ++v4744_n1) {
            int32_t v4745_a = v4744_n1 * 4;
            int32_t v4746_a = v4743_n0 + v4745_a;
            float v4747_data = ir14[v4746_a];
            float v4748_data = r9[(v4743_n0 + v4745_a)];
            r14[v4746_a] = (v4748_data + v4747_data);
          }
        }
        // glb_m2 = store{r>g}(r14);
        if (v3117_g) {
          #pragma unroll
          for (int32_t v4756_i1 = 0; v4756_i1 < 9; ++v4756_i1) {
            float v4759_data = r14[(v4756_i1 * 4)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[(v2483_lead + (v4756_i1 * 56))], v4759_data);
            }
          }
        }
        #pragma unroll
        for (int32_t v4764_i0 = 1; v4764_i0 < 3; ++v4764_i0) {
          #pragma unroll
          for (int32_t v4765_i1 = 0; v4765_i1 < 9; ++v4765_i1) {
            float v4768_data = r14[(v4764_i0 + (v4765_i1 * 4))];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[((v2483_lead + (v4764_i0 * 16)) + (v4765_i1 * 56))], v4768_data);
            }
          }
        }
        if (v4728_g) {
          #pragma unroll
          for (int32_t v4773_i1 = 0; v4773_i1 < 9; ++v4773_i1) {
            float v4776_data = ir14[((v4773_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[(v2768_lead + (v4773_i1 * 56))], v4776_data);
            }
          }
        }
      }
    }
  }
}

