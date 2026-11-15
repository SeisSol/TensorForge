// === base name ===
kernel_c25956c76cb6077d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c25956c76cb6077d = {{16, 4, 1}, 16, 55, 1, 4, 58368, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c25956c76cb6077d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c25956c76cb6077d(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c25956c76cb6077d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c25956c76cb6077d, block.x * block.y * block.z, 7296 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (7296 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c25956c76cb6077d, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (7296 * sizeof(double)));
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
  config.block[1] = 4;
  config.block[2] = 1;
  config.sharedMemBytes = 7296 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c25956c76cb6077d(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c25956c76cb6077d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c25956c76cb6077d), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_c25956c76cb6077d, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, m5Arg, m6Arg, m6_extraOffset, m7Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(64)
 kernel_kernel_c25956c76cb6077d(const double *const * __restrict__ m0, size_t m0_extraOffset, const double *const * __restrict__ m1, size_t m1_extraOffset, double *const * __restrict__ m2, size_t m2_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m3, const double *const * __restrict__ m4, size_t m4_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m5, const double *const * __restrict__ m6, size_t m6_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m7, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (55 active) x 4 per block = block 16x4x1, 58368 B shared, occupancy grid
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
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":55,"block":[16,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":7296}],"shared_bytes":58368,"shared_elements":7296,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[56,9]],"name":"m0","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"pointer_based","alias":"star0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[56,9]],"name":"m2","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"none","alias":"kDivM0","bbox":[[0,0],[56,35]],"name":"m3","ordered":false,"parts":1,"shape":[56,56],"variant":false},{"addressing":"pointer_based","alias":"star1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"kDivM1","bbox":[[0,0],[56,35]],"name":"m5","ordered":false,"parts":1,"shape":[56,56],"variant":false},{"addressing":"pointer_based","alias":"star2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"kDivM2","bbox":[[0,0],[56,35]],"name":"m7","ordered":false,"parts":1,"shape":[56,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[54,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[54,35]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[55,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[55,35]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[35,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[1,0],[56,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[1,0],[56,35]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[56,56]},{"addressing":"strided","bbox":[[0,0],[35,9]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[336 * threadIdx.y + 5952];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m3[0];
      double * __restrict__ glb_m3 = &totalShrMem[0];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      int32_t v4104_lin = threadIdx.x + (threadIdx.y * 16);
      int32_t v4106_h = v4104_lin * 2;
      #pragma unroll
      for (int32_t v4105_i = 0; v4105_i < 30; v4105_i += 2) {
        int32_t v4108_a = v4106_h + (v4105_i * 64);
        tensorforge::VectorT<double, 2> v4109_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m3[v4108_a];
        *(tensorforge::VectorT<double, 2>*)&glb_m3[v4108_a] = v4109_ld;
      }
      bool v4110_g = v4104_lin < 40;
      if (v4110_g) {
        double v4111_ld = ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1920];
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1920] = v4111_ld;
      }
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m5 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m5[0];
      double * __restrict__ glb_m5 = &totalShrMem[1984];
      // glb_m5 = load{g>s}(ptr_glb_m5[0, 1])
      #pragma unroll
      for (int32_t v4114_i = 0; v4114_i < 30; v4114_i += 2) {
        int32_t v4117_a = v4106_h + (v4114_i * 64);
        tensorforge::VectorT<double, 2> v4118_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m5[v4117_a];
        *(tensorforge::VectorT<double, 2>*)&glb_m5[v4117_a] = v4118_ld;
      }
      if (v4110_g) {
        double v4119_ld = ptr_glb_m5[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1920];
        glb_m5[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1920] = v4119_ld;
      }
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m7 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m7[0];
      double * __restrict__ glb_m7 = &totalShrMem[3968];
      // glb_m7 = load{g>s}(ptr_glb_m7[0, 1])
      #pragma unroll
      for (int32_t v4122_i = 0; v4122_i < 30; v4122_i += 2) {
        int32_t v4125_a = v4106_h + (v4122_i * 64);
        tensorforge::VectorT<double, 2> v4126_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m7[v4125_a];
        *(tensorforge::VectorT<double, 2>*)&glb_m7[v4125_a] = v4126_ld;
      }
      if (v4110_g) {
        double v4127_ld = ptr_glb_m7[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1920];
        glb_m7[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1920] = v4127_ld;
      }
      __syncthreads();
      double * __restrict__ s0 = &localShrMem0[0];
      double * __restrict__ s1 = &localShrMem0[0];
      double * __restrict__ s2 = &localShrMem0[0];
      size_t v4132_batchIdLane0 = threadIdx.y % 4;
      size_t v4135_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v4154_lead = threadIdx.x % 16;
      bool v4164_g = v4154_lead < 3;
      bool v4179_g = v4154_lead < 9;
      int32_t v4267_m = threadIdx.y % 4;
      int32_t v4279_sh = (threadIdx.x / 4) * 35;
      int32_t v4281_sh = ((threadIdx.x % 4) - v4267_m) * 336;
      int32_t v4282_sh = (v4267_m + v4279_sh) + v4281_sh;
      int32_t v4286_a = v4282_sh + 140;
      int32_t v4288_r = v4267_m * 56;
      int32_t v4291_a = v4154_lead + v4288_r;
      int32_t v4297_lead = v4154_lead + 16_i32;
      int32_t v4298_a = v4297_lead + v4288_r;
      int32_t v4304_lead = v4154_lead + 32_i32;
      int32_t v4305_a = v4304_lead + v4288_r;
      int32_t v4311_lead = v4154_lead + 48_i32;
      int32_t v4312_a = v4311_lead + v4288_r;
      int32_t v4317_a = v4282_sh + 4;
      int32_t v4320_a = v4317_a + 140;
      int32_t v4322_a = v4291_a + 224;
      int32_t v4326_a = v4298_a + 224;
      int32_t v4330_a = v4305_a + 224;
      int32_t v4334_a = v4312_a + 224;
      int32_t v4338_a = v4282_sh + 8;
      int32_t v4341_a = v4338_a + 140;
      int32_t v4343_a = v4291_a + 448;
      int32_t v4347_a = v4298_a + 448;
      int32_t v4351_a = v4305_a + 448;
      int32_t v4355_a = v4312_a + 448;
      int32_t v4359_a = v4282_sh + 12;
      int32_t v4362_a = v4359_a + 140;
      int32_t v4364_a = v4298_a + 672;
      int32_t v4368_a = v4305_a + 672;
      int32_t v4372_a = v4312_a + 672;
      int32_t v4376_a = v4282_sh + 16;
      int32_t v4379_a = v4376_a + 140;
      int32_t v4381_a = v4298_a + 896;
      int32_t v4385_a = v4305_a + 896;
      int32_t v4389_a = v4312_a + 896;
      int32_t v4393_a = v4282_sh + 20;
      int32_t v4396_a = v4393_a + 140;
      int32_t v4398_a = v4305_a + 1120;
      int32_t v4402_a = v4312_a + 1120;
      int32_t v4406_a = v4282_sh + 24;
      int32_t v4409_a = v4406_a + 140;
      int32_t v4411_a = v4305_a + 1344;
      int32_t v4415_a = v4312_a + 1344;
      int32_t v4419_a = v4282_sh + 28;
      int32_t v4422_a = v4419_a + 140;
      int32_t v4424_a = v4312_a + 1568;
      int32_t v4476_a = v4154_lead + 56;
      int32_t v4478_a = v4297_lead + 56;
      int32_t v4480_a = v4304_lead + 56;
      int32_t v4482_a = v4311_lead + 56;
      int32_t v4484_a = v4154_lead + 112;
      int32_t v4486_a = v4297_lead + 112;
      int32_t v4488_a = v4304_lead + 112;
      int32_t v4490_a = v4311_lead + 112;
      int32_t v4492_a = v4154_lead + 168;
      int32_t v4494_a = v4297_lead + 168;
      int32_t v4496_a = v4304_lead + 168;
      int32_t v4498_a = v4311_lead + 168;
      int32_t v4500_a = v4154_lead + 224;
      int32_t v4502_a = v4297_lead + 224;
      int32_t v4504_a = v4304_lead + 224;
      int32_t v4506_a = v4311_lead + 224;
      int32_t v4508_a = v4154_lead + 280;
      int32_t v4510_a = v4297_lead + 280;
      int32_t v4512_a = v4304_lead + 280;
      int32_t v4514_a = v4311_lead + 280;
      int32_t v4516_a = v4154_lead + 336;
      int32_t v4518_a = v4297_lead + 336;
      int32_t v4520_a = v4304_lead + 336;
      int32_t v4522_a = v4311_lead + 336;
      int32_t v4524_a = v4154_lead + 392;
      int32_t v4526_a = v4297_lead + 392;
      int32_t v4528_a = v4304_lead + 392;
      int32_t v4530_a = v4311_lead + 392;
      int32_t v4532_a = v4154_lead + 448;
      int32_t v4534_a = v4297_lead + 448;
      int32_t v4536_a = v4304_lead + 448;
      int32_t v4538_a = v4311_lead + 448;
      int32_t v4540_a = v4154_lead + 504;
      int32_t v4542_a = v4297_lead + 504;
      int32_t v4544_a = v4304_lead + 504;
      int32_t v4546_a = v4311_lead + 504;
      int32_t v4548_a = v4154_lead + 560;
      int32_t v4550_a = v4297_lead + 560;
      int32_t v4552_a = v4304_lead + 560;
      int32_t v4554_a = v4311_lead + 560;
      int32_t v4556_a = v4154_lead + 616;
      int32_t v4558_a = v4297_lead + 616;
      int32_t v4560_a = v4304_lead + 616;
      int32_t v4562_a = v4311_lead + 616;
      int32_t v4564_a = v4297_lead + 672;
      int32_t v4566_a = v4304_lead + 672;
      int32_t v4568_a = v4311_lead + 672;
      int32_t v4570_a = v4297_lead + 728;
      int32_t v4572_a = v4304_lead + 728;
      int32_t v4574_a = v4311_lead + 728;
      int32_t v4576_a = v4297_lead + 784;
      int32_t v4578_a = v4304_lead + 784;
      int32_t v4580_a = v4311_lead + 784;
      int32_t v4582_a = v4297_lead + 840;
      int32_t v4584_a = v4304_lead + 840;
      int32_t v4586_a = v4311_lead + 840;
      int32_t v4588_a = v4297_lead + 896;
      int32_t v4590_a = v4304_lead + 896;
      int32_t v4592_a = v4311_lead + 896;
      int32_t v4594_a = v4297_lead + 952;
      int32_t v4596_a = v4304_lead + 952;
      int32_t v4598_a = v4311_lead + 952;
      int32_t v4600_a = v4297_lead + 1008;
      int32_t v4602_a = v4304_lead + 1008;
      int32_t v4604_a = v4311_lead + 1008;
      int32_t v4606_a = v4297_lead + 1064;
      int32_t v4608_a = v4304_lead + 1064;
      int32_t v4610_a = v4311_lead + 1064;
      int32_t v4612_a = v4304_lead + 1120;
      int32_t v4614_a = v4311_lead + 1120;
      int32_t v4616_a = v4304_lead + 1176;
      int32_t v4618_a = v4311_lead + 1176;
      int32_t v4620_a = v4304_lead + 1232;
      int32_t v4622_a = v4311_lead + 1232;
      int32_t v4624_a = v4304_lead + 1288;
      int32_t v4626_a = v4311_lead + 1288;
      int32_t v4628_a = v4304_lead + 1344;
      int32_t v4630_a = v4311_lead + 1344;
      int32_t v4632_a = v4304_lead + 1400;
      int32_t v4634_a = v4311_lead + 1400;
      int32_t v4636_a = v4304_lead + 1456;
      int32_t v4638_a = v4311_lead + 1456;
      int32_t v4640_a = v4304_lead + 1512;
      int32_t v4642_a = v4311_lead + 1512;
      int32_t v4644_a = v4311_lead + 1568;
      int32_t v4646_a = v4311_lead + 1624;
      int32_t v4648_a = v4311_lead + 1680;
      int32_t v4650_a = v4311_lead + 1736;
      bool v4660_g = v4154_lead >= 1;
      bool v4678_g = v4154_lead < 6;
      int32_t v4940_a = v4291_a + 672;
      int32_t v4978_a = v4298_a + 1120;
      int32_t v5008_a = v4305_a + 1568;
      int32_t v5022_a = ((((min((max((v4267_m + 32), 0)), 34)) - 32) + v4279_sh) + v4281_sh) + 32;
      bool v5025_g = v4267_m == 2;
      bool v5027_g = v4267_m == 1;
      bool v5029_g = v4267_m == 0;
      int32_t v5031_a = v5022_a + 140;
      int32_t v5036_a = v4311_lead + 1792;
      int32_t v5038_a = v4311_lead + 1848;
      int32_t v5040_a = v4311_lead + 1904;
      int32_t v5183_a = v4154_lead + 672;
      int32_t v5191_a = v4154_lead + 728;
      int32_t v5199_a = v4154_lead + 784;
      int32_t v5207_a = v4154_lead + 840;
      int32_t v5239_a = v4297_lead + 1120;
      int32_t v5245_a = v4297_lead + 1176;
      int32_t v5251_a = v4297_lead + 1232;
      int32_t v5257_a = v4297_lead + 1288;
      int32_t v5279_a = v4304_lead + 1568;
      int32_t v5283_a = v4304_lead + 1624;
      int32_t v5287_a = v4304_lead + 1680;
      int32_t v5291_a = v4304_lead + 1736;
      bool v5326_g = v4154_lead < 7;
      int32_t v5609_a = v4291_a + 896;
      int32_t v5647_a = v4298_a + 1344;
      int32_t v5692_a = v4304_lead + 1792;
      int32_t v5694_a = v4304_lead + 1848;
      int32_t v5696_a = v4304_lead + 1904;
      int32_t v5882_a = v4154_lead + 896;
      int32_t v5890_a = v4154_lead + 952;
      int32_t v5898_a = v4154_lead + 1008;
      int32_t v5906_a = v4154_lead + 1064;
      int32_t v5938_a = v4297_lead + 1344;
      int32_t v5944_a = v4297_lead + 1400;
      int32_t v5950_a = v4297_lead + 1456;
      int32_t v5956_a = v4297_lead + 1512;
      bool v6015_g = v4154_lead < 8;
      double r0[27]{};
      size_t v6085_row = v4135_batchIdGroup0First + v4132_batchIdLane0;
      const bool peel_batchIdActive0 = v6085_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v6085_row]));
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[(peel_batchIdActive0 ? v6085_row : v4135_batchIdGroup0First)][0 + m0_extraOffset];
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v6093_i0 = 0; v6093_i0 < 2; ++v6093_i0) {
        int32_t v6095_lead = v4154_lead + (v6093_i0 * 16);
        #pragma unroll
        for (int32_t v6096_i1 = 0; v6096_i1 < 9; ++v6096_i1) {
          double v6099_data = __builtin_nontemporal_load(&peel_glb_m0[(v6095_lead + (v6096_i1 * 56))]);
          r0[(v6093_i0 + (v6096_i1 * 3))] = v6099_data;
        }
      }
      if (v4164_g) {
        #pragma unroll
        for (int32_t v6102_i1 = 0; v6102_i1 < 9; ++v6102_i1) {
          double v6105_data = __builtin_nontemporal_load(&peel_glb_m0[(v4304_lead + (v6102_i1 * 56))]);
          r0[(2 + (v6102_i1 * 3))] = v6105_data;
        }
      }
      for (size_t v4136_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v4136_batchIdGroup0 < numElements0; v4136_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v6082_groupNextAhead = v4136_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v6084_groupNext = (v6082_groupNextAhead < numElements0) ? v6082_groupNextAhead : v4136_batchIdGroup0;
        size_t v6089_row = v6084_groupNext + v4132_batchIdLane0;
        const bool wrap_batchIdActive0 = v6089_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v6089_row]));
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[(wrap_batchIdActive0 ? v6089_row : v6084_groupNext)][0 + m0_extraOffset];
        size_t v4137_row = v4136_batchIdGroup0 + v4132_batchIdLane0;
        const bool batchIdActive0 = v4137_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v4137_row]));
        size_t v4139_batchId0 = batchIdActive0 ? v4137_row : v4136_batchIdGroup0;
        size_t v4140_ahead1 = v4139_batchId0 + (gridDim.x * blockDim.y);
        size_t v4142_batchId1 = (v4140_ahead1 < numElements0) ? v4140_ahead1 : v4139_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v4139_batchId0][0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m2[v4139_batchId0][0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m4[v4139_batchId0][0 + m4_extraOffset];
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m6[v4139_batchId0][0 + m6_extraOffset];
        double r1[9]{};
        // r1 = load{g>r}(glb_m1);
        #pragma unroll
        for (int32_t v4174_i1 = 0; v4174_i1 < 9; ++v4174_i1) {
          double v4180_data = v4179_g ? (__builtin_nontemporal_load(&glb_m1[(v4154_lead + (v4174_i1 * 9))])) : (0.0);
          r1[v4174_i1] = v4180_data;
        }
        double r2[27]{};
        // r2 = +(r0 * r1) + None
        // [(0, 35), (0, 9)] [(0, 9)]
        double v4183_data = r0[0];
        double v4184_data = r0[1];
        double v4185_data = r0[2];
        double v4186_data = r0[3];
        double v4187_data = r0[4];
        double v4188_data = r0[5];
        double v4189_data = r0[6];
        double v4190_data = r0[7];
        double v4191_data = r0[8];
        double v4192_data = r0[9];
        double v4193_data = r0[10];
        double v4194_data = r0[11];
        double v4195_data = r0[12];
        double v4196_data = r0[13];
        double v4197_data = r0[14];
        double v4198_data = r0[15];
        double v4199_data = r0[16];
        double v4200_data = r0[17];
        double v4201_data = r0[18];
        double v4202_data = r0[19];
        double v4203_data = r0[20];
        double v4204_data = r0[21];
        double v4205_data = r0[22];
        double v4206_data = r0[23];
        double v4207_data = r0[24];
        double v4208_data = r0[25];
        double v4209_data = r0[26];
        double v4210_acc{};
        double v4211_acc{};
        double v4212_acc{};
        double v4213_acc{};
        double v4214_acc{};
        double v4215_acc{};
        double v4216_acc{};
        double v4217_acc{};
        double v4218_acc{};
        double v4219_acc{};
        double v4220_acc{};
        double v4221_acc{};
        double v4222_acc{};
        double v4223_acc{};
        double v4224_acc{};
        double v4225_acc{};
        double v4226_acc{};
        double v4227_acc{};
        double v4228_acc{};
        double v4229_acc{};
        double v4230_acc{};
        double v4231_acc{};
        double v4232_acc{};
        double v4233_acc{};
        double v4234_acc{};
        double v4235_acc{};
        double v4236_acc{};
        double v4237_data = r1[0];
        double v4238_data = r1[1];
        double v4239_data = r1[2];
        double v4240_data = r1[3];
        double v4241_data = r1[4];
        double v4242_data = r1[5];
        double v4243_data = r1[6];
        double v4244_data = r1[7];
        double v4245_data = r1[8];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %29 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %30 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %31 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %32 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %33 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %34, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %34, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %34, %9 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %34, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %34, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4210_acc), "+&v"(v4211_acc), "+&v"(v4212_acc), "+&v"(v4213_acc), "+&v"(v4214_acc), "+&v"(v4215_acc)
        : "v"(v4237_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4238_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %30 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %31 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %32 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %33 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %34 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4215_acc), "+&v"(v4213_acc), "+&v"(v4214_acc), "+&v"(v4216_acc), "+&v"(v4217_acc), "+&v"(v4218_acc)
        : "v"(v4238_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4239_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %25 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %26 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %27 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %28 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %29 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %30 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %31 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %32 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %33 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %34 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4217_acc), "+&v"(v4218_acc), "+&v"(v4216_acc), "+&v"(v4219_acc), "+&v"(v4220_acc), "+&v"(v4221_acc)
        : "v"(v4239_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4240_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %22 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %24 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %25 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %27 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %28 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %29 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %30 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %31 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %32 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %33 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %34 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %9 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4219_acc), "+&v"(v4220_acc), "+&v"(v4221_acc), "+&v"(v4222_acc), "+&v"(v4223_acc), "+&v"(v4224_acc)
        : "v"(v4240_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4241_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %15 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %16 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %17 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %18 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %19 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %20 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %21 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %22 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %23 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %24 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %25 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %26 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %27 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %28 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %29 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %30 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %31 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %32 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %33 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %34 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %10 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %11 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4224_acc), "+&v"(v4222_acc), "+&v"(v4223_acc), "+&v"(v4225_acc), "+&v"(v4226_acc), "+&v"(v4227_acc)
        : "v"(v4241_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4242_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %8, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %8, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %14 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %15 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %16 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %17 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %18 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %19 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %20 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %21 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %22 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %23 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %25 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %26 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %29 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %36 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %37, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %6, %37, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %7, %37, %14 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4226_acc), "+&v"(v4227_acc), "+&v"(v4228_acc), "+&v"(v4229_acc), "+&v"(v4230_acc), "+&v"(v4231_acc), "+&v"(v4232_acc), "+&v"(v4233_acc)
        : "v"(v4242_data), "v"(v4208_data), "v"(v4209_data), "v"(v4243_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4244_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %30 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %32 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %33 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %31, %34 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %31, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4231_acc), "+&v"(v4232_acc), "+&v"(v4233_acc), "+&v"(v4234_acc), "+&v"(v4235_acc), "+&v"(v4236_acc)
        : "v"(v4244_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4245_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4236_acc), "+&v"(v4234_acc), "+&v"(v4235_acc)
        : "v"(v4245_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data)
        );
        r2[0] = v4210_acc;
        r2[1] = v4211_acc;
        r2[2] = v4212_acc;
        r2[3] = v4213_acc;
        r2[4] = v4214_acc;
        r2[5] = v4215_acc;
        r2[6] = v4216_acc;
        r2[7] = v4217_acc;
        r2[8] = v4218_acc;
        r2[9] = v4219_acc;
        r2[10] = v4220_acc;
        r2[11] = v4221_acc;
        r2[12] = v4222_acc;
        r2[13] = v4223_acc;
        r2[14] = v4224_acc;
        r2[15] = v4225_acc;
        r2[16] = v4226_acc;
        r2[17] = v4227_acc;
        r2[18] = v4228_acc;
        r2[19] = v4229_acc;
        r2[20] = v4230_acc;
        r2[21] = v4231_acc;
        r2[22] = v4232_acc;
        r2[23] = v4233_acc;
        r2[24] = v4234_acc;
        r2[25] = v4235_acc;
        r2[26] = v4236_acc;
        // r2 = 0 past (0, 35)
        // s0 = store{r>s}(localShrMem0, r2);
        #pragma unroll
        for (int32_t v4247_i0 = 0; v4247_i0 < 2; ++v4247_i0) {
          int32_t v4253_lead = v4154_lead + (v4247_i0 * 16);
          #pragma unroll
          for (int32_t v4248_i1 = 0; v4248_i1 < 9; ++v4248_i1) {
            double v4251_data = r2[(v4247_i0 + (v4248_i1 * 3))];
            s0[(v4253_lead + (v4248_i1 * 35))] = v4251_data;
          }
        }
        if (v4164_g) {
          #pragma unroll
          for (int32_t v4256_i1 = 0; v4256_i1 < 9; ++v4256_i1) {
            double v4259_data = r2[(2 + (v4256_i1 * 3))];
            s0[(v4304_lead + (v4256_i1 * 35))] = v4259_data;
          }
        }
        double r4[36]{};
        // ir4 = +(glb_m3 * s0)
        // [(1, 54), (0, 9)] [(0, 35)]
        double ir4[36]{};
        tensorforge::VectorT<double, 4> v4271_acc{};
        tensorforge::VectorT<double, 4> v4272_acc{};
        tensorforge::VectorT<double, 4> v4273_acc{};
        tensorforge::VectorT<double, 4> v4274_acc{};
        tensorforge::VectorT<double, 4> v4275_acc{};
        tensorforge::VectorT<double, 4> v4276_acc{};
        tensorforge::VectorT<double, 4> v4277_acc{};
        tensorforge::VectorT<double, 4> v4278_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        double v4285_data = s0[v4282_sh];
        double v4287_data = s0[v4286_a];
        double v4293_data = glb_m3[v4291_a];
        tensorforge::VectorT<double, 4> v4294_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4285_data, v4293_data, v4271_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4295_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4287_data, v4293_data, v4272_acc, 0, 0, 0);
        double v4300_data = glb_m3[v4298_a];
        tensorforge::VectorT<double, 4> v4301_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4285_data, v4300_data, v4273_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4302_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4287_data, v4300_data, v4274_acc, 0, 0, 0);
        double v4307_data = glb_m3[v4305_a];
        tensorforge::VectorT<double, 4> v4308_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4285_data, v4307_data, v4275_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4309_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4287_data, v4307_data, v4276_acc, 0, 0, 0);
        double v4314_data = glb_m3[v4312_a];
        tensorforge::VectorT<double, 4> v4315_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4285_data, v4314_data, v4277_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4316_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4287_data, v4314_data, v4278_acc, 0, 0, 0);
        double v4319_data = s0[v4317_a];
        double v4321_data = s0[v4320_a];
        double v4323_data = glb_m3[v4322_a];
        tensorforge::VectorT<double, 4> v4324_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4319_data, v4323_data, v4294_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4325_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4321_data, v4323_data, v4295_acc, 0, 0, 0);
        double v4327_data = glb_m3[v4326_a];
        tensorforge::VectorT<double, 4> v4328_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4319_data, v4327_data, v4301_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4329_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4321_data, v4327_data, v4302_acc, 0, 0, 0);
        double v4331_data = glb_m3[v4330_a];
        tensorforge::VectorT<double, 4> v4332_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4319_data, v4331_data, v4308_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4333_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4321_data, v4331_data, v4309_acc, 0, 0, 0);
        double v4335_data = glb_m3[v4334_a];
        tensorforge::VectorT<double, 4> v4336_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4319_data, v4335_data, v4315_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4337_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4321_data, v4335_data, v4316_acc, 0, 0, 0);
        double v4340_data = s0[v4338_a];
        double v4342_data = s0[v4341_a];
        double v4344_data = glb_m3[v4343_a];
        tensorforge::VectorT<double, 4> v4345_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4340_data, v4344_data, v4324_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4346_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4342_data, v4344_data, v4325_acc, 0, 0, 0);
        double v4348_data = glb_m3[v4347_a];
        tensorforge::VectorT<double, 4> v4349_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4340_data, v4348_data, v4328_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4350_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4342_data, v4348_data, v4329_acc, 0, 0, 0);
        double v4352_data = glb_m3[v4351_a];
        tensorforge::VectorT<double, 4> v4353_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4340_data, v4352_data, v4332_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4354_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4342_data, v4352_data, v4333_acc, 0, 0, 0);
        double v4356_data = glb_m3[v4355_a];
        tensorforge::VectorT<double, 4> v4357_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4340_data, v4356_data, v4336_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4358_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4342_data, v4356_data, v4337_acc, 0, 0, 0);
        double v4361_data = s0[v4359_a];
        double v4363_data = s0[v4362_a];
        double v4365_data = glb_m3[v4364_a];
        tensorforge::VectorT<double, 4> v4366_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4361_data, v4365_data, v4349_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4367_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4363_data, v4365_data, v4350_acc, 0, 0, 0);
        double v4369_data = glb_m3[v4368_a];
        tensorforge::VectorT<double, 4> v4370_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4361_data, v4369_data, v4353_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4371_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4363_data, v4369_data, v4354_acc, 0, 0, 0);
        double v4373_data = glb_m3[v4372_a];
        tensorforge::VectorT<double, 4> v4374_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4361_data, v4373_data, v4357_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4375_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4363_data, v4373_data, v4358_acc, 0, 0, 0);
        double v4378_data = s0[v4376_a];
        double v4380_data = s0[v4379_a];
        double v4382_data = glb_m3[v4381_a];
        tensorforge::VectorT<double, 4> v4383_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4378_data, v4382_data, v4366_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4384_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4380_data, v4382_data, v4367_acc, 0, 0, 0);
        double v4386_data = glb_m3[v4385_a];
        tensorforge::VectorT<double, 4> v4387_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4378_data, v4386_data, v4370_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4388_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4380_data, v4386_data, v4371_acc, 0, 0, 0);
        double v4390_data = glb_m3[v4389_a];
        tensorforge::VectorT<double, 4> v4391_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4378_data, v4390_data, v4374_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4392_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4380_data, v4390_data, v4375_acc, 0, 0, 0);
        double v4395_data = s0[v4393_a];
        double v4397_data = s0[v4396_a];
        double v4399_data = glb_m3[v4398_a];
        tensorforge::VectorT<double, 4> v4400_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4395_data, v4399_data, v4387_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4401_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4397_data, v4399_data, v4388_acc, 0, 0, 0);
        double v4403_data = glb_m3[v4402_a];
        tensorforge::VectorT<double, 4> v4404_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4395_data, v4403_data, v4391_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4405_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4397_data, v4403_data, v4392_acc, 0, 0, 0);
        double v4408_data = s0[v4406_a];
        double v4410_data = s0[v4409_a];
        double v4412_data = glb_m3[v4411_a];
        tensorforge::VectorT<double, 4> v4413_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4408_data, v4412_data, v4400_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4414_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4410_data, v4412_data, v4401_acc, 0, 0, 0);
        double v4416_data = glb_m3[v4415_a];
        tensorforge::VectorT<double, 4> v4417_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4408_data, v4416_data, v4404_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4418_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4410_data, v4416_data, v4405_acc, 0, 0, 0);
        double v4421_data = s0[v4419_a];
        double v4423_data = s0[v4422_a];
        double v4425_data = glb_m3[v4424_a];
        tensorforge::VectorT<double, 4> v4426_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4421_data, v4425_data, v4417_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4427_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4423_data, v4425_data, v4418_acc, 0, 0, 0);
        ir4[0] = (v4345_acc[0]);
        ir4[4] = (v4345_acc[1]);
        ir4[8] = (v4345_acc[2]);
        ir4[12] = (v4345_acc[3]);
        ir4[16] = (v4346_acc[0]);
        ir4[20] = (v4346_acc[1]);
        ir4[24] = (v4346_acc[2]);
        ir4[28] = (v4346_acc[3]);
        ir4[1] = (v4383_acc[0]);
        ir4[5] = (v4383_acc[1]);
        ir4[9] = (v4383_acc[2]);
        ir4[13] = (v4383_acc[3]);
        ir4[17] = (v4384_acc[0]);
        ir4[21] = (v4384_acc[1]);
        ir4[25] = (v4384_acc[2]);
        ir4[29] = (v4384_acc[3]);
        ir4[2] = (v4413_acc[0]);
        ir4[6] = (v4413_acc[1]);
        ir4[10] = (v4413_acc[2]);
        ir4[14] = (v4413_acc[3]);
        ir4[18] = (v4414_acc[0]);
        ir4[22] = (v4414_acc[1]);
        ir4[26] = (v4414_acc[2]);
        ir4[30] = (v4414_acc[3]);
        ir4[3] = (v4426_acc[0]);
        ir4[7] = (v4426_acc[1]);
        ir4[11] = (v4426_acc[2]);
        ir4[15] = (v4426_acc[3]);
        ir4[19] = (v4427_acc[0]);
        ir4[23] = (v4427_acc[1]);
        ir4[27] = (v4427_acc[2]);
        ir4[31] = (v4427_acc[3]);
        double v4463_data = glb_m3[v4154_lead];
        double v4467_data = glb_m3[v4297_lead];
        double v4471_data = glb_m3[v4304_lead];
        double v4475_data = glb_m3[v4311_lead];
        double v4477_data = glb_m3[v4476_a];
        double v4479_data = glb_m3[v4478_a];
        double v4481_data = glb_m3[v4480_a];
        double v4483_data = glb_m3[v4482_a];
        double v4485_data = glb_m3[v4484_a];
        double v4487_data = glb_m3[v4486_a];
        double v4489_data = glb_m3[v4488_a];
        double v4491_data = glb_m3[v4490_a];
        double v4493_data = glb_m3[v4492_a];
        double v4495_data = glb_m3[v4494_a];
        double v4497_data = glb_m3[v4496_a];
        double v4499_data = glb_m3[v4498_a];
        double v4501_data = glb_m3[v4500_a];
        double v4503_data = glb_m3[v4502_a];
        double v4505_data = glb_m3[v4504_a];
        double v4507_data = glb_m3[v4506_a];
        double v4509_data = glb_m3[v4508_a];
        double v4511_data = glb_m3[v4510_a];
        double v4513_data = glb_m3[v4512_a];
        double v4515_data = glb_m3[v4514_a];
        double v4517_data = glb_m3[v4516_a];
        double v4519_data = glb_m3[v4518_a];
        double v4521_data = glb_m3[v4520_a];
        double v4523_data = glb_m3[v4522_a];
        double v4525_data = glb_m3[v4524_a];
        double v4527_data = glb_m3[v4526_a];
        double v4529_data = glb_m3[v4528_a];
        double v4531_data = glb_m3[v4530_a];
        double v4533_data = glb_m3[v4532_a];
        double v4535_data = glb_m3[v4534_a];
        double v4537_data = glb_m3[v4536_a];
        double v4539_data = glb_m3[v4538_a];
        double v4541_data = glb_m3[v4540_a];
        double v4543_data = glb_m3[v4542_a];
        double v4545_data = glb_m3[v4544_a];
        double v4547_data = glb_m3[v4546_a];
        double v4549_data = glb_m3[v4548_a];
        double v4551_data = glb_m3[v4550_a];
        double v4553_data = glb_m3[v4552_a];
        double v4555_data = glb_m3[v4554_a];
        double v4557_data = glb_m3[v4556_a];
        double v4559_data = glb_m3[v4558_a];
        double v4561_data = glb_m3[v4560_a];
        double v4563_data = glb_m3[v4562_a];
        double v4565_data = glb_m3[v4564_a];
        double v4567_data = glb_m3[v4566_a];
        double v4569_data = glb_m3[v4568_a];
        double v4571_data = glb_m3[v4570_a];
        double v4573_data = glb_m3[v4572_a];
        double v4575_data = glb_m3[v4574_a];
        double v4577_data = glb_m3[v4576_a];
        double v4579_data = glb_m3[v4578_a];
        double v4581_data = glb_m3[v4580_a];
        double v4583_data = glb_m3[v4582_a];
        double v4585_data = glb_m3[v4584_a];
        double v4587_data = glb_m3[v4586_a];
        double v4589_data = glb_m3[v4588_a];
        double v4591_data = glb_m3[v4590_a];
        double v4593_data = glb_m3[v4592_a];
        double v4595_data = glb_m3[v4594_a];
        double v4597_data = glb_m3[v4596_a];
        double v4599_data = glb_m3[v4598_a];
        double v4601_data = glb_m3[v4600_a];
        double v4603_data = glb_m3[v4602_a];
        double v4605_data = glb_m3[v4604_a];
        double v4607_data = glb_m3[v4606_a];
        double v4609_data = glb_m3[v4608_a];
        double v4611_data = glb_m3[v4610_a];
        double v4613_data = glb_m3[v4612_a];
        double v4615_data = glb_m3[v4614_a];
        double v4617_data = glb_m3[v4616_a];
        double v4619_data = glb_m3[v4618_a];
        double v4621_data = glb_m3[v4620_a];
        double v4623_data = glb_m3[v4622_a];
        double v4625_data = glb_m3[v4624_a];
        double v4627_data = glb_m3[v4626_a];
        double v4629_data = glb_m3[v4628_a];
        double v4631_data = glb_m3[v4630_a];
        double v4633_data = glb_m3[v4632_a];
        double v4635_data = glb_m3[v4634_a];
        double v4637_data = glb_m3[v4636_a];
        double v4639_data = glb_m3[v4638_a];
        double v4641_data = glb_m3[v4640_a];
        double v4643_data = glb_m3[v4642_a];
        double v4645_data = glb_m3[v4644_a];
        double v4647_data = glb_m3[v4646_a];
        double v4649_data = glb_m3[v4648_a];
        double v4651_data = glb_m3[v4650_a];
        double v4652_acc{};
        double v4653_acc{};
        double v4654_acc{};
        double v4655_acc{};
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
        : "+&v"(v4652_acc), "+&v"(v4653_acc), "+&v"(v4654_acc), "+&v"(v4655_acc)
        : "v"(v4234_acc), "v"(v4463_data), "v"(v4467_data), "v"(v4471_data), "v"(v4475_data), "v"(v4477_data), "v"(v4479_data), "v"(v4481_data), "v"(v4483_data), "v"(v4485_data), "v"(v4487_data), "v"(v4489_data), "v"(v4491_data), "v"(v4493_data), "v"(v4495_data), "v"(v4497_data), "v"(v4499_data), "v"(v4501_data), "v"(v4503_data), "v"(v4505_data), "v"(v4507_data), "v"(v4509_data), "v"(v4511_data), "v"(v4513_data), "v"(v4515_data), "v"(v4517_data), "v"(v4519_data), "v"(v4521_data), "v"(v4523_data), "v"(v4525_data), "v"(v4527_data), "v"(v4529_data), "v"(v4531_data)
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
        "v_fmac_f64_dpp %2, %4, %19 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %20 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %21 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %22 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %23 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %24 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %25 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %26 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %27 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %28 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %29 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %30 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %31 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %32 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %33, %34 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %33, %35 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %33, %36 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %33, %37 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4652_acc), "+&v"(v4653_acc), "+&v"(v4654_acc), "+&v"(v4655_acc)
        : "v"(v4234_acc), "v"(v4533_data), "v"(v4535_data), "v"(v4537_data), "v"(v4539_data), "v"(v4541_data), "v"(v4543_data), "v"(v4545_data), "v"(v4547_data), "v"(v4549_data), "v"(v4551_data), "v"(v4553_data), "v"(v4555_data), "v"(v4557_data), "v"(v4559_data), "v"(v4561_data), "v"(v4563_data), "v"(v4565_data), "v"(v4567_data), "v"(v4569_data), "v"(v4571_data), "v"(v4573_data), "v"(v4575_data), "v"(v4577_data), "v"(v4579_data), "v"(v4581_data), "v"(v4583_data), "v"(v4585_data), "v"(v4587_data), "v"(v4235_acc), "v"(v4589_data), "v"(v4591_data), "v"(v4593_data), "v"(v4595_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %12 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %13 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %15 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %22 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %23 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %24 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %25 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %26 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %27 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %28 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %29 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %30 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %31 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4654_acc), "+&v"(v4655_acc), "+&v"(v4653_acc)
        : "v"(v4235_acc), "v"(v4597_data), "v"(v4599_data), "v"(v4601_data), "v"(v4603_data), "v"(v4605_data), "v"(v4607_data), "v"(v4609_data), "v"(v4611_data), "v"(v4613_data), "v"(v4615_data), "v"(v4617_data), "v"(v4619_data), "v"(v4621_data), "v"(v4623_data), "v"(v4625_data), "v"(v4627_data), "v"(v4629_data), "v"(v4631_data), "v"(v4633_data), "v"(v4635_data), "v"(v4637_data), "v"(v4639_data), "v"(v4641_data), "v"(v4643_data), "v"(v4645_data), "v"(v4647_data), "v"(v4649_data), "v"(v4651_data)
        );
        ir4[32] = v4652_acc;
        ir4[33] = v4653_acc;
        ir4[34] = v4654_acc;
        ir4[35] = v4655_acc;
        double r3[36]{};
        // r3 = load{g>r}(glb_m2);
        double r5[27]{};
        // r5 = load{g>r}(glb_m0);
        // r4 = ir4 + r3
        // glb_m2 = store{r>g}(r4);
        if (v4678_g) {
          #pragma unroll
          for (int32_t v4723_i1 = 0; v4723_i1 < 9; ++v4723_i1) {
            double v4726_data = ir4[((v4723_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m2[(v4311_lead + (v4723_i1 * 56))], v4726_data);
            }
          }
        }
        double r6[9]{};
        // r6 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v4750_i1 = 0; v4750_i1 < 9; ++v4750_i1) {
          double v4756_data = v4179_g ? (__builtin_nontemporal_load(&glb_m4[(v4154_lead + (v4750_i1 * 9))])) : (0.0);
          r6[v4750_i1] = v4756_data;
        }
        double r7[27]{};
        // r7 = +(r5 * r6) + None
        // [(0, 35), (0, 9)] [(0, 9)]
        double v4786_acc{};
        double v4787_acc{};
        double v4788_acc{};
        double v4789_acc{};
        double v4790_acc{};
        double v4791_acc{};
        double v4792_acc{};
        double v4793_acc{};
        double v4794_acc{};
        double v4795_acc{};
        double v4796_acc{};
        double v4797_acc{};
        double v4798_acc{};
        double v4799_acc{};
        double v4800_acc{};
        double v4801_acc{};
        double v4802_acc{};
        double v4803_acc{};
        double v4804_acc{};
        double v4805_acc{};
        double v4806_acc{};
        double v4807_acc{};
        double v4808_acc{};
        double v4809_acc{};
        double v4810_acc{};
        double v4811_acc{};
        double v4812_acc{};
        double v4813_data = r6[0];
        double v4814_data = r6[1];
        double v4815_data = r6[2];
        double v4816_data = r6[3];
        double v4817_data = r6[4];
        double v4818_data = r6[5];
        double v4819_data = r6[6];
        double v4820_data = r6[7];
        double v4821_data = r6[8];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %29 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %30 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %31 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %32 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %33 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %34, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %34, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %34, %9 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %34, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %34, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4786_acc), "+&v"(v4787_acc), "+&v"(v4788_acc), "+&v"(v4789_acc), "+&v"(v4790_acc), "+&v"(v4791_acc)
        : "v"(v4813_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4814_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %30 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %31 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %32 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %33 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %34 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4791_acc), "+&v"(v4789_acc), "+&v"(v4790_acc), "+&v"(v4792_acc), "+&v"(v4793_acc), "+&v"(v4794_acc)
        : "v"(v4814_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4815_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %25 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %26 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %27 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %28 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %29 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %30 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %31 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %32 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %33 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %34 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4793_acc), "+&v"(v4794_acc), "+&v"(v4792_acc), "+&v"(v4795_acc), "+&v"(v4796_acc), "+&v"(v4797_acc)
        : "v"(v4815_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4816_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %22 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %24 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %25 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %27 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %28 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %29 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %30 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %31 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %32 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %33 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %34 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %9 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4795_acc), "+&v"(v4796_acc), "+&v"(v4797_acc), "+&v"(v4798_acc), "+&v"(v4799_acc), "+&v"(v4800_acc)
        : "v"(v4816_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4817_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %15 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %16 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %17 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %18 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %19 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %20 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %21 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %22 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %23 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %24 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %25 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %26 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %27 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %28 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %29 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %30 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %31 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %32 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %33 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %34 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %10 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %11 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4800_acc), "+&v"(v4798_acc), "+&v"(v4799_acc), "+&v"(v4801_acc), "+&v"(v4802_acc), "+&v"(v4803_acc)
        : "v"(v4817_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4818_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %8, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %8, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %14 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %15 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %16 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %17 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %18 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %19 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %20 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %21 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %22 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %23 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %25 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %26 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %29 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %36 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %37, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %6, %37, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %7, %37, %14 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4802_acc), "+&v"(v4803_acc), "+&v"(v4804_acc), "+&v"(v4805_acc), "+&v"(v4806_acc), "+&v"(v4807_acc), "+&v"(v4808_acc), "+&v"(v4809_acc)
        : "v"(v4818_data), "v"(v4208_data), "v"(v4209_data), "v"(v4819_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4820_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %30 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %32 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %33 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %31, %34 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %31, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4807_acc), "+&v"(v4808_acc), "+&v"(v4809_acc), "+&v"(v4810_acc), "+&v"(v4811_acc), "+&v"(v4812_acc)
        : "v"(v4820_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v4821_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v4812_acc), "+&v"(v4810_acc), "+&v"(v4811_acc)
        : "v"(v4821_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data)
        );
        r7[0] = v4786_acc;
        r7[1] = v4787_acc;
        r7[2] = v4788_acc;
        r7[3] = v4789_acc;
        r7[4] = v4790_acc;
        r7[5] = v4791_acc;
        r7[6] = v4792_acc;
        r7[7] = v4793_acc;
        r7[8] = v4794_acc;
        r7[9] = v4795_acc;
        r7[10] = v4796_acc;
        r7[11] = v4797_acc;
        r7[12] = v4798_acc;
        r7[13] = v4799_acc;
        r7[14] = v4800_acc;
        r7[15] = v4801_acc;
        r7[16] = v4802_acc;
        r7[17] = v4803_acc;
        r7[18] = v4804_acc;
        r7[19] = v4805_acc;
        r7[20] = v4806_acc;
        r7[21] = v4807_acc;
        r7[22] = v4808_acc;
        r7[23] = v4809_acc;
        r7[24] = v4810_acc;
        r7[25] = v4811_acc;
        r7[26] = v4812_acc;
        // r7 = 0 past (0, 35)
        // s1 = store{r>s}(localShrMem0, r7);
        #pragma unroll
        for (int32_t v4823_i0 = 0; v4823_i0 < 2; ++v4823_i0) {
          int32_t v4829_lead = v4154_lead + (v4823_i0 * 16);
          #pragma unroll
          for (int32_t v4824_i1 = 0; v4824_i1 < 9; ++v4824_i1) {
            double v4827_data = r7[(v4823_i0 + (v4824_i1 * 3))];
            s1[(v4829_lead + (v4824_i1 * 35))] = v4827_data;
          }
        }
        if (v4164_g) {
          #pragma unroll
          for (int32_t v4832_i1 = 0; v4832_i1 < 9; ++v4832_i1) {
            double v4835_data = r7[(2 + (v4832_i1 * 3))];
            s1[(v4304_lead + (v4832_i1 * 35))] = v4835_data;
          }
        }
        double r9[36]{};
        // ir9 = +(glb_m5 * s1)
        // [(1, 55), (0, 9)] [(0, 35)]
        double ir9[36]{};
        tensorforge::VectorT<double, 4> v4847_acc{};
        tensorforge::VectorT<double, 4> v4848_acc{};
        tensorforge::VectorT<double, 4> v4849_acc{};
        tensorforge::VectorT<double, 4> v4850_acc{};
        tensorforge::VectorT<double, 4> v4851_acc{};
        tensorforge::VectorT<double, 4> v4852_acc{};
        tensorforge::VectorT<double, 4> v4853_acc{};
        tensorforge::VectorT<double, 4> v4854_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        double v4861_data = s1[v4282_sh];
        double v4863_data = s1[v4286_a];
        double v4869_data = glb_m5[v4291_a];
        tensorforge::VectorT<double, 4> v4870_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4861_data, v4869_data, v4847_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4871_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4863_data, v4869_data, v4848_acc, 0, 0, 0);
        double v4876_data = glb_m5[v4298_a];
        tensorforge::VectorT<double, 4> v4877_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4861_data, v4876_data, v4849_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4878_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4863_data, v4876_data, v4850_acc, 0, 0, 0);
        double v4883_data = glb_m5[v4305_a];
        tensorforge::VectorT<double, 4> v4884_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4861_data, v4883_data, v4851_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4885_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4863_data, v4883_data, v4852_acc, 0, 0, 0);
        double v4890_data = glb_m5[v4312_a];
        tensorforge::VectorT<double, 4> v4891_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4861_data, v4890_data, v4853_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4892_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4863_data, v4890_data, v4854_acc, 0, 0, 0);
        double v4895_data = s1[v4317_a];
        double v4897_data = s1[v4320_a];
        double v4899_data = glb_m5[v4322_a];
        tensorforge::VectorT<double, 4> v4900_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4895_data, v4899_data, v4870_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4901_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4897_data, v4899_data, v4871_acc, 0, 0, 0);
        double v4903_data = glb_m5[v4326_a];
        tensorforge::VectorT<double, 4> v4904_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4895_data, v4903_data, v4877_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4905_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4897_data, v4903_data, v4878_acc, 0, 0, 0);
        double v4907_data = glb_m5[v4330_a];
        tensorforge::VectorT<double, 4> v4908_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4895_data, v4907_data, v4884_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4909_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4897_data, v4907_data, v4885_acc, 0, 0, 0);
        double v4911_data = glb_m5[v4334_a];
        tensorforge::VectorT<double, 4> v4912_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4895_data, v4911_data, v4891_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4913_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4897_data, v4911_data, v4892_acc, 0, 0, 0);
        double v4916_data = s1[v4338_a];
        double v4918_data = s1[v4341_a];
        double v4920_data = glb_m5[v4343_a];
        tensorforge::VectorT<double, 4> v4921_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4916_data, v4920_data, v4900_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4922_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4918_data, v4920_data, v4901_acc, 0, 0, 0);
        double v4924_data = glb_m5[v4347_a];
        tensorforge::VectorT<double, 4> v4925_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4916_data, v4924_data, v4904_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4926_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4918_data, v4924_data, v4905_acc, 0, 0, 0);
        double v4928_data = glb_m5[v4351_a];
        tensorforge::VectorT<double, 4> v4929_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4916_data, v4928_data, v4908_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4930_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4918_data, v4928_data, v4909_acc, 0, 0, 0);
        double v4932_data = glb_m5[v4355_a];
        tensorforge::VectorT<double, 4> v4933_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4916_data, v4932_data, v4912_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4934_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4918_data, v4932_data, v4913_acc, 0, 0, 0);
        double v4937_data = s1[v4359_a];
        double v4939_data = s1[v4362_a];
        double v4941_data = glb_m5[v4940_a];
        tensorforge::VectorT<double, 4> v4942_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4937_data, v4941_data, v4921_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4943_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4939_data, v4941_data, v4922_acc, 0, 0, 0);
        double v4945_data = glb_m5[v4364_a];
        tensorforge::VectorT<double, 4> v4946_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4937_data, v4945_data, v4925_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4947_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4939_data, v4945_data, v4926_acc, 0, 0, 0);
        double v4949_data = glb_m5[v4368_a];
        tensorforge::VectorT<double, 4> v4950_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4937_data, v4949_data, v4929_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4951_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4939_data, v4949_data, v4930_acc, 0, 0, 0);
        double v4953_data = glb_m5[v4372_a];
        tensorforge::VectorT<double, 4> v4954_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4937_data, v4953_data, v4933_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4955_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4939_data, v4953_data, v4934_acc, 0, 0, 0);
        double v4958_data = s1[v4376_a];
        double v4960_data = s1[v4379_a];
        double v4962_data = glb_m5[v4381_a];
        tensorforge::VectorT<double, 4> v4963_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4958_data, v4962_data, v4946_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4964_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4960_data, v4962_data, v4947_acc, 0, 0, 0);
        double v4966_data = glb_m5[v4385_a];
        tensorforge::VectorT<double, 4> v4967_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4958_data, v4966_data, v4950_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4968_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4960_data, v4966_data, v4951_acc, 0, 0, 0);
        double v4970_data = glb_m5[v4389_a];
        tensorforge::VectorT<double, 4> v4971_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4958_data, v4970_data, v4954_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4972_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4960_data, v4970_data, v4955_acc, 0, 0, 0);
        double v4975_data = s1[v4393_a];
        double v4977_data = s1[v4396_a];
        double v4979_data = glb_m5[v4978_a];
        tensorforge::VectorT<double, 4> v4980_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4975_data, v4979_data, v4963_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4981_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4977_data, v4979_data, v4964_acc, 0, 0, 0);
        double v4983_data = glb_m5[v4398_a];
        tensorforge::VectorT<double, 4> v4984_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4975_data, v4983_data, v4967_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4985_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4977_data, v4983_data, v4968_acc, 0, 0, 0);
        double v4987_data = glb_m5[v4402_a];
        tensorforge::VectorT<double, 4> v4988_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4975_data, v4987_data, v4971_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4989_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4977_data, v4987_data, v4972_acc, 0, 0, 0);
        double v4992_data = s1[v4406_a];
        double v4994_data = s1[v4409_a];
        double v4996_data = glb_m5[v4411_a];
        tensorforge::VectorT<double, 4> v4997_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4992_data, v4996_data, v4984_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v4998_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4994_data, v4996_data, v4985_acc, 0, 0, 0);
        double v5000_data = glb_m5[v4415_a];
        tensorforge::VectorT<double, 4> v5001_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4992_data, v5000_data, v4988_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5002_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v4994_data, v5000_data, v4989_acc, 0, 0, 0);
        double v5005_data = s1[v4419_a];
        double v5007_data = s1[v4422_a];
        double v5009_data = glb_m5[v5008_a];
        tensorforge::VectorT<double, 4> v5010_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5005_data, v5009_data, v4997_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5011_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5007_data, v5009_data, v4998_acc, 0, 0, 0);
        double v5013_data = glb_m5[v4424_a];
        tensorforge::VectorT<double, 4> v5014_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5005_data, v5013_data, v5001_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5015_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5007_data, v5013_data, v5002_acc, 0, 0, 0);
        double v5024_data = s1[v5022_a];
        double v5032_data = s1[v5031_a];
        double v5037_data = glb_m5[v5036_a];
        double v5039_data = glb_m5[v5038_a];
        double v5041_data = glb_m5[v5040_a];
        double v5044_ks = v5029_g ? v5037_data : (v5027_g ? v5039_data : (v5025_g ? v5041_data : 0.0));
        tensorforge::VectorT<double, 4> v5045_acc = __builtin_amdgcn_mfma_f64_16x16x4f64((v5029_g ? v5024_data : (v5027_g ? v5024_data : (v5025_g ? v5024_data : 0.0))), v5044_ks, v5014_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5046_acc = __builtin_amdgcn_mfma_f64_16x16x4f64((v5029_g ? v5032_data : (v5027_g ? v5032_data : (v5025_g ? v5032_data : 0.0))), v5044_ks, v5015_acc, 0, 0, 0);
        ir9[0] = (v4942_acc[0]);
        ir9[4] = (v4942_acc[1]);
        ir9[8] = (v4942_acc[2]);
        ir9[12] = (v4942_acc[3]);
        ir9[16] = (v4943_acc[0]);
        ir9[20] = (v4943_acc[1]);
        ir9[24] = (v4943_acc[2]);
        ir9[28] = (v4943_acc[3]);
        ir9[1] = (v4980_acc[0]);
        ir9[5] = (v4980_acc[1]);
        ir9[9] = (v4980_acc[2]);
        ir9[13] = (v4980_acc[3]);
        ir9[17] = (v4981_acc[0]);
        ir9[21] = (v4981_acc[1]);
        ir9[25] = (v4981_acc[2]);
        ir9[29] = (v4981_acc[3]);
        ir9[2] = (v5010_acc[0]);
        ir9[6] = (v5010_acc[1]);
        ir9[10] = (v5010_acc[2]);
        ir9[14] = (v5010_acc[3]);
        ir9[18] = (v5011_acc[0]);
        ir9[22] = (v5011_acc[1]);
        ir9[26] = (v5011_acc[2]);
        ir9[30] = (v5011_acc[3]);
        ir9[3] = (v5045_acc[0]);
        ir9[7] = (v5045_acc[1]);
        ir9[11] = (v5045_acc[2]);
        ir9[15] = (v5045_acc[3]);
        ir9[19] = (v5046_acc[0]);
        ir9[23] = (v5046_acc[1]);
        ir9[27] = (v5046_acc[2]);
        ir9[31] = (v5046_acc[3]);
        double v5082_data = glb_m5[v4154_lead];
        double v5086_data = glb_m5[v4297_lead];
        double v5090_data = glb_m5[v4304_lead];
        double v5094_data = glb_m5[v4311_lead];
        double v5096_data = glb_m5[v4476_a];
        double v5098_data = glb_m5[v4478_a];
        double v5100_data = glb_m5[v4480_a];
        double v5102_data = glb_m5[v4482_a];
        double v5104_data = glb_m5[v4484_a];
        double v5106_data = glb_m5[v4486_a];
        double v5108_data = glb_m5[v4488_a];
        double v5110_data = glb_m5[v4490_a];
        double v5112_data = glb_m5[v4492_a];
        double v5114_data = glb_m5[v4494_a];
        double v5116_data = glb_m5[v4496_a];
        double v5118_data = glb_m5[v4498_a];
        double v5120_data = glb_m5[v4500_a];
        double v5122_data = glb_m5[v4502_a];
        double v5124_data = glb_m5[v4504_a];
        double v5126_data = glb_m5[v4506_a];
        double v5128_data = glb_m5[v4508_a];
        double v5130_data = glb_m5[v4510_a];
        double v5132_data = glb_m5[v4512_a];
        double v5134_data = glb_m5[v4514_a];
        double v5136_data = glb_m5[v4516_a];
        double v5138_data = glb_m5[v4518_a];
        double v5140_data = glb_m5[v4520_a];
        double v5142_data = glb_m5[v4522_a];
        double v5144_data = glb_m5[v4524_a];
        double v5146_data = glb_m5[v4526_a];
        double v5148_data = glb_m5[v4528_a];
        double v5150_data = glb_m5[v4530_a];
        double v5152_data = glb_m5[v4532_a];
        double v5154_data = glb_m5[v4534_a];
        double v5156_data = glb_m5[v4536_a];
        double v5158_data = glb_m5[v4538_a];
        double v5160_data = glb_m5[v4540_a];
        double v5162_data = glb_m5[v4542_a];
        double v5164_data = glb_m5[v4544_a];
        double v5166_data = glb_m5[v4546_a];
        double v5168_data = glb_m5[v4548_a];
        double v5170_data = glb_m5[v4550_a];
        double v5172_data = glb_m5[v4552_a];
        double v5174_data = glb_m5[v4554_a];
        double v5176_data = glb_m5[v4556_a];
        double v5178_data = glb_m5[v4558_a];
        double v5180_data = glb_m5[v4560_a];
        double v5182_data = glb_m5[v4562_a];
        double v5184_data = glb_m5[v5183_a];
        double v5186_data = glb_m5[v4564_a];
        double v5188_data = glb_m5[v4566_a];
        double v5190_data = glb_m5[v4568_a];
        double v5192_data = glb_m5[v5191_a];
        double v5194_data = glb_m5[v4570_a];
        double v5196_data = glb_m5[v4572_a];
        double v5198_data = glb_m5[v4574_a];
        double v5200_data = glb_m5[v5199_a];
        double v5202_data = glb_m5[v4576_a];
        double v5204_data = glb_m5[v4578_a];
        double v5206_data = glb_m5[v4580_a];
        double v5208_data = glb_m5[v5207_a];
        double v5210_data = glb_m5[v4582_a];
        double v5212_data = glb_m5[v4584_a];
        double v5214_data = glb_m5[v4586_a];
        double v5216_data = glb_m5[v4588_a];
        double v5218_data = glb_m5[v4590_a];
        double v5220_data = glb_m5[v4592_a];
        double v5222_data = glb_m5[v4594_a];
        double v5224_data = glb_m5[v4596_a];
        double v5226_data = glb_m5[v4598_a];
        double v5228_data = glb_m5[v4600_a];
        double v5230_data = glb_m5[v4602_a];
        double v5232_data = glb_m5[v4604_a];
        double v5234_data = glb_m5[v4606_a];
        double v5236_data = glb_m5[v4608_a];
        double v5238_data = glb_m5[v4610_a];
        double v5240_data = glb_m5[v5239_a];
        double v5242_data = glb_m5[v4612_a];
        double v5244_data = glb_m5[v4614_a];
        double v5246_data = glb_m5[v5245_a];
        double v5248_data = glb_m5[v4616_a];
        double v5250_data = glb_m5[v4618_a];
        double v5252_data = glb_m5[v5251_a];
        double v5254_data = glb_m5[v4620_a];
        double v5256_data = glb_m5[v4622_a];
        double v5258_data = glb_m5[v5257_a];
        double v5260_data = glb_m5[v4624_a];
        double v5262_data = glb_m5[v4626_a];
        double v5264_data = glb_m5[v4628_a];
        double v5266_data = glb_m5[v4630_a];
        double v5268_data = glb_m5[v4632_a];
        double v5270_data = glb_m5[v4634_a];
        double v5272_data = glb_m5[v4636_a];
        double v5274_data = glb_m5[v4638_a];
        double v5276_data = glb_m5[v4640_a];
        double v5278_data = glb_m5[v4642_a];
        double v5280_data = glb_m5[v5279_a];
        double v5282_data = glb_m5[v4644_a];
        double v5284_data = glb_m5[v5283_a];
        double v5286_data = glb_m5[v4646_a];
        double v5288_data = glb_m5[v5287_a];
        double v5290_data = glb_m5[v4648_a];
        double v5292_data = glb_m5[v5291_a];
        double v5294_data = glb_m5[v4650_a];
        double v5301_acc{};
        double v5302_acc{};
        double v5303_acc{};
        double v5304_acc{};
        double v5307_data = r7[26];
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
        : "+&v"(v5301_acc), "+&v"(v5302_acc), "+&v"(v5303_acc), "+&v"(v5304_acc)
        : "v"(v4810_acc), "v"(v5082_data), "v"(v5086_data), "v"(v5090_data), "v"(v5094_data), "v"(v5096_data), "v"(v5098_data), "v"(v5100_data), "v"(v5102_data), "v"(v5104_data), "v"(v5106_data), "v"(v5108_data), "v"(v5110_data), "v"(v5112_data), "v"(v5114_data), "v"(v5116_data), "v"(v5118_data), "v"(v5120_data), "v"(v5122_data), "v"(v5124_data), "v"(v5126_data), "v"(v5128_data), "v"(v5130_data), "v"(v5132_data), "v"(v5134_data), "v"(v5136_data), "v"(v5138_data), "v"(v5140_data), "v"(v5142_data), "v"(v5144_data), "v"(v5146_data), "v"(v5148_data), "v"(v5150_data)
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
        "v_fmac_f64_dpp %2, %4, %19 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %20 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %21 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %22 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %23 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %24 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %25 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %26 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %27 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %28 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %29 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %30 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %35 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %36 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5301_acc), "+&v"(v5302_acc), "+&v"(v5303_acc), "+&v"(v5304_acc)
        : "v"(v4810_acc), "v"(v5152_data), "v"(v5154_data), "v"(v5156_data), "v"(v5158_data), "v"(v5160_data), "v"(v5162_data), "v"(v5164_data), "v"(v5166_data), "v"(v5168_data), "v"(v5170_data), "v"(v5172_data), "v"(v5174_data), "v"(v5176_data), "v"(v5178_data), "v"(v5180_data), "v"(v5182_data), "v"(v5184_data), "v"(v5186_data), "v"(v5188_data), "v"(v5190_data), "v"(v5192_data), "v"(v5194_data), "v"(v5196_data), "v"(v5198_data), "v"(v5200_data), "v"(v5202_data), "v"(v5204_data), "v"(v5206_data), "v"(v5208_data), "v"(v5210_data), "v"(v5212_data), "v"(v5214_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %30 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %31 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %32 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %33 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %34 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %35 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5302_acc), "+&v"(v5303_acc), "+&v"(v5304_acc)
        : "v"(v4811_acc), "v"(v5216_data), "v"(v5218_data), "v"(v5220_data), "v"(v5222_data), "v"(v5224_data), "v"(v5226_data), "v"(v5228_data), "v"(v5230_data), "v"(v5232_data), "v"(v5234_data), "v"(v5236_data), "v"(v5238_data), "v"(v5240_data), "v"(v5242_data), "v"(v5244_data), "v"(v5246_data), "v"(v5248_data), "v"(v5250_data), "v"(v5252_data), "v"(v5254_data), "v"(v5256_data), "v"(v5258_data), "v"(v5260_data), "v"(v5262_data), "v"(v5264_data), "v"(v5266_data), "v"(v5268_data), "v"(v5270_data), "v"(v5272_data), "v"(v5274_data), "v"(v5276_data), "v"(v5278_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %5 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %6 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %7 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %8 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %9 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %10 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %11, %13 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %11, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5303_acc), "+&v"(v5304_acc)
        : "v"(v4811_acc), "v"(v5280_data), "v"(v5282_data), "v"(v5284_data), "v"(v5286_data), "v"(v5288_data), "v"(v5290_data), "v"(v5292_data), "v"(v5294_data), "v"(v5307_data), "v"(v5037_data), "v"(v5039_data), "v"(v5041_data)
        );
        ir9[32] = v5301_acc;
        ir9[33] = v5302_acc;
        ir9[34] = v5303_acc;
        ir9[35] = v5304_acc;
        double r8[36]{};
        // r8 = load{g>r}(glb_m2);
        double r10[27]{};
        // r10 = load{g>r}(glb_m0);
        // r9 = ir9 + r8
        if (v4660_g) {
          #pragma unroll
          for (int32_t v5335_n1 = 0; v5335_n1 < 9; ++v5335_n1) {
            int32_t v5336_a = v5335_n1 * 4;
            double v5338_data = ir9[v5336_a];
            double v5339_data = ir4[v5336_a];
            r9[v5336_a] = (v5339_data + v5338_data);
          }
        }
        #pragma unroll
        for (int32_t v5341_n0 = 1; v5341_n0 < 3; ++v5341_n0) {
          #pragma unroll
          for (int32_t v5342_n1 = 0; v5342_n1 < 9; ++v5342_n1) {
            int32_t v5343_a = v5342_n1 * 4;
            int32_t v5344_a = v5341_n0 + v5343_a;
            double v5345_data = ir9[v5344_a];
            double v5346_data = ir4[(v5341_n0 + v5343_a)];
            r9[v5344_a] = (v5346_data + v5345_data);
          }
        }
        // glb_m2 = store{r>g}(r9);
        if (v5326_g) {
          #pragma unroll
          for (int32_t v5371_i1 = 0; v5371_i1 < 9; ++v5371_i1) {
            double v5374_data = ir9[((v5371_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m2[(v4311_lead + (v5371_i1 * 56))], v5374_data);
            }
          }
        }
        double r11[9]{};
        // r11 = load{g>r}(glb_m6);
        #pragma unroll
        for (int32_t v5398_i1 = 0; v5398_i1 < 9; ++v5398_i1) {
          double v5404_data = v4179_g ? (__builtin_nontemporal_load(&glb_m6[(v4154_lead + (v5398_i1 * 9))])) : (0.0);
          r11[v5398_i1] = v5404_data;
        }
        double r12[27]{};
        // r12 = +(r10 * r11) + None
        // [(0, 35), (0, 9)] [(0, 9)]
        double v5434_acc{};
        double v5435_acc{};
        double v5436_acc{};
        double v5437_acc{};
        double v5438_acc{};
        double v5439_acc{};
        double v5440_acc{};
        double v5441_acc{};
        double v5442_acc{};
        double v5443_acc{};
        double v5444_acc{};
        double v5445_acc{};
        double v5446_acc{};
        double v5447_acc{};
        double v5448_acc{};
        double v5449_acc{};
        double v5450_acc{};
        double v5451_acc{};
        double v5452_acc{};
        double v5453_acc{};
        double v5454_acc{};
        double v5455_acc{};
        double v5456_acc{};
        double v5457_acc{};
        double v5458_acc{};
        double v5459_acc{};
        double v5460_acc{};
        double v5461_data = r11[0];
        double v5462_data = r11[1];
        double v5463_data = r11[2];
        double v5464_data = r11[3];
        double v5465_data = r11[4];
        double v5466_data = r11[5];
        double v5467_data = r11[6];
        double v5468_data = r11[7];
        double v5469_data = r11[8];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %29 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %30 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %31 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %32 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %33 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %34, %7 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %34, %8 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %34, %9 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %34, %10 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %34, %11 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5434_acc), "+&v"(v5435_acc), "+&v"(v5436_acc), "+&v"(v5437_acc), "+&v"(v5438_acc), "+&v"(v5439_acc)
        : "v"(v5461_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v5462_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %30 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %31 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %32 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %33 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %34 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %29, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %29, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %29, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5439_acc), "+&v"(v5437_acc), "+&v"(v5438_acc), "+&v"(v5440_acc), "+&v"(v5441_acc), "+&v"(v5442_acc)
        : "v"(v5462_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v5463_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %25 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %26 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %27 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %28 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %29 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %30 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %31 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %32 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %33 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %34 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %8 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %24, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %24, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %24, %11 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5441_acc), "+&v"(v5442_acc), "+&v"(v5440_acc), "+&v"(v5443_acc), "+&v"(v5444_acc), "+&v"(v5445_acc)
        : "v"(v5463_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v5464_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %22 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %24 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %25 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %27 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %28 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %29 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %30 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %31 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %32 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %33 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %34 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %8 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %19, %9 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %19, %10 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %19, %11 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5443_acc), "+&v"(v5444_acc), "+&v"(v5445_acc), "+&v"(v5446_acc), "+&v"(v5447_acc), "+&v"(v5448_acc)
        : "v"(v5464_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v5465_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %15 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %16 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %17 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %18 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %19 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %20 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %21 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %22 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %23 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %24 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %25 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %26 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %27 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %28 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %29 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %30 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %31 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %32 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %33 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %34 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %7 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %8 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %14, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %14, %10 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %14, %11 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5448_acc), "+&v"(v5446_acc), "+&v"(v5447_acc), "+&v"(v5449_acc), "+&v"(v5450_acc), "+&v"(v5451_acc)
        : "v"(v5465_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v5466_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %8, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %8, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %14 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %15 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %16 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %17 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %18 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %19 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %20 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %21 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %22 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %23 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %24 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %25 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %26 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %27 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %28 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %29 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %30 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %31 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %32 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %33 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %34 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %35 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %11, %36 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %11, %9 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %11, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %37, %12 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %6, %37, %13 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %7, %37, %14 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5450_acc), "+&v"(v5451_acc), "+&v"(v5452_acc), "+&v"(v5453_acc), "+&v"(v5454_acc), "+&v"(v5455_acc), "+&v"(v5456_acc), "+&v"(v5457_acc)
        : "v"(v5466_data), "v"(v4208_data), "v"(v4209_data), "v"(v5467_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v5468_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %6, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %12 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %13 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %14 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %15 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %16 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %17 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %18 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %19 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %20 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %21 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %22 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %23 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %24 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %25 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %26 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %27 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %6, %28 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %6, %29 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %6, %30 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %32 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %33 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %31, %34 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %7 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %8 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %5, %31, %9 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %31, %10 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %4, %31, %11 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5455_acc), "+&v"(v5456_acc), "+&v"(v5457_acc), "+&v"(v5458_acc), "+&v"(v5459_acc), "+&v"(v5460_acc)
        : "v"(v5468_data), "v"(v4186_data), "v"(v4187_data), "v"(v4188_data), "v"(v4189_data), "v"(v4190_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data), "v"(v5469_data), "v"(v4183_data), "v"(v4184_data), "v"(v4185_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %18 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %19 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %20 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %21 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %22 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5460_acc), "+&v"(v5458_acc), "+&v"(v5459_acc)
        : "v"(v5469_data), "v"(v4191_data), "v"(v4192_data), "v"(v4193_data), "v"(v4194_data), "v"(v4195_data), "v"(v4196_data), "v"(v4197_data), "v"(v4198_data), "v"(v4199_data), "v"(v4200_data), "v"(v4201_data), "v"(v4202_data), "v"(v4203_data), "v"(v4204_data), "v"(v4205_data), "v"(v4206_data), "v"(v4207_data), "v"(v4208_data), "v"(v4209_data)
        );
        r12[0] = v5434_acc;
        r12[1] = v5435_acc;
        r12[2] = v5436_acc;
        r12[3] = v5437_acc;
        r12[4] = v5438_acc;
        r12[5] = v5439_acc;
        r12[6] = v5440_acc;
        r12[7] = v5441_acc;
        r12[8] = v5442_acc;
        r12[9] = v5443_acc;
        r12[10] = v5444_acc;
        r12[11] = v5445_acc;
        r12[12] = v5446_acc;
        r12[13] = v5447_acc;
        r12[14] = v5448_acc;
        r12[15] = v5449_acc;
        r12[16] = v5450_acc;
        r12[17] = v5451_acc;
        r12[18] = v5452_acc;
        r12[19] = v5453_acc;
        r12[20] = v5454_acc;
        r12[21] = v5455_acc;
        r12[22] = v5456_acc;
        r12[23] = v5457_acc;
        r12[24] = v5458_acc;
        r12[25] = v5459_acc;
        r12[26] = v5460_acc;
        // r12 = 0 past (0, 35)
        // s2 = store{r>s}(localShrMem0, r12);
        #pragma unroll
        for (int32_t v5471_i0 = 0; v5471_i0 < 2; ++v5471_i0) {
          int32_t v5477_lead = v4154_lead + (v5471_i0 * 16);
          #pragma unroll
          for (int32_t v5472_i1 = 0; v5472_i1 < 9; ++v5472_i1) {
            double v5475_data = r12[(v5471_i0 + (v5472_i1 * 3))];
            s2[(v5477_lead + (v5472_i1 * 35))] = v5475_data;
          }
        }
        if (v4164_g) {
          #pragma unroll
          for (int32_t v5480_i1 = 0; v5480_i1 < 9; ++v5480_i1) {
            double v5483_data = r12[(2 + (v5480_i1 * 3))];
            s2[(v4304_lead + (v5480_i1 * 35))] = v5483_data;
          }
        }
        double r14[36]{};
        // ir14 = +(glb_m7 * s2)
        // [(1, 56), (0, 9)] [(0, 35)]
        double ir14[36]{};
        tensorforge::VectorT<double, 4> v5495_acc{};
        tensorforge::VectorT<double, 4> v5496_acc{};
        tensorforge::VectorT<double, 4> v5497_acc{};
        tensorforge::VectorT<double, 4> v5498_acc{};
        tensorforge::VectorT<double, 4> v5499_acc{};
        tensorforge::VectorT<double, 4> v5500_acc{};
        tensorforge::VectorT<double, 4> v5501_acc{};
        tensorforge::VectorT<double, 4> v5502_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        double v5509_data = s2[v4282_sh];
        double v5511_data = s2[v4286_a];
        double v5517_data = glb_m7[v4291_a];
        tensorforge::VectorT<double, 4> v5518_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5509_data, v5517_data, v5495_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5519_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5511_data, v5517_data, v5496_acc, 0, 0, 0);
        double v5524_data = glb_m7[v4298_a];
        tensorforge::VectorT<double, 4> v5525_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5509_data, v5524_data, v5497_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5526_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5511_data, v5524_data, v5498_acc, 0, 0, 0);
        double v5531_data = glb_m7[v4305_a];
        tensorforge::VectorT<double, 4> v5532_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5509_data, v5531_data, v5499_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5533_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5511_data, v5531_data, v5500_acc, 0, 0, 0);
        double v5538_data = glb_m7[v4312_a];
        tensorforge::VectorT<double, 4> v5539_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5509_data, v5538_data, v5501_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5540_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5511_data, v5538_data, v5502_acc, 0, 0, 0);
        double v5543_data = s2[v4317_a];
        double v5545_data = s2[v4320_a];
        double v5547_data = glb_m7[v4322_a];
        tensorforge::VectorT<double, 4> v5548_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5543_data, v5547_data, v5518_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5549_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5545_data, v5547_data, v5519_acc, 0, 0, 0);
        double v5551_data = glb_m7[v4326_a];
        tensorforge::VectorT<double, 4> v5552_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5543_data, v5551_data, v5525_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5553_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5545_data, v5551_data, v5526_acc, 0, 0, 0);
        double v5555_data = glb_m7[v4330_a];
        tensorforge::VectorT<double, 4> v5556_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5543_data, v5555_data, v5532_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5557_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5545_data, v5555_data, v5533_acc, 0, 0, 0);
        double v5559_data = glb_m7[v4334_a];
        tensorforge::VectorT<double, 4> v5560_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5543_data, v5559_data, v5539_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5561_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5545_data, v5559_data, v5540_acc, 0, 0, 0);
        double v5564_data = s2[v4338_a];
        double v5566_data = s2[v4341_a];
        double v5568_data = glb_m7[v4343_a];
        tensorforge::VectorT<double, 4> v5569_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5564_data, v5568_data, v5548_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5570_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5566_data, v5568_data, v5549_acc, 0, 0, 0);
        double v5572_data = glb_m7[v4347_a];
        tensorforge::VectorT<double, 4> v5573_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5564_data, v5572_data, v5552_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5574_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5566_data, v5572_data, v5553_acc, 0, 0, 0);
        double v5576_data = glb_m7[v4351_a];
        tensorforge::VectorT<double, 4> v5577_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5564_data, v5576_data, v5556_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5578_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5566_data, v5576_data, v5557_acc, 0, 0, 0);
        double v5580_data = glb_m7[v4355_a];
        tensorforge::VectorT<double, 4> v5581_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5564_data, v5580_data, v5560_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5582_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5566_data, v5580_data, v5561_acc, 0, 0, 0);
        double v5585_data = s2[v4359_a];
        double v5587_data = s2[v4362_a];
        double v5589_data = glb_m7[v4940_a];
        tensorforge::VectorT<double, 4> v5590_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5585_data, v5589_data, v5569_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5591_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5587_data, v5589_data, v5570_acc, 0, 0, 0);
        double v5593_data = glb_m7[v4364_a];
        tensorforge::VectorT<double, 4> v5594_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5585_data, v5593_data, v5573_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5595_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5587_data, v5593_data, v5574_acc, 0, 0, 0);
        double v5597_data = glb_m7[v4368_a];
        tensorforge::VectorT<double, 4> v5598_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5585_data, v5597_data, v5577_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5599_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5587_data, v5597_data, v5578_acc, 0, 0, 0);
        double v5601_data = glb_m7[v4372_a];
        tensorforge::VectorT<double, 4> v5602_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5585_data, v5601_data, v5581_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5603_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5587_data, v5601_data, v5582_acc, 0, 0, 0);
        double v5606_data = s2[v4376_a];
        double v5608_data = s2[v4379_a];
        double v5610_data = glb_m7[v5609_a];
        tensorforge::VectorT<double, 4> v5611_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5606_data, v5610_data, v5590_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5612_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5608_data, v5610_data, v5591_acc, 0, 0, 0);
        double v5614_data = glb_m7[v4381_a];
        tensorforge::VectorT<double, 4> v5615_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5606_data, v5614_data, v5594_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5616_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5608_data, v5614_data, v5595_acc, 0, 0, 0);
        double v5618_data = glb_m7[v4385_a];
        tensorforge::VectorT<double, 4> v5619_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5606_data, v5618_data, v5598_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5620_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5608_data, v5618_data, v5599_acc, 0, 0, 0);
        double v5622_data = glb_m7[v4389_a];
        tensorforge::VectorT<double, 4> v5623_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5606_data, v5622_data, v5602_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5624_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5608_data, v5622_data, v5603_acc, 0, 0, 0);
        double v5627_data = s2[v4393_a];
        double v5629_data = s2[v4396_a];
        double v5631_data = glb_m7[v4978_a];
        tensorforge::VectorT<double, 4> v5632_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5627_data, v5631_data, v5615_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5633_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5629_data, v5631_data, v5616_acc, 0, 0, 0);
        double v5635_data = glb_m7[v4398_a];
        tensorforge::VectorT<double, 4> v5636_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5627_data, v5635_data, v5619_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5637_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5629_data, v5635_data, v5620_acc, 0, 0, 0);
        double v5639_data = glb_m7[v4402_a];
        tensorforge::VectorT<double, 4> v5640_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5627_data, v5639_data, v5623_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5641_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5629_data, v5639_data, v5624_acc, 0, 0, 0);
        double v5644_data = s2[v4406_a];
        double v5646_data = s2[v4409_a];
        double v5648_data = glb_m7[v5647_a];
        tensorforge::VectorT<double, 4> v5649_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5644_data, v5648_data, v5632_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5650_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5646_data, v5648_data, v5633_acc, 0, 0, 0);
        double v5652_data = glb_m7[v4411_a];
        tensorforge::VectorT<double, 4> v5653_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5644_data, v5652_data, v5636_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5654_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5646_data, v5652_data, v5637_acc, 0, 0, 0);
        double v5656_data = glb_m7[v4415_a];
        tensorforge::VectorT<double, 4> v5657_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5644_data, v5656_data, v5640_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5658_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5646_data, v5656_data, v5641_acc, 0, 0, 0);
        double v5661_data = s2[v4419_a];
        double v5663_data = s2[v4422_a];
        double v5665_data = glb_m7[v5008_a];
        tensorforge::VectorT<double, 4> v5666_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5661_data, v5665_data, v5653_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5667_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5663_data, v5665_data, v5654_acc, 0, 0, 0);
        double v5669_data = glb_m7[v4424_a];
        tensorforge::VectorT<double, 4> v5670_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5661_data, v5669_data, v5657_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5671_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5663_data, v5669_data, v5658_acc, 0, 0, 0);
        double v5680_data = s2[v5022_a];
        double v5686_ks = v5029_g ? v5680_data : (v5027_g ? v5680_data : (v5025_g ? v5680_data : 0.0));
        double v5688_data = s2[v5031_a];
        double v5691_ks = v5029_g ? v5688_data : (v5027_g ? v5688_data : (v5025_g ? v5688_data : 0.0));
        double v5693_data = glb_m7[v5692_a];
        double v5695_data = glb_m7[v5694_a];
        double v5697_data = glb_m7[v5696_a];
        double v5700_ks = v5029_g ? v5693_data : (v5027_g ? v5695_data : (v5025_g ? v5697_data : 0.0));
        tensorforge::VectorT<double, 4> v5701_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5686_ks, v5700_ks, v5666_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5702_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5691_ks, v5700_ks, v5667_acc, 0, 0, 0);
        double v5704_data = glb_m7[v5036_a];
        double v5706_data = glb_m7[v5038_a];
        double v5708_data = glb_m7[v5040_a];
        double v5711_ks = v5029_g ? v5704_data : (v5027_g ? v5706_data : (v5025_g ? v5708_data : 0.0));
        tensorforge::VectorT<double, 4> v5712_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5686_ks, v5711_ks, v5670_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v5713_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v5691_ks, v5711_ks, v5671_acc, 0, 0, 0);
        ir14[0] = (v5611_acc[0]);
        ir14[4] = (v5611_acc[1]);
        ir14[8] = (v5611_acc[2]);
        ir14[12] = (v5611_acc[3]);
        ir14[16] = (v5612_acc[0]);
        ir14[20] = (v5612_acc[1]);
        ir14[24] = (v5612_acc[2]);
        ir14[28] = (v5612_acc[3]);
        ir14[1] = (v5649_acc[0]);
        ir14[5] = (v5649_acc[1]);
        ir14[9] = (v5649_acc[2]);
        ir14[13] = (v5649_acc[3]);
        ir14[17] = (v5650_acc[0]);
        ir14[21] = (v5650_acc[1]);
        ir14[25] = (v5650_acc[2]);
        ir14[29] = (v5650_acc[3]);
        ir14[2] = (v5701_acc[0]);
        ir14[6] = (v5701_acc[1]);
        ir14[10] = (v5701_acc[2]);
        ir14[14] = (v5701_acc[3]);
        ir14[18] = (v5702_acc[0]);
        ir14[22] = (v5702_acc[1]);
        ir14[26] = (v5702_acc[2]);
        ir14[30] = (v5702_acc[3]);
        ir14[3] = (v5712_acc[0]);
        ir14[7] = (v5712_acc[1]);
        ir14[11] = (v5712_acc[2]);
        ir14[15] = (v5712_acc[3]);
        ir14[19] = (v5713_acc[0]);
        ir14[23] = (v5713_acc[1]);
        ir14[27] = (v5713_acc[2]);
        ir14[31] = (v5713_acc[3]);
        double v5749_data = glb_m7[v4154_lead];
        double v5753_data = glb_m7[v4297_lead];
        double v5757_data = glb_m7[v4304_lead];
        double v5761_data = glb_m7[v4311_lead];
        double v5763_data = glb_m7[v4476_a];
        double v5765_data = glb_m7[v4478_a];
        double v5767_data = glb_m7[v4480_a];
        double v5769_data = glb_m7[v4482_a];
        double v5771_data = glb_m7[v4484_a];
        double v5773_data = glb_m7[v4486_a];
        double v5775_data = glb_m7[v4488_a];
        double v5777_data = glb_m7[v4490_a];
        double v5779_data = glb_m7[v4492_a];
        double v5781_data = glb_m7[v4494_a];
        double v5783_data = glb_m7[v4496_a];
        double v5785_data = glb_m7[v4498_a];
        double v5787_data = glb_m7[v4500_a];
        double v5789_data = glb_m7[v4502_a];
        double v5791_data = glb_m7[v4504_a];
        double v5793_data = glb_m7[v4506_a];
        double v5795_data = glb_m7[v4508_a];
        double v5797_data = glb_m7[v4510_a];
        double v5799_data = glb_m7[v4512_a];
        double v5801_data = glb_m7[v4514_a];
        double v5803_data = glb_m7[v4516_a];
        double v5805_data = glb_m7[v4518_a];
        double v5807_data = glb_m7[v4520_a];
        double v5809_data = glb_m7[v4522_a];
        double v5811_data = glb_m7[v4524_a];
        double v5813_data = glb_m7[v4526_a];
        double v5815_data = glb_m7[v4528_a];
        double v5817_data = glb_m7[v4530_a];
        double v5819_data = glb_m7[v4532_a];
        double v5821_data = glb_m7[v4534_a];
        double v5823_data = glb_m7[v4536_a];
        double v5825_data = glb_m7[v4538_a];
        double v5827_data = glb_m7[v4540_a];
        double v5829_data = glb_m7[v4542_a];
        double v5831_data = glb_m7[v4544_a];
        double v5833_data = glb_m7[v4546_a];
        double v5835_data = glb_m7[v4548_a];
        double v5837_data = glb_m7[v4550_a];
        double v5839_data = glb_m7[v4552_a];
        double v5841_data = glb_m7[v4554_a];
        double v5843_data = glb_m7[v4556_a];
        double v5845_data = glb_m7[v4558_a];
        double v5847_data = glb_m7[v4560_a];
        double v5849_data = glb_m7[v4562_a];
        double v5851_data = glb_m7[v5183_a];
        double v5853_data = glb_m7[v4564_a];
        double v5855_data = glb_m7[v4566_a];
        double v5857_data = glb_m7[v4568_a];
        double v5859_data = glb_m7[v5191_a];
        double v5861_data = glb_m7[v4570_a];
        double v5863_data = glb_m7[v4572_a];
        double v5865_data = glb_m7[v4574_a];
        double v5867_data = glb_m7[v5199_a];
        double v5869_data = glb_m7[v4576_a];
        double v5871_data = glb_m7[v4578_a];
        double v5873_data = glb_m7[v4580_a];
        double v5875_data = glb_m7[v5207_a];
        double v5877_data = glb_m7[v4582_a];
        double v5879_data = glb_m7[v4584_a];
        double v5881_data = glb_m7[v4586_a];
        double v5883_data = glb_m7[v5882_a];
        double v5885_data = glb_m7[v4588_a];
        double v5887_data = glb_m7[v4590_a];
        double v5889_data = glb_m7[v4592_a];
        double v5891_data = glb_m7[v5890_a];
        double v5893_data = glb_m7[v4594_a];
        double v5895_data = glb_m7[v4596_a];
        double v5897_data = glb_m7[v4598_a];
        double v5899_data = glb_m7[v5898_a];
        double v5901_data = glb_m7[v4600_a];
        double v5903_data = glb_m7[v4602_a];
        double v5905_data = glb_m7[v4604_a];
        double v5907_data = glb_m7[v5906_a];
        double v5909_data = glb_m7[v4606_a];
        double v5911_data = glb_m7[v4608_a];
        double v5913_data = glb_m7[v4610_a];
        double v5915_data = glb_m7[v5239_a];
        double v5917_data = glb_m7[v4612_a];
        double v5919_data = glb_m7[v4614_a];
        double v5921_data = glb_m7[v5245_a];
        double v5923_data = glb_m7[v4616_a];
        double v5925_data = glb_m7[v4618_a];
        double v5927_data = glb_m7[v5251_a];
        double v5929_data = glb_m7[v4620_a];
        double v5931_data = glb_m7[v4622_a];
        double v5933_data = glb_m7[v5257_a];
        double v5935_data = glb_m7[v4624_a];
        double v5937_data = glb_m7[v4626_a];
        double v5939_data = glb_m7[v5938_a];
        double v5941_data = glb_m7[v4628_a];
        double v5943_data = glb_m7[v4630_a];
        double v5945_data = glb_m7[v5944_a];
        double v5947_data = glb_m7[v4632_a];
        double v5949_data = glb_m7[v4634_a];
        double v5951_data = glb_m7[v5950_a];
        double v5953_data = glb_m7[v4636_a];
        double v5955_data = glb_m7[v4638_a];
        double v5957_data = glb_m7[v5956_a];
        double v5959_data = glb_m7[v4640_a];
        double v5961_data = glb_m7[v4642_a];
        double v5963_data = glb_m7[v5279_a];
        double v5965_data = glb_m7[v4644_a];
        double v5967_data = glb_m7[v5283_a];
        double v5969_data = glb_m7[v4646_a];
        double v5971_data = glb_m7[v5287_a];
        double v5973_data = glb_m7[v4648_a];
        double v5975_data = glb_m7[v5291_a];
        double v5977_data = glb_m7[v4650_a];
        double v5990_acc{};
        double v5991_acc{};
        double v5992_acc{};
        double v5993_acc{};
        double v5996_data = r12[26];
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
        : "+&v"(v5990_acc), "+&v"(v5991_acc), "+&v"(v5992_acc), "+&v"(v5993_acc)
        : "v"(v5458_acc), "v"(v5749_data), "v"(v5753_data), "v"(v5757_data), "v"(v5761_data), "v"(v5763_data), "v"(v5765_data), "v"(v5767_data), "v"(v5769_data), "v"(v5771_data), "v"(v5773_data), "v"(v5775_data), "v"(v5777_data), "v"(v5779_data), "v"(v5781_data), "v"(v5783_data), "v"(v5785_data), "v"(v5787_data), "v"(v5789_data), "v"(v5791_data), "v"(v5793_data), "v"(v5795_data), "v"(v5797_data), "v"(v5799_data), "v"(v5801_data), "v"(v5803_data), "v"(v5805_data), "v"(v5807_data), "v"(v5809_data), "v"(v5811_data), "v"(v5813_data), "v"(v5815_data), "v"(v5817_data)
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
        "v_fmac_f64_dpp %2, %4, %19 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %20 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %21 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %22 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %23 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %24 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %25 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %26 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %27 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %28 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %29 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %30 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %31 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %32 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %4, %33 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %34 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %35 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %36 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5990_acc), "+&v"(v5991_acc), "+&v"(v5992_acc), "+&v"(v5993_acc)
        : "v"(v5458_acc), "v"(v5819_data), "v"(v5821_data), "v"(v5823_data), "v"(v5825_data), "v"(v5827_data), "v"(v5829_data), "v"(v5831_data), "v"(v5833_data), "v"(v5835_data), "v"(v5837_data), "v"(v5839_data), "v"(v5841_data), "v"(v5843_data), "v"(v5845_data), "v"(v5847_data), "v"(v5849_data), "v"(v5851_data), "v"(v5853_data), "v"(v5855_data), "v"(v5857_data), "v"(v5859_data), "v"(v5861_data), "v"(v5863_data), "v"(v5865_data), "v"(v5867_data), "v"(v5869_data), "v"(v5871_data), "v"(v5873_data), "v"(v5875_data), "v"(v5877_data), "v"(v5879_data), "v"(v5881_data)
        );
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
        "v_fmac_f64_dpp %1, %4, %21 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %22 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %23 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %24 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %25 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %26 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %27 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %28 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %29 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %30 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %31 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %32 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %33 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %4, %34 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %3, %4, %35 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %4, %36 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5990_acc), "+&v"(v5991_acc), "+&v"(v5992_acc), "+&v"(v5993_acc)
        : "v"(v5459_acc), "v"(v5883_data), "v"(v5885_data), "v"(v5887_data), "v"(v5889_data), "v"(v5891_data), "v"(v5893_data), "v"(v5895_data), "v"(v5897_data), "v"(v5899_data), "v"(v5901_data), "v"(v5903_data), "v"(v5905_data), "v"(v5907_data), "v"(v5909_data), "v"(v5911_data), "v"(v5913_data), "v"(v5915_data), "v"(v5917_data), "v"(v5919_data), "v"(v5921_data), "v"(v5923_data), "v"(v5925_data), "v"(v5927_data), "v"(v5929_data), "v"(v5931_data), "v"(v5933_data), "v"(v5935_data), "v"(v5937_data), "v"(v5939_data), "v"(v5941_data), "v"(v5943_data), "v"(v5945_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %12 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %13 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %14 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %15 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %18 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %19 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %20, %21 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %20, %22 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %20, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %20, %24 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %20, %25 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %20, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v5992_acc), "+&v"(v5993_acc), "+&v"(v5991_acc)
        : "v"(v5459_acc), "v"(v5947_data), "v"(v5949_data), "v"(v5951_data), "v"(v5953_data), "v"(v5955_data), "v"(v5957_data), "v"(v5959_data), "v"(v5961_data), "v"(v5963_data), "v"(v5965_data), "v"(v5967_data), "v"(v5969_data), "v"(v5971_data), "v"(v5973_data), "v"(v5975_data), "v"(v5977_data), "v"(v5996_data), "v"(v5693_data), "v"(v5704_data), "v"(v5695_data), "v"(v5706_data), "v"(v5697_data), "v"(v5708_data)
        );
        ir14[32] = v5990_acc;
        ir14[33] = v5991_acc;
        ir14[34] = v5992_acc;
        ir14[35] = v5993_acc;
        double r13[36]{};
        // r13 = load{g>r}(glb_m2);
        // r14 = ir14 + r13
        // r0 = load{g>r}(wrap_glb_m0);
        #pragma unroll
        for (int32_t v6108_i0 = 0; v6108_i0 < 2; ++v6108_i0) {
          int32_t v6110_lead = v4154_lead + (v6108_i0 * 16);
          #pragma unroll
          for (int32_t v6111_i1 = 0; v6111_i1 < 9; ++v6111_i1) {
            double v6114_data = __builtin_nontemporal_load(&wrap_glb_m0[(v6110_lead + (v6111_i1 * 56))]);
            r0[(v6108_i0 + (v6111_i1 * 3))] = v6114_data;
          }
        }
        if (v4164_g) {
          #pragma unroll
          for (int32_t v6117_i1 = 0; v6117_i1 < 9; ++v6117_i1) {
            double v6120_data = __builtin_nontemporal_load(&wrap_glb_m0[(v4304_lead + (v6117_i1 * 56))]);
            r0[(2 + (v6117_i1 * 3))] = v6120_data;
          }
        }
        if (v4660_g) {
          #pragma unroll
          for (int32_t v6024_n1 = 0; v6024_n1 < 9; ++v6024_n1) {
            int32_t v6025_a = v6024_n1 * 4;
            double v6027_data = ir14[v6025_a];
            double v6028_data = r9[v6025_a];
            r14[v6025_a] = (v6028_data + v6027_data);
          }
        }
        #pragma unroll
        for (int32_t v6030_n0 = 1; v6030_n0 < 3; ++v6030_n0) {
          #pragma unroll
          for (int32_t v6031_n1 = 0; v6031_n1 < 9; ++v6031_n1) {
            int32_t v6032_a = v6031_n1 * 4;
            int32_t v6033_a = v6030_n0 + v6032_a;
            double v6034_data = ir14[v6033_a];
            double v6035_data = r9[(v6030_n0 + v6032_a)];
            r14[v6033_a] = (v6035_data + v6034_data);
          }
        }
        // glb_m2 = store{r>g}(r14);
        if (v4660_g) {
          #pragma unroll
          for (int32_t v6043_i1 = 0; v6043_i1 < 9; ++v6043_i1) {
            double v6046_data = r14[(v6043_i1 * 4)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m2[(v4154_lead + (v6043_i1 * 56))], v6046_data);
            }
          }
        }
        #pragma unroll
        for (int32_t v6051_i0 = 1; v6051_i0 < 3; ++v6051_i0) {
          #pragma unroll
          for (int32_t v6052_i1 = 0; v6052_i1 < 9; ++v6052_i1) {
            double v6055_data = r14[(v6051_i0 + (v6052_i1 * 4))];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m2[((v4154_lead + (v6051_i0 * 16)) + (v6052_i1 * 56))], v6055_data);
            }
          }
        }
        if (v6015_g) {
          #pragma unroll
          for (int32_t v6060_i1 = 0; v6060_i1 < 9; ++v6060_i1) {
            double v6063_data = ir14[((v6060_i1 * 4) + 3)];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f64(&glb_m2[(v4311_lead + (v6060_i1 * 56))], v6063_data);
            }
          }
        }
      }
    }
  }
}

