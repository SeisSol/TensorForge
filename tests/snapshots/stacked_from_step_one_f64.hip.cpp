// === base name ===
kernel_44ac75e6739bab56

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_44ac75e6739bab56 = {{16, 16, 1}, 16, 40, 1, 16, 48640, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_44ac75e6739bab56(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_44ac75e6739bab56(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_44ac75e6739bab56(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_44ac75e6739bab56, block.x * block.y * block.z, 6080 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6080 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_44ac75e6739bab56, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (6080 * sizeof(double)));
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
  config.sharedMemBytes = 6080 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_44ac75e6739bab56(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_44ac75e6739bab56(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_44ac75e6739bab56), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m3;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_44ac75e6739bab56, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_44ac75e6739bab56(tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m3, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (40 active) x 16 per block = block 16x16x1, 48640 B shared, occupancy grid
    // operands:
    //   m0 30×9(30×9) {0..30}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 40×9(40×9) {0..40}×{0..9} strided
    //   m3 40×30(40×30) {0..40}×{0..30} none
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k]@{0..40}×{1..30} × t0[k,j]@{1..30}×{0..9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":40,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":6080}],"shared_bytes":48640,"shared_elements":6080,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0,0],[30,9]],"name":"m0","ordered":false,"parts":1,"shape":[30,9],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[40,9]],"name":"m2","ordered":false,"parts":1,"shape":[40,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,30]],"name":"m3","ordered":false,"parts":1,"shape":[40,30],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[30,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[40,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[40,30]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[40,30]},{"addressing":"strided","bbox":[[1,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[304 * threadIdx.y + 1216];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m3[0];
      double * __restrict__ glb_m3 = &totalShrMem[0];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      tensorforge::VectorT<double, 2> v609_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m3[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<double, 2>*)&glb_m3[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v609_ld;
      tensorforge::VectorT<double, 2> v610_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m3[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512];
      *(tensorforge::VectorT<double, 2>*)&glb_m3[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512] = v610_ld;
      if ((threadIdx.x + (threadIdx.y * 16)) < 176) {
        double v616_ld = ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v616_ld;
      }
      __syncthreads();
      double * __restrict__ s0 = &localShrMem0[0];
      size_t v619_batchIdLane0 = threadIdx.y % 4;
      size_t v622_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v639_lead = threadIdx.x % 16;
      int32_t v651_lead = v639_lead + 16_i32;
      bool v654_g = v639_lead < 14;
      bool v664_g = v639_lead < 9;
      int32_t v735_m = threadIdx.y % 4;
      int32_t v749_sh = (threadIdx.x / 4) * 30;
      int32_t v751_sh = ((threadIdx.x % 4) - v735_m) * 304;
      int32_t v753_a = (((min((max((v735_m + -1), 0)), 28)) + v749_sh) + v751_sh) + 1;
      bool v756_g = v735_m == 2;
      bool v758_g = v735_m == 1;
      bool v760_g = v735_m == 0;
      int32_t v762_a = v753_a + 120;
      int32_t v769_a = v639_lead + 40;
      int32_t v771_a = v639_lead + 80;
      int32_t v773_a = v639_lead + 120;
      int32_t v782_a = v651_lead + 40;
      int32_t v784_a = v651_lead + 80;
      int32_t v786_a = v651_lead + 120;
      int32_t v794_lead = v639_lead + 32_i32;
      int32_t v795_a = v794_lead + 40;
      int32_t v797_a = v794_lead + 80;
      int32_t v799_a = v794_lead + 120;
      int32_t v807_sh = (v735_m + v749_sh) + v751_sh;
      int32_t v808_a = v807_sh + 4;
      int32_t v811_a = v808_a + 120;
      int32_t v813_r = v735_m * 40;
      int32_t v814_a = v639_lead + v813_r;
      int32_t v815_a = v814_a + 160;
      int32_t v819_a = v651_lead + v813_r;
      int32_t v820_a = v819_a + 160;
      int32_t v824_a = v794_lead + v813_r;
      int32_t v825_a = v824_a + 160;
      int32_t v829_a = v807_sh + 8;
      int32_t v832_a = v829_a + 120;
      int32_t v834_a = v814_a + 320;
      int32_t v838_a = v819_a + 320;
      int32_t v842_a = v824_a + 320;
      int32_t v846_a = v807_sh + 12;
      int32_t v849_a = v846_a + 120;
      int32_t v851_a = v814_a + 480;
      int32_t v855_a = v819_a + 480;
      int32_t v859_a = v824_a + 480;
      int32_t v863_a = v807_sh + 16;
      int32_t v866_a = v863_a + 120;
      int32_t v868_a = v814_a + 640;
      int32_t v872_a = v819_a + 640;
      int32_t v876_a = v824_a + 640;
      int32_t v880_a = v807_sh + 20;
      int32_t v883_a = v880_a + 120;
      int32_t v885_a = v819_a + 800;
      int32_t v889_a = v824_a + 800;
      int32_t v893_a = v807_sh + 24;
      int32_t v896_a = v893_a + 120;
      int32_t v898_a = v819_a + 960;
      int32_t v902_a = v824_a + 960;
      int32_t v912_a = ((((min((max((v735_m + 27), 0)), 28)) - 27) + v749_sh) + v751_sh) + 28;
      double v915_ks = v756_g ? 0.0 : 0.0;
      int32_t v918_a = v912_a + 120;
      int32_t v922_a = v651_lead + 1120;
      int32_t v924_a = v651_lead + 1160;
      int32_t v930_a = v794_lead + 1120;
      int32_t v932_a = v794_lead + 1160;
      int32_t v986_a = v639_lead + 160;
      int32_t v988_a = v651_lead + 160;
      int32_t v990_a = v794_lead + 160;
      int32_t v992_a = v639_lead + 200;
      int32_t v994_a = v651_lead + 200;
      int32_t v996_a = v794_lead + 200;
      int32_t v998_a = v639_lead + 240;
      int32_t v1000_a = v651_lead + 240;
      int32_t v1002_a = v794_lead + 240;
      int32_t v1004_a = v639_lead + 280;
      int32_t v1006_a = v651_lead + 280;
      int32_t v1008_a = v794_lead + 280;
      int32_t v1010_a = v639_lead + 320;
      int32_t v1012_a = v651_lead + 320;
      int32_t v1014_a = v794_lead + 320;
      int32_t v1016_a = v639_lead + 360;
      int32_t v1018_a = v651_lead + 360;
      int32_t v1020_a = v794_lead + 360;
      int32_t v1022_a = v639_lead + 400;
      int32_t v1024_a = v651_lead + 400;
      int32_t v1026_a = v794_lead + 400;
      int32_t v1028_a = v639_lead + 440;
      int32_t v1030_a = v651_lead + 440;
      int32_t v1032_a = v794_lead + 440;
      int32_t v1034_a = v639_lead + 480;
      int32_t v1036_a = v651_lead + 480;
      int32_t v1038_a = v794_lead + 480;
      int32_t v1040_a = v639_lead + 520;
      int32_t v1042_a = v651_lead + 520;
      int32_t v1044_a = v794_lead + 520;
      int32_t v1046_a = v639_lead + 560;
      int32_t v1048_a = v651_lead + 560;
      int32_t v1050_a = v794_lead + 560;
      int32_t v1052_a = v639_lead + 600;
      int32_t v1054_a = v651_lead + 600;
      int32_t v1056_a = v794_lead + 600;
      int32_t v1058_a = v639_lead + 640;
      int32_t v1060_a = v651_lead + 640;
      int32_t v1062_a = v794_lead + 640;
      int32_t v1064_a = v639_lead + 680;
      int32_t v1066_a = v651_lead + 680;
      int32_t v1068_a = v794_lead + 680;
      int32_t v1070_a = v639_lead + 720;
      int32_t v1072_a = v651_lead + 720;
      int32_t v1074_a = v794_lead + 720;
      int32_t v1076_a = v639_lead + 760;
      int32_t v1078_a = v651_lead + 760;
      int32_t v1080_a = v794_lead + 760;
      int32_t v1082_a = v651_lead + 800;
      int32_t v1084_a = v794_lead + 800;
      int32_t v1086_a = v651_lead + 840;
      int32_t v1088_a = v794_lead + 840;
      int32_t v1090_a = v651_lead + 880;
      int32_t v1092_a = v794_lead + 880;
      int32_t v1094_a = v651_lead + 920;
      int32_t v1096_a = v794_lead + 920;
      int32_t v1098_a = v651_lead + 960;
      int32_t v1100_a = v794_lead + 960;
      int32_t v1102_a = v651_lead + 1000;
      int32_t v1104_a = v794_lead + 1000;
      int32_t v1106_a = v651_lead + 1040;
      int32_t v1108_a = v794_lead + 1040;
      int32_t v1110_a = v651_lead + 1080;
      int32_t v1112_a = v794_lead + 1080;
      bool v1132_g = v639_lead < 8;
      double r0[18]{};
      size_t v1158_row = v622_batchIdGroup0First + v619_batchIdLane0;
      const bool peel_batchIdActive0 = v1158_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1158_row]));
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[(peel_batchIdActive0 ? v1158_row : v622_batchIdGroup0First) * 270 + 0 + m0_extraOffset];
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v1166_i0 = 0; v1166_i0 < 1; ++v1166_i0) {
        int32_t v1168_lead = v639_lead + (v1166_i0 * 16);
        #pragma unroll
        for (int32_t v1169_i1 = 0; v1169_i1 < 9; ++v1169_i1) {
          double v1172_data = __builtin_nontemporal_load(&peel_glb_m0[(v1168_lead + (v1169_i1 * 30))]);
          r0[(v1166_i0 + (v1169_i1 * 2))] = v1172_data;
        }
      }
      #pragma unroll
      for (int32_t v1175_i1 = 0; v1175_i1 < 9; ++v1175_i1) {
        double v1178_data = v654_g ? (__builtin_nontemporal_load(&peel_glb_m0[(v651_lead + (v1175_i1 * 30))])) : (0.0);
        r0[(1 + (v1175_i1 * 2))] = v1178_data;
      }
      for (size_t v623_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v623_batchIdGroup0 < numElements0; v623_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v1155_groupNextAhead = v623_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v1157_groupNext = (v1155_groupNextAhead < numElements0) ? v1155_groupNextAhead : v623_batchIdGroup0;
        size_t v1162_row = v1157_groupNext + v619_batchIdLane0;
        const bool wrap_batchIdActive0 = v1162_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1162_row]));
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_m0 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m0[(wrap_batchIdActive0 ? v1162_row : v1157_groupNext) * 270 + 0 + m0_extraOffset];
        size_t v624_row = v623_batchIdGroup0 + v619_batchIdLane0;
        const bool batchIdActive0 = v624_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v624_row]));
        size_t v626_batchId0 = batchIdActive0 ? v624_row : v623_batchIdGroup0;
        size_t v627_ahead1 = v626_batchId0 + (gridDim.x * blockDim.y);
        size_t v629_batchId1 = (v627_ahead1 < numElements0) ? v627_ahead1 : v626_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v626_batchId0 * 81 + 0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m2[v626_batchId0 * 360 + 0 + m2_extraOffset];
        double r1[9]{};
        // r1 = load{g>r}(glb_m1);
        #pragma unroll
        for (int32_t v659_i1 = 0; v659_i1 < 9; ++v659_i1) {
          double v665_data = v664_g ? (__builtin_nontemporal_load(&glb_m1[(v639_lead + (v659_i1 * 9))])) : (0.0);
          r1[v659_i1] = v665_data;
        }
        double r2[18]{};
        // r2 = +(r0 * r1) + None
        // [(0, 30), (0, 9)] [(0, 9)]
        double v668_data = r0[0];
        double v669_data = r0[1];
        double v670_data = r0[2];
        double v671_data = r0[3];
        double v672_data = r0[4];
        double v673_data = r0[5];
        double v674_data = r0[6];
        double v675_data = r0[7];
        double v676_data = r0[8];
        double v677_data = r0[9];
        double v678_data = r0[10];
        double v679_data = r0[11];
        double v680_data = r0[12];
        double v681_data = r0[13];
        double v682_data = r0[14];
        double v683_data = r0[15];
        double v684_data = r0[16];
        double v685_data = r0[17];
        double v686_acc{};
        double v687_acc{};
        double v688_acc{};
        double v689_acc{};
        double v690_acc{};
        double v691_acc{};
        double v692_acc{};
        double v693_acc{};
        double v694_acc{};
        double v695_acc{};
        double v696_acc{};
        double v697_acc{};
        double v698_acc{};
        double v699_acc{};
        double v700_acc{};
        double v701_acc{};
        double v702_acc{};
        double v703_acc{};
        double v704_data = r1[0];
        double v705_data = r1[1];
        double v706_data = r1[2];
        double v707_data = r1[3];
        double v708_data = r1[4];
        double v709_data = r1[5];
        double v710_data = r1[6];
        double v711_data = r1[7];
        double v712_data = r1[8];
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
        : "+&v"(v686_acc), "+&v"(v687_acc), "+&v"(v688_acc), "+&v"(v689_acc)
        : "v"(v704_data), "v"(v668_data), "v"(v669_data), "v"(v670_data), "v"(v671_data), "v"(v672_data), "v"(v673_data), "v"(v674_data), "v"(v675_data), "v"(v676_data), "v"(v677_data), "v"(v678_data), "v"(v679_data), "v"(v680_data), "v"(v681_data), "v"(v682_data), "v"(v683_data), "v"(v684_data), "v"(v685_data), "v"(v705_data)
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
        : "+&v"(v688_acc), "+&v"(v689_acc), "+&v"(v690_acc), "+&v"(v691_acc), "+&v"(v692_acc), "+&v"(v693_acc)
        : "v"(v705_data), "v"(v682_data), "v"(v683_data), "v"(v684_data), "v"(v685_data), "v"(v706_data), "v"(v668_data), "v"(v669_data), "v"(v670_data), "v"(v671_data), "v"(v672_data), "v"(v673_data), "v"(v674_data), "v"(v675_data), "v"(v676_data), "v"(v677_data), "v"(v678_data), "v"(v679_data), "v"(v680_data), "v"(v681_data), "v"(v707_data)
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
        : "+&v"(v692_acc), "+&v"(v693_acc), "+&v"(v694_acc), "+&v"(v695_acc), "+&v"(v696_acc), "+&v"(v697_acc)
        : "v"(v707_data), "v"(v678_data), "v"(v679_data), "v"(v680_data), "v"(v681_data), "v"(v682_data), "v"(v683_data), "v"(v684_data), "v"(v685_data), "v"(v708_data), "v"(v668_data), "v"(v669_data), "v"(v670_data), "v"(v671_data), "v"(v672_data), "v"(v673_data), "v"(v674_data), "v"(v675_data), "v"(v676_data), "v"(v677_data), "v"(v709_data)
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
        : "+&v"(v696_acc), "+&v"(v697_acc), "+&v"(v698_acc), "+&v"(v699_acc), "+&v"(v700_acc), "+&v"(v701_acc)
        : "v"(v709_data), "v"(v674_data), "v"(v675_data), "v"(v676_data), "v"(v677_data), "v"(v678_data), "v"(v679_data), "v"(v680_data), "v"(v681_data), "v"(v682_data), "v"(v683_data), "v"(v684_data), "v"(v685_data), "v"(v710_data), "v"(v668_data), "v"(v669_data), "v"(v670_data), "v"(v671_data), "v"(v672_data), "v"(v673_data), "v"(v711_data)
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
        : "+&v"(v700_acc), "+&v"(v701_acc), "+&v"(v702_acc), "+&v"(v703_acc)
        : "v"(v711_data), "v"(v670_data), "v"(v671_data), "v"(v672_data), "v"(v673_data), "v"(v674_data), "v"(v675_data), "v"(v676_data), "v"(v677_data), "v"(v678_data), "v"(v679_data), "v"(v680_data), "v"(v681_data), "v"(v682_data), "v"(v683_data), "v"(v684_data), "v"(v685_data), "v"(v712_data), "v"(v668_data), "v"(v669_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v702_acc), "+&v"(v703_acc)
        : "v"(v712_data), "v"(v684_data), "v"(v685_data)
        );
        r2[0] = v686_acc;
        r2[1] = v687_acc;
        r2[2] = v688_acc;
        r2[3] = v689_acc;
        r2[4] = v690_acc;
        r2[5] = v691_acc;
        r2[6] = v692_acc;
        r2[7] = v693_acc;
        r2[8] = v694_acc;
        r2[9] = v695_acc;
        r2[10] = v696_acc;
        r2[11] = v697_acc;
        r2[12] = v698_acc;
        r2[13] = v699_acc;
        r2[14] = v700_acc;
        r2[15] = v701_acc;
        r2[16] = v702_acc;
        r2[17] = v703_acc;
        // r2 = 0 past (0, 30)
        // s0 = store{r>s}(localShrMem0, r2);
        #pragma unroll
        for (int32_t v714_i0 = 0; v714_i0 < 1; ++v714_i0) {
          int32_t v720_lead = v639_lead + (v714_i0 * 16);
          #pragma unroll
          for (int32_t v715_i1 = 0; v715_i1 < 9; ++v715_i1) {
            double v718_data = r2[(v714_i0 + (v715_i1 * 2))];
            s0[(v720_lead + (v715_i1 * 30))] = v718_data;
          }
        }
        if (v654_g) {
          #pragma unroll
          for (int32_t v724_i1 = 0; v724_i1 < 9; ++v724_i1) {
            double v727_data = r2[(1 + (v724_i1 * 2))];
            s0[(v651_lead + (v724_i1 * 30))] = v727_data;
          }
        }
        double r3[27]{};
        // ir3 = +(glb_m3 * s0)
        // [(0, 40), (0, 9)] [(1, 30)]
        double ir3[27]{};
        tensorforge::VectorT<double, 4> v739_acc{};
        tensorforge::VectorT<double, 4> v740_acc{};
        tensorforge::VectorT<double, 4> v741_acc{};
        tensorforge::VectorT<double, 4> v742_acc{};
        tensorforge::VectorT<double, 4> v743_acc{};
        tensorforge::VectorT<double, 4> v744_acc{};
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
        double v755_data = s0[v753_a];
        double v761_ks = v760_g ? 0.0 : (v758_g ? v755_data : (v756_g ? v755_data : v755_data));
        double v763_data = s0[v762_a];
        double v766_ks = v760_g ? 0.0 : (v758_g ? v763_data : (v756_g ? v763_data : v763_data));
        double v770_data = glb_m3[v769_a];
        double v772_data = glb_m3[v771_a];
        double v774_data = glb_m3[v773_a];
        double v777_ks = v760_g ? 0.0 : (v758_g ? v770_data : (v756_g ? v772_data : v774_data));
        tensorforge::VectorT<double, 4> v778_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v761_ks, v777_ks, v739_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v779_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v766_ks, v777_ks, v740_acc, 0, 0, 0);
        double v783_data = glb_m3[v782_a];
        double v785_data = glb_m3[v784_a];
        double v787_data = glb_m3[v786_a];
        double v790_ks = v760_g ? 0.0 : (v758_g ? v783_data : (v756_g ? v785_data : v787_data));
        tensorforge::VectorT<double, 4> v791_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v761_ks, v790_ks, v741_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v792_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v766_ks, v790_ks, v742_acc, 0, 0, 0);
        double v796_data = glb_m3[v795_a];
        double v798_data = glb_m3[v797_a];
        double v800_data = glb_m3[v799_a];
        double v803_ks = v760_g ? 0.0 : (v758_g ? v796_data : (v756_g ? v798_data : v800_data));
        tensorforge::VectorT<double, 4> v804_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v761_ks, v803_ks, v743_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v805_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v766_ks, v803_ks, v744_acc, 0, 0, 0);
        double v810_data = s0[v808_a];
        double v812_data = s0[v811_a];
        double v816_data = glb_m3[v815_a];
        tensorforge::VectorT<double, 4> v817_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v810_data, v816_data, v778_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v818_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v812_data, v816_data, v779_acc, 0, 0, 0);
        double v821_data = glb_m3[v820_a];
        tensorforge::VectorT<double, 4> v822_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v810_data, v821_data, v791_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v823_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v812_data, v821_data, v792_acc, 0, 0, 0);
        double v826_data = glb_m3[v825_a];
        tensorforge::VectorT<double, 4> v827_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v810_data, v826_data, v804_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v828_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v812_data, v826_data, v805_acc, 0, 0, 0);
        double v831_data = s0[v829_a];
        double v833_data = s0[v832_a];
        double v835_data = glb_m3[v834_a];
        tensorforge::VectorT<double, 4> v836_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v831_data, v835_data, v817_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v837_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v833_data, v835_data, v818_acc, 0, 0, 0);
        double v839_data = glb_m3[v838_a];
        tensorforge::VectorT<double, 4> v840_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v831_data, v839_data, v822_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v841_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v833_data, v839_data, v823_acc, 0, 0, 0);
        double v843_data = glb_m3[v842_a];
        tensorforge::VectorT<double, 4> v844_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v831_data, v843_data, v827_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v845_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v833_data, v843_data, v828_acc, 0, 0, 0);
        double v848_data = s0[v846_a];
        double v850_data = s0[v849_a];
        double v852_data = glb_m3[v851_a];
        tensorforge::VectorT<double, 4> v853_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v848_data, v852_data, v836_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v854_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v850_data, v852_data, v837_acc, 0, 0, 0);
        double v856_data = glb_m3[v855_a];
        tensorforge::VectorT<double, 4> v857_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v848_data, v856_data, v840_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v858_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v850_data, v856_data, v841_acc, 0, 0, 0);
        double v860_data = glb_m3[v859_a];
        tensorforge::VectorT<double, 4> v861_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v848_data, v860_data, v844_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v862_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v850_data, v860_data, v845_acc, 0, 0, 0);
        double v865_data = s0[v863_a];
        double v867_data = s0[v866_a];
        double v869_data = glb_m3[v868_a];
        tensorforge::VectorT<double, 4> v870_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v865_data, v869_data, v853_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v871_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v867_data, v869_data, v854_acc, 0, 0, 0);
        double v873_data = glb_m3[v872_a];
        tensorforge::VectorT<double, 4> v874_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v865_data, v873_data, v857_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v875_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v867_data, v873_data, v858_acc, 0, 0, 0);
        double v877_data = glb_m3[v876_a];
        tensorforge::VectorT<double, 4> v878_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v865_data, v877_data, v861_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v879_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v867_data, v877_data, v862_acc, 0, 0, 0);
        double v882_data = s0[v880_a];
        double v884_data = s0[v883_a];
        double v886_data = glb_m3[v885_a];
        tensorforge::VectorT<double, 4> v887_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v882_data, v886_data, v874_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v888_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v884_data, v886_data, v875_acc, 0, 0, 0);
        double v890_data = glb_m3[v889_a];
        tensorforge::VectorT<double, 4> v891_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v882_data, v890_data, v878_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v892_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v884_data, v890_data, v879_acc, 0, 0, 0);
        double v895_data = s0[v893_a];
        double v897_data = s0[v896_a];
        double v899_data = glb_m3[v898_a];
        tensorforge::VectorT<double, 4> v900_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v895_data, v899_data, v887_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v901_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v897_data, v899_data, v888_acc, 0, 0, 0);
        double v903_data = glb_m3[v902_a];
        tensorforge::VectorT<double, 4> v904_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v895_data, v903_data, v891_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v905_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v897_data, v903_data, v892_acc, 0, 0, 0);
        double v914_data = s0[v912_a];
        double v917_ks = v760_g ? v914_data : (v758_g ? v914_data : v915_ks);
        double v919_data = s0[v918_a];
        double v921_ks = v760_g ? v919_data : (v758_g ? v919_data : v915_ks);
        double v923_data = glb_m3[v922_a];
        double v925_data = glb_m3[v924_a];
        double v927_ks = v760_g ? v923_data : (v758_g ? v925_data : v915_ks);
        tensorforge::VectorT<double, 4> v928_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v917_ks, v927_ks, v900_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v929_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v921_ks, v927_ks, v901_acc, 0, 0, 0);
        double v931_data = glb_m3[v930_a];
        double v933_data = glb_m3[v932_a];
        double v935_ks = v760_g ? v931_data : (v758_g ? v933_data : v915_ks);
        tensorforge::VectorT<double, 4> v936_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v917_ks, v935_ks, v904_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v937_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v921_ks, v935_ks, v905_acc, 0, 0, 0);
        ir3[0] = (v870_acc[0]);
        ir3[3] = (v870_acc[1]);
        ir3[6] = (v870_acc[2]);
        ir3[9] = (v870_acc[3]);
        ir3[12] = (v871_acc[0]);
        ir3[15] = (v871_acc[1]);
        ir3[18] = (v871_acc[2]);
        ir3[21] = (v871_acc[3]);
        ir3[1] = (v928_acc[0]);
        ir3[4] = (v928_acc[1]);
        ir3[7] = (v928_acc[2]);
        ir3[10] = (v928_acc[3]);
        ir3[13] = (v929_acc[0]);
        ir3[16] = (v929_acc[1]);
        ir3[19] = (v929_acc[2]);
        ir3[22] = (v929_acc[3]);
        ir3[2] = (v936_acc[0]);
        ir3[5] = (v936_acc[1]);
        ir3[8] = (v936_acc[2]);
        ir3[11] = (v936_acc[3]);
        ir3[14] = (v937_acc[0]);
        ir3[17] = (v937_acc[1]);
        ir3[20] = (v937_acc[2]);
        ir3[23] = (v937_acc[3]);
        double v987_data = glb_m3[v986_a];
        double v989_data = glb_m3[v988_a];
        double v991_data = glb_m3[v990_a];
        double v993_data = glb_m3[v992_a];
        double v995_data = glb_m3[v994_a];
        double v997_data = glb_m3[v996_a];
        double v999_data = glb_m3[v998_a];
        double v1001_data = glb_m3[v1000_a];
        double v1003_data = glb_m3[v1002_a];
        double v1005_data = glb_m3[v1004_a];
        double v1007_data = glb_m3[v1006_a];
        double v1009_data = glb_m3[v1008_a];
        double v1011_data = glb_m3[v1010_a];
        double v1013_data = glb_m3[v1012_a];
        double v1015_data = glb_m3[v1014_a];
        double v1017_data = glb_m3[v1016_a];
        double v1019_data = glb_m3[v1018_a];
        double v1021_data = glb_m3[v1020_a];
        double v1023_data = glb_m3[v1022_a];
        double v1025_data = glb_m3[v1024_a];
        double v1027_data = glb_m3[v1026_a];
        double v1029_data = glb_m3[v1028_a];
        double v1031_data = glb_m3[v1030_a];
        double v1033_data = glb_m3[v1032_a];
        double v1035_data = glb_m3[v1034_a];
        double v1037_data = glb_m3[v1036_a];
        double v1039_data = glb_m3[v1038_a];
        double v1041_data = glb_m3[v1040_a];
        double v1043_data = glb_m3[v1042_a];
        double v1045_data = glb_m3[v1044_a];
        double v1047_data = glb_m3[v1046_a];
        double v1049_data = glb_m3[v1048_a];
        double v1051_data = glb_m3[v1050_a];
        double v1053_data = glb_m3[v1052_a];
        double v1055_data = glb_m3[v1054_a];
        double v1057_data = glb_m3[v1056_a];
        // r0 = load{g>r}(wrap_glb_m0);
        #pragma unroll
        for (int32_t v1181_i0 = 0; v1181_i0 < 1; ++v1181_i0) {
          int32_t v1183_lead = v639_lead + (v1181_i0 * 16);
          #pragma unroll
          for (int32_t v1184_i1 = 0; v1184_i1 < 9; ++v1184_i1) {
            double v1187_data = __builtin_nontemporal_load(&wrap_glb_m0[(v1183_lead + (v1184_i1 * 30))]);
            r0[(v1181_i0 + (v1184_i1 * 2))] = v1187_data;
          }
        }
        #pragma unroll
        for (int32_t v1190_i1 = 0; v1190_i1 < 9; ++v1190_i1) {
          double v1193_data = v654_g ? (__builtin_nontemporal_load(&wrap_glb_m0[(v651_lead + (v1190_i1 * 30))])) : (0.0);
          r0[(1 + (v1190_i1 * 2))] = v1193_data;
        }
        double v1059_data = glb_m3[v1058_a];
        double v1061_data = glb_m3[v1060_a];
        double v1063_data = glb_m3[v1062_a];
        double v1065_data = glb_m3[v1064_a];
        double v1067_data = glb_m3[v1066_a];
        double v1069_data = glb_m3[v1068_a];
        double v1071_data = glb_m3[v1070_a];
        double v1073_data = glb_m3[v1072_a];
        double v1075_data = glb_m3[v1074_a];
        double v1077_data = glb_m3[v1076_a];
        double v1079_data = glb_m3[v1078_a];
        double v1081_data = glb_m3[v1080_a];
        double v1083_data = glb_m3[v1082_a];
        double v1085_data = glb_m3[v1084_a];
        double v1087_data = glb_m3[v1086_a];
        double v1089_data = glb_m3[v1088_a];
        double v1091_data = glb_m3[v1090_a];
        double v1093_data = glb_m3[v1092_a];
        double v1095_data = glb_m3[v1094_a];
        double v1097_data = glb_m3[v1096_a];
        double v1099_data = glb_m3[v1098_a];
        double v1101_data = glb_m3[v1100_a];
        double v1103_data = glb_m3[v1102_a];
        double v1105_data = glb_m3[v1104_a];
        double v1107_data = glb_m3[v1106_a];
        double v1109_data = glb_m3[v1108_a];
        double v1111_data = glb_m3[v1110_a];
        double v1113_data = glb_m3[v1112_a];
        double v1122_acc{};
        double v1123_acc{};
        double v1124_acc{};
        double v1126_data = r2[17];
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %17 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %18 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %19 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %20 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %21 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %22 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %23 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %24 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %25 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %26 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %27 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %28 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %29 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %30 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %31 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %32 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %33 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %34 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %35 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v1122_acc), "+&v"(v1123_acc), "+&v"(v1124_acc)
        : "v"(v702_acc), "v"(v770_data), "v"(v783_data), "v"(v796_data), "v"(v772_data), "v"(v785_data), "v"(v798_data), "v"(v774_data), "v"(v787_data), "v"(v800_data), "v"(v987_data), "v"(v989_data), "v"(v991_data), "v"(v993_data), "v"(v995_data), "v"(v997_data), "v"(v999_data), "v"(v1001_data), "v"(v1003_data), "v"(v1005_data), "v"(v1007_data), "v"(v1009_data), "v"(v1011_data), "v"(v1013_data), "v"(v1015_data), "v"(v1017_data), "v"(v1019_data), "v"(v1021_data), "v"(v1023_data), "v"(v1025_data), "v"(v1027_data), "v"(v1029_data), "v"(v1031_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %3, %4 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %5 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %6 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %7 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %8 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %9 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %10 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %11 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %12 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %13 row_newbcast:14 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %3, %14 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %3, %15 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %3, %16 row_newbcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %17, %18 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %19 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %20 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %17, %21 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %22 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %23 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %17, %24 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %25 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %26 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %17, %27 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %28 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %29 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %30 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %31 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %32 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %33 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %34 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %17, %35 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %2, %17, %36 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v1124_acc), "+&v"(v1122_acc), "+&v"(v1123_acc)
        : "v"(v702_acc), "v"(v1033_data), "v"(v1035_data), "v"(v1037_data), "v"(v1039_data), "v"(v1041_data), "v"(v1043_data), "v"(v1045_data), "v"(v1047_data), "v"(v1049_data), "v"(v1051_data), "v"(v1053_data), "v"(v1055_data), "v"(v1057_data), "v"(v1126_data), "v"(v1059_data), "v"(v1061_data), "v"(v1063_data), "v"(v1065_data), "v"(v1067_data), "v"(v1069_data), "v"(v1071_data), "v"(v1073_data), "v"(v1075_data), "v"(v1077_data), "v"(v1079_data), "v"(v1081_data), "v"(v1083_data), "v"(v1085_data), "v"(v1087_data), "v"(v1089_data), "v"(v1091_data), "v"(v1093_data), "v"(v1095_data)
        );
        asm("s_nop 1\n\t"
        "v_fmac_f64_dpp %0, %2, %3 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %4 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %5 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %6 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %7 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %8 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %9 row_newbcast:10 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %10 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %11 row_newbcast:11 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %12 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %13 row_newbcast:12 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %1, %2, %14 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f64_dpp %0, %2, %15 row_newbcast:13 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v1124_acc), "+&v"(v1123_acc)
        : "v"(v1126_data), "v"(v1097_data), "v"(v1099_data), "v"(v1101_data), "v"(v1103_data), "v"(v1105_data), "v"(v1107_data), "v"(v1109_data), "v"(v1111_data), "v"(v1113_data), "v"(v923_data), "v"(v931_data), "v"(v925_data), "v"(v933_data)
        );
        ir3[24] = v1122_acc;
        ir3[25] = v1123_acc;
        ir3[26] = v1124_acc;
        // r3 = ir3
        // glb_m2 = store{r>g}(r3);
        #pragma unroll
        for (int32_t v1137_i0 = 0; v1137_i0 < 2; ++v1137_i0) {
          #pragma unroll
          for (int32_t v1138_i1 = 0; v1138_i1 < 9; ++v1138_i1) {
            double v1141_data = ir3[(v1137_i0 + (v1138_i1 * 3))];
            if (batchIdActive0) {
              glb_m2[((v639_lead + (v1137_i0 * 16)) + (v1138_i1 * 40))] = v1141_data;
            }
          }
        }
        if (v1132_g) {
          #pragma unroll
          for (int32_t v1147_i1 = 0; v1147_i1 < 9; ++v1147_i1) {
            double v1150_data = ir3[((v1147_i1 * 3) + 2)];
            if (batchIdActive0) {
              glb_m2[(v794_lead + (v1147_i1 * 40))] = v1150_data;
            }
          }
        }
      }
    }
  }
}

