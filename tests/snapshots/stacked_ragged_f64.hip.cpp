// === base name ===
kernel_e1f0515c44b625f1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e1f0515c44b625f1 = {{16, 16, 1}, 16, 48, 1, 16, 11264, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e1f0515c44b625f1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e1f0515c44b625f1(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e1f0515c44b625f1(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e1f0515c44b625f1, block.x * block.y * block.z, 1408 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1408 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e1f0515c44b625f1, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (1408 * sizeof(double)));
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
  config.sharedMemBytes = 1408 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e1f0515c44b625f1(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e1f0515c44b625f1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e1f0515c44b625f1), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_e1f0515c44b625f1, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e1f0515c44b625f1(tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (48 active) x 16 per block = block 16x16x1, 11264 B shared, occupancy grid
    // operands:
    //   m0 48×8(48×8) {0..48}×{0..8} strided
    //   m1 48×24(48×24) {0..48}×{0..24} none
    //   m2 24×8(24×8) {0..24}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":48,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1408}],"shared_bytes":11264,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[48,8]],"name":"m0","ordered":false,"parts":1,"shape":[48,8],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[48,24]],"name":"m1","ordered":false,"parts":1,"shape":[48,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,8]],"name":"m2","ordered":false,"parts":1,"shape":[24,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[48,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[48,8]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[48,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[48,24]},{"addressing":"strided","bbox":[[0,0],[24,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 1152];
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m1[0];
      double * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<double, 2> v9_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      *(tensorforge::VectorT<double, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      tensorforge::VectorT<double, 2> v10_ld = *(tensorforge::VectorRelaxedT<double, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512];
      *(tensorforge::VectorT<double, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 512] = v10_ld;
      if ((threadIdx.x + (threadIdx.y * 16)) < 128) {
        double v16_ld = ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024];
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v16_ld;
      }
      __syncthreads();
      size_t v18_batchIdLane0 = threadIdx.y % 4;
      size_t v21_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v37_lead = threadIdx.x % 16;
      int32_t v49_lead = v37_lead + 16_i32;
      bool v52_g = v37_lead < 8;
      int32_t v59_m = threadIdx.y % 4;
      int32_t v61_c = threadIdx.x / 4;
      int32_t v63_src = ((threadIdx.x % 4) * 16) + v59_m;
      bool v79_g = v61_c == 2;
      bool v81_g = v61_c == 1;
      bool v83_g = v61_c == 0;
      int32_t v96_r = v59_m * 48;
      int32_t v99_a = v37_lead + v96_r;
      int32_t v106_a = v49_lead + v96_r;
      int32_t v113_a = (v37_lead + 32_i32) + v96_r;
      int32_t v118_src = v63_src + 4;
      int32_t v141_a = v99_a + 192;
      int32_t v145_a = v106_a + 192;
      int32_t v149_a = v113_a + 192;
      int32_t v153_src = v63_src + 8;
      int32_t v176_a = v99_a + 384;
      int32_t v180_a = v106_a + 384;
      int32_t v184_a = v113_a + 384;
      int32_t v188_src = v63_src + 12;
      int32_t v211_a = v99_a + 576;
      int32_t v215_a = v106_a + 576;
      int32_t v219_a = v113_a + 576;
      int32_t v245_a = v99_a + 768;
      int32_t v249_a = v106_a + 768;
      int32_t v253_a = v113_a + 768;
      int32_t v279_a = v99_a + 960;
      int32_t v283_a = v106_a + 960;
      int32_t v287_a = v113_a + 960;
      double r0[16]{};
      size_t v332_row = v21_batchIdGroup0First + v18_batchIdLane0;
      const bool peel_batchIdActive0 = v332_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v332_row]));
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v332_row : v21_batchIdGroup0First) * 192 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v340_i0 = 0; v340_i0 < 1; ++v340_i0) {
        int32_t v342_lead = v37_lead + (v340_i0 * 16);
        #pragma unroll
        for (int32_t v343_i1 = 0; v343_i1 < 8; ++v343_i1) {
          double v346_data = __builtin_nontemporal_load(&peel_glb_m2[(v342_lead + (v343_i1 * 24))]);
          r0[(v340_i0 + (v343_i1 * 2))] = v346_data;
        }
      }
      #pragma unroll
      for (int32_t v349_i1 = 0; v349_i1 < 8; ++v349_i1) {
        double v352_data = v52_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v49_lead + (v349_i1 * 24))])) : (0.0);
        r0[(1 + (v349_i1 * 2))] = v352_data;
      }
      for (size_t v22_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v22_batchIdGroup0 < numElements0; v22_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v329_groupNextAhead = v22_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v331_groupNext = (v329_groupNextAhead < numElements0) ? v329_groupNextAhead : v22_batchIdGroup0;
        size_t v336_row = v331_groupNext + v18_batchIdLane0;
        const bool wrap_batchIdActive0 = v336_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v336_row]));
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v336_row : v331_groupNext) * 192 + 0 + m2_extraOffset];
        size_t v23_row = v22_batchIdGroup0 + v18_batchIdLane0;
        const bool batchIdActive0 = v23_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v23_row]));
        size_t v25_batchId0 = batchIdActive0 ? v23_row : v22_batchIdGroup0;
        size_t v26_ahead1 = v25_batchId0 + (gridDim.x * blockDim.y);
        size_t v28_batchId1 = (v26_ahead1 < numElements0) ? v26_ahead1 : v25_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v25_batchId0 * 384 + 0 + m0_extraOffset];
        double r1[24]{};
        // ir1 = +(glb_m1 * r0)
        // [(0, 48), (0, 8)] [(0, 24)]
        double ir1[24]{};
        tensorforge::VectorT<double, 4> v64_acc{};
        tensorforge::VectorT<double, 4> v65_acc{};
        tensorforge::VectorT<double, 4> v66_acc{};
        tensorforge::VectorT<double, 4> v67_acc{};
        tensorforge::VectorT<double, 4> v68_acc{};
        tensorforge::VectorT<double, 4> v69_acc{};
        double v71_data = r0[0];
        double v72_ga = tensorforge::gather<64>(v71_data, v63_src);
        double v73_data = r0[2];
        double v74_ga = tensorforge::gather<64>(v73_data, v63_src);
        double v75_data = r0[4];
        double v76_ga = tensorforge::gather<64>(v75_data, v63_src);
        double v77_data = r0[6];
        double v78_ga = tensorforge::gather<64>(v77_data, v63_src);
        double v84_af = v83_g ? v72_ga : (v81_g ? v74_ga : (v79_g ? v76_ga : v78_ga));
        double v85_data = r0[8];
        double v86_ga = tensorforge::gather<64>(v85_data, v63_src);
        double v87_data = r0[10];
        double v88_ga = tensorforge::gather<64>(v87_data, v63_src);
        double v89_data = r0[12];
        double v90_ga = tensorforge::gather<64>(v89_data, v63_src);
        double v91_data = r0[14];
        double v92_ga = tensorforge::gather<64>(v91_data, v63_src);
        double v95_af = v83_g ? v86_ga : (v81_g ? v88_ga : (v79_g ? v90_ga : v92_ga));
        double v101_data = glb_m1[v99_a];
        tensorforge::VectorT<double, 4> v102_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v84_af, v101_data, v64_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v103_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v95_af, v101_data, v65_acc, 0, 0, 0);
        double v108_data = glb_m1[v106_a];
        tensorforge::VectorT<double, 4> v109_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v84_af, v108_data, v66_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v110_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v95_af, v108_data, v67_acc, 0, 0, 0);
        double v115_data = glb_m1[v113_a];
        tensorforge::VectorT<double, 4> v116_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v84_af, v115_data, v68_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v117_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v95_af, v115_data, v69_acc, 0, 0, 0);
        double v120_ga = tensorforge::gather<64>(v71_data, v118_src);
        double v122_ga = tensorforge::gather<64>(v73_data, v118_src);
        double v124_ga = tensorforge::gather<64>(v75_data, v118_src);
        double v126_ga = tensorforge::gather<64>(v77_data, v118_src);
        double v129_af = v83_g ? v120_ga : (v81_g ? v122_ga : (v79_g ? v124_ga : v126_ga));
        double v131_ga = tensorforge::gather<64>(v85_data, v118_src);
        double v133_ga = tensorforge::gather<64>(v87_data, v118_src);
        double v135_ga = tensorforge::gather<64>(v89_data, v118_src);
        double v137_ga = tensorforge::gather<64>(v91_data, v118_src);
        double v140_af = v83_g ? v131_ga : (v81_g ? v133_ga : (v79_g ? v135_ga : v137_ga));
        double v142_data = glb_m1[v141_a];
        tensorforge::VectorT<double, 4> v143_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v129_af, v142_data, v102_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v144_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v140_af, v142_data, v103_acc, 0, 0, 0);
        double v146_data = glb_m1[v145_a];
        tensorforge::VectorT<double, 4> v147_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v129_af, v146_data, v109_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v148_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v140_af, v146_data, v110_acc, 0, 0, 0);
        double v150_data = glb_m1[v149_a];
        tensorforge::VectorT<double, 4> v151_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v129_af, v150_data, v116_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v152_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v140_af, v150_data, v117_acc, 0, 0, 0);
        double v155_ga = tensorforge::gather<64>(v71_data, v153_src);
        double v157_ga = tensorforge::gather<64>(v73_data, v153_src);
        double v159_ga = tensorforge::gather<64>(v75_data, v153_src);
        double v161_ga = tensorforge::gather<64>(v77_data, v153_src);
        double v164_af = v83_g ? v155_ga : (v81_g ? v157_ga : (v79_g ? v159_ga : v161_ga));
        double v166_ga = tensorforge::gather<64>(v85_data, v153_src);
        double v168_ga = tensorforge::gather<64>(v87_data, v153_src);
        double v170_ga = tensorforge::gather<64>(v89_data, v153_src);
        double v172_ga = tensorforge::gather<64>(v91_data, v153_src);
        double v175_af = v83_g ? v166_ga : (v81_g ? v168_ga : (v79_g ? v170_ga : v172_ga));
        double v177_data = glb_m1[v176_a];
        tensorforge::VectorT<double, 4> v178_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v164_af, v177_data, v143_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v179_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v175_af, v177_data, v144_acc, 0, 0, 0);
        double v181_data = glb_m1[v180_a];
        tensorforge::VectorT<double, 4> v182_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v164_af, v181_data, v147_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v183_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v175_af, v181_data, v148_acc, 0, 0, 0);
        double v185_data = glb_m1[v184_a];
        tensorforge::VectorT<double, 4> v186_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v164_af, v185_data, v151_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v187_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v175_af, v185_data, v152_acc, 0, 0, 0);
        double v190_ga = tensorforge::gather<64>(v71_data, v188_src);
        double v192_ga = tensorforge::gather<64>(v73_data, v188_src);
        double v194_ga = tensorforge::gather<64>(v75_data, v188_src);
        double v196_ga = tensorforge::gather<64>(v77_data, v188_src);
        double v199_af = v83_g ? v190_ga : (v81_g ? v192_ga : (v79_g ? v194_ga : v196_ga));
        double v201_ga = tensorforge::gather<64>(v85_data, v188_src);
        double v203_ga = tensorforge::gather<64>(v87_data, v188_src);
        double v205_ga = tensorforge::gather<64>(v89_data, v188_src);
        double v207_ga = tensorforge::gather<64>(v91_data, v188_src);
        double v210_af = v83_g ? v201_ga : (v81_g ? v203_ga : (v79_g ? v205_ga : v207_ga));
        double v212_data = glb_m1[v211_a];
        tensorforge::VectorT<double, 4> v213_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v199_af, v212_data, v178_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v214_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v210_af, v212_data, v179_acc, 0, 0, 0);
        double v216_data = glb_m1[v215_a];
        tensorforge::VectorT<double, 4> v217_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v199_af, v216_data, v182_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v218_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v210_af, v216_data, v183_acc, 0, 0, 0);
        double v220_data = glb_m1[v219_a];
        tensorforge::VectorT<double, 4> v221_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v199_af, v220_data, v186_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v222_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v210_af, v220_data, v187_acc, 0, 0, 0);
        double v223_data = r0[1];
        double v224_ga = tensorforge::gather<64>(v223_data, v63_src);
        double v225_data = r0[3];
        double v226_ga = tensorforge::gather<64>(v225_data, v63_src);
        double v227_data = r0[5];
        double v228_ga = tensorforge::gather<64>(v227_data, v63_src);
        double v229_data = r0[7];
        double v230_ga = tensorforge::gather<64>(v229_data, v63_src);
        double v233_af = v83_g ? v224_ga : (v81_g ? v226_ga : (v79_g ? v228_ga : v230_ga));
        double v234_data = r0[9];
        double v235_ga = tensorforge::gather<64>(v234_data, v63_src);
        double v236_data = r0[11];
        double v237_ga = tensorforge::gather<64>(v236_data, v63_src);
        double v238_data = r0[13];
        double v239_ga = tensorforge::gather<64>(v238_data, v63_src);
        double v240_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v355_i0 = 0; v355_i0 < 1; ++v355_i0) {
          int32_t v357_lead = v37_lead + (v355_i0 * 16);
          #pragma unroll
          for (int32_t v358_i1 = 0; v358_i1 < 8; ++v358_i1) {
            double v361_data = __builtin_nontemporal_load(&wrap_glb_m2[(v357_lead + (v358_i1 * 24))]);
            r0[(v355_i0 + (v358_i1 * 2))] = v361_data;
          }
        }
        #pragma unroll
        for (int32_t v364_i1 = 0; v364_i1 < 8; ++v364_i1) {
          double v367_data = v52_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v49_lead + (v364_i1 * 24))])) : (0.0);
          r0[(1 + (v364_i1 * 2))] = v367_data;
        }
        double v241_ga = tensorforge::gather<64>(v240_data, v63_src);
        double v244_af = v83_g ? v235_ga : (v81_g ? v237_ga : (v79_g ? v239_ga : v241_ga));
        double v246_data = glb_m1[v245_a];
        tensorforge::VectorT<double, 4> v247_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v233_af, v246_data, v213_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v248_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v244_af, v246_data, v214_acc, 0, 0, 0);
        double v250_data = glb_m1[v249_a];
        tensorforge::VectorT<double, 4> v251_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v233_af, v250_data, v217_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v252_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v244_af, v250_data, v218_acc, 0, 0, 0);
        double v254_data = glb_m1[v253_a];
        tensorforge::VectorT<double, 4> v255_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v233_af, v254_data, v221_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v256_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v244_af, v254_data, v222_acc, 0, 0, 0);
        double v258_ga = tensorforge::gather<64>(v223_data, v118_src);
        double v260_ga = tensorforge::gather<64>(v225_data, v118_src);
        double v262_ga = tensorforge::gather<64>(v227_data, v118_src);
        double v264_ga = tensorforge::gather<64>(v229_data, v118_src);
        double v267_af = v83_g ? v258_ga : (v81_g ? v260_ga : (v79_g ? v262_ga : v264_ga));
        double v269_ga = tensorforge::gather<64>(v234_data, v118_src);
        double v271_ga = tensorforge::gather<64>(v236_data, v118_src);
        double v273_ga = tensorforge::gather<64>(v238_data, v118_src);
        double v275_ga = tensorforge::gather<64>(v240_data, v118_src);
        double v278_af = v83_g ? v269_ga : (v81_g ? v271_ga : (v79_g ? v273_ga : v275_ga));
        double v280_data = glb_m1[v279_a];
        tensorforge::VectorT<double, 4> v281_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v267_af, v280_data, v247_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v282_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v278_af, v280_data, v248_acc, 0, 0, 0);
        double v284_data = glb_m1[v283_a];
        tensorforge::VectorT<double, 4> v285_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v267_af, v284_data, v251_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v286_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v278_af, v284_data, v252_acc, 0, 0, 0);
        double v288_data = glb_m1[v287_a];
        tensorforge::VectorT<double, 4> v289_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v267_af, v288_data, v255_acc, 0, 0, 0);
        tensorforge::VectorT<double, 4> v290_acc = __builtin_amdgcn_mfma_f64_16x16x4f64(v278_af, v288_data, v256_acc, 0, 0, 0);
        ir1[0] = (v281_acc[0]);
        ir1[3] = (v281_acc[1]);
        ir1[6] = (v281_acc[2]);
        ir1[9] = (v281_acc[3]);
        ir1[12] = (v282_acc[0]);
        ir1[15] = (v282_acc[1]);
        ir1[18] = (v282_acc[2]);
        ir1[21] = (v282_acc[3]);
        ir1[1] = (v285_acc[0]);
        ir1[4] = (v285_acc[1]);
        ir1[7] = (v285_acc[2]);
        ir1[10] = (v285_acc[3]);
        ir1[13] = (v286_acc[0]);
        ir1[16] = (v286_acc[1]);
        ir1[19] = (v286_acc[2]);
        ir1[22] = (v286_acc[3]);
        ir1[2] = (v289_acc[0]);
        ir1[5] = (v289_acc[1]);
        ir1[8] = (v289_acc[2]);
        ir1[11] = (v289_acc[3]);
        ir1[14] = (v290_acc[0]);
        ir1[17] = (v290_acc[1]);
        ir1[20] = (v290_acc[2]);
        ir1[23] = (v290_acc[3]);
        // r1 = ir1
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v320_i0 = 0; v320_i0 < 3; ++v320_i0) {
          #pragma unroll
          for (int32_t v321_i1 = 0; v321_i1 < 8; ++v321_i1) {
            double v324_data = ir1[(v320_i0 + (v321_i1 * 3))];
            if (batchIdActive0) {
              glb_m0[((v37_lead + (v320_i0 * 16)) + (v321_i1 * 48))] = v324_data;
            }
          }
        }
      }
    }
  }
}

