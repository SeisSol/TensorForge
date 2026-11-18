// === base name ===
kernel_baab1af32f4f8956

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_baab1af32f4f8956 = {{16, 16, 1}, 16, 9, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_baab1af32f4f8956(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_baab1af32f4f8956(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_baab1af32f4f8956(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_baab1af32f4f8956, block.x * block.y * block.z, 256 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_baab1af32f4f8956, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(double)));
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
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_baab1af32f4f8956(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_baab1af32f4f8956(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_baab1af32f4f8956), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_baab1af32f4f8956, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_baab1af32f4f8956(tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const double, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (9 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 9×9(9×9) {0..9}×{0..9} indexed
    //   m1 9×9(9×9) {0..9}×{0..9} indexed(offset=8)
    //   m2 9×9(9×9) {0..9}×{0..9} indexed(stride=96)
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":9,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(offset=8)","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(stride=96)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"indexed(offset=8)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"indexed(stride=96)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      double r0[9]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v81_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v81_flagSecondAhead < numElements0) ? v81_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
      int32_t v91_lead = threadIdx.x % 16;
      bool v92_g = v91_lead < 9;
      bool v109_g = allowed_peel && v92_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v101_i1 = 0; v101_i1 < 9; ++v101_i1) {
        double v104_data = v109_g ? (__builtin_nontemporal_load(&peel_glb_m1[(v91_lead + (v101_i1 * 9))])) : (0.0);
        r0[v101_i1] = v104_data;
      }
      uint32_t v85_flagWord = flagWordFirst;
      uint32_t v86_flagWordNext = flagWordSecond;
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[static_cast<size_t>(elementIndices[v10_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
        size_t v114_flagAheadAhead = v10_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v114_flagAheadAhead < numElements0) ? v114_flagAheadAhead : v10_batchId1)];
        const bool allowed = static_cast<bool>(v85_flagWord);
        const bool allowed_next = static_cast<bool>(v86_flagWordNext);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[static_cast<size_t>(elementIndices[v7_batchId0]) * 81 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[v7_batchId0]) * 96 + 0 + m2_extraOffset];
          double r1[9]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
            double v37_data = v92_g ? (__builtin_nontemporal_load(&glb_m2[(v91_lead + (v31_i1 * 9))])) : (0.0);
            r1[v31_i1] = v37_data;
          }
          double r2[9]{};
          // ir2 = +(r0 * r1)
          // [(0, 9), (0, 9)] [(0, 9)]
          double ir2[9]{};
          double v41_data = r0[0];
          double v42_data = r0[1];
          double v43_data = r0[2];
          double v44_data = r0[3];
          double v45_data = r0[4];
          double v46_data = r0[5];
          double v47_data = r0[6];
          double v48_data = r0[7];
          double v49_data = r0[8];
          bool v110_g = allowed_next && v92_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v105_i1 = 0; v105_i1 < 9; ++v105_i1) {
            double v108_data = v110_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v91_lead + (v105_i1 * 9))])) : (0.0);
            r0[v105_i1] = v108_data;
          }
          double v50_acc{};
          double v51_acc{};
          double v52_acc{};
          double v53_acc{};
          double v54_acc{};
          double v55_acc{};
          double v56_acc{};
          double v57_acc{};
          double v58_acc{};
          double v59_data = r1[0];
          double v60_data = r1[1];
          double v61_data = r1[2];
          double v62_data = r1[3];
          double v63_data = r1[4];
          double v64_data = r1[5];
          double v65_data = r1[6];
          double v66_data = r1[7];
          double v67_data = r1[8];
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
          : "+&v"(v50_acc), "+&v"(v51_acc), "+&v"(v52_acc), "+&v"(v53_acc)
          : "v"(v59_data), "v"(v41_data), "v"(v42_data), "v"(v43_data), "v"(v44_data), "v"(v45_data), "v"(v46_data), "v"(v47_data), "v"(v48_data), "v"(v49_data), "v"(v60_data), "v"(v61_data), "v"(v62_data)
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
          : "+&v"(v53_acc), "+&v"(v54_acc), "+&v"(v55_acc), "+&v"(v56_acc), "+&v"(v57_acc)
          : "v"(v62_data), "v"(v46_data), "v"(v47_data), "v"(v48_data), "v"(v49_data), "v"(v63_data), "v"(v41_data), "v"(v42_data), "v"(v43_data), "v"(v44_data), "v"(v45_data), "v"(v64_data), "v"(v65_data), "v"(v66_data)
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
          : "+&v"(v57_acc), "+&v"(v58_acc)
          : "v"(v66_data), "v"(v42_data), "v"(v43_data), "v"(v44_data), "v"(v45_data), "v"(v46_data), "v"(v47_data), "v"(v48_data), "v"(v49_data), "v"(v67_data), "v"(v41_data)
          );
          ir2[0] = v50_acc;
          ir2[1] = v51_acc;
          ir2[2] = v52_acc;
          ir2[3] = v53_acc;
          ir2[4] = v54_acc;
          ir2[5] = v55_acc;
          ir2[6] = v56_acc;
          ir2[7] = v57_acc;
          ir2[8] = v58_acc;
          // r2 = ir2
          // glb_m0 = store{r>g}(r2);
          if (v92_g) {
            #pragma unroll
            for (int32_t v73_i1 = 0; v73_i1 < 9; ++v73_i1) {
              double v75_data = ir2[v73_i1];
              glb_m0[(v91_lead + (v73_i1 * 9))] = v75_data;
            }
          }
        }
        else {
          bool v111_g = allowed_next && v92_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v97_i1 = 0; v97_i1 < 9; ++v97_i1) {
            double v100_data = v111_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v91_lead + (v97_i1 * 9))])) : (0.0);
            r0[v97_i1] = v100_data;
          }
        }
        v85_flagWord = v86_flagWordNext;
        v86_flagWordNext = flagWordAhead;
      }
    }
  }
}

