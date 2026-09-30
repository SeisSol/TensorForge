// === base name ===
kernel_9c02697dd1a707af

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9c02697dd1a707af = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9c02697dd1a707af(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9c02697dd1a707af(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9c02697dd1a707af(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9c02697dd1a707af, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_9c02697dd1a707af, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (3328 * sizeof(float)));
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
  config.sharedMemBytes = 3328 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_9c02697dd1a707af(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9c02697dd1a707af(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9c02697dd1a707af), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_9c02697dd1a707af, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9c02697dd1a707af(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 6×12(6×12) {0..6}×{0..12} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 6×12(6×12) {0..6}×{0..12} strided
    //   m3 12×12(12×12) {0..12}×{0..12} strided
    //   m4 2×12(2×12) {0..2}×{0..12} strided
    //   m5 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = t0[i,j]
    //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
    //   m5[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v5_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v5_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 24 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v5_batchId0 * 144 + 0 + m5_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 6;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
              float v29_data = __builtin_nontemporal_load(&glb_m0[(v22_lead + (v24_i1 * 6))]);
              r0[v24_i1] = v29_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v32_g = v22_lead < 12;
          if (v32_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m1[(v22_lead + (v33_i1 * 12))]);
              r1[v33_i1] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v23_g) {
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
              float v46_data = __builtin_nontemporal_load(&glb_m2[(v22_lead + (v41_i1 * 6))]);
              r3[v41_i1] = v46_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v49_data = r1[0];
          float v50_data = r1[1];
          float v51_data = r1[2];
          float v52_data = r1[3];
          float v53_tp{};
          float v54_tp{};
          float v55_tp{};
          float v56_tp{};
          tensorforge::transpose4x4b32(v53_tp, v54_tp, v55_tp, v56_tp, v49_data, v50_data, v51_data, v52_data);
          tensorforge::VectorT<float, 4> v57_acc{};
          float v58_data = r0[0];
          float v59_data = r0[1];
          float v60_data = r0[2];
          float v61_data = r0[3];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v57_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v62_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v63_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 2, 0, 0);
          float v66_data = r0[4];
          float v67_data = r0[5];
          float v68_data = r0[6];
          float v69_data = r0[7];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v65_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v71_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 2, 1, 0);
          float v74_data = r0[8];
          float v75_data = r0[9];
          float v76_data = r0[10];
          float v77_data = r0[11];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v73_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v78_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v79_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v80_acc, 2, 2, 0);
          r2[0] = (v81_acc[0]);
          r2[1] = (v81_acc[1]);
          r2[2] = (v81_acc[2]);
          r2[3] = (v81_acc[3]);
          float v86_data = r1[4];
          float v87_data = r1[5];
          float v88_data = r1[6];
          float v89_data = r1[7];
          float v90_tp{};
          float v91_tp{};
          float v92_tp{};
          float v93_tp{};
          tensorforge::transpose4x4b32(v90_tp, v91_tp, v92_tp, v93_tp, v86_data, v87_data, v88_data, v89_data);
          tensorforge::VectorT<float, 4> v94_acc{};
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v94_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v100_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v102_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v107_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v108_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v110_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v115_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v116_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v117_acc, 2, 2, 0);
          r2[4] = (v118_acc[0]);
          r2[5] = (v118_acc[1]);
          r2[6] = (v118_acc[2]);
          r2[7] = (v118_acc[3]);
          float v123_data = r1[8];
          float v124_data = r1[9];
          float v125_data = r1[10];
          float v126_data = r1[11];
          float v127_tp{};
          float v128_tp{};
          float v129_tp{};
          float v130_tp{};
          tensorforge::transpose4x4b32(v127_tp, v128_tp, v129_tp, v130_tp, v123_data, v124_data, v125_data, v126_data);
          tensorforge::VectorT<float, 4> v131_acc{};
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v58_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v59_data, v136_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v60_data, v137_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v61_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v66_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v67_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v68_data, v145_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v69_data, v146_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v74_data, v147_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v75_data, v152_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v76_data, v153_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v77_data, v154_acc, 2, 2, 0);
          r2[8] = (v155_acc[0]);
          r2[9] = (v155_acc[1]);
          r2[10] = (v155_acc[2]);
          r2[11] = (v155_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v23_g) {
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 12; ++v160_i1) {
              float v162_data = r2[v160_i1];
              int32_t v166_a = v22_lead + (v160_i1 * 12);
              s0[(v166_a ^ ((v166_a >> 4) & 15))] = v162_data;
            }
          }
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          bool v171_g = v22_lead < 2;
          if (v171_g) {
            #pragma unroll
            for (int32_t v172_i1 = 0; v172_i1 < 12; ++v172_i1) {
              float v177_data = __builtin_nontemporal_load(&glb_m4[(v22_lead + (v172_i1 * 2))]);
              r6[v172_i1] = v177_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          tensorforge::transpose4x4b32(v184_tp, v185_tp, v186_tp, v187_tp, v49_data, v50_data, v51_data, v52_data);
          tensorforge::VectorT<float, 4> v188_acc{};
          float v189_data = r3[0];
          float v190_data = r3[1];
          float v191_data = r3[2];
          float v192_data = r3[3];
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v189_data, v188_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v190_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v191_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v192_data, v195_acc, 2, 0, 0);
          float v197_data = r3[4];
          float v198_data = r3[5];
          float v199_data = r3[6];
          float v200_data = r3[7];
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v197_data, v196_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v198_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v199_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v200_data, v203_acc, 2, 1, 0);
          float v205_data = r3[8];
          float v206_data = r3[9];
          float v207_data = r3[10];
          float v208_data = r3[11];
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v205_data, v204_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v206_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v207_data, v210_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v208_data, v211_acc, 2, 2, 0);
          r4[0] = (v212_acc[0]);
          r4[1] = (v212_acc[1]);
          r4[2] = (v212_acc[2]);
          r4[3] = (v212_acc[3]);
          float v221_tp{};
          float v222_tp{};
          float v223_tp{};
          float v224_tp{};
          tensorforge::transpose4x4b32(v221_tp, v222_tp, v223_tp, v224_tp, v86_data, v87_data, v88_data, v89_data);
          tensorforge::VectorT<float, 4> v225_acc{};
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v189_data, v225_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v190_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v191_data, v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v192_data, v232_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v197_data, v233_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v198_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v199_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v200_data, v240_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v205_data, v241_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v206_data, v246_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v207_data, v247_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v208_data, v248_acc, 2, 2, 0);
          r4[4] = (v249_acc[0]);
          r4[5] = (v249_acc[1]);
          r4[6] = (v249_acc[2]);
          r4[7] = (v249_acc[3]);
          float v258_tp{};
          float v259_tp{};
          float v260_tp{};
          float v261_tp{};
          tensorforge::transpose4x4b32(v258_tp, v259_tp, v260_tp, v261_tp, v123_data, v124_data, v125_data, v126_data);
          tensorforge::VectorT<float, 4> v262_acc{};
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v189_data, v262_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v190_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v191_data, v268_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v192_data, v269_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v197_data, v270_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v198_data, v275_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v199_data, v276_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v200_data, v277_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v205_data, v278_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v206_data, v283_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v207_data, v284_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v208_data, v285_acc, 2, 2, 0);
          r4[8] = (v286_acc[0]);
          r4[9] = (v286_acc[1]);
          r4[10] = (v286_acc[2]);
          r4[11] = (v286_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v23_g) {
            int32_t v296_off = v22_lead + 6;
            #pragma unroll
            for (int32_t v291_i1 = 0; v291_i1 < 12; ++v291_i1) {
              float v293_data = r4[v291_i1];
              int32_t v298_a = v296_off + (v291_i1 * 12);
              s0[(v298_a ^ ((v298_a >> 4) & 15))] = v293_data;
            }
          }
          float r5[12]{};
          // r5 = +(s0) + None
          // [(0, 12), (0, 12)] []
          int32_t v307_sw = (v22_lead >> 4) & 15;
          float v309_data = v32_g ? (s0[(v22_lead ^ v307_sw)]) : (0.0f);
          float v310_data = r5[0];
          r5[0] = (v310_data + v309_data);
          int32_t v312_a = v22_lead + 12;
          int32_t v313_sw = v312_a >> 4;
          float v316_data = v32_g ? (s0[(v312_a ^ (v313_sw & 15))]) : (0.0f);
          float v317_data = r5[1];
          r5[1] = (v317_data + v316_data);
          int32_t v319_a = v22_lead + 24;
          int32_t v320_sw = v319_a >> 4;
          float v323_data = v32_g ? (s0[(v319_a ^ (v320_sw & 15))]) : (0.0f);
          float v324_data = r5[2];
          r5[2] = (v324_data + v323_data);
          int32_t v326_a = v22_lead + 36;
          int32_t v327_sw = v326_a >> 4;
          float v330_data = v32_g ? (s0[(v326_a ^ (v327_sw & 15))]) : (0.0f);
          float v331_data = r5[3];
          r5[3] = (v331_data + v330_data);
          int32_t v333_a = v22_lead + 48;
          int32_t v334_sw = v333_a >> 4;
          float v337_data = v32_g ? (s0[(v333_a ^ (v334_sw & 15))]) : (0.0f);
          float v338_data = r5[4];
          r5[4] = (v338_data + v337_data);
          int32_t v340_a = v22_lead + 60;
          int32_t v341_sw = v340_a >> 4;
          float v344_data = v32_g ? (s0[(v340_a ^ (v341_sw & 15))]) : (0.0f);
          float v345_data = r5[5];
          r5[5] = (v345_data + v344_data);
          int32_t v347_a = v22_lead + 72;
          int32_t v348_sw = v347_a >> 4;
          float v351_data = v32_g ? (s0[(v347_a ^ (v348_sw & 15))]) : (0.0f);
          float v352_data = r5[6];
          r5[6] = (v352_data + v351_data);
          int32_t v354_a = v22_lead + 84;
          int32_t v355_sw = v354_a >> 4;
          float v358_data = v32_g ? (s0[(v354_a ^ (v355_sw & 15))]) : (0.0f);
          float v359_data = r5[7];
          r5[7] = (v359_data + v358_data);
          int32_t v361_a = v22_lead + 96;
          int32_t v362_sw = v361_a >> 4;
          float v365_data = v32_g ? (s0[(v361_a ^ (v362_sw & 15))]) : (0.0f);
          float v366_data = r5[8];
          r5[8] = (v366_data + v365_data);
          int32_t v368_a = v22_lead + 108;
          int32_t v369_sw = v368_a >> 4;
          float v372_data = v32_g ? (s0[(v368_a ^ (v369_sw & 15))]) : (0.0f);
          float v373_data = r5[9];
          r5[9] = (v373_data + v372_data);
          int32_t v375_a = v22_lead + 120;
          int32_t v376_sw = v375_a >> 4;
          float v379_data = v32_g ? (s0[(v375_a ^ (v376_sw & 15))]) : (0.0f);
          float v380_data = r5[10];
          r5[10] = (v380_data + v379_data);
          int32_t v382_a = v22_lead + 132;
          int32_t v383_sw = v382_a >> 4;
          float v386_data = v32_g ? (s0[(v382_a ^ (v383_sw & 15))]) : (0.0f);
          float v387_data = r5[11];
          r5[11] = (v387_data + v386_data);
          // glb_m3 = store{r>g}(r5);
          if (v32_g) {
            #pragma unroll
            for (int32_t v389_i1 = 0; v389_i1 < 12; ++v389_i1) {
              float v391_data = r5[v389_i1];
              glb_m3[(v22_lead + (v389_i1 * 12))] = v391_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[12]{};
          // r7 = +(r6 * r1) + None
          // [(0, 2), (0, 12)] [(0, 12)]
          float v401_tp{};
          float v402_tp{};
          float v403_tp{};
          float v404_tp{};
          tensorforge::transpose4x4b32(v401_tp, v402_tp, v403_tp, v404_tp, v49_data, v50_data, v51_data, v52_data);
          tensorforge::VectorT<float, 4> v405_acc{};
          float v406_data = r6[0];
          float v407_data = r6[1];
          float v408_data = r6[2];
          float v409_data = r6[3];
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v406_data, v405_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v407_data, v410_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v408_data, v411_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v409_data, v412_acc, 2, 0, 0);
          float v414_data = r6[4];
          float v415_data = r6[5];
          float v416_data = r6[6];
          float v417_data = r6[7];
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v414_data, v413_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v415_data, v418_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v416_data, v419_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v417_data, v420_acc, 2, 1, 0);
          float v422_data = r6[8];
          float v423_data = r6[9];
          float v424_data = r6[10];
          float v425_data = r6[11];
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v422_data, v421_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v423_data, v426_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v424_data, v427_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v425_data, v428_acc, 2, 2, 0);
          r7[0] = (v429_acc[0]);
          r7[1] = (v429_acc[1]);
          r7[2] = (v429_acc[2]);
          r7[3] = (v429_acc[3]);
          float v438_tp{};
          float v439_tp{};
          float v440_tp{};
          float v441_tp{};
          tensorforge::transpose4x4b32(v438_tp, v439_tp, v440_tp, v441_tp, v86_data, v87_data, v88_data, v89_data);
          tensorforge::VectorT<float, 4> v442_acc{};
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v406_data, v442_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v407_data, v447_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v408_data, v448_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v409_data, v449_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v414_data, v450_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v415_data, v455_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v416_data, v456_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v417_data, v457_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v422_data, v458_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v423_data, v463_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v424_data, v464_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v425_data, v465_acc, 2, 2, 0);
          r7[4] = (v466_acc[0]);
          r7[5] = (v466_acc[1]);
          r7[6] = (v466_acc[2]);
          r7[7] = (v466_acc[3]);
          float v475_tp{};
          float v476_tp{};
          float v477_tp{};
          float v478_tp{};
          tensorforge::transpose4x4b32(v475_tp, v476_tp, v477_tp, v478_tp, v123_data, v124_data, v125_data, v126_data);
          tensorforge::VectorT<float, 4> v479_acc{};
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v406_data, v479_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v407_data, v484_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v408_data, v485_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v409_data, v486_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v414_data, v487_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v415_data, v492_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v416_data, v493_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v417_data, v494_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v422_data, v495_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v476_tp, v423_data, v500_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v424_data, v501_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v425_data, v502_acc, 2, 2, 0);
          r7[8] = (v503_acc[0]);
          r7[9] = (v503_acc[1]);
          r7[10] = (v503_acc[2]);
          r7[11] = (v503_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r7);
          if ((v22_lead >= 8) && v32_g) {
            #pragma unroll
            for (int32_t v510_z1 = 0; v510_z1 < 12; ++v510_z1) {
              int32_t v515_a = v22_lead + (v510_z1 * 12);
              s0[(v515_a ^ ((v515_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v171_g) {
            int32_t v524_off = v22_lead + 6;
            #pragma unroll
            for (int32_t v519_i1 = 0; v519_i1 < 12; ++v519_i1) {
              float v521_data = r7[v519_i1];
              int32_t v526_a = v524_off + (v519_i1 * 12);
              s0[(v526_a ^ ((v526_a >> 4) & 15))] = v521_data;
            }
          }
          float r8[12]{};
          // r8 = +(s0) + None
          // [(0, 12), (0, 12)] []
          float v537_data = v32_g ? (s0[(v22_lead ^ v307_sw)]) : (0.0f);
          float v538_data = r8[0];
          r8[0] = (v538_data + v537_data);
          float v544_data = v32_g ? (s0[(v312_a ^ (v313_sw & 15))]) : (0.0f);
          float v545_data = r8[1];
          r8[1] = (v545_data + v544_data);
          float v551_data = v32_g ? (s0[(v319_a ^ (v320_sw & 15))]) : (0.0f);
          float v552_data = r8[2];
          r8[2] = (v552_data + v551_data);
          float v558_data = v32_g ? (s0[(v326_a ^ (v327_sw & 15))]) : (0.0f);
          float v559_data = r8[3];
          r8[3] = (v559_data + v558_data);
          float v565_data = v32_g ? (s0[(v333_a ^ (v334_sw & 15))]) : (0.0f);
          float v566_data = r8[4];
          r8[4] = (v566_data + v565_data);
          float v572_data = v32_g ? (s0[(v340_a ^ (v341_sw & 15))]) : (0.0f);
          float v573_data = r8[5];
          r8[5] = (v573_data + v572_data);
          float v579_data = v32_g ? (s0[(v347_a ^ (v348_sw & 15))]) : (0.0f);
          float v580_data = r8[6];
          r8[6] = (v580_data + v579_data);
          float v586_data = v32_g ? (s0[(v354_a ^ (v355_sw & 15))]) : (0.0f);
          float v587_data = r8[7];
          r8[7] = (v587_data + v586_data);
          float v593_data = v32_g ? (s0[(v361_a ^ (v362_sw & 15))]) : (0.0f);
          float v594_data = r8[8];
          r8[8] = (v594_data + v593_data);
          float v600_data = v32_g ? (s0[(v368_a ^ (v369_sw & 15))]) : (0.0f);
          float v601_data = r8[9];
          r8[9] = (v601_data + v600_data);
          float v607_data = v32_g ? (s0[(v375_a ^ (v376_sw & 15))]) : (0.0f);
          float v608_data = r8[10];
          r8[10] = (v608_data + v607_data);
          float v614_data = v32_g ? (s0[(v382_a ^ (v383_sw & 15))]) : (0.0f);
          float v615_data = r8[11];
          r8[11] = (v615_data + v614_data);
          // glb_m5 = store{r>g}(r8);
          if (v32_g) {
            #pragma unroll
            for (int32_t v617_i1 = 0; v617_i1 < 12; ++v617_i1) {
              float v619_data = r8[v617_i1];
              glb_m5[(v22_lead + (v617_i1 * 12))] = v619_data;
            }
          }
        }
      }
    }
  }
}

