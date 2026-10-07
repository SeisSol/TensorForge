// === base name ===
kernel_c514bdb803598f86

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c514bdb803598f86 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c514bdb803598f86(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c514bdb803598f86(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c514bdb803598f86(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c514bdb803598f86, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c514bdb803598f86, block.x * block.y * block.z, 0));
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
void launcher_kernel_c514bdb803598f86(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c514bdb803598f86(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c514bdb803598f86), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_c514bdb803598f86, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c514bdb803598f86(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 32×32(6×12) {0..6}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(6×12) {0..6}×{0..12} strided
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    //   m4 32×32(12×12) {0..12}×{0..12} strided
    //   m5 32×32(6×12) {0..6}×{0..12} strided
    //   m6 32×32(12×12) {0..12}×{0..12} strided
    //   m7 32×32(6×12) {0..6}×{0..12} strided
    //   m8 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = t0[i,k] × m4[k,j]
    //   t1[i,j]@{0..6}×{0..12} = m5[i,k] × m6[k,j]
    //   t1[i,j]@{6..12}×{0..12} = m7[i,k] × m6[k,j]
    //   m3[i,j] += t1[i,k] × m8[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"F1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"G","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"H","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m6","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m7","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m8","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v9_batchId0 * 144 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v9_batchId0 * 72 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v9_batchId0 * 144 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v9_batchId0 * 72 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v9_batchId0 * 144 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v29_lead = threadIdx.x % 16;
          bool v30_g = v29_lead < 6;
          if (v30_g) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m0[(v29_lead + (v31_i1 * 6))]);
              r0[v31_i1] = v36_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v39_g = v29_lead < 12;
          if (v39_g) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v40_i1 * 12))]);
              r1[v40_i1] = v45_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v30_g) {
            #pragma unroll
            for (int32_t v170_i1 = 0; v170_i1 < 12; ++v170_i1) {
              float v175_data = __builtin_nontemporal_load(&glb_m2[(v29_lead + (v170_i1 * 6))]);
              r3[v170_i1] = v175_data;
            }
          }
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v48_data = r1[0];
          float v49_data = r1[1];
          float v50_data = r1[2];
          float v51_data = r1[3];
          float v52_tp{};
          float v53_tp{};
          float v54_tp{};
          float v55_tp{};
          tensorforge::transpose4x4b32(v52_tp, v53_tp, v54_tp, v55_tp, v48_data, v49_data, v50_data, v51_data);
          tensorforge::VectorT<float, 4> v56_acc{};
          float v57_data = r0[0];
          float v58_data = r0[1];
          float v59_data = r0[2];
          float v60_data = r0[3];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v62_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v63_acc, 2, 0, 0);
          float v65_data = r0[4];
          float v66_data = r0[5];
          float v67_data = r0[6];
          float v68_data = r0[7];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v71_acc, 2, 1, 0);
          float v73_data = r0[8];
          float v74_data = r0[9];
          float v75_data = r0[10];
          float v76_data = r0[11];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v78_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v79_acc, 2, 2, 0);
          r2[0] = (v80_acc[0]);
          r2[1] = (v80_acc[1]);
          r2[2] = (v80_acc[2]);
          r2[3] = (v80_acc[3]);
          float v85_data = r1[4];
          float v86_data = r1[5];
          float v87_data = r1[6];
          float v88_data = r1[7];
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          float v92_tp{};
          tensorforge::transpose4x4b32(v89_tp, v90_tp, v91_tp, v92_tp, v85_data, v86_data, v87_data, v88_data);
          tensorforge::VectorT<float, 4> v93_acc{};
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v100_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v107_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v108_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v109_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v115_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v116_acc, 2, 2, 0);
          r2[4] = (v117_acc[0]);
          r2[5] = (v117_acc[1]);
          r2[6] = (v117_acc[2]);
          r2[7] = (v117_acc[3]);
          float v122_data = r1[8];
          float v123_data = r1[9];
          float v124_data = r1[10];
          float v125_data = r1[11];
          float v126_tp{};
          float v127_tp{};
          float v128_tp{};
          float v129_tp{};
          tensorforge::transpose4x4b32(v126_tp, v127_tp, v128_tp, v129_tp, v122_data, v123_data, v124_data, v125_data);
          tensorforge::VectorT<float, 4> v130_acc{};
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v57_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v58_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v59_data, v136_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v60_data, v137_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v65_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v66_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v67_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v68_data, v145_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v73_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v74_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v75_data, v152_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v76_data, v153_acc, 2, 2, 0);
          r2[8] = (v154_acc[0]);
          r2[9] = (v154_acc[1]);
          r2[10] = (v154_acc[2]);
          r2[11] = (v154_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v30_g) {
            #pragma unroll
            for (int32_t v159_i1 = 0; v159_i1 < 12; ++v159_i1) {
              float v161_data = r2[v159_i1];
              int32_t v165_a = v29_lead + (v159_i1 * 12);
              s0[(v165_a ^ ((v165_a >> 4) & 15))] = v161_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v39_g) {
            #pragma unroll
            for (int32_t v301_i1 = 0; v301_i1 < 12; ++v301_i1) {
              float v306_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v301_i1 * 12))]);
              r5[v301_i1] = v306_data;
            }
          }
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v182_tp{};
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          tensorforge::transpose4x4b32(v182_tp, v183_tp, v184_tp, v185_tp, v48_data, v49_data, v50_data, v51_data);
          tensorforge::VectorT<float, 4> v186_acc{};
          float v187_data = r3[0];
          float v188_data = r3[1];
          float v189_data = r3[2];
          float v190_data = r3[3];
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v187_data, v186_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v188_data, v191_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v189_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v190_data, v193_acc, 2, 0, 0);
          float v195_data = r3[4];
          float v196_data = r3[5];
          float v197_data = r3[6];
          float v198_data = r3[7];
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v195_data, v194_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v196_data, v199_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v197_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v198_data, v201_acc, 2, 1, 0);
          float v203_data = r3[8];
          float v204_data = r3[9];
          float v205_data = r3[10];
          float v206_data = r3[11];
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v203_data, v202_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v204_data, v207_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v205_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v206_data, v209_acc, 2, 2, 0);
          r4[0] = (v210_acc[0]);
          r4[1] = (v210_acc[1]);
          r4[2] = (v210_acc[2]);
          r4[3] = (v210_acc[3]);
          float v219_tp{};
          float v220_tp{};
          float v221_tp{};
          float v222_tp{};
          tensorforge::transpose4x4b32(v219_tp, v220_tp, v221_tp, v222_tp, v85_data, v86_data, v87_data, v88_data);
          tensorforge::VectorT<float, 4> v223_acc{};
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v187_data, v223_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v188_data, v228_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v189_data, v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v190_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v195_data, v231_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v196_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v197_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v198_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v203_data, v239_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v204_data, v244_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v205_data, v245_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v206_data, v246_acc, 2, 2, 0);
          r4[4] = (v247_acc[0]);
          r4[5] = (v247_acc[1]);
          r4[6] = (v247_acc[2]);
          r4[7] = (v247_acc[3]);
          float v256_tp{};
          float v257_tp{};
          float v258_tp{};
          float v259_tp{};
          tensorforge::transpose4x4b32(v256_tp, v257_tp, v258_tp, v259_tp, v122_data, v123_data, v124_data, v125_data);
          tensorforge::VectorT<float, 4> v260_acc{};
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v187_data, v260_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v188_data, v265_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v189_data, v266_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v190_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v195_data, v268_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v196_data, v273_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v197_data, v274_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v198_data, v275_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v203_data, v276_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v204_data, v281_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v205_data, v282_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v206_data, v283_acc, 2, 2, 0);
          r4[8] = (v284_acc[0]);
          r4[9] = (v284_acc[1]);
          r4[10] = (v284_acc[2]);
          r4[11] = (v284_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v30_g) {
            int32_t v294_off = v29_lead + 6;
            #pragma unroll
            for (int32_t v289_i1 = 0; v289_i1 < 12; ++v289_i1) {
              float v291_data = r4[v289_i1];
              int32_t v296_a = v294_off + (v289_i1 * 12);
              s0[(v296_a ^ ((v296_a >> 4) & 15))] = v291_data;
            }
          }
          float r7[12]{};
          // r7 = load{g>r}(glb_m5);
          if (v30_g) {
            #pragma unroll
            for (int32_t v571_i1 = 0; v571_i1 < 12; ++v571_i1) {
              float v576_data = __builtin_nontemporal_load(&glb_m5[(v29_lead + (v571_i1 * 6))]);
              r7[v571_i1] = v576_data;
            }
          }
          float r6[12]{};
          // r6 = +(s0 * r5) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v309_data = r5[0];
          float v310_data = r5[1];
          float v311_data = r5[2];
          float v312_data = r5[3];
          float v313_tp{};
          float v314_tp{};
          float v315_tp{};
          float v316_tp{};
          tensorforge::transpose4x4b32(v313_tp, v314_tp, v315_tp, v316_tp, v309_data, v310_data, v311_data, v312_data);
          tensorforge::VectorT<float, 4> v317_acc{};
          int32_t v322_sw = (v29_lead >> 4) & 15;
          int32_t v323_sw = v29_lead ^ v322_sw;
          float v324_data = s0[v323_sw];
          int32_t v325_a = v29_lead + 12;
          int32_t v326_sw = v325_a >> 4;
          float v329_data = s0[(v325_a ^ (v326_sw & 15))];
          int32_t v330_a = v29_lead + 24;
          int32_t v331_sw = v330_a >> 4;
          float v334_data = s0[(v330_a ^ (v331_sw & 15))];
          int32_t v335_a = v29_lead + 36;
          int32_t v336_sw = v335_a >> 4;
          float v339_data = s0[(v335_a ^ (v336_sw & 15))];
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v324_data, v317_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v329_data, v340_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v334_data, v341_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v339_data, v342_acc, 2, 0, 0);
          int32_t v344_a = v29_lead + 48;
          int32_t v345_sw = v344_a >> 4;
          float v348_data = s0[(v344_a ^ (v345_sw & 15))];
          int32_t v349_a = v29_lead + 60;
          int32_t v350_sw = v349_a >> 4;
          float v353_data = s0[(v349_a ^ (v350_sw & 15))];
          int32_t v354_a = v29_lead + 72;
          int32_t v355_sw = v354_a >> 4;
          float v358_data = s0[(v354_a ^ (v355_sw & 15))];
          int32_t v359_a = v29_lead + 84;
          int32_t v360_sw = v359_a >> 4;
          float v363_data = s0[(v359_a ^ (v360_sw & 15))];
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v348_data, v343_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v353_data, v364_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v358_data, v365_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v363_data, v366_acc, 2, 1, 0);
          int32_t v368_a = v29_lead + 96;
          int32_t v369_sw = v368_a >> 4;
          float v372_data = s0[(v368_a ^ (v369_sw & 15))];
          int32_t v373_a = v29_lead + 108;
          int32_t v374_sw = v373_a >> 4;
          float v377_data = s0[(v373_a ^ (v374_sw & 15))];
          int32_t v378_a = v29_lead + 120;
          int32_t v379_sw = v378_a >> 4;
          float v382_data = s0[(v378_a ^ (v379_sw & 15))];
          int32_t v383_a = v29_lead + 132;
          int32_t v384_sw = v383_a >> 4;
          float v387_data = s0[(v383_a ^ (v384_sw & 15))];
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v372_data, v367_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v377_data, v388_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v382_data, v389_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v387_data, v390_acc, 2, 2, 0);
          r6[0] = (v391_acc[0]);
          r6[1] = (v391_acc[1]);
          r6[2] = (v391_acc[2]);
          r6[3] = (v391_acc[3]);
          float v396_data = r5[4];
          float v397_data = r5[5];
          float v398_data = r5[6];
          float v399_data = r5[7];
          float v400_tp{};
          float v401_tp{};
          float v402_tp{};
          float v403_tp{};
          tensorforge::transpose4x4b32(v400_tp, v401_tp, v402_tp, v403_tp, v396_data, v397_data, v398_data, v399_data);
          tensorforge::VectorT<float, 4> v404_acc{};
          float v411_data = s0[(v29_lead ^ v322_sw)];
          float v416_data = s0[(v325_a ^ (v326_sw & 15))];
          float v421_data = s0[(v330_a ^ (v331_sw & 15))];
          float v426_data = s0[(v335_a ^ (v336_sw & 15))];
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v411_data, v404_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v416_data, v427_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v421_data, v428_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v426_data, v429_acc, 2, 0, 0);
          float v435_data = s0[(v344_a ^ (v345_sw & 15))];
          float v440_data = s0[(v349_a ^ (v350_sw & 15))];
          float v445_data = s0[(v354_a ^ (v355_sw & 15))];
          float v450_data = s0[(v359_a ^ (v360_sw & 15))];
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v435_data, v430_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v440_data, v451_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v445_data, v452_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v450_data, v453_acc, 2, 1, 0);
          float v459_data = s0[(v368_a ^ (v369_sw & 15))];
          float v464_data = s0[(v373_a ^ (v374_sw & 15))];
          float v469_data = s0[(v378_a ^ (v379_sw & 15))];
          float v474_data = s0[(v383_a ^ (v384_sw & 15))];
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v459_data, v454_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v464_data, v475_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v469_data, v476_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v474_data, v477_acc, 2, 2, 0);
          r6[4] = (v478_acc[0]);
          r6[5] = (v478_acc[1]);
          r6[6] = (v478_acc[2]);
          r6[7] = (v478_acc[3]);
          float v483_data = r5[8];
          float v484_data = r5[9];
          float v485_data = r5[10];
          float v486_data = r5[11];
          float v487_tp{};
          float v488_tp{};
          float v489_tp{};
          float v490_tp{};
          tensorforge::transpose4x4b32(v487_tp, v488_tp, v489_tp, v490_tp, v483_data, v484_data, v485_data, v486_data);
          tensorforge::VectorT<float, 4> v491_acc{};
          float v498_data = s0[(v29_lead ^ v322_sw)];
          float v503_data = s0[(v325_a ^ (v326_sw & 15))];
          float v508_data = s0[(v330_a ^ (v331_sw & 15))];
          float v513_data = s0[(v335_a ^ (v336_sw & 15))];
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v498_data, v491_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v503_data, v514_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v508_data, v515_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v513_data, v516_acc, 2, 0, 0);
          float v522_data = s0[(v344_a ^ (v345_sw & 15))];
          float v527_data = s0[(v349_a ^ (v350_sw & 15))];
          float v532_data = s0[(v354_a ^ (v355_sw & 15))];
          float v537_data = s0[(v359_a ^ (v360_sw & 15))];
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v522_data, v517_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v527_data, v538_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v532_data, v539_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v537_data, v540_acc, 2, 1, 0);
          float v546_data = s0[(v368_a ^ (v369_sw & 15))];
          float v551_data = s0[(v373_a ^ (v374_sw & 15))];
          float v556_data = s0[(v378_a ^ (v379_sw & 15))];
          float v561_data = s0[(v383_a ^ (v384_sw & 15))];
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v546_data, v541_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v551_data, v562_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v556_data, v563_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v490_tp, v561_data, v564_acc, 2, 2, 0);
          r6[8] = (v565_acc[0]);
          r6[9] = (v565_acc[1]);
          r6[10] = (v565_acc[2]);
          r6[11] = (v565_acc[3]);
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v39_g) {
            #pragma unroll
            for (int32_t v579_i1 = 0; v579_i1 < 12; ++v579_i1) {
              float v584_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v579_i1 * 12))]);
              r8[v579_i1] = v584_data;
            }
          }
          float r10[12]{};
          // r10 = load{g>r}(glb_m7);
          if (v30_g) {
            #pragma unroll
            for (int32_t v709_i1 = 0; v709_i1 < 12; ++v709_i1) {
              float v714_data = __builtin_nontemporal_load(&glb_m7[(v29_lead + (v709_i1 * 6))]);
              r10[v709_i1] = v714_data;
            }
          }
          float r9[12]{};
          // r9 = +(r7 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v587_data = r8[0];
          float v588_data = r8[1];
          float v589_data = r8[2];
          float v590_data = r8[3];
          float v591_tp{};
          float v592_tp{};
          float v593_tp{};
          float v594_tp{};
          tensorforge::transpose4x4b32(v591_tp, v592_tp, v593_tp, v594_tp, v587_data, v588_data, v589_data, v590_data);
          tensorforge::VectorT<float, 4> v595_acc{};
          float v596_data = r7[0];
          float v597_data = r7[1];
          float v598_data = r7[2];
          float v599_data = r7[3];
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v591_tp, v596_data, v595_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v592_tp, v597_data, v600_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v593_tp, v598_data, v601_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v594_tp, v599_data, v602_acc, 2, 0, 0);
          float v604_data = r7[4];
          float v605_data = r7[5];
          float v606_data = r7[6];
          float v607_data = r7[7];
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v591_tp, v604_data, v603_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v592_tp, v605_data, v608_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v593_tp, v606_data, v609_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v594_tp, v607_data, v610_acc, 2, 1, 0);
          float v612_data = r7[8];
          float v613_data = r7[9];
          float v614_data = r7[10];
          float v615_data = r7[11];
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v591_tp, v612_data, v611_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v592_tp, v613_data, v616_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v593_tp, v614_data, v617_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v594_tp, v615_data, v618_acc, 2, 2, 0);
          r9[0] = (v619_acc[0]);
          r9[1] = (v619_acc[1]);
          r9[2] = (v619_acc[2]);
          r9[3] = (v619_acc[3]);
          float v624_data = r8[4];
          float v625_data = r8[5];
          float v626_data = r8[6];
          float v627_data = r8[7];
          float v628_tp{};
          float v629_tp{};
          float v630_tp{};
          float v631_tp{};
          tensorforge::transpose4x4b32(v628_tp, v629_tp, v630_tp, v631_tp, v624_data, v625_data, v626_data, v627_data);
          tensorforge::VectorT<float, 4> v632_acc{};
          tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v596_data, v632_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v629_tp, v597_data, v637_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v598_data, v638_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v599_data, v639_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v604_data, v640_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v629_tp, v605_data, v645_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v606_data, v646_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v607_data, v647_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v612_data, v648_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v629_tp, v613_data, v653_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v614_data, v654_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v615_data, v655_acc, 2, 2, 0);
          r9[4] = (v656_acc[0]);
          r9[5] = (v656_acc[1]);
          r9[6] = (v656_acc[2]);
          r9[7] = (v656_acc[3]);
          float v661_data = r8[8];
          float v662_data = r8[9];
          float v663_data = r8[10];
          float v664_data = r8[11];
          float v665_tp{};
          float v666_tp{};
          float v667_tp{};
          float v668_tp{};
          tensorforge::transpose4x4b32(v665_tp, v666_tp, v667_tp, v668_tp, v661_data, v662_data, v663_data, v664_data);
          tensorforge::VectorT<float, 4> v669_acc{};
          tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v596_data, v669_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v597_data, v674_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v598_data, v675_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v599_data, v676_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v604_data, v677_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v605_data, v682_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v606_data, v683_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v607_data, v684_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v612_data, v685_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v613_data, v690_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v614_data, v691_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v615_data, v692_acc, 2, 2, 0);
          r9[8] = (v693_acc[0]);
          r9[9] = (v693_acc[1]);
          r9[10] = (v693_acc[2]);
          r9[11] = (v693_acc[3]);
          // s1 = store{r>s}(localShrMem0, r9);
          if (v30_g) {
            #pragma unroll
            for (int32_t v698_i1 = 0; v698_i1 < 12; ++v698_i1) {
              float v700_data = r9[v698_i1];
              int32_t v704_a = v29_lead + (v698_i1 * 12);
              s1[(v704_a ^ ((v704_a >> 4) & 15))] = v700_data;
            }
          }
          float r12[12]{};
          // r12 = load{g>r}(glb_m8);
          if (v39_g) {
            #pragma unroll
            for (int32_t v840_i1 = 0; v840_i1 < 12; ++v840_i1) {
              float v845_data = __builtin_nontemporal_load(&glb_m8[(v29_lead + (v840_i1 * 12))]);
              r12[v840_i1] = v845_data;
            }
          }
          float r11[12]{};
          // r11 = +(r10 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v721_tp{};
          float v722_tp{};
          float v723_tp{};
          float v724_tp{};
          tensorforge::transpose4x4b32(v721_tp, v722_tp, v723_tp, v724_tp, v587_data, v588_data, v589_data, v590_data);
          tensorforge::VectorT<float, 4> v725_acc{};
          float v726_data = r10[0];
          float v727_data = r10[1];
          float v728_data = r10[2];
          float v729_data = r10[3];
          tensorforge::VectorT<float, 4> v730_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v721_tp, v726_data, v725_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v731_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v722_tp, v727_data, v730_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v732_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v723_tp, v728_data, v731_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v733_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v724_tp, v729_data, v732_acc, 2, 0, 0);
          float v734_data = r10[4];
          float v735_data = r10[5];
          float v736_data = r10[6];
          float v737_data = r10[7];
          tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v721_tp, v734_data, v733_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v739_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v722_tp, v735_data, v738_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v740_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v723_tp, v736_data, v739_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v741_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v724_tp, v737_data, v740_acc, 2, 1, 0);
          float v742_data = r10[8];
          float v743_data = r10[9];
          float v744_data = r10[10];
          float v745_data = r10[11];
          tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v721_tp, v742_data, v741_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v747_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v722_tp, v743_data, v746_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v748_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v723_tp, v744_data, v747_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v749_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v724_tp, v745_data, v748_acc, 2, 2, 0);
          r11[0] = (v749_acc[0]);
          r11[1] = (v749_acc[1]);
          r11[2] = (v749_acc[2]);
          r11[3] = (v749_acc[3]);
          float v758_tp{};
          float v759_tp{};
          float v760_tp{};
          float v761_tp{};
          tensorforge::transpose4x4b32(v758_tp, v759_tp, v760_tp, v761_tp, v624_data, v625_data, v626_data, v627_data);
          tensorforge::VectorT<float, 4> v762_acc{};
          tensorforge::VectorT<float, 4> v767_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v758_tp, v726_data, v762_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v759_tp, v727_data, v767_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v769_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v760_tp, v728_data, v768_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v729_data, v769_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v758_tp, v734_data, v770_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v759_tp, v735_data, v775_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v777_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v760_tp, v736_data, v776_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v737_data, v777_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v758_tp, v742_data, v778_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v759_tp, v743_data, v783_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v785_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v760_tp, v744_data, v784_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v745_data, v785_acc, 2, 2, 0);
          r11[4] = (v786_acc[0]);
          r11[5] = (v786_acc[1]);
          r11[6] = (v786_acc[2]);
          r11[7] = (v786_acc[3]);
          float v795_tp{};
          float v796_tp{};
          float v797_tp{};
          float v798_tp{};
          tensorforge::transpose4x4b32(v795_tp, v796_tp, v797_tp, v798_tp, v661_data, v662_data, v663_data, v664_data);
          tensorforge::VectorT<float, 4> v799_acc{};
          tensorforge::VectorT<float, 4> v804_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v795_tp, v726_data, v799_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v805_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v796_tp, v727_data, v804_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v806_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v797_tp, v728_data, v805_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v807_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v798_tp, v729_data, v806_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v795_tp, v734_data, v807_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v813_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v796_tp, v735_data, v812_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v814_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v797_tp, v736_data, v813_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v815_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v798_tp, v737_data, v814_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v795_tp, v742_data, v815_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v821_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v796_tp, v743_data, v820_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v797_tp, v744_data, v821_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v823_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v798_tp, v745_data, v822_acc, 2, 2, 0);
          r11[8] = (v823_acc[0]);
          r11[9] = (v823_acc[1]);
          r11[10] = (v823_acc[2]);
          r11[11] = (v823_acc[3]);
          // s1 = store{r>s}(localShrMem0, r11);
          if (v30_g) {
            int32_t v833_off = v29_lead + 6;
            #pragma unroll
            for (int32_t v828_i1 = 0; v828_i1 < 12; ++v828_i1) {
              float v830_data = r11[v828_i1];
              int32_t v835_a = v833_off + (v828_i1 * 12);
              s1[(v835_a ^ ((v835_a >> 4) & 15))] = v830_data;
            }
          }
          float r13[12]{};
          // ir13 = +(s1 * r12)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir13[12]{};
          float v849_data = r12[0];
          float v850_data = r12[1];
          float v851_data = r12[2];
          float v852_data = r12[3];
          float v853_tp{};
          float v854_tp{};
          float v855_tp{};
          float v856_tp{};
          tensorforge::transpose4x4b32(v853_tp, v854_tp, v855_tp, v856_tp, v849_data, v850_data, v851_data, v852_data);
          tensorforge::VectorT<float, 4> v857_acc{};
          int32_t v863_sw = v29_lead ^ v322_sw;
          float v864_data = s1[v863_sw];
          float v869_data = s1[(v325_a ^ (v326_sw & 15))];
          float v874_data = s1[(v330_a ^ (v331_sw & 15))];
          float v879_data = s1[(v335_a ^ (v336_sw & 15))];
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v864_data, v857_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v881_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v869_data, v880_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v882_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v874_data, v881_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v879_data, v882_acc, 2, 0, 0);
          float v888_data = s1[(v344_a ^ (v345_sw & 15))];
          float v893_data = s1[(v349_a ^ (v350_sw & 15))];
          float v898_data = s1[(v354_a ^ (v355_sw & 15))];
          float v903_data = s1[(v359_a ^ (v360_sw & 15))];
          tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v888_data, v883_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v905_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v893_data, v904_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v906_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v898_data, v905_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v903_data, v906_acc, 2, 1, 0);
          float v912_data = s1[(v368_a ^ (v369_sw & 15))];
          float v917_data = s1[(v373_a ^ (v374_sw & 15))];
          float v922_data = s1[(v378_a ^ (v379_sw & 15))];
          float v927_data = s1[(v383_a ^ (v384_sw & 15))];
          tensorforge::VectorT<float, 4> v928_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v912_data, v907_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v929_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v917_data, v928_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v930_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v922_data, v929_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v931_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v927_data, v930_acc, 2, 2, 0);
          ir13[0] = (v931_acc[0]);
          ir13[1] = (v931_acc[1]);
          ir13[2] = (v931_acc[2]);
          ir13[3] = (v931_acc[3]);
          float v936_data = r12[4];
          float v937_data = r12[5];
          float v938_data = r12[6];
          float v939_data = r12[7];
          float v940_tp{};
          float v941_tp{};
          float v942_tp{};
          float v943_tp{};
          tensorforge::transpose4x4b32(v940_tp, v941_tp, v942_tp, v943_tp, v936_data, v937_data, v938_data, v939_data);
          tensorforge::VectorT<float, 4> v944_acc{};
          float v951_data = s1[(v29_lead ^ v322_sw)];
          float v956_data = s1[(v325_a ^ (v326_sw & 15))];
          float v961_data = s1[(v330_a ^ (v331_sw & 15))];
          float v966_data = s1[(v335_a ^ (v336_sw & 15))];
          tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v951_data, v944_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v968_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v941_tp, v956_data, v967_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v942_tp, v961_data, v968_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v970_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v966_data, v969_acc, 2, 0, 0);
          float v975_data = s1[(v344_a ^ (v345_sw & 15))];
          float v980_data = s1[(v349_a ^ (v350_sw & 15))];
          float v985_data = s1[(v354_a ^ (v355_sw & 15))];
          float v990_data = s1[(v359_a ^ (v360_sw & 15))];
          tensorforge::VectorT<float, 4> v991_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v975_data, v970_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v992_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v941_tp, v980_data, v991_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v993_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v942_tp, v985_data, v992_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v994_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v990_data, v993_acc, 2, 1, 0);
          float v999_data = s1[(v368_a ^ (v369_sw & 15))];
          float v1004_data = s1[(v373_a ^ (v374_sw & 15))];
          float v1009_data = s1[(v378_a ^ (v379_sw & 15))];
          float v1014_data = s1[(v383_a ^ (v384_sw & 15))];
          tensorforge::VectorT<float, 4> v1015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v999_data, v994_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1016_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v941_tp, v1004_data, v1015_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1017_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v942_tp, v1009_data, v1016_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1018_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v1014_data, v1017_acc, 2, 2, 0);
          ir13[4] = (v1018_acc[0]);
          ir13[5] = (v1018_acc[1]);
          ir13[6] = (v1018_acc[2]);
          ir13[7] = (v1018_acc[3]);
          float v1023_data = r12[8];
          float v1024_data = r12[9];
          float v1025_data = r12[10];
          float v1026_data = r12[11];
          float v1027_tp{};
          float v1028_tp{};
          float v1029_tp{};
          float v1030_tp{};
          tensorforge::transpose4x4b32(v1027_tp, v1028_tp, v1029_tp, v1030_tp, v1023_data, v1024_data, v1025_data, v1026_data);
          tensorforge::VectorT<float, 4> v1031_acc{};
          float v1038_data = s1[(v29_lead ^ v322_sw)];
          float v1043_data = s1[(v325_a ^ (v326_sw & 15))];
          float v1048_data = s1[(v330_a ^ (v331_sw & 15))];
          float v1053_data = s1[(v335_a ^ (v336_sw & 15))];
          tensorforge::VectorT<float, 4> v1054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1038_data, v1031_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1055_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1043_data, v1054_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1056_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1048_data, v1055_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1053_data, v1056_acc, 2, 0, 0);
          float v1062_data = s1[(v344_a ^ (v345_sw & 15))];
          float v1067_data = s1[(v349_a ^ (v350_sw & 15))];
          float v1072_data = s1[(v354_a ^ (v355_sw & 15))];
          float v1077_data = s1[(v359_a ^ (v360_sw & 15))];
          tensorforge::VectorT<float, 4> v1078_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1062_data, v1057_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1079_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1067_data, v1078_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1080_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1072_data, v1079_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1081_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1077_data, v1080_acc, 2, 1, 0);
          float v1086_data = s1[(v368_a ^ (v369_sw & 15))];
          float v1091_data = s1[(v373_a ^ (v374_sw & 15))];
          float v1096_data = s1[(v378_a ^ (v379_sw & 15))];
          float v1101_data = s1[(v383_a ^ (v384_sw & 15))];
          tensorforge::VectorT<float, 4> v1102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1086_data, v1081_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1091_data, v1102_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1096_data, v1103_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1101_data, v1104_acc, 2, 2, 0);
          ir13[8] = (v1105_acc[0]);
          ir13[9] = (v1105_acc[1]);
          ir13[10] = (v1105_acc[2]);
          ir13[11] = (v1105_acc[3]);
          // r13 = ir13 + r6
          if (v39_g) {
            #pragma unroll
            for (int32_t v1110_n1 = 0; v1110_n1 < 12; ++v1110_n1) {
              float v1112_data = ir13[v1110_n1];
              float v1113_data = r6[v1110_n1];
              r13[v1110_n1] = (v1113_data + v1112_data);
            }
          }
          // glb_m3 = store{r>g}(r13);
          if (v39_g) {
            #pragma unroll
            for (int32_t v1115_i1 = 0; v1115_i1 < 12; ++v1115_i1) {
              float v1117_data = r13[v1115_i1];
              glb_m3[(v29_lead + (v1115_i1 * 12))] = v1117_data;
            }
          }
        }
      }
    }
  }
}

