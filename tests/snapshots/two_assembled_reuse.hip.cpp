// === base name ===
kernel_dccfee18aa640389

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dccfee18aa640389 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dccfee18aa640389(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dccfee18aa640389(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dccfee18aa640389(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_dccfee18aa640389, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_dccfee18aa640389, block.x * block.y * block.z, 0));
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
void launcher_kernel_dccfee18aa640389(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dccfee18aa640389(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_dccfee18aa640389), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_dccfee18aa640389, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_dccfee18aa640389(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v30_g) {
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 12; ++v48_i1) {
              float v53_data = __builtin_nontemporal_load(&glb_m2[(v29_lead + (v48_i1 * 6))]);
              r3[v48_i1] = v53_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v56_data = r1[0];
          float v57_data = r1[1];
          float v58_data = r1[2];
          float v59_data = r1[3];
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v56_data, v57_data, v58_data, v59_data);
          tensorforge::VectorT<float, 4> v64_acc{};
          float v65_data = r0[0];
          float v66_data = r0[1];
          float v67_data = r0[2];
          float v68_data = r0[3];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 2, 0, 0);
          float v73_data = r0[4];
          float v74_data = r0[5];
          float v75_data = r0[6];
          float v76_data = r0[7];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v72_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v79_acc, 2, 1, 0);
          float v81_data = r0[8];
          float v82_data = r0[9];
          float v83_data = r0[10];
          float v84_data = r0[11];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v80_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v87_acc, 2, 2, 0);
          r2[0] = (v88_acc[0]);
          r2[1] = (v88_acc[1]);
          r2[2] = (v88_acc[2]);
          r2[3] = (v88_acc[3]);
          float v93_data = r1[4];
          float v94_data = r1[5];
          float v95_data = r1[6];
          float v96_data = r1[7];
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          float v100_tp{};
          tensorforge::transpose4x4b32(v97_tp, v98_tp, v99_tp, v100_tp, v93_data, v94_data, v95_data, v96_data);
          tensorforge::VectorT<float, 4> v101_acc{};
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v116_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v123_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v124_acc, 2, 2, 0);
          r2[4] = (v125_acc[0]);
          r2[5] = (v125_acc[1]);
          r2[6] = (v125_acc[2]);
          r2[7] = (v125_acc[3]);
          float v130_data = r1[8];
          float v131_data = r1[9];
          float v132_data = r1[10];
          float v133_data = r1[11];
          float v134_tp{};
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          tensorforge::transpose4x4b32(v134_tp, v135_tp, v136_tp, v137_tp, v130_data, v131_data, v132_data, v133_data);
          tensorforge::VectorT<float, 4> v138_acc{};
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v143_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v144_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v146_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v151_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v152_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v81_data, v154_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v82_data, v159_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v83_data, v160_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v84_data, v161_acc, 2, 2, 0);
          r2[8] = (v162_acc[0]);
          r2[9] = (v162_acc[1]);
          r2[10] = (v162_acc[2]);
          r2[11] = (v162_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v30_g) {
            #pragma unroll
            for (int32_t v167_i1 = 0; v167_i1 < 12; ++v167_i1) {
              float v169_data = r2[v167_i1];
              int32_t v173_a = v29_lead + (v167_i1 * 12);
              s0[(v173_a ^ ((v173_a >> 4) & 15))] = v169_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v39_g) {
            #pragma unroll
            for (int32_t v178_i1 = 0; v178_i1 < 12; ++v178_i1) {
              float v183_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v178_i1 * 12))]);
              r5[v178_i1] = v183_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          float v193_tp{};
          tensorforge::transpose4x4b32(v190_tp, v191_tp, v192_tp, v193_tp, v56_data, v57_data, v58_data, v59_data);
          tensorforge::VectorT<float, 4> v194_acc{};
          float v195_data = r3[0];
          float v196_data = r3[1];
          float v197_data = r3[2];
          float v198_data = r3[3];
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v195_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v196_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v197_data, v200_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v198_data, v201_acc, 2, 0, 0);
          float v203_data = r3[4];
          float v204_data = r3[5];
          float v205_data = r3[6];
          float v206_data = r3[7];
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v203_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v204_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v205_data, v208_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v206_data, v209_acc, 2, 1, 0);
          float v211_data = r3[8];
          float v212_data = r3[9];
          float v213_data = r3[10];
          float v214_data = r3[11];
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v211_data, v210_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v212_data, v215_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v213_data, v216_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v214_data, v217_acc, 2, 2, 0);
          r4[0] = (v218_acc[0]);
          r4[1] = (v218_acc[1]);
          r4[2] = (v218_acc[2]);
          r4[3] = (v218_acc[3]);
          float v227_tp{};
          float v228_tp{};
          float v229_tp{};
          float v230_tp{};
          tensorforge::transpose4x4b32(v227_tp, v228_tp, v229_tp, v230_tp, v93_data, v94_data, v95_data, v96_data);
          tensorforge::VectorT<float, 4> v231_acc{};
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v195_data, v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v196_data, v236_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v197_data, v237_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v198_data, v238_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v203_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v204_data, v244_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v205_data, v245_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v206_data, v246_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v211_data, v247_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v212_data, v252_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v213_data, v253_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v214_data, v254_acc, 2, 2, 0);
          r4[4] = (v255_acc[0]);
          r4[5] = (v255_acc[1]);
          r4[6] = (v255_acc[2]);
          r4[7] = (v255_acc[3]);
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          float v267_tp{};
          tensorforge::transpose4x4b32(v264_tp, v265_tp, v266_tp, v267_tp, v130_data, v131_data, v132_data, v133_data);
          tensorforge::VectorT<float, 4> v268_acc{};
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v195_data, v268_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v196_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v197_data, v274_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v198_data, v275_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v203_data, v276_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v204_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v205_data, v282_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v206_data, v283_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v211_data, v284_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v212_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v213_data, v290_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v214_data, v291_acc, 2, 2, 0);
          r4[8] = (v292_acc[0]);
          r4[9] = (v292_acc[1]);
          r4[10] = (v292_acc[2]);
          r4[11] = (v292_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v30_g) {
            int32_t v302_off = v29_lead + 6;
            #pragma unroll
            for (int32_t v297_i1 = 0; v297_i1 < 12; ++v297_i1) {
              float v299_data = r4[v297_i1];
              int32_t v304_a = v302_off + (v297_i1 * 12);
              s0[(v304_a ^ ((v304_a >> 4) & 15))] = v299_data;
            }
          }
          float r7[12]{};
          // r7 = load{g>r}(glb_m5);
          if (v30_g) {
            #pragma unroll
            for (int32_t v309_i1 = 0; v309_i1 < 12; ++v309_i1) {
              float v314_data = __builtin_nontemporal_load(&glb_m5[(v29_lead + (v309_i1 * 6))]);
              r7[v309_i1] = v314_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(s0 * r5) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v317_data = r5[0];
          float v318_data = r5[1];
          float v319_data = r5[2];
          float v320_data = r5[3];
          float v321_tp{};
          float v322_tp{};
          float v323_tp{};
          float v324_tp{};
          tensorforge::transpose4x4b32(v321_tp, v322_tp, v323_tp, v324_tp, v317_data, v318_data, v319_data, v320_data);
          tensorforge::VectorT<float, 4> v325_acc{};
          int32_t v330_sw = (v29_lead >> 4) & 15;
          int32_t v331_sw = v29_lead ^ v330_sw;
          float v332_data = s0[v331_sw];
          int32_t v333_a = v29_lead + 12;
          int32_t v334_sw = v333_a >> 4;
          float v337_data = s0[(v333_a ^ (v334_sw & 15))];
          int32_t v338_a = v29_lead + 24;
          int32_t v339_sw = v338_a >> 4;
          float v342_data = s0[(v338_a ^ (v339_sw & 15))];
          int32_t v343_a = v29_lead + 36;
          int32_t v344_sw = v343_a >> 4;
          float v347_data = s0[(v343_a ^ (v344_sw & 15))];
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v332_data, v325_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v337_data, v348_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v342_data, v349_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v347_data, v350_acc, 2, 0, 0);
          int32_t v352_a = v29_lead + 48;
          int32_t v353_sw = v352_a >> 4;
          float v356_data = s0[(v352_a ^ (v353_sw & 15))];
          int32_t v357_a = v29_lead + 60;
          int32_t v358_sw = v357_a >> 4;
          float v361_data = s0[(v357_a ^ (v358_sw & 15))];
          int32_t v362_a = v29_lead + 72;
          int32_t v363_sw = v362_a >> 4;
          float v366_data = s0[(v362_a ^ (v363_sw & 15))];
          int32_t v367_a = v29_lead + 84;
          int32_t v368_sw = v367_a >> 4;
          float v371_data = s0[(v367_a ^ (v368_sw & 15))];
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v356_data, v351_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v361_data, v372_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v366_data, v373_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v371_data, v374_acc, 2, 1, 0);
          int32_t v376_a = v29_lead + 96;
          int32_t v377_sw = v376_a >> 4;
          float v380_data = s0[(v376_a ^ (v377_sw & 15))];
          int32_t v381_a = v29_lead + 108;
          int32_t v382_sw = v381_a >> 4;
          float v385_data = s0[(v381_a ^ (v382_sw & 15))];
          int32_t v386_a = v29_lead + 120;
          int32_t v387_sw = v386_a >> 4;
          float v390_data = s0[(v386_a ^ (v387_sw & 15))];
          int32_t v391_a = v29_lead + 132;
          int32_t v392_sw = v391_a >> 4;
          float v395_data = s0[(v391_a ^ (v392_sw & 15))];
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v380_data, v375_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v385_data, v396_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v390_data, v397_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v395_data, v398_acc, 2, 2, 0);
          r6[0] = (v399_acc[0]);
          r6[1] = (v399_acc[1]);
          r6[2] = (v399_acc[2]);
          r6[3] = (v399_acc[3]);
          float v404_data = r5[4];
          float v405_data = r5[5];
          float v406_data = r5[6];
          float v407_data = r5[7];
          float v408_tp{};
          float v409_tp{};
          float v410_tp{};
          float v411_tp{};
          tensorforge::transpose4x4b32(v408_tp, v409_tp, v410_tp, v411_tp, v404_data, v405_data, v406_data, v407_data);
          tensorforge::VectorT<float, 4> v412_acc{};
          float v419_data = s0[(v29_lead ^ v330_sw)];
          float v424_data = s0[(v333_a ^ (v334_sw & 15))];
          float v429_data = s0[(v338_a ^ (v339_sw & 15))];
          float v434_data = s0[(v343_a ^ (v344_sw & 15))];
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v419_data, v412_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v424_data, v435_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v429_data, v436_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v434_data, v437_acc, 2, 0, 0);
          float v443_data = s0[(v352_a ^ (v353_sw & 15))];
          float v448_data = s0[(v357_a ^ (v358_sw & 15))];
          float v453_data = s0[(v362_a ^ (v363_sw & 15))];
          float v458_data = s0[(v367_a ^ (v368_sw & 15))];
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v443_data, v438_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v448_data, v459_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v453_data, v460_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v458_data, v461_acc, 2, 1, 0);
          float v467_data = s0[(v376_a ^ (v377_sw & 15))];
          float v472_data = s0[(v381_a ^ (v382_sw & 15))];
          float v477_data = s0[(v386_a ^ (v387_sw & 15))];
          float v482_data = s0[(v391_a ^ (v392_sw & 15))];
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v467_data, v462_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v472_data, v483_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v477_data, v484_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v482_data, v485_acc, 2, 2, 0);
          r6[4] = (v486_acc[0]);
          r6[5] = (v486_acc[1]);
          r6[6] = (v486_acc[2]);
          r6[7] = (v486_acc[3]);
          float v491_data = r5[8];
          float v492_data = r5[9];
          float v493_data = r5[10];
          float v494_data = r5[11];
          float v495_tp{};
          float v496_tp{};
          float v497_tp{};
          float v498_tp{};
          tensorforge::transpose4x4b32(v495_tp, v496_tp, v497_tp, v498_tp, v491_data, v492_data, v493_data, v494_data);
          tensorforge::VectorT<float, 4> v499_acc{};
          float v506_data = s0[(v29_lead ^ v330_sw)];
          float v511_data = s0[(v333_a ^ (v334_sw & 15))];
          float v516_data = s0[(v338_a ^ (v339_sw & 15))];
          float v521_data = s0[(v343_a ^ (v344_sw & 15))];
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v506_data, v499_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v511_data, v522_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v516_data, v523_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v521_data, v524_acc, 2, 0, 0);
          float v530_data = s0[(v352_a ^ (v353_sw & 15))];
          float v535_data = s0[(v357_a ^ (v358_sw & 15))];
          float v540_data = s0[(v362_a ^ (v363_sw & 15))];
          float v545_data = s0[(v367_a ^ (v368_sw & 15))];
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v530_data, v525_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v535_data, v546_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v540_data, v547_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v545_data, v548_acc, 2, 1, 0);
          float v554_data = s0[(v376_a ^ (v377_sw & 15))];
          float v559_data = s0[(v381_a ^ (v382_sw & 15))];
          float v564_data = s0[(v386_a ^ (v387_sw & 15))];
          float v569_data = s0[(v391_a ^ (v392_sw & 15))];
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v554_data, v549_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v559_data, v570_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v564_data, v571_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v569_data, v572_acc, 2, 2, 0);
          r6[8] = (v573_acc[0]);
          r6[9] = (v573_acc[1]);
          r6[10] = (v573_acc[2]);
          r6[11] = (v573_acc[3]);
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v39_g) {
            #pragma unroll
            for (int32_t v579_i1 = 0; v579_i1 < 12; ++v579_i1) {
              float v584_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v579_i1 * 12))]);
              r8[v579_i1] = v584_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r10[12]{};
          // r10 = load{g>r}(glb_m7);
          if (v30_g) {
            #pragma unroll
            for (int32_t v587_i1 = 0; v587_i1 < 12; ++v587_i1) {
              float v592_data = __builtin_nontemporal_load(&glb_m7[(v29_lead + (v587_i1 * 6))]);
              r10[v587_i1] = v592_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(r7 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v595_data = r8[0];
          float v596_data = r8[1];
          float v597_data = r8[2];
          float v598_data = r8[3];
          float v599_tp{};
          float v600_tp{};
          float v601_tp{};
          float v602_tp{};
          tensorforge::transpose4x4b32(v599_tp, v600_tp, v601_tp, v602_tp, v595_data, v596_data, v597_data, v598_data);
          tensorforge::VectorT<float, 4> v603_acc{};
          float v604_data = r7[0];
          float v605_data = r7[1];
          float v606_data = r7[2];
          float v607_data = r7[3];
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v604_data, v603_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v600_tp, v605_data, v608_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v601_tp, v606_data, v609_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v607_data, v610_acc, 2, 0, 0);
          float v612_data = r7[4];
          float v613_data = r7[5];
          float v614_data = r7[6];
          float v615_data = r7[7];
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v612_data, v611_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v600_tp, v613_data, v616_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v601_tp, v614_data, v617_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v615_data, v618_acc, 2, 1, 0);
          float v620_data = r7[8];
          float v621_data = r7[9];
          float v622_data = r7[10];
          float v623_data = r7[11];
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v620_data, v619_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v600_tp, v621_data, v624_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v601_tp, v622_data, v625_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v623_data, v626_acc, 2, 2, 0);
          r9[0] = (v627_acc[0]);
          r9[1] = (v627_acc[1]);
          r9[2] = (v627_acc[2]);
          r9[3] = (v627_acc[3]);
          float v632_data = r8[4];
          float v633_data = r8[5];
          float v634_data = r8[6];
          float v635_data = r8[7];
          float v636_tp{};
          float v637_tp{};
          float v638_tp{};
          float v639_tp{};
          tensorforge::transpose4x4b32(v636_tp, v637_tp, v638_tp, v639_tp, v632_data, v633_data, v634_data, v635_data);
          tensorforge::VectorT<float, 4> v640_acc{};
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v604_data, v640_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v605_data, v645_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v606_data, v646_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v607_data, v647_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v612_data, v648_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v613_data, v653_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v614_data, v654_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v615_data, v655_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v620_data, v656_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v621_data, v661_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v622_data, v662_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v623_data, v663_acc, 2, 2, 0);
          r9[4] = (v664_acc[0]);
          r9[5] = (v664_acc[1]);
          r9[6] = (v664_acc[2]);
          r9[7] = (v664_acc[3]);
          float v669_data = r8[8];
          float v670_data = r8[9];
          float v671_data = r8[10];
          float v672_data = r8[11];
          float v673_tp{};
          float v674_tp{};
          float v675_tp{};
          float v676_tp{};
          tensorforge::transpose4x4b32(v673_tp, v674_tp, v675_tp, v676_tp, v669_data, v670_data, v671_data, v672_data);
          tensorforge::VectorT<float, 4> v677_acc{};
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v604_data, v677_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v674_tp, v605_data, v682_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v675_tp, v606_data, v683_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v607_data, v684_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v612_data, v685_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v674_tp, v613_data, v690_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v675_tp, v614_data, v691_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v615_data, v692_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v620_data, v693_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v674_tp, v621_data, v698_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v675_tp, v622_data, v699_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v623_data, v700_acc, 2, 2, 0);
          r9[8] = (v701_acc[0]);
          r9[9] = (v701_acc[1]);
          r9[10] = (v701_acc[2]);
          r9[11] = (v701_acc[3]);
          // s1 = store{r>s}(localShrMem0, r9);
          if (v30_g) {
            #pragma unroll
            for (int32_t v706_i1 = 0; v706_i1 < 12; ++v706_i1) {
              float v708_data = r9[v706_i1];
              int32_t v712_a = v29_lead + (v706_i1 * 12);
              s1[(v712_a ^ ((v712_a >> 4) & 15))] = v708_data;
            }
          }
          float r12[12]{};
          // r12 = load{g>r}(glb_m8);
          if (v39_g) {
            #pragma unroll
            for (int32_t v717_i1 = 0; v717_i1 < 12; ++v717_i1) {
              float v722_data = __builtin_nontemporal_load(&glb_m8[(v29_lead + (v717_i1 * 12))]);
              r12[v717_i1] = v722_data;
            }
          }
          // wait(r10 = load{g>r}(glb_m7););
          float r11[12]{};
          // r11 = +(r10 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v729_tp{};
          float v730_tp{};
          float v731_tp{};
          float v732_tp{};
          tensorforge::transpose4x4b32(v729_tp, v730_tp, v731_tp, v732_tp, v595_data, v596_data, v597_data, v598_data);
          tensorforge::VectorT<float, 4> v733_acc{};
          float v734_data = r10[0];
          float v735_data = r10[1];
          float v736_data = r10[2];
          float v737_data = r10[3];
          tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v734_data, v733_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v739_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v730_tp, v735_data, v738_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v740_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v731_tp, v736_data, v739_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v741_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v732_tp, v737_data, v740_acc, 2, 0, 0);
          float v742_data = r10[4];
          float v743_data = r10[5];
          float v744_data = r10[6];
          float v745_data = r10[7];
          tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v742_data, v741_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v747_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v730_tp, v743_data, v746_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v748_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v731_tp, v744_data, v747_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v749_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v732_tp, v745_data, v748_acc, 2, 1, 0);
          float v750_data = r10[8];
          float v751_data = r10[9];
          float v752_data = r10[10];
          float v753_data = r10[11];
          tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v750_data, v749_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v755_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v730_tp, v751_data, v754_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v756_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v731_tp, v752_data, v755_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v757_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v732_tp, v753_data, v756_acc, 2, 2, 0);
          r11[0] = (v757_acc[0]);
          r11[1] = (v757_acc[1]);
          r11[2] = (v757_acc[2]);
          r11[3] = (v757_acc[3]);
          float v766_tp{};
          float v767_tp{};
          float v768_tp{};
          float v769_tp{};
          tensorforge::transpose4x4b32(v766_tp, v767_tp, v768_tp, v769_tp, v632_data, v633_data, v634_data, v635_data);
          tensorforge::VectorT<float, 4> v770_acc{};
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v734_data, v770_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v735_data, v775_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v777_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v768_tp, v736_data, v776_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v769_tp, v737_data, v777_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v742_data, v778_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v743_data, v783_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v785_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v768_tp, v744_data, v784_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v769_tp, v745_data, v785_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v750_data, v786_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v751_data, v791_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v793_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v768_tp, v752_data, v792_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v769_tp, v753_data, v793_acc, 2, 2, 0);
          r11[4] = (v794_acc[0]);
          r11[5] = (v794_acc[1]);
          r11[6] = (v794_acc[2]);
          r11[7] = (v794_acc[3]);
          float v803_tp{};
          float v804_tp{};
          float v805_tp{};
          float v806_tp{};
          tensorforge::transpose4x4b32(v803_tp, v804_tp, v805_tp, v806_tp, v669_data, v670_data, v671_data, v672_data);
          tensorforge::VectorT<float, 4> v807_acc{};
          tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v734_data, v807_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v813_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v735_data, v812_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v814_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v805_tp, v736_data, v813_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v815_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v737_data, v814_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v742_data, v815_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v821_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v743_data, v820_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v805_tp, v744_data, v821_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v823_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v745_data, v822_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v750_data, v823_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v829_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v751_data, v828_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v805_tp, v752_data, v829_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v753_data, v830_acc, 2, 2, 0);
          r11[8] = (v831_acc[0]);
          r11[9] = (v831_acc[1]);
          r11[10] = (v831_acc[2]);
          r11[11] = (v831_acc[3]);
          // s1 = store{r>s}(localShrMem0, r11);
          if (v30_g) {
            int32_t v841_off = v29_lead + 6;
            #pragma unroll
            for (int32_t v836_i1 = 0; v836_i1 < 12; ++v836_i1) {
              float v838_data = r11[v836_i1];
              int32_t v843_a = v841_off + (v836_i1 * 12);
              s1[(v843_a ^ ((v843_a >> 4) & 15))] = v838_data;
            }
          }
          // wait(r12 = load{g>r}(glb_m8););
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
          int32_t v863_sw = v29_lead ^ v330_sw;
          float v864_data = s1[v863_sw];
          float v869_data = s1[(v333_a ^ (v334_sw & 15))];
          float v874_data = s1[(v338_a ^ (v339_sw & 15))];
          float v879_data = s1[(v343_a ^ (v344_sw & 15))];
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v864_data, v857_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v881_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v869_data, v880_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v882_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v874_data, v881_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v879_data, v882_acc, 2, 0, 0);
          float v888_data = s1[(v352_a ^ (v353_sw & 15))];
          float v893_data = s1[(v357_a ^ (v358_sw & 15))];
          float v898_data = s1[(v362_a ^ (v363_sw & 15))];
          float v903_data = s1[(v367_a ^ (v368_sw & 15))];
          tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v888_data, v883_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v905_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v854_tp, v893_data, v904_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v906_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v855_tp, v898_data, v905_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v903_data, v906_acc, 2, 1, 0);
          float v912_data = s1[(v376_a ^ (v377_sw & 15))];
          float v917_data = s1[(v381_a ^ (v382_sw & 15))];
          float v922_data = s1[(v386_a ^ (v387_sw & 15))];
          float v927_data = s1[(v391_a ^ (v392_sw & 15))];
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
          float v951_data = s1[(v29_lead ^ v330_sw)];
          float v956_data = s1[(v333_a ^ (v334_sw & 15))];
          float v961_data = s1[(v338_a ^ (v339_sw & 15))];
          float v966_data = s1[(v343_a ^ (v344_sw & 15))];
          tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v951_data, v944_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v968_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v941_tp, v956_data, v967_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v942_tp, v961_data, v968_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v970_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v966_data, v969_acc, 2, 0, 0);
          float v975_data = s1[(v352_a ^ (v353_sw & 15))];
          float v980_data = s1[(v357_a ^ (v358_sw & 15))];
          float v985_data = s1[(v362_a ^ (v363_sw & 15))];
          float v990_data = s1[(v367_a ^ (v368_sw & 15))];
          tensorforge::VectorT<float, 4> v991_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v975_data, v970_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v992_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v941_tp, v980_data, v991_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v993_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v942_tp, v985_data, v992_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v994_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v990_data, v993_acc, 2, 1, 0);
          float v999_data = s1[(v376_a ^ (v377_sw & 15))];
          float v1004_data = s1[(v381_a ^ (v382_sw & 15))];
          float v1009_data = s1[(v386_a ^ (v387_sw & 15))];
          float v1014_data = s1[(v391_a ^ (v392_sw & 15))];
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
          float v1038_data = s1[(v29_lead ^ v330_sw)];
          float v1043_data = s1[(v333_a ^ (v334_sw & 15))];
          float v1048_data = s1[(v338_a ^ (v339_sw & 15))];
          float v1053_data = s1[(v343_a ^ (v344_sw & 15))];
          tensorforge::VectorT<float, 4> v1054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1038_data, v1031_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1055_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1043_data, v1054_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1056_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1048_data, v1055_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1053_data, v1056_acc, 2, 0, 0);
          float v1062_data = s1[(v352_a ^ (v353_sw & 15))];
          float v1067_data = s1[(v357_a ^ (v358_sw & 15))];
          float v1072_data = s1[(v362_a ^ (v363_sw & 15))];
          float v1077_data = s1[(v367_a ^ (v368_sw & 15))];
          tensorforge::VectorT<float, 4> v1078_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1062_data, v1057_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1079_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1067_data, v1078_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1080_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1072_data, v1079_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1081_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1077_data, v1080_acc, 2, 1, 0);
          float v1086_data = s1[(v376_a ^ (v377_sw & 15))];
          float v1091_data = s1[(v381_a ^ (v382_sw & 15))];
          float v1096_data = s1[(v386_a ^ (v387_sw & 15))];
          float v1101_data = s1[(v391_a ^ (v392_sw & 15))];
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

