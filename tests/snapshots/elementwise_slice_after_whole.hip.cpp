// === base name ===
kernel_c26278c6deaf648a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c26278c6deaf648a = {{16, 16, 1}, 16, 12, 1, 16, 25600, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c26278c6deaf648a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c26278c6deaf648a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c26278c6deaf648a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c26278c6deaf648a, block.x * block.y * block.z, 6400 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6400 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c26278c6deaf648a, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (6400 * sizeof(float)));
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
  config.sharedMemBytes = 6400 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c26278c6deaf648a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c26278c6deaf648a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c26278c6deaf648a), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_c26278c6deaf648a, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c26278c6deaf648a(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 25600 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(6×12) {0..6}×{0..12} strided
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    //   m4 32×32(6×12) {0..6}×{0..12} strided
    //   m5 32×32(12×12) {0..12}×{0..12} strided
    //   m6 32×32(12×12) {0..12}×{0..12} strided
    //   m7 32×32(4×12) {0..4}×{0..12} strided
    //   m8 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j]@{0..6}×{0..12} = m2[i,k] × m3[k,j]
    //   t1[i,j]@{6..12}×{0..12} = m4[i,k] × m3[k,j]
    //   m5[i,j] = t1[i,k] × m6[k,j]
    //   t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..4}×{0..12}) = abs(K)
    //   m5[i,j] += m8[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":6400}],"shared_bytes":25600,"shared_elements":6400,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"G","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[6,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"H","bbox":[[0,0],[12,12]],"name":"m6","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"K","bbox":[[0,0],[4,12]],"name":"m7","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m8","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[4,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[4,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[400 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      float * __restrict__ s0 = &localShrMem0[192];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v6_batchId0 * 72 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v6_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v6_batchId0 * 144 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v6_batchId0 * 48 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v6_batchId0 * 144 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 16;
          bool v27_g = v26_lead < 12;
          if (v27_g) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m0[(v26_lead + (v28_i1 * 12))]);
              r0[v28_i1] = v33_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v27_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v26_lead + (v36_i1 * 12))]);
              r1[v36_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          bool v44_g = v26_lead < 6;
          if (v44_g) {
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              float v50_data = __builtin_nontemporal_load(&glb_m2[(v26_lead + (v45_i1 * 6))]);
              r3[v45_i1] = v50_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v53_data = r1[0];
          float v54_data = r1[1];
          float v55_data = r1[2];
          float v56_data = r1[3];
          float v57_tp{};
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          tensorforge::transpose4x4b32(v57_tp, v58_tp, v59_tp, v60_tp, v53_data, v54_data, v55_data, v56_data);
          tensorforge::VectorT<float, 4> v61_acc{};
          float v62_data = r0[0];
          float v63_data = r0[1];
          float v64_data = r0[2];
          float v65_data = r0[3];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 2, 0, 0);
          float v70_data = r0[4];
          float v71_data = r0[5];
          float v72_data = r0[6];
          float v73_data = r0[7];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 2, 1, 0);
          float v78_data = r0[8];
          float v79_data = r0[9];
          float v80_data = r0[10];
          float v81_data = r0[11];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v77_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 2, 2, 0);
          r2[0] = (v85_acc[0]);
          r2[1] = (v85_acc[1]);
          r2[2] = (v85_acc[2]);
          r2[3] = (v85_acc[3]);
          float v90_data = r1[4];
          float v91_data = r1[5];
          float v92_data = r1[6];
          float v93_data = r1[7];
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          float v97_tp{};
          tensorforge::transpose4x4b32(v94_tp, v95_tp, v96_tp, v97_tp, v90_data, v91_data, v92_data, v93_data);
          tensorforge::VectorT<float, 4> v98_acc{};
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v113_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v119_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v121_acc, 2, 2, 0);
          r2[4] = (v122_acc[0]);
          r2[5] = (v122_acc[1]);
          r2[6] = (v122_acc[2]);
          r2[7] = (v122_acc[3]);
          float v127_data = r1[8];
          float v128_data = r1[9];
          float v129_data = r1[10];
          float v130_data = r1[11];
          float v131_tp{};
          float v132_tp{};
          float v133_tp{};
          float v134_tp{};
          tensorforge::transpose4x4b32(v131_tp, v132_tp, v133_tp, v134_tp, v127_data, v128_data, v129_data, v130_data);
          tensorforge::VectorT<float, 4> v135_acc{};
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v140_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v142_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v148_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v150_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v78_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v79_data, v156_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v80_data, v157_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v81_data, v158_acc, 2, 2, 0);
          r2[8] = (v159_acc[0]);
          r2[9] = (v159_acc[1]);
          r2[10] = (v159_acc[2]);
          r2[11] = (v159_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v164_i1 = 0; v164_i1 < 12; ++v164_i1) {
              float v166_data = r2[v164_i1];
              int32_t v170_a = v26_lead + (v164_i1 * 12);
              s0[(v170_a ^ ((v170_a >> 4) & 15))] = v166_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m3);
          if (v27_g) {
            #pragma unroll
            for (int32_t v175_i1 = 0; v175_i1 < 12; ++v175_i1) {
              float v180_data = __builtin_nontemporal_load(&glb_m3[(v26_lead + (v175_i1 * 12))]);
              r4[v175_i1] = v180_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          if (v44_g) {
            #pragma unroll
            for (int32_t v183_i1 = 0; v183_i1 < 12; ++v183_i1) {
              float v188_data = __builtin_nontemporal_load(&glb_m4[(v26_lead + (v183_i1 * 6))]);
              r6[v183_i1] = v188_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[12]{};
          // r5 = +(r3 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v191_data = r4[0];
          float v192_data = r4[1];
          float v193_data = r4[2];
          float v194_data = r4[3];
          float v195_tp{};
          float v196_tp{};
          float v197_tp{};
          float v198_tp{};
          tensorforge::transpose4x4b32(v195_tp, v196_tp, v197_tp, v198_tp, v191_data, v192_data, v193_data, v194_data);
          tensorforge::VectorT<float, 4> v199_acc{};
          float v200_data = r3[0];
          float v201_data = r3[1];
          float v202_data = r3[2];
          float v203_data = r3[3];
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v200_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v201_data, v204_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v202_data, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v203_data, v206_acc, 2, 0, 0);
          float v208_data = r3[4];
          float v209_data = r3[5];
          float v210_data = r3[6];
          float v211_data = r3[7];
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v208_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v209_data, v212_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v210_data, v213_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v211_data, v214_acc, 2, 1, 0);
          float v216_data = r3[8];
          float v217_data = r3[9];
          float v218_data = r3[10];
          float v219_data = r3[11];
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v216_data, v215_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v217_data, v220_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v218_data, v221_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v219_data, v222_acc, 2, 2, 0);
          r5[0] = (v223_acc[0]);
          r5[1] = (v223_acc[1]);
          r5[2] = (v223_acc[2]);
          r5[3] = (v223_acc[3]);
          float v228_data = r4[4];
          float v229_data = r4[5];
          float v230_data = r4[6];
          float v231_data = r4[7];
          float v232_tp{};
          float v233_tp{};
          float v234_tp{};
          float v235_tp{};
          tensorforge::transpose4x4b32(v232_tp, v233_tp, v234_tp, v235_tp, v228_data, v229_data, v230_data, v231_data);
          tensorforge::VectorT<float, 4> v236_acc{};
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v200_data, v236_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v201_data, v241_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v202_data, v242_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v203_data, v243_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v208_data, v244_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v209_data, v249_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v210_data, v250_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v211_data, v251_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v216_data, v252_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v217_data, v257_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v218_data, v258_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v219_data, v259_acc, 2, 2, 0);
          r5[4] = (v260_acc[0]);
          r5[5] = (v260_acc[1]);
          r5[6] = (v260_acc[2]);
          r5[7] = (v260_acc[3]);
          float v265_data = r4[8];
          float v266_data = r4[9];
          float v267_data = r4[10];
          float v268_data = r4[11];
          float v269_tp{};
          float v270_tp{};
          float v271_tp{};
          float v272_tp{};
          tensorforge::transpose4x4b32(v269_tp, v270_tp, v271_tp, v272_tp, v265_data, v266_data, v267_data, v268_data);
          tensorforge::VectorT<float, 4> v273_acc{};
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v200_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v201_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v202_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v203_data, v280_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v208_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v209_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v210_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v211_data, v288_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v216_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v217_data, v294_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v218_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v219_data, v296_acc, 2, 2, 0);
          r5[8] = (v297_acc[0]);
          r5[9] = (v297_acc[1]);
          r5[10] = (v297_acc[2]);
          r5[11] = (v297_acc[3]);
          // s1 = store{r>s}(localShrMem0, r5);
          if (v44_g) {
            #pragma unroll
            for (int32_t v302_i1 = 0; v302_i1 < 12; ++v302_i1) {
              float v304_data = r5[v302_i1];
              int32_t v308_a = v26_lead + (v302_i1 * 12);
              s1[(v308_a ^ ((v308_a >> 4) & 15))] = v304_data;
            }
          }
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v27_g) {
            #pragma unroll
            for (int32_t v313_i1 = 0; v313_i1 < 12; ++v313_i1) {
              float v318_data = __builtin_nontemporal_load(&glb_m6[(v26_lead + (v313_i1 * 12))]);
              r8[v313_i1] = v318_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[12]{};
          // r7 = +(r6 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v325_tp{};
          float v326_tp{};
          float v327_tp{};
          float v328_tp{};
          tensorforge::transpose4x4b32(v325_tp, v326_tp, v327_tp, v328_tp, v191_data, v192_data, v193_data, v194_data);
          tensorforge::VectorT<float, 4> v329_acc{};
          float v330_data = r6[0];
          float v331_data = r6[1];
          float v332_data = r6[2];
          float v333_data = r6[3];
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v330_data, v329_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v331_data, v334_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v332_data, v335_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v333_data, v336_acc, 2, 0, 0);
          float v338_data = r6[4];
          float v339_data = r6[5];
          float v340_data = r6[6];
          float v341_data = r6[7];
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v338_data, v337_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v339_data, v342_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v340_data, v343_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v341_data, v344_acc, 2, 1, 0);
          float v346_data = r6[8];
          float v347_data = r6[9];
          float v348_data = r6[10];
          float v349_data = r6[11];
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v346_data, v345_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v347_data, v350_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v348_data, v351_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v349_data, v352_acc, 2, 2, 0);
          r7[0] = (v353_acc[0]);
          r7[1] = (v353_acc[1]);
          r7[2] = (v353_acc[2]);
          r7[3] = (v353_acc[3]);
          float v362_tp{};
          float v363_tp{};
          float v364_tp{};
          float v365_tp{};
          tensorforge::transpose4x4b32(v362_tp, v363_tp, v364_tp, v365_tp, v228_data, v229_data, v230_data, v231_data);
          tensorforge::VectorT<float, 4> v366_acc{};
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v330_data, v366_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v331_data, v371_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v332_data, v372_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v333_data, v373_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v338_data, v374_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v339_data, v379_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v340_data, v380_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v341_data, v381_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v346_data, v382_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v347_data, v387_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v348_data, v388_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v349_data, v389_acc, 2, 2, 0);
          r7[4] = (v390_acc[0]);
          r7[5] = (v390_acc[1]);
          r7[6] = (v390_acc[2]);
          r7[7] = (v390_acc[3]);
          float v399_tp{};
          float v400_tp{};
          float v401_tp{};
          float v402_tp{};
          tensorforge::transpose4x4b32(v399_tp, v400_tp, v401_tp, v402_tp, v265_data, v266_data, v267_data, v268_data);
          tensorforge::VectorT<float, 4> v403_acc{};
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v330_data, v403_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v331_data, v408_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v332_data, v409_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v333_data, v410_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v338_data, v411_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v339_data, v416_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v340_data, v417_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v341_data, v418_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v399_tp, v346_data, v419_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v347_data, v424_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v401_tp, v348_data, v425_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v349_data, v426_acc, 2, 2, 0);
          r7[8] = (v427_acc[0]);
          r7[9] = (v427_acc[1]);
          r7[10] = (v427_acc[2]);
          r7[11] = (v427_acc[3]);
          // s1 = store{r>s}(localShrMem0, r7);
          if (v44_g) {
            int32_t v437_off = v26_lead + 6;
            #pragma unroll
            for (int32_t v432_i1 = 0; v432_i1 < 12; ++v432_i1) {
              float v434_data = r7[v432_i1];
              int32_t v439_a = v437_off + (v432_i1 * 12);
              s1[(v439_a ^ ((v439_a >> 4) & 15))] = v434_data;
            }
          }
          float r11[12]{};
          // r11 = load{g>r}(glb_m8);
          if (v27_g) {
            #pragma unroll
            for (int32_t v444_i1 = 0; v444_i1 < 12; ++v444_i1) {
              float v449_data = __builtin_nontemporal_load(&glb_m8[(v26_lead + (v444_i1 * 12))]);
              r11[v444_i1] = v449_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(s1 * r8) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v452_data = r8[0];
          float v453_data = r8[1];
          float v454_data = r8[2];
          float v455_data = r8[3];
          float v456_tp{};
          float v457_tp{};
          float v458_tp{};
          float v459_tp{};
          tensorforge::transpose4x4b32(v456_tp, v457_tp, v458_tp, v459_tp, v452_data, v453_data, v454_data, v455_data);
          tensorforge::VectorT<float, 4> v460_acc{};
          int32_t v465_sw = (v26_lead >> 4) & 15;
          float v467_data = s1[(v26_lead ^ v465_sw)];
          int32_t v468_a = v26_lead + 12;
          int32_t v469_sw = v468_a >> 4;
          float v472_data = s1[(v468_a ^ (v469_sw & 15))];
          int32_t v473_a = v26_lead + 24;
          int32_t v474_sw = v473_a >> 4;
          float v477_data = s1[(v473_a ^ (v474_sw & 15))];
          int32_t v478_a = v26_lead + 36;
          int32_t v479_sw = v478_a >> 4;
          float v482_data = s1[(v478_a ^ (v479_sw & 15))];
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v467_data, v460_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v472_data, v483_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v477_data, v484_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v482_data, v485_acc, 2, 0, 0);
          int32_t v487_a = v26_lead + 48;
          int32_t v488_sw = v487_a >> 4;
          float v491_data = s1[(v487_a ^ (v488_sw & 15))];
          int32_t v492_a = v26_lead + 60;
          int32_t v493_sw = v492_a >> 4;
          float v496_data = s1[(v492_a ^ (v493_sw & 15))];
          int32_t v497_a = v26_lead + 72;
          int32_t v498_sw = v497_a >> 4;
          float v501_data = s1[(v497_a ^ (v498_sw & 15))];
          int32_t v502_a = v26_lead + 84;
          int32_t v503_sw = v502_a >> 4;
          float v506_data = s1[(v502_a ^ (v503_sw & 15))];
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v491_data, v486_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v496_data, v507_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v501_data, v508_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v506_data, v509_acc, 2, 1, 0);
          int32_t v511_a = v26_lead + 96;
          int32_t v512_sw = v511_a >> 4;
          float v515_data = s1[(v511_a ^ (v512_sw & 15))];
          int32_t v516_a = v26_lead + 108;
          int32_t v517_sw = v516_a >> 4;
          float v520_data = s1[(v516_a ^ (v517_sw & 15))];
          int32_t v521_a = v26_lead + 120;
          int32_t v522_sw = v521_a >> 4;
          float v525_data = s1[(v521_a ^ (v522_sw & 15))];
          int32_t v526_a = v26_lead + 132;
          int32_t v527_sw = v526_a >> 4;
          float v530_data = s1[(v526_a ^ (v527_sw & 15))];
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v515_data, v510_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v520_data, v531_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v525_data, v532_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v530_data, v533_acc, 2, 2, 0);
          r9[0] = (v534_acc[0]);
          r9[1] = (v534_acc[1]);
          r9[2] = (v534_acc[2]);
          r9[3] = (v534_acc[3]);
          float v539_data = r8[4];
          float v540_data = r8[5];
          float v541_data = r8[6];
          float v542_data = r8[7];
          float v543_tp{};
          float v544_tp{};
          float v545_tp{};
          float v546_tp{};
          tensorforge::transpose4x4b32(v543_tp, v544_tp, v545_tp, v546_tp, v539_data, v540_data, v541_data, v542_data);
          tensorforge::VectorT<float, 4> v547_acc{};
          float v554_data = s1[(v26_lead ^ v465_sw)];
          float v559_data = s1[(v468_a ^ (v469_sw & 15))];
          float v564_data = s1[(v473_a ^ (v474_sw & 15))];
          float v569_data = s1[(v478_a ^ (v479_sw & 15))];
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v543_tp, v554_data, v547_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v559_data, v570_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v564_data, v571_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v569_data, v572_acc, 2, 0, 0);
          float v578_data = s1[(v487_a ^ (v488_sw & 15))];
          float v583_data = s1[(v492_a ^ (v493_sw & 15))];
          float v588_data = s1[(v497_a ^ (v498_sw & 15))];
          float v593_data = s1[(v502_a ^ (v503_sw & 15))];
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v543_tp, v578_data, v573_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v583_data, v594_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v588_data, v595_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v593_data, v596_acc, 2, 1, 0);
          float v602_data = s1[(v511_a ^ (v512_sw & 15))];
          float v607_data = s1[(v516_a ^ (v517_sw & 15))];
          float v612_data = s1[(v521_a ^ (v522_sw & 15))];
          float v617_data = s1[(v526_a ^ (v527_sw & 15))];
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v543_tp, v602_data, v597_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v607_data, v618_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v612_data, v619_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v617_data, v620_acc, 2, 2, 0);
          r9[4] = (v621_acc[0]);
          r9[5] = (v621_acc[1]);
          r9[6] = (v621_acc[2]);
          r9[7] = (v621_acc[3]);
          float v626_data = r8[8];
          float v627_data = r8[9];
          float v628_data = r8[10];
          float v629_data = r8[11];
          float v630_tp{};
          float v631_tp{};
          float v632_tp{};
          float v633_tp{};
          tensorforge::transpose4x4b32(v630_tp, v631_tp, v632_tp, v633_tp, v626_data, v627_data, v628_data, v629_data);
          tensorforge::VectorT<float, 4> v634_acc{};
          float v641_data = s1[(v26_lead ^ v465_sw)];
          float v646_data = s1[(v468_a ^ (v469_sw & 15))];
          float v651_data = s1[(v473_a ^ (v474_sw & 15))];
          float v656_data = s1[(v478_a ^ (v479_sw & 15))];
          tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v641_data, v634_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v646_data, v657_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v651_data, v658_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v656_data, v659_acc, 2, 0, 0);
          float v665_data = s1[(v487_a ^ (v488_sw & 15))];
          float v670_data = s1[(v492_a ^ (v493_sw & 15))];
          float v675_data = s1[(v497_a ^ (v498_sw & 15))];
          float v680_data = s1[(v502_a ^ (v503_sw & 15))];
          tensorforge::VectorT<float, 4> v681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v665_data, v660_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v670_data, v681_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v675_data, v682_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v680_data, v683_acc, 2, 1, 0);
          float v689_data = s1[(v511_a ^ (v512_sw & 15))];
          float v694_data = s1[(v516_a ^ (v517_sw & 15))];
          float v699_data = s1[(v521_a ^ (v522_sw & 15))];
          float v704_data = s1[(v526_a ^ (v527_sw & 15))];
          tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v689_data, v684_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v694_data, v705_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v699_data, v706_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v704_data, v707_acc, 2, 2, 0);
          r9[8] = (v708_acc[0]);
          r9[9] = (v708_acc[1]);
          r9[10] = (v708_acc[2]);
          r9[11] = (v708_acc[3]);
          float r10[12]{};
          // r10 = abs(glb_m7)
          bool v714_g = v26_lead < 4;
          if (v714_g) {
            #pragma unroll
            for (int32_t v715_k1 = 0; v715_k1 < 12; ++v715_k1) {
              float v720_data = glb_m7[(v26_lead + (v715_k1 * 4))];
              r10[v715_k1] = (fabsf(v720_data));
            }
          }
          // s0 = store{r>s}(localShrMem0, r10);
          if (v714_g) {
            #pragma unroll
            for (int32_t v724_i1 = 0; v724_i1 < 12; ++v724_i1) {
              float v726_data = r10[v724_i1];
              int32_t v730_a = v26_lead + (v724_i1 * 12);
              s0[(v730_a ^ ((v730_a >> 4) & 15))] = v726_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m8););
          float r12[12]{};
          // ir12 = +(r11 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir12[12]{};
          float v742_data = s0[(v26_lead ^ v465_sw)];
          float v747_data = s0[(v468_a ^ (v469_sw & 15))];
          float v752_data = s0[(v473_a ^ (v474_sw & 15))];
          float v757_data = s0[(v478_a ^ (v479_sw & 15))];
          float v758_tp{};
          float v759_tp{};
          float v760_tp{};
          float v761_tp{};
          tensorforge::transpose4x4b32(v758_tp, v759_tp, v760_tp, v761_tp, v742_data, v747_data, v752_data, v757_data);
          tensorforge::VectorT<float, 4> v762_acc{};
          float v763_data = r11[0];
          float v764_data = r11[1];
          float v765_data = r11[2];
          float v766_data = r11[3];
          tensorforge::VectorT<float, 4> v767_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v758_tp, v763_data, v762_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v759_tp, v764_data, v767_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v769_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v760_tp, v765_data, v768_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v766_data, v769_acc, 2, 0, 0);
          float v771_data = r11[4];
          float v772_data = r11[5];
          float v773_data = r11[6];
          float v774_data = r11[7];
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v758_tp, v771_data, v770_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v759_tp, v772_data, v775_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v777_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v760_tp, v773_data, v776_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v774_data, v777_acc, 2, 1, 0);
          float v779_data = r11[8];
          float v780_data = r11[9];
          float v781_data = r11[10];
          float v782_data = r11[11];
          tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v758_tp, v779_data, v778_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v759_tp, v780_data, v783_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v785_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v760_tp, v781_data, v784_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v782_data, v785_acc, 2, 2, 0);
          ir12[0] = (v786_acc[0]);
          ir12[1] = (v786_acc[1]);
          ir12[2] = (v786_acc[2]);
          ir12[3] = (v786_acc[3]);
          float v797_data = s0[(v487_a ^ (v488_sw & 15))];
          float v802_data = s0[(v492_a ^ (v493_sw & 15))];
          float v807_data = s0[(v497_a ^ (v498_sw & 15))];
          float v812_data = s0[(v502_a ^ (v503_sw & 15))];
          float v813_tp{};
          float v814_tp{};
          float v815_tp{};
          float v816_tp{};
          tensorforge::transpose4x4b32(v813_tp, v814_tp, v815_tp, v816_tp, v797_data, v802_data, v807_data, v812_data);
          tensorforge::VectorT<float, 4> v817_acc{};
          tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v813_tp, v763_data, v817_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v823_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v814_tp, v764_data, v822_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v824_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v815_tp, v765_data, v823_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v825_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v816_tp, v766_data, v824_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v813_tp, v771_data, v825_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v814_tp, v772_data, v830_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v832_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v815_tp, v773_data, v831_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v833_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v816_tp, v774_data, v832_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v838_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v813_tp, v779_data, v833_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v839_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v814_tp, v780_data, v838_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v840_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v815_tp, v781_data, v839_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v841_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v816_tp, v782_data, v840_acc, 2, 2, 0);
          ir12[4] = (v841_acc[0]);
          ir12[5] = (v841_acc[1]);
          ir12[6] = (v841_acc[2]);
          ir12[7] = (v841_acc[3]);
          float v852_data = s0[(v511_a ^ (v512_sw & 15))];
          float v857_data = s0[(v516_a ^ (v517_sw & 15))];
          float v862_data = s0[(v521_a ^ (v522_sw & 15))];
          float v867_data = s0[(v526_a ^ (v527_sw & 15))];
          float v868_tp{};
          float v869_tp{};
          float v870_tp{};
          float v871_tp{};
          tensorforge::transpose4x4b32(v868_tp, v869_tp, v870_tp, v871_tp, v852_data, v857_data, v862_data, v867_data);
          tensorforge::VectorT<float, 4> v872_acc{};
          tensorforge::VectorT<float, 4> v877_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v868_tp, v763_data, v872_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v869_tp, v764_data, v877_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v879_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v870_tp, v765_data, v878_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v766_data, v879_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v868_tp, v771_data, v880_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v869_tp, v772_data, v885_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v870_tp, v773_data, v886_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v774_data, v887_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v893_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v868_tp, v779_data, v888_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v894_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v869_tp, v780_data, v893_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v895_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v870_tp, v781_data, v894_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v896_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v782_data, v895_acc, 2, 2, 0);
          ir12[8] = (v896_acc[0]);
          ir12[9] = (v896_acc[1]);
          ir12[10] = (v896_acc[2]);
          ir12[11] = (v896_acc[3]);
          // r12 = ir12 + r9
          if (v27_g) {
            #pragma unroll
            for (int32_t v901_n1 = 0; v901_n1 < 12; ++v901_n1) {
              float v903_data = ir12[v901_n1];
              float v904_data = r9[v901_n1];
              r12[v901_n1] = (v904_data + v903_data);
            }
          }
          // glb_m5 = store{r>g}(r12);
          if (v27_g) {
            #pragma unroll
            for (int32_t v906_i1 = 0; v906_i1 < 12; ++v906_i1) {
              float v908_data = r12[v906_i1];
              glb_m5[(v26_lead + (v906_i1 * 12))] = v908_data;
            }
          }
        }
      }
    }
  }
}

