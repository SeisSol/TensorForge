// === base name ===
kernel_c41efb00409da53e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c41efb00409da53e = {{16, 16, 1}, 16, 12, 1, 16, 25600, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c41efb00409da53e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c41efb00409da53e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c41efb00409da53e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c41efb00409da53e, block.x * block.y * block.z, 6400 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6400 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c41efb00409da53e, block.x * block.y * block.z, 0));
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
void launcher_kernel_c41efb00409da53e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c41efb00409da53e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c41efb00409da53e), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_c41efb00409da53e, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c41efb00409da53e(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    //   t0[i,j]@{0..4}×{0..12} = m7[i,k] × m1[k,j]
    //   m5[i,j] += m8[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":6400}],"shared_bytes":25600,"shared_elements":6400,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"G","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[6,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"H","bbox":[[0,0],[12,12]],"name":"m6","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[4,12]],"name":"m7","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m8","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[4,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          float r10[12]{};
          // r10 = load{g>r}(glb_m7);
          bool v444_g = v26_lead < 4;
          if (v444_g) {
            #pragma unroll
            for (int32_t v445_i1 = 0; v445_i1 < 12; ++v445_i1) {
              float v450_data = __builtin_nontemporal_load(&glb_m7[(v26_lead + (v445_i1 * 4))]);
              r10[v445_i1] = v450_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(s1 * r8) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v453_data = r8[0];
          float v454_data = r8[1];
          float v455_data = r8[2];
          float v456_data = r8[3];
          float v457_tp{};
          float v458_tp{};
          float v459_tp{};
          float v460_tp{};
          tensorforge::transpose4x4b32(v457_tp, v458_tp, v459_tp, v460_tp, v453_data, v454_data, v455_data, v456_data);
          tensorforge::VectorT<float, 4> v461_acc{};
          int32_t v466_sw = (v26_lead >> 4) & 15;
          float v468_data = s1[(v26_lead ^ v466_sw)];
          int32_t v469_a = v26_lead + 12;
          int32_t v470_sw = v469_a >> 4;
          float v473_data = s1[(v469_a ^ (v470_sw & 15))];
          int32_t v474_a = v26_lead + 24;
          int32_t v475_sw = v474_a >> 4;
          float v478_data = s1[(v474_a ^ (v475_sw & 15))];
          int32_t v479_a = v26_lead + 36;
          int32_t v480_sw = v479_a >> 4;
          float v483_data = s1[(v479_a ^ (v480_sw & 15))];
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v468_data, v461_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v473_data, v484_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v478_data, v485_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v483_data, v486_acc, 2, 0, 0);
          int32_t v488_a = v26_lead + 48;
          int32_t v489_sw = v488_a >> 4;
          float v492_data = s1[(v488_a ^ (v489_sw & 15))];
          int32_t v493_a = v26_lead + 60;
          int32_t v494_sw = v493_a >> 4;
          float v497_data = s1[(v493_a ^ (v494_sw & 15))];
          int32_t v498_a = v26_lead + 72;
          int32_t v499_sw = v498_a >> 4;
          float v502_data = s1[(v498_a ^ (v499_sw & 15))];
          int32_t v503_a = v26_lead + 84;
          int32_t v504_sw = v503_a >> 4;
          float v507_data = s1[(v503_a ^ (v504_sw & 15))];
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v492_data, v487_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v497_data, v508_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v502_data, v509_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v507_data, v510_acc, 2, 1, 0);
          int32_t v512_a = v26_lead + 96;
          int32_t v513_sw = v512_a >> 4;
          float v516_data = s1[(v512_a ^ (v513_sw & 15))];
          int32_t v517_a = v26_lead + 108;
          int32_t v518_sw = v517_a >> 4;
          float v521_data = s1[(v517_a ^ (v518_sw & 15))];
          int32_t v522_a = v26_lead + 120;
          int32_t v523_sw = v522_a >> 4;
          float v526_data = s1[(v522_a ^ (v523_sw & 15))];
          int32_t v527_a = v26_lead + 132;
          int32_t v528_sw = v527_a >> 4;
          float v531_data = s1[(v527_a ^ (v528_sw & 15))];
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v516_data, v511_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v521_data, v532_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v526_data, v533_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v531_data, v534_acc, 2, 2, 0);
          r9[0] = (v535_acc[0]);
          r9[1] = (v535_acc[1]);
          r9[2] = (v535_acc[2]);
          r9[3] = (v535_acc[3]);
          float v540_data = r8[4];
          float v541_data = r8[5];
          float v542_data = r8[6];
          float v543_data = r8[7];
          float v544_tp{};
          float v545_tp{};
          float v546_tp{};
          float v547_tp{};
          tensorforge::transpose4x4b32(v544_tp, v545_tp, v546_tp, v547_tp, v540_data, v541_data, v542_data, v543_data);
          tensorforge::VectorT<float, 4> v548_acc{};
          float v555_data = s1[(v26_lead ^ v466_sw)];
          float v560_data = s1[(v469_a ^ (v470_sw & 15))];
          float v565_data = s1[(v474_a ^ (v475_sw & 15))];
          float v570_data = s1[(v479_a ^ (v480_sw & 15))];
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v555_data, v548_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v560_data, v571_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v565_data, v572_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v570_data, v573_acc, 2, 0, 0);
          float v579_data = s1[(v488_a ^ (v489_sw & 15))];
          float v584_data = s1[(v493_a ^ (v494_sw & 15))];
          float v589_data = s1[(v498_a ^ (v499_sw & 15))];
          float v594_data = s1[(v503_a ^ (v504_sw & 15))];
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v579_data, v574_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v584_data, v595_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v589_data, v596_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v594_data, v597_acc, 2, 1, 0);
          float v603_data = s1[(v512_a ^ (v513_sw & 15))];
          float v608_data = s1[(v517_a ^ (v518_sw & 15))];
          float v613_data = s1[(v522_a ^ (v523_sw & 15))];
          float v618_data = s1[(v527_a ^ (v528_sw & 15))];
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v603_data, v598_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v608_data, v619_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v613_data, v620_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v618_data, v621_acc, 2, 2, 0);
          r9[4] = (v622_acc[0]);
          r9[5] = (v622_acc[1]);
          r9[6] = (v622_acc[2]);
          r9[7] = (v622_acc[3]);
          float v627_data = r8[8];
          float v628_data = r8[9];
          float v629_data = r8[10];
          float v630_data = r8[11];
          float v631_tp{};
          float v632_tp{};
          float v633_tp{};
          float v634_tp{};
          tensorforge::transpose4x4b32(v631_tp, v632_tp, v633_tp, v634_tp, v627_data, v628_data, v629_data, v630_data);
          tensorforge::VectorT<float, 4> v635_acc{};
          float v642_data = s1[(v26_lead ^ v466_sw)];
          float v647_data = s1[(v469_a ^ (v470_sw & 15))];
          float v652_data = s1[(v474_a ^ (v475_sw & 15))];
          float v657_data = s1[(v479_a ^ (v480_sw & 15))];
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v642_data, v635_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v647_data, v658_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v652_data, v659_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v657_data, v660_acc, 2, 0, 0);
          float v666_data = s1[(v488_a ^ (v489_sw & 15))];
          float v671_data = s1[(v493_a ^ (v494_sw & 15))];
          float v676_data = s1[(v498_a ^ (v499_sw & 15))];
          float v681_data = s1[(v503_a ^ (v504_sw & 15))];
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v666_data, v661_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v671_data, v682_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v676_data, v683_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v681_data, v684_acc, 2, 1, 0);
          float v690_data = s1[(v512_a ^ (v513_sw & 15))];
          float v695_data = s1[(v517_a ^ (v518_sw & 15))];
          float v700_data = s1[(v522_a ^ (v523_sw & 15))];
          float v705_data = s1[(v527_a ^ (v528_sw & 15))];
          tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v690_data, v685_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v695_data, v706_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v700_data, v707_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v705_data, v708_acc, 2, 2, 0);
          r9[8] = (v709_acc[0]);
          r9[9] = (v709_acc[1]);
          r9[10] = (v709_acc[2]);
          r9[11] = (v709_acc[3]);
          float r12[12]{};
          // r12 = load{g>r}(glb_m8);
          if (v27_g) {
            #pragma unroll
            for (int32_t v715_i1 = 0; v715_i1 < 12; ++v715_i1) {
              float v720_data = __builtin_nontemporal_load(&glb_m8[(v26_lead + (v715_i1 * 12))]);
              r12[v715_i1] = v720_data;
            }
          }
          // wait(r10 = load{g>r}(glb_m7););
          float r11[12]{};
          // r11 = +(r10 * r1) + None
          // [(0, 4), (0, 12)] [(0, 12)]
          float v727_tp{};
          float v728_tp{};
          float v729_tp{};
          float v730_tp{};
          tensorforge::transpose4x4b32(v727_tp, v728_tp, v729_tp, v730_tp, v53_data, v54_data, v55_data, v56_data);
          tensorforge::VectorT<float, 4> v731_acc{};
          float v732_data = r10[0];
          float v733_data = r10[1];
          float v734_data = r10[2];
          float v735_data = r10[3];
          tensorforge::VectorT<float, 4> v736_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v727_tp, v732_data, v731_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v737_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v728_tp, v733_data, v736_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v734_data, v737_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v739_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v730_tp, v735_data, v738_acc, 2, 0, 0);
          float v740_data = r10[4];
          float v741_data = r10[5];
          float v742_data = r10[6];
          float v743_data = r10[7];
          tensorforge::VectorT<float, 4> v744_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v727_tp, v740_data, v739_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v745_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v728_tp, v741_data, v744_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v742_data, v745_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v747_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v730_tp, v743_data, v746_acc, 2, 1, 0);
          float v748_data = r10[8];
          float v749_data = r10[9];
          float v750_data = r10[10];
          float v751_data = r10[11];
          tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v727_tp, v748_data, v747_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v753_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v728_tp, v749_data, v752_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v750_data, v753_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v755_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v730_tp, v751_data, v754_acc, 2, 2, 0);
          r11[0] = (v755_acc[0]);
          r11[1] = (v755_acc[1]);
          r11[2] = (v755_acc[2]);
          r11[3] = (v755_acc[3]);
          float v764_tp{};
          float v765_tp{};
          float v766_tp{};
          float v767_tp{};
          tensorforge::transpose4x4b32(v764_tp, v765_tp, v766_tp, v767_tp, v90_data, v91_data, v92_data, v93_data);
          tensorforge::VectorT<float, 4> v768_acc{};
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v732_data, v768_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v733_data, v773_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v734_data, v774_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v735_data, v775_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v740_data, v776_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v741_data, v781_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v742_data, v782_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v743_data, v783_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v748_data, v784_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v749_data, v789_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v750_data, v790_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v751_data, v791_acc, 2, 2, 0);
          r11[4] = (v792_acc[0]);
          r11[5] = (v792_acc[1]);
          r11[6] = (v792_acc[2]);
          r11[7] = (v792_acc[3]);
          float v801_tp{};
          float v802_tp{};
          float v803_tp{};
          float v804_tp{};
          tensorforge::transpose4x4b32(v801_tp, v802_tp, v803_tp, v804_tp, v127_data, v128_data, v129_data, v130_data);
          tensorforge::VectorT<float, 4> v805_acc{};
          tensorforge::VectorT<float, 4> v810_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v801_tp, v732_data, v805_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v811_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v802_tp, v733_data, v810_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v734_data, v811_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v813_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v735_data, v812_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v801_tp, v740_data, v813_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v819_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v802_tp, v741_data, v818_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v742_data, v819_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v821_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v743_data, v820_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v801_tp, v748_data, v821_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v827_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v802_tp, v749_data, v826_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v750_data, v827_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v829_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v751_data, v828_acc, 2, 2, 0);
          r11[8] = (v829_acc[0]);
          r11[9] = (v829_acc[1]);
          r11[10] = (v829_acc[2]);
          r11[11] = (v829_acc[3]);
          // s0 = store{r>s}(localShrMem0, r11);
          if (v444_g) {
            #pragma unroll
            for (int32_t v834_i1 = 0; v834_i1 < 12; ++v834_i1) {
              float v836_data = r11[v834_i1];
              int32_t v840_a = v26_lead + (v834_i1 * 12);
              s0[(v840_a ^ ((v840_a >> 4) & 15))] = v836_data;
            }
          }
          // wait(r12 = load{g>r}(glb_m8););
          float r13[12]{};
          // ir13 = +(r12 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir13[12]{};
          float v852_data = s0[(v26_lead ^ v466_sw)];
          float v857_data = s0[(v469_a ^ (v470_sw & 15))];
          float v862_data = s0[(v474_a ^ (v475_sw & 15))];
          float v867_data = s0[(v479_a ^ (v480_sw & 15))];
          float v868_tp{};
          float v869_tp{};
          float v870_tp{};
          float v871_tp{};
          tensorforge::transpose4x4b32(v868_tp, v869_tp, v870_tp, v871_tp, v852_data, v857_data, v862_data, v867_data);
          tensorforge::VectorT<float, 4> v872_acc{};
          float v873_data = r12[0];
          float v874_data = r12[1];
          float v875_data = r12[2];
          float v876_data = r12[3];
          tensorforge::VectorT<float, 4> v877_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v868_tp, v873_data, v872_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v869_tp, v874_data, v877_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v879_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v870_tp, v875_data, v878_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v876_data, v879_acc, 2, 0, 0);
          float v881_data = r12[4];
          float v882_data = r12[5];
          float v883_data = r12[6];
          float v884_data = r12[7];
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v868_tp, v881_data, v880_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v869_tp, v882_data, v885_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v870_tp, v883_data, v886_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v884_data, v887_acc, 2, 1, 0);
          float v889_data = r12[8];
          float v890_data = r12[9];
          float v891_data = r12[10];
          float v892_data = r12[11];
          tensorforge::VectorT<float, 4> v893_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v868_tp, v889_data, v888_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v894_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v869_tp, v890_data, v893_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v895_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v870_tp, v891_data, v894_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v896_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v892_data, v895_acc, 2, 2, 0);
          ir13[0] = (v896_acc[0]);
          ir13[1] = (v896_acc[1]);
          ir13[2] = (v896_acc[2]);
          ir13[3] = (v896_acc[3]);
          float v907_data = s0[(v488_a ^ (v489_sw & 15))];
          float v912_data = s0[(v493_a ^ (v494_sw & 15))];
          float v917_data = s0[(v498_a ^ (v499_sw & 15))];
          float v922_data = s0[(v503_a ^ (v504_sw & 15))];
          float v923_tp{};
          float v924_tp{};
          float v925_tp{};
          float v926_tp{};
          tensorforge::transpose4x4b32(v923_tp, v924_tp, v925_tp, v926_tp, v907_data, v912_data, v917_data, v922_data);
          tensorforge::VectorT<float, 4> v927_acc{};
          tensorforge::VectorT<float, 4> v932_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v923_tp, v873_data, v927_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v933_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v924_tp, v874_data, v932_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v934_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v925_tp, v875_data, v933_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v935_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v926_tp, v876_data, v934_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v940_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v923_tp, v881_data, v935_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v941_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v924_tp, v882_data, v940_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v942_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v925_tp, v883_data, v941_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v943_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v926_tp, v884_data, v942_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v948_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v923_tp, v889_data, v943_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v949_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v924_tp, v890_data, v948_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v950_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v925_tp, v891_data, v949_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v951_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v926_tp, v892_data, v950_acc, 2, 2, 0);
          ir13[4] = (v951_acc[0]);
          ir13[5] = (v951_acc[1]);
          ir13[6] = (v951_acc[2]);
          ir13[7] = (v951_acc[3]);
          float v962_data = s0[(v512_a ^ (v513_sw & 15))];
          float v967_data = s0[(v517_a ^ (v518_sw & 15))];
          float v972_data = s0[(v522_a ^ (v523_sw & 15))];
          float v977_data = s0[(v527_a ^ (v528_sw & 15))];
          float v978_tp{};
          float v979_tp{};
          float v980_tp{};
          float v981_tp{};
          tensorforge::transpose4x4b32(v978_tp, v979_tp, v980_tp, v981_tp, v962_data, v967_data, v972_data, v977_data);
          tensorforge::VectorT<float, 4> v982_acc{};
          tensorforge::VectorT<float, 4> v987_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v978_tp, v873_data, v982_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v988_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v979_tp, v874_data, v987_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v989_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v980_tp, v875_data, v988_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v990_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v981_tp, v876_data, v989_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v995_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v978_tp, v881_data, v990_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v996_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v979_tp, v882_data, v995_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v997_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v980_tp, v883_data, v996_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v998_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v981_tp, v884_data, v997_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1003_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v978_tp, v889_data, v998_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1004_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v979_tp, v890_data, v1003_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1005_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v980_tp, v891_data, v1004_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1006_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v981_tp, v892_data, v1005_acc, 2, 2, 0);
          ir13[8] = (v1006_acc[0]);
          ir13[9] = (v1006_acc[1]);
          ir13[10] = (v1006_acc[2]);
          ir13[11] = (v1006_acc[3]);
          // r13 = ir13 + r9
          if (v27_g) {
            #pragma unroll
            for (int32_t v1011_n1 = 0; v1011_n1 < 12; ++v1011_n1) {
              float v1013_data = ir13[v1011_n1];
              float v1014_data = r9[v1011_n1];
              r13[v1011_n1] = (v1014_data + v1013_data);
            }
          }
          // glb_m5 = store{r>g}(r13);
          if (v27_g) {
            #pragma unroll
            for (int32_t v1016_i1 = 0; v1016_i1 < 12; ++v1016_i1) {
              float v1018_data = r13[v1016_i1];
              glb_m5[(v26_lead + (v1016_i1 * 12))] = v1018_data;
            }
          }
        }
      }
    }
  }
}

