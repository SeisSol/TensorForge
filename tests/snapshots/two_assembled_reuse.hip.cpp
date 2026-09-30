// === base name ===
kernel_b5358aca1bc0503a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b5358aca1bc0503a = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b5358aca1bc0503a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b5358aca1bc0503a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b5358aca1bc0503a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b5358aca1bc0503a, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b5358aca1bc0503a, block.x * block.y * block.z, 0));
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
void launcher_kernel_b5358aca1bc0503a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b5358aca1bc0503a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b5358aca1bc0503a), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_b5358aca1bc0503a, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b5358aca1bc0503a(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v6_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v6_batchId0 * 144 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v6_batchId0 * 72 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v6_batchId0 * 144 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v6_batchId0 * 72 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v6_batchId0 * 144 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 16;
          bool v27_g = v26_lead < 6;
          if (v27_g) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m0[(v26_lead + (v28_i1 * 6))]);
              r0[v28_i1] = v33_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v36_g = v26_lead < 12;
          if (v36_g) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v26_lead + (v37_i1 * 12))]);
              r1[v37_i1] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              float v50_data = __builtin_nontemporal_load(&glb_m2[(v26_lead + (v45_i1 * 6))]);
              r3[v45_i1] = v50_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
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
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v36_g) {
            #pragma unroll
            for (int32_t v175_i1 = 0; v175_i1 < 12; ++v175_i1) {
              float v180_data = __builtin_nontemporal_load(&glb_m4[(v26_lead + (v175_i1 * 12))]);
              r5[v175_i1] = v180_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          tensorforge::transpose4x4b32(v187_tp, v188_tp, v189_tp, v190_tp, v53_data, v54_data, v55_data, v56_data);
          tensorforge::VectorT<float, 4> v191_acc{};
          float v192_data = r3[0];
          float v193_data = r3[1];
          float v194_data = r3[2];
          float v195_data = r3[3];
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v192_data, v191_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v193_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v194_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v195_data, v198_acc, 2, 0, 0);
          float v200_data = r3[4];
          float v201_data = r3[5];
          float v202_data = r3[6];
          float v203_data = r3[7];
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v200_data, v199_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v201_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v202_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v203_data, v206_acc, 2, 1, 0);
          float v208_data = r3[8];
          float v209_data = r3[9];
          float v210_data = r3[10];
          float v211_data = r3[11];
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v208_data, v207_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v209_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v210_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v211_data, v214_acc, 2, 2, 0);
          r4[0] = (v215_acc[0]);
          r4[1] = (v215_acc[1]);
          r4[2] = (v215_acc[2]);
          r4[3] = (v215_acc[3]);
          float v224_tp{};
          float v225_tp{};
          float v226_tp{};
          float v227_tp{};
          tensorforge::transpose4x4b32(v224_tp, v225_tp, v226_tp, v227_tp, v90_data, v91_data, v92_data, v93_data);
          tensorforge::VectorT<float, 4> v228_acc{};
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v192_data, v228_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v193_data, v233_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v194_data, v234_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v195_data, v235_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v200_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v201_data, v241_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v202_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v203_data, v243_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v208_data, v244_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v209_data, v249_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v210_data, v250_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v211_data, v251_acc, 2, 2, 0);
          r4[4] = (v252_acc[0]);
          r4[5] = (v252_acc[1]);
          r4[6] = (v252_acc[2]);
          r4[7] = (v252_acc[3]);
          float v261_tp{};
          float v262_tp{};
          float v263_tp{};
          float v264_tp{};
          tensorforge::transpose4x4b32(v261_tp, v262_tp, v263_tp, v264_tp, v127_data, v128_data, v129_data, v130_data);
          tensorforge::VectorT<float, 4> v265_acc{};
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v192_data, v265_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v193_data, v270_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v194_data, v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v195_data, v272_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v200_data, v273_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v201_data, v278_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v202_data, v279_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v203_data, v280_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v208_data, v281_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v209_data, v286_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v210_data, v287_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v211_data, v288_acc, 2, 2, 0);
          r4[8] = (v289_acc[0]);
          r4[9] = (v289_acc[1]);
          r4[10] = (v289_acc[2]);
          r4[11] = (v289_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v27_g) {
            int32_t v299_off = v26_lead + 6;
            #pragma unroll
            for (int32_t v294_i1 = 0; v294_i1 < 12; ++v294_i1) {
              float v296_data = r4[v294_i1];
              int32_t v301_a = v299_off + (v294_i1 * 12);
              s0[(v301_a ^ ((v301_a >> 4) & 15))] = v296_data;
            }
          }
          float r7[12]{};
          // r7 = load{g>r}(glb_m5);
          if (v27_g) {
            #pragma unroll
            for (int32_t v306_i1 = 0; v306_i1 < 12; ++v306_i1) {
              float v311_data = __builtin_nontemporal_load(&glb_m5[(v26_lead + (v306_i1 * 6))]);
              r7[v306_i1] = v311_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(s0 * r5) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v314_data = r5[0];
          float v315_data = r5[1];
          float v316_data = r5[2];
          float v317_data = r5[3];
          float v318_tp{};
          float v319_tp{};
          float v320_tp{};
          float v321_tp{};
          tensorforge::transpose4x4b32(v318_tp, v319_tp, v320_tp, v321_tp, v314_data, v315_data, v316_data, v317_data);
          tensorforge::VectorT<float, 4> v322_acc{};
          int32_t v327_sw = (v26_lead >> 4) & 15;
          float v329_data = s0[(v26_lead ^ v327_sw)];
          int32_t v330_a = v26_lead + 12;
          int32_t v331_sw = v330_a >> 4;
          float v334_data = s0[(v330_a ^ (v331_sw & 15))];
          int32_t v335_a = v26_lead + 24;
          int32_t v336_sw = v335_a >> 4;
          float v339_data = s0[(v335_a ^ (v336_sw & 15))];
          int32_t v340_a = v26_lead + 36;
          int32_t v341_sw = v340_a >> 4;
          float v344_data = s0[(v340_a ^ (v341_sw & 15))];
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v329_data, v322_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v334_data, v345_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v339_data, v346_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v344_data, v347_acc, 2, 0, 0);
          int32_t v349_a = v26_lead + 48;
          int32_t v350_sw = v349_a >> 4;
          float v353_data = s0[(v349_a ^ (v350_sw & 15))];
          int32_t v354_a = v26_lead + 60;
          int32_t v355_sw = v354_a >> 4;
          float v358_data = s0[(v354_a ^ (v355_sw & 15))];
          int32_t v359_a = v26_lead + 72;
          int32_t v360_sw = v359_a >> 4;
          float v363_data = s0[(v359_a ^ (v360_sw & 15))];
          int32_t v364_a = v26_lead + 84;
          int32_t v365_sw = v364_a >> 4;
          float v368_data = s0[(v364_a ^ (v365_sw & 15))];
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v353_data, v348_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v358_data, v369_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v363_data, v370_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v368_data, v371_acc, 2, 1, 0);
          int32_t v373_a = v26_lead + 96;
          int32_t v374_sw = v373_a >> 4;
          float v377_data = s0[(v373_a ^ (v374_sw & 15))];
          int32_t v378_a = v26_lead + 108;
          int32_t v379_sw = v378_a >> 4;
          float v382_data = s0[(v378_a ^ (v379_sw & 15))];
          int32_t v383_a = v26_lead + 120;
          int32_t v384_sw = v383_a >> 4;
          float v387_data = s0[(v383_a ^ (v384_sw & 15))];
          int32_t v388_a = v26_lead + 132;
          int32_t v389_sw = v388_a >> 4;
          float v392_data = s0[(v388_a ^ (v389_sw & 15))];
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v377_data, v372_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v382_data, v393_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v387_data, v394_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v392_data, v395_acc, 2, 2, 0);
          r6[0] = (v396_acc[0]);
          r6[1] = (v396_acc[1]);
          r6[2] = (v396_acc[2]);
          r6[3] = (v396_acc[3]);
          float v401_data = r5[4];
          float v402_data = r5[5];
          float v403_data = r5[6];
          float v404_data = r5[7];
          float v405_tp{};
          float v406_tp{};
          float v407_tp{};
          float v408_tp{};
          tensorforge::transpose4x4b32(v405_tp, v406_tp, v407_tp, v408_tp, v401_data, v402_data, v403_data, v404_data);
          tensorforge::VectorT<float, 4> v409_acc{};
          float v416_data = s0[(v26_lead ^ v327_sw)];
          float v421_data = s0[(v330_a ^ (v331_sw & 15))];
          float v426_data = s0[(v335_a ^ (v336_sw & 15))];
          float v431_data = s0[(v340_a ^ (v341_sw & 15))];
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v416_data, v409_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v421_data, v432_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v426_data, v433_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v431_data, v434_acc, 2, 0, 0);
          float v440_data = s0[(v349_a ^ (v350_sw & 15))];
          float v445_data = s0[(v354_a ^ (v355_sw & 15))];
          float v450_data = s0[(v359_a ^ (v360_sw & 15))];
          float v455_data = s0[(v364_a ^ (v365_sw & 15))];
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v440_data, v435_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v445_data, v456_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v450_data, v457_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v455_data, v458_acc, 2, 1, 0);
          float v464_data = s0[(v373_a ^ (v374_sw & 15))];
          float v469_data = s0[(v378_a ^ (v379_sw & 15))];
          float v474_data = s0[(v383_a ^ (v384_sw & 15))];
          float v479_data = s0[(v388_a ^ (v389_sw & 15))];
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v464_data, v459_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v469_data, v480_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v474_data, v481_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v479_data, v482_acc, 2, 2, 0);
          r6[4] = (v483_acc[0]);
          r6[5] = (v483_acc[1]);
          r6[6] = (v483_acc[2]);
          r6[7] = (v483_acc[3]);
          float v488_data = r5[8];
          float v489_data = r5[9];
          float v490_data = r5[10];
          float v491_data = r5[11];
          float v492_tp{};
          float v493_tp{};
          float v494_tp{};
          float v495_tp{};
          tensorforge::transpose4x4b32(v492_tp, v493_tp, v494_tp, v495_tp, v488_data, v489_data, v490_data, v491_data);
          tensorforge::VectorT<float, 4> v496_acc{};
          float v503_data = s0[(v26_lead ^ v327_sw)];
          float v508_data = s0[(v330_a ^ (v331_sw & 15))];
          float v513_data = s0[(v335_a ^ (v336_sw & 15))];
          float v518_data = s0[(v340_a ^ (v341_sw & 15))];
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v503_data, v496_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v493_tp, v508_data, v519_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v513_data, v520_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v518_data, v521_acc, 2, 0, 0);
          float v527_data = s0[(v349_a ^ (v350_sw & 15))];
          float v532_data = s0[(v354_a ^ (v355_sw & 15))];
          float v537_data = s0[(v359_a ^ (v360_sw & 15))];
          float v542_data = s0[(v364_a ^ (v365_sw & 15))];
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v527_data, v522_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v493_tp, v532_data, v543_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v537_data, v544_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v542_data, v545_acc, 2, 1, 0);
          float v551_data = s0[(v373_a ^ (v374_sw & 15))];
          float v556_data = s0[(v378_a ^ (v379_sw & 15))];
          float v561_data = s0[(v383_a ^ (v384_sw & 15))];
          float v566_data = s0[(v388_a ^ (v389_sw & 15))];
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v551_data, v546_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v493_tp, v556_data, v567_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v561_data, v568_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v566_data, v569_acc, 2, 2, 0);
          r6[8] = (v570_acc[0]);
          r6[9] = (v570_acc[1]);
          r6[10] = (v570_acc[2]);
          r6[11] = (v570_acc[3]);
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v36_g) {
            #pragma unroll
            for (int32_t v576_i1 = 0; v576_i1 < 12; ++v576_i1) {
              float v581_data = __builtin_nontemporal_load(&glb_m6[(v26_lead + (v576_i1 * 12))]);
              r8[v576_i1] = v581_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r10[12]{};
          // r10 = load{g>r}(glb_m7);
          if (v27_g) {
            #pragma unroll
            for (int32_t v584_i1 = 0; v584_i1 < 12; ++v584_i1) {
              float v589_data = __builtin_nontemporal_load(&glb_m7[(v26_lead + (v584_i1 * 6))]);
              r10[v584_i1] = v589_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(r7 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v592_data = r8[0];
          float v593_data = r8[1];
          float v594_data = r8[2];
          float v595_data = r8[3];
          float v596_tp{};
          float v597_tp{};
          float v598_tp{};
          float v599_tp{};
          tensorforge::transpose4x4b32(v596_tp, v597_tp, v598_tp, v599_tp, v592_data, v593_data, v594_data, v595_data);
          tensorforge::VectorT<float, 4> v600_acc{};
          float v601_data = r7[0];
          float v602_data = r7[1];
          float v603_data = r7[2];
          float v604_data = r7[3];
          tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v596_tp, v601_data, v600_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v597_tp, v602_data, v605_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v607_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v598_tp, v603_data, v606_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v604_data, v607_acc, 2, 0, 0);
          float v609_data = r7[4];
          float v610_data = r7[5];
          float v611_data = r7[6];
          float v612_data = r7[7];
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v596_tp, v609_data, v608_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v597_tp, v610_data, v613_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v598_tp, v611_data, v614_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v612_data, v615_acc, 2, 1, 0);
          float v617_data = r7[8];
          float v618_data = r7[9];
          float v619_data = r7[10];
          float v620_data = r7[11];
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v596_tp, v617_data, v616_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v597_tp, v618_data, v621_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v598_tp, v619_data, v622_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v620_data, v623_acc, 2, 2, 0);
          r9[0] = (v624_acc[0]);
          r9[1] = (v624_acc[1]);
          r9[2] = (v624_acc[2]);
          r9[3] = (v624_acc[3]);
          float v629_data = r8[4];
          float v630_data = r8[5];
          float v631_data = r8[6];
          float v632_data = r8[7];
          float v633_tp{};
          float v634_tp{};
          float v635_tp{};
          float v636_tp{};
          tensorforge::transpose4x4b32(v633_tp, v634_tp, v635_tp, v636_tp, v629_data, v630_data, v631_data, v632_data);
          tensorforge::VectorT<float, 4> v637_acc{};
          tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v601_data, v637_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v643_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v602_data, v642_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v644_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v635_tp, v603_data, v643_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v604_data, v644_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v609_data, v645_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v610_data, v650_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v635_tp, v611_data, v651_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v612_data, v652_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v617_data, v653_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v618_data, v658_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v635_tp, v619_data, v659_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v620_data, v660_acc, 2, 2, 0);
          r9[4] = (v661_acc[0]);
          r9[5] = (v661_acc[1]);
          r9[6] = (v661_acc[2]);
          r9[7] = (v661_acc[3]);
          float v666_data = r8[8];
          float v667_data = r8[9];
          float v668_data = r8[10];
          float v669_data = r8[11];
          float v670_tp{};
          float v671_tp{};
          float v672_tp{};
          float v673_tp{};
          tensorforge::transpose4x4b32(v670_tp, v671_tp, v672_tp, v673_tp, v666_data, v667_data, v668_data, v669_data);
          tensorforge::VectorT<float, 4> v674_acc{};
          tensorforge::VectorT<float, 4> v679_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v601_data, v674_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v680_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v602_data, v679_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v603_data, v680_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v604_data, v681_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v609_data, v682_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v610_data, v687_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v689_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v611_data, v688_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v612_data, v689_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v695_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v617_data, v690_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v618_data, v695_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v697_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v619_data, v696_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v620_data, v697_acc, 2, 2, 0);
          r9[8] = (v698_acc[0]);
          r9[9] = (v698_acc[1]);
          r9[10] = (v698_acc[2]);
          r9[11] = (v698_acc[3]);
          // s1 = store{r>s}(localShrMem0, r9);
          if (v27_g) {
            #pragma unroll
            for (int32_t v703_i1 = 0; v703_i1 < 12; ++v703_i1) {
              float v705_data = r9[v703_i1];
              int32_t v709_a = v26_lead + (v703_i1 * 12);
              s1[(v709_a ^ ((v709_a >> 4) & 15))] = v705_data;
            }
          }
          float r12[12]{};
          // r12 = load{g>r}(glb_m8);
          if (v36_g) {
            #pragma unroll
            for (int32_t v714_i1 = 0; v714_i1 < 12; ++v714_i1) {
              float v719_data = __builtin_nontemporal_load(&glb_m8[(v26_lead + (v714_i1 * 12))]);
              r12[v714_i1] = v719_data;
            }
          }
          // wait(r10 = load{g>r}(glb_m7););
          float r11[12]{};
          // r11 = +(r10 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v726_tp{};
          float v727_tp{};
          float v728_tp{};
          float v729_tp{};
          tensorforge::transpose4x4b32(v726_tp, v727_tp, v728_tp, v729_tp, v592_data, v593_data, v594_data, v595_data);
          tensorforge::VectorT<float, 4> v730_acc{};
          float v731_data = r10[0];
          float v732_data = r10[1];
          float v733_data = r10[2];
          float v734_data = r10[3];
          tensorforge::VectorT<float, 4> v735_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v726_tp, v731_data, v730_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v736_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v727_tp, v732_data, v735_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v737_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v728_tp, v733_data, v736_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v734_data, v737_acc, 2, 0, 0);
          float v739_data = r10[4];
          float v740_data = r10[5];
          float v741_data = r10[6];
          float v742_data = r10[7];
          tensorforge::VectorT<float, 4> v743_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v726_tp, v739_data, v738_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v744_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v727_tp, v740_data, v743_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v745_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v728_tp, v741_data, v744_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v742_data, v745_acc, 2, 1, 0);
          float v747_data = r10[8];
          float v748_data = r10[9];
          float v749_data = r10[10];
          float v750_data = r10[11];
          tensorforge::VectorT<float, 4> v751_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v726_tp, v747_data, v746_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v727_tp, v748_data, v751_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v753_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v728_tp, v749_data, v752_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v729_tp, v750_data, v753_acc, 2, 2, 0);
          r11[0] = (v754_acc[0]);
          r11[1] = (v754_acc[1]);
          r11[2] = (v754_acc[2]);
          r11[3] = (v754_acc[3]);
          float v763_tp{};
          float v764_tp{};
          float v765_tp{};
          float v766_tp{};
          tensorforge::transpose4x4b32(v763_tp, v764_tp, v765_tp, v766_tp, v629_data, v630_data, v631_data, v632_data);
          tensorforge::VectorT<float, 4> v767_acc{};
          tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v763_tp, v731_data, v767_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v732_data, v772_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v733_data, v773_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v734_data, v774_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v763_tp, v739_data, v775_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v740_data, v780_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v741_data, v781_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v742_data, v782_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v788_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v763_tp, v747_data, v783_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v748_data, v788_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v749_data, v789_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v750_data, v790_acc, 2, 2, 0);
          r11[4] = (v791_acc[0]);
          r11[5] = (v791_acc[1]);
          r11[6] = (v791_acc[2]);
          r11[7] = (v791_acc[3]);
          float v800_tp{};
          float v801_tp{};
          float v802_tp{};
          float v803_tp{};
          tensorforge::transpose4x4b32(v800_tp, v801_tp, v802_tp, v803_tp, v666_data, v667_data, v668_data, v669_data);
          tensorforge::VectorT<float, 4> v804_acc{};
          tensorforge::VectorT<float, 4> v809_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v800_tp, v731_data, v804_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v810_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v801_tp, v732_data, v809_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v811_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v802_tp, v733_data, v810_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v734_data, v811_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v817_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v800_tp, v739_data, v812_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v801_tp, v740_data, v817_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v819_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v802_tp, v741_data, v818_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v742_data, v819_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v825_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v800_tp, v747_data, v820_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v801_tp, v748_data, v825_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v827_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v802_tp, v749_data, v826_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v750_data, v827_acc, 2, 2, 0);
          r11[8] = (v828_acc[0]);
          r11[9] = (v828_acc[1]);
          r11[10] = (v828_acc[2]);
          r11[11] = (v828_acc[3]);
          // s1 = store{r>s}(localShrMem0, r11);
          if (v27_g) {
            int32_t v838_off = v26_lead + 6;
            #pragma unroll
            for (int32_t v833_i1 = 0; v833_i1 < 12; ++v833_i1) {
              float v835_data = r11[v833_i1];
              int32_t v840_a = v838_off + (v833_i1 * 12);
              s1[(v840_a ^ ((v840_a >> 4) & 15))] = v835_data;
            }
          }
          // wait(r12 = load{g>r}(glb_m8););
          float r13[12]{};
          // ir13 = +(s1 * r12)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir13[12]{};
          float v846_data = r12[0];
          float v847_data = r12[1];
          float v848_data = r12[2];
          float v849_data = r12[3];
          float v850_tp{};
          float v851_tp{};
          float v852_tp{};
          float v853_tp{};
          tensorforge::transpose4x4b32(v850_tp, v851_tp, v852_tp, v853_tp, v846_data, v847_data, v848_data, v849_data);
          tensorforge::VectorT<float, 4> v854_acc{};
          float v861_data = s1[(v26_lead ^ v327_sw)];
          float v866_data = s1[(v330_a ^ (v331_sw & 15))];
          float v871_data = s1[(v335_a ^ (v336_sw & 15))];
          float v876_data = s1[(v340_a ^ (v341_sw & 15))];
          tensorforge::VectorT<float, 4> v877_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v861_data, v854_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v866_data, v877_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v879_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v871_data, v878_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v876_data, v879_acc, 2, 0, 0);
          float v885_data = s1[(v349_a ^ (v350_sw & 15))];
          float v890_data = s1[(v354_a ^ (v355_sw & 15))];
          float v895_data = s1[(v359_a ^ (v360_sw & 15))];
          float v900_data = s1[(v364_a ^ (v365_sw & 15))];
          tensorforge::VectorT<float, 4> v901_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v885_data, v880_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v902_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v890_data, v901_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v903_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v895_data, v902_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v900_data, v903_acc, 2, 1, 0);
          float v909_data = s1[(v373_a ^ (v374_sw & 15))];
          float v914_data = s1[(v378_a ^ (v379_sw & 15))];
          float v919_data = s1[(v383_a ^ (v384_sw & 15))];
          float v924_data = s1[(v388_a ^ (v389_sw & 15))];
          tensorforge::VectorT<float, 4> v925_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v909_data, v904_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v926_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v914_data, v925_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v927_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v919_data, v926_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v928_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v924_data, v927_acc, 2, 2, 0);
          ir13[0] = (v928_acc[0]);
          ir13[1] = (v928_acc[1]);
          ir13[2] = (v928_acc[2]);
          ir13[3] = (v928_acc[3]);
          float v933_data = r12[4];
          float v934_data = r12[5];
          float v935_data = r12[6];
          float v936_data = r12[7];
          float v937_tp{};
          float v938_tp{};
          float v939_tp{};
          float v940_tp{};
          tensorforge::transpose4x4b32(v937_tp, v938_tp, v939_tp, v940_tp, v933_data, v934_data, v935_data, v936_data);
          tensorforge::VectorT<float, 4> v941_acc{};
          float v948_data = s1[(v26_lead ^ v327_sw)];
          float v953_data = s1[(v330_a ^ (v331_sw & 15))];
          float v958_data = s1[(v335_a ^ (v336_sw & 15))];
          float v963_data = s1[(v340_a ^ (v341_sw & 15))];
          tensorforge::VectorT<float, 4> v964_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v937_tp, v948_data, v941_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v938_tp, v953_data, v964_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v966_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v939_tp, v958_data, v965_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v963_data, v966_acc, 2, 0, 0);
          float v972_data = s1[(v349_a ^ (v350_sw & 15))];
          float v977_data = s1[(v354_a ^ (v355_sw & 15))];
          float v982_data = s1[(v359_a ^ (v360_sw & 15))];
          float v987_data = s1[(v364_a ^ (v365_sw & 15))];
          tensorforge::VectorT<float, 4> v988_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v937_tp, v972_data, v967_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v989_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v938_tp, v977_data, v988_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v990_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v939_tp, v982_data, v989_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v991_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v987_data, v990_acc, 2, 1, 0);
          float v996_data = s1[(v373_a ^ (v374_sw & 15))];
          float v1001_data = s1[(v378_a ^ (v379_sw & 15))];
          float v1006_data = s1[(v383_a ^ (v384_sw & 15))];
          float v1011_data = s1[(v388_a ^ (v389_sw & 15))];
          tensorforge::VectorT<float, 4> v1012_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v937_tp, v996_data, v991_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1013_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v938_tp, v1001_data, v1012_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1014_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v939_tp, v1006_data, v1013_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v940_tp, v1011_data, v1014_acc, 2, 2, 0);
          ir13[4] = (v1015_acc[0]);
          ir13[5] = (v1015_acc[1]);
          ir13[6] = (v1015_acc[2]);
          ir13[7] = (v1015_acc[3]);
          float v1020_data = r12[8];
          float v1021_data = r12[9];
          float v1022_data = r12[10];
          float v1023_data = r12[11];
          float v1024_tp{};
          float v1025_tp{};
          float v1026_tp{};
          float v1027_tp{};
          tensorforge::transpose4x4b32(v1024_tp, v1025_tp, v1026_tp, v1027_tp, v1020_data, v1021_data, v1022_data, v1023_data);
          tensorforge::VectorT<float, 4> v1028_acc{};
          float v1035_data = s1[(v26_lead ^ v327_sw)];
          float v1040_data = s1[(v330_a ^ (v331_sw & 15))];
          float v1045_data = s1[(v335_a ^ (v336_sw & 15))];
          float v1050_data = s1[(v340_a ^ (v341_sw & 15))];
          tensorforge::VectorT<float, 4> v1051_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1024_tp, v1035_data, v1028_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1052_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1025_tp, v1040_data, v1051_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1053_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1045_data, v1052_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1050_data, v1053_acc, 2, 0, 0);
          float v1059_data = s1[(v349_a ^ (v350_sw & 15))];
          float v1064_data = s1[(v354_a ^ (v355_sw & 15))];
          float v1069_data = s1[(v359_a ^ (v360_sw & 15))];
          float v1074_data = s1[(v364_a ^ (v365_sw & 15))];
          tensorforge::VectorT<float, 4> v1075_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1024_tp, v1059_data, v1054_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1076_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1025_tp, v1064_data, v1075_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1077_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1069_data, v1076_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1078_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1074_data, v1077_acc, 2, 1, 0);
          float v1083_data = s1[(v373_a ^ (v374_sw & 15))];
          float v1088_data = s1[(v378_a ^ (v379_sw & 15))];
          float v1093_data = s1[(v383_a ^ (v384_sw & 15))];
          float v1098_data = s1[(v388_a ^ (v389_sw & 15))];
          tensorforge::VectorT<float, 4> v1099_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1024_tp, v1083_data, v1078_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1025_tp, v1088_data, v1099_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1093_data, v1100_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1098_data, v1101_acc, 2, 2, 0);
          ir13[8] = (v1102_acc[0]);
          ir13[9] = (v1102_acc[1]);
          ir13[10] = (v1102_acc[2]);
          ir13[11] = (v1102_acc[3]);
          // r13 = ir13 + r6
          if (v36_g) {
            #pragma unroll
            for (int32_t v1107_n1 = 0; v1107_n1 < 12; ++v1107_n1) {
              float v1109_data = ir13[v1107_n1];
              float v1110_data = r6[v1107_n1];
              r13[v1107_n1] = (v1110_data + v1109_data);
            }
          }
          // glb_m3 = store{r>g}(r13);
          if (v36_g) {
            #pragma unroll
            for (int32_t v1112_i1 = 0; v1112_i1 < 12; ++v1112_i1) {
              float v1114_data = r13[v1112_i1];
              glb_m3[(v26_lead + (v1112_i1 * 12))] = v1114_data;
            }
          }
        }
      }
    }
  }
}

