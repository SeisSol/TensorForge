// === base name ===
kernel_6fd1f8e626b35c60

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6fd1f8e626b35c60 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6fd1f8e626b35c60(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6fd1f8e626b35c60(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, float * m9, size_t m9_extraOffset, const float * m10, size_t m10_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6fd1f8e626b35c60(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6fd1f8e626b35c60, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6fd1f8e626b35c60, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_6fd1f8e626b35c60(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, float * m9, size_t m9_extraOffset, const float * m10, size_t m10_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6fd1f8e626b35c60(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6fd1f8e626b35c60), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m9Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m9;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m10Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m10;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_6fd1f8e626b35c60, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, m9Arg, m9_extraOffset, m10Arg, m10_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6fd1f8e626b35c60(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m9, size_t m9_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m10, size_t m10_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×13(32×13) {0..32}×{0..13} strided
    //   m2 13×13(13×13) {0..13}×{0..13} strided
    //   m3 13×13(13×13) {0..13}×{0..13} strided
    //   m4 16×32(16×32) {0..16}×{0..32} strided
    //   m5 13×13(13×13) {0..13}×{0..13} strided
    //   m6 16×32(16×32) {0..16}×{0..32} strided
    //   m7 13×13(13×13) {0..13}×{0..13} strided
    //   m8 16×32(16×32) {0..16}×{0..32} strided
    //   m9 32×13(32×13) {0..32}×{0..13} strided
    //   m10 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   t0[i,j] = m1[i,k] × m3[k,j]
    //   m0[i,j]@{0..16}×{0..13} += m4[i,k] × t0[k,j]
    //   t1[i,j] = m1[i,k] × m5[k,j]
    //   m0[i,j]@{0..16}×{0..13} += m6[i,k] × t1[k,j]
    //   t2[i,j] = m1[i,k] × m7[k,j]
    //   m0[i,j]@{0..16}×{0..13} += m8[i,k] × t2[k,j]
    //   m9[i,j] = m0[i,k] × m10[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[13,13]],"name":"m3","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"W0","bbox":[[0,0],[16,32]],"name":"m4","ordered":false,"parts":1,"shape":[16,32],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[13,13]],"name":"m5","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"W1","bbox":[[0,0],[16,32]],"name":"m6","ordered":false,"parts":1,"shape":[16,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[13,13]],"name":"m7","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"W2","bbox":[[0,0],[16,32]],"name":"m8","ordered":false,"parts":1,"shape":[16,32],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m9","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m10","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[16,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,32]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,32]},{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[16,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,32]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[16,32]},{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[16,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,32]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[16,32]},{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m9","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m10","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 416 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 416 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 169 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 169 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 512 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v7_batchId0 * 169 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v7_batchId0 * 512 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v7_batchId0 * 169 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v7_batchId0 * 512 + 0 + m8_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m9 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m9[v7_batchId0 * 416 + 0 + m9_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m10 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m10[v7_batchId0 * 169 + 0 + m10_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v29_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
            int32_t v33_lead = v29_lead + (v30_i0 * 32);
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 13; ++v31_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m1[(v33_lead + (v31_i1 * 32))]);
              r0[(v30_i0 + v31_i1)] = v36_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          bool v39_g = v29_lead < 13;
          if (v39_g) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 13; ++v40_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m2[(v29_lead + (v40_i1 * 13))]);
              r1[v40_i1] = v45_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m3);
          if (v39_g) {
            #pragma unroll
            for (int32_t v257_i1 = 0; v257_i1 < 13; ++v257_i1) {
              float v262_data = __builtin_nontemporal_load(&glb_m3[(v29_lead + (v257_i1 * 13))]);
              r3[v257_i1] = v262_data;
            }
          }
          float r2[13]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v48_data = r1[0];
          float v49_data = r1[1];
          float v50_data = r1[2];
          float v51_data = r1[3];
          float v52_data = r1[4];
          float v53_data = r1[5];
          float v54_data = r1[6];
          float v55_data = r1[7];
          float v56_data = r1[8];
          float v57_data = r1[9];
          float v58_data = r1[10];
          float v59_data = r1[11];
          float v60_data = r1[12];
          float v61_pad{};
          float v62_pad{};
          float v63_pad{};
          tensorforge::transpose16x16b32(v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_pad, v62_pad, v63_pad);
          tensorforge::VectorT<float, 16> v64_acc{};
          float v65_data = r0[0];
          float v66_data = r0[1];
          float v67_data = r0[2];
          float v68_data = r0[3];
          float v69_data = r0[4];
          float v70_data = r0[5];
          float v71_data = r0[6];
          float v72_data = r0[7];
          float v73_data = r0[8];
          float v74_data = r0[9];
          float v75_data = r0[10];
          float v76_data = r0[11];
          float v77_data = r0[12];
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v64_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v90_acc, 1, 0, 0);
          float v92_el = v91_acc[0];
          float v94_el = v91_acc[4];
          float v95_sw = tensorforge::swap<32>(v94_el);
          float v97_el = v91_acc[8];
          float v100_el = v91_acc[12];
          float v101_sw = tensorforge::swap<32>(v100_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v101_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v95_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v92_el, v92_el))))))));
          float v104_el = v91_acc[1];
          float v106_el = v91_acc[5];
          float v107_sw = tensorforge::swap<32>(v106_el);
          float v109_el = v91_acc[9];
          float v112_el = v91_acc[13];
          float v113_sw = tensorforge::swap<32>(v112_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v113_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v107_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v104_el, v104_el))))))));
          float v116_el = v91_acc[2];
          float v118_el = v91_acc[6];
          float v119_sw = tensorforge::swap<32>(v118_el);
          float v121_el = v91_acc[10];
          float v124_el = v91_acc[14];
          float v125_sw = tensorforge::swap<32>(v124_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v125_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v119_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v116_el, v116_el))))))));
          float v128_el = v91_acc[3];
          float v130_el = v91_acc[7];
          float v131_sw = tensorforge::swap<32>(v130_el);
          float v133_el = v91_acc[11];
          float v136_el = v91_acc[15];
          float v137_sw = tensorforge::swap<32>(v136_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v137_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v133_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v131_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v128_el, v128_el))))))));
          float v141_sw = tensorforge::swap<32>(v92_el);
          float v146_sw = tensorforge::swap<32>(v97_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v146_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v94_el, (tensorforge::dppUpdate<228, 1, 15, false>(v141_sw, v141_sw))))))));
          float v153_sw = tensorforge::swap<32>(v104_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v109_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v106_el, (tensorforge::dppUpdate<228, 1, 15, false>(v153_sw, v153_sw))))))));
          float v165_sw = tensorforge::swap<32>(v116_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v121_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v118_el, (tensorforge::dppUpdate<228, 1, 15, false>(v165_sw, v165_sw))))))));
          float v177_sw = tensorforge::swap<32>(v128_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v136_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v133_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v130_el, (tensorforge::dppUpdate<228, 1, 15, false>(v177_sw, v177_sw))))))));
          float v189_sw = tensorforge::swap<64>(v92_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v101_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v97_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v95_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v189_sw, v189_sw))))))));
          float v201_sw = tensorforge::swap<64>(v104_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v113_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v109_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v107_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v201_sw, v201_sw))))))));
          float v213_sw = tensorforge::swap<64>(v116_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v125_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v121_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v119_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v213_sw, v213_sw))))))));
          float v225_sw = tensorforge::swap<64>(v128_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v137_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v133_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v131_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v225_sw, v225_sw))))))));
          float v238_sw = tensorforge::swap<64>(v141_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v100_el, (tensorforge::dppUpdate<228, 4, 15, false>(v146_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v94_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v238_sw, v238_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v248_i0 = 0; v248_i0 < 1; ++v248_i0) {
            int32_t v253_lead = v29_lead + (v248_i0 * 32);
            #pragma unroll
            for (int32_t v249_i1 = 0; v249_i1 < 13; ++v249_i1) {
              float v251_data = r2[(v248_i0 + v249_i1)];
              glb_m0[(v253_lead + (v249_i1 * 32))] = v251_data;
            }
          }
          float r5[32]{};
          // r5 = load{g>r}(glb_m4);
          bool v466_g = v29_lead < 16;
          if (v466_g) {
            #pragma unroll
            for (int32_t v467_i1 = 0; v467_i1 < 32; ++v467_i1) {
              float v472_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v467_i1 * 16))]);
              r5[v467_i1] = v472_data;
            }
          }
          float r4[13]{};
          // r4 = +(r0 * r3) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v265_data = r3[0];
          float v266_data = r3[1];
          float v267_data = r3[2];
          float v268_data = r3[3];
          float v269_data = r3[4];
          float v270_data = r3[5];
          float v271_data = r3[6];
          float v272_data = r3[7];
          float v273_data = r3[8];
          float v274_data = r3[9];
          float v275_data = r3[10];
          float v276_data = r3[11];
          float v277_data = r3[12];
          float v278_pad{};
          float v279_pad{};
          float v280_pad{};
          tensorforge::transpose16x16b32(v265_data, v266_data, v267_data, v268_data, v269_data, v270_data, v271_data, v272_data, v273_data, v274_data, v275_data, v276_data, v277_data, v278_pad, v279_pad, v280_pad);
          tensorforge::VectorT<float, 16> v281_acc{};
          tensorforge::VectorT<float, 16> v296_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v265_data, v65_data, v281_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v297_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v266_data, v66_data, v296_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v298_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v267_data, v67_data, v297_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v299_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v268_data, v68_data, v298_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v300_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v269_data, v69_data, v299_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v301_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v270_data, v70_data, v300_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v302_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v271_data, v71_data, v301_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v303_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v272_data, v72_data, v302_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v304_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v273_data, v73_data, v303_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v305_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v274_data, v74_data, v304_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v306_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v275_data, v75_data, v305_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v307_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v276_data, v76_data, v306_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v308_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v277_data, v77_data, v307_acc, 1, 0, 0);
          float v309_el = v308_acc[0];
          float v311_el = v308_acc[4];
          float v312_sw = tensorforge::swap<32>(v311_el);
          float v314_el = v308_acc[8];
          float v317_el = v308_acc[12];
          float v318_sw = tensorforge::swap<32>(v317_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v318_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v314_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v312_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v309_el, v309_el))))))));
          float v321_el = v308_acc[1];
          float v323_el = v308_acc[5];
          float v324_sw = tensorforge::swap<32>(v323_el);
          float v326_el = v308_acc[9];
          float v329_el = v308_acc[13];
          float v330_sw = tensorforge::swap<32>(v329_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v330_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v326_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v324_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v321_el, v321_el))))))));
          float v333_el = v308_acc[2];
          float v335_el = v308_acc[6];
          float v336_sw = tensorforge::swap<32>(v335_el);
          float v338_el = v308_acc[10];
          float v341_el = v308_acc[14];
          float v342_sw = tensorforge::swap<32>(v341_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v342_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v338_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v336_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v333_el, v333_el))))))));
          float v345_el = v308_acc[3];
          float v347_el = v308_acc[7];
          float v348_sw = tensorforge::swap<32>(v347_el);
          float v350_el = v308_acc[11];
          float v353_el = v308_acc[15];
          float v354_sw = tensorforge::swap<32>(v353_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v354_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v350_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v348_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v345_el, v345_el))))))));
          float v358_sw = tensorforge::swap<32>(v309_el);
          float v363_sw = tensorforge::swap<32>(v314_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v317_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v363_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v311_el, (tensorforge::dppUpdate<228, 1, 15, false>(v358_sw, v358_sw))))))));
          float v370_sw = tensorforge::swap<32>(v321_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v329_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v326_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v323_el, (tensorforge::dppUpdate<228, 1, 15, false>(v370_sw, v370_sw))))))));
          float v382_sw = tensorforge::swap<32>(v333_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v341_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v338_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v335_el, (tensorforge::dppUpdate<228, 1, 15, false>(v382_sw, v382_sw))))))));
          float v394_sw = tensorforge::swap<32>(v345_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v353_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v350_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v347_el, (tensorforge::dppUpdate<228, 1, 15, false>(v394_sw, v394_sw))))))));
          float v406_sw = tensorforge::swap<64>(v309_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v318_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v314_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v312_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v406_sw, v406_sw))))))));
          float v418_sw = tensorforge::swap<64>(v321_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v330_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v326_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v324_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v418_sw, v418_sw))))))));
          float v430_sw = tensorforge::swap<64>(v333_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v342_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v338_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v336_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v430_sw, v430_sw))))))));
          float v442_sw = tensorforge::swap<64>(v345_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v354_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v350_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v348_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v442_sw, v442_sw))))))));
          float v455_sw = tensorforge::swap<64>(v358_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v317_el, (tensorforge::dppUpdate<228, 4, 15, false>(v363_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v311_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v455_sw, v455_sw))))))));
          float r7[13]{};
          // r7 = load{g>r}(glb_m5);
          if (v39_g) {
            #pragma unroll
            for (int32_t v720_i1 = 0; v720_i1 < 13; ++v720_i1) {
              float v725_data = __builtin_nontemporal_load(&glb_m5[(v29_lead + (v720_i1 * 13))]);
              r7[v720_i1] = v725_data;
            }
          }
          float r6[13]{};
          // r6 = +(r5 * r4) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v475_data = r4[0];
          float v476_data = r4[1];
          float v477_data = r4[2];
          float v478_data = r4[3];
          float v479_data = r4[4];
          float v480_data = r4[5];
          float v481_data = r4[6];
          float v482_data = r4[7];
          float v483_data = r4[8];
          float v484_data = r4[9];
          float v485_data = r4[10];
          float v486_data = r4[11];
          float v487_data = r4[12];
          float v488_pad{};
          float v489_pad{};
          float v490_pad{};
          tensorforge::transpose16x16b32(v475_data, v476_data, v477_data, v478_data, v479_data, v480_data, v481_data, v482_data, v483_data, v484_data, v485_data, v486_data, v487_data, v488_pad, v489_pad, v490_pad);
          tensorforge::VectorT<float, 16> v491_acc{};
          float v492_data = r5[0];
          float v493_data = r5[1];
          float v494_data = r5[2];
          float v495_data = r5[3];
          float v496_data = r5[4];
          float v497_data = r5[5];
          float v498_data = r5[6];
          float v499_data = r5[7];
          float v500_data = r5[8];
          float v501_data = r5[9];
          float v502_data = r5[10];
          float v503_data = r5[11];
          float v504_data = r5[12];
          float v505_data = r5[13];
          float v506_data = r5[14];
          float v507_data = r5[15];
          tensorforge::VectorT<float, 16> v508_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v475_data, v492_data, v491_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v509_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v476_data, v493_data, v508_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v510_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v477_data, v494_data, v509_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v511_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v478_data, v495_data, v510_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v512_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v479_data, v496_data, v511_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v513_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v480_data, v497_data, v512_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v514_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v481_data, v498_data, v513_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v515_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v482_data, v499_data, v514_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v516_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v483_data, v500_data, v515_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v517_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v484_data, v501_data, v516_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v518_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v485_data, v502_data, v517_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v519_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v486_data, v503_data, v518_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v520_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v487_data, v504_data, v519_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v521_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v488_pad, v505_data, v520_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v522_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v489_pad, v506_data, v521_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v523_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v490_pad, v507_data, v522_acc, 1, 0, 0);
          float v524_data = r5[16];
          float v525_data = r5[17];
          float v526_data = r5[18];
          float v527_data = r5[19];
          float v528_data = r5[20];
          float v529_data = r5[21];
          float v530_data = r5[22];
          float v531_data = r5[23];
          float v532_data = r5[24];
          float v533_data = r5[25];
          float v534_data = r5[26];
          float v535_data = r5[27];
          float v536_data = r5[28];
          float v537_data = r5[29];
          float v538_data = r5[30];
          float v539_data = r5[31];
          tensorforge::VectorT<float, 16> v540_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v475_data, v524_data, v523_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v541_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v476_data, v525_data, v540_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v542_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v477_data, v526_data, v541_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v543_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v478_data, v527_data, v542_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v544_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v479_data, v528_data, v543_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v545_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v480_data, v529_data, v544_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v546_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v481_data, v530_data, v545_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v547_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v482_data, v531_data, v546_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v548_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v483_data, v532_data, v547_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v549_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v484_data, v533_data, v548_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v550_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v485_data, v534_data, v549_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v551_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v486_data, v535_data, v550_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v552_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v487_data, v536_data, v551_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v553_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v488_pad, v537_data, v552_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v554_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v489_pad, v538_data, v553_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v555_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v490_pad, v539_data, v554_acc, 1, 1, 0);
          float v556_el = v555_acc[0];
          float v558_el = v555_acc[4];
          float v559_sw = tensorforge::swap<32>(v558_el);
          float v561_el = v555_acc[8];
          float v564_el = v555_acc[12];
          float v565_sw = tensorforge::swap<32>(v564_el);
          r6[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v565_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v561_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v559_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v556_el, v556_el))))))));
          float v568_el = v555_acc[1];
          float v570_el = v555_acc[5];
          float v571_sw = tensorforge::swap<32>(v570_el);
          float v573_el = v555_acc[9];
          float v576_el = v555_acc[13];
          float v577_sw = tensorforge::swap<32>(v576_el);
          r6[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v577_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v573_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v571_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v568_el, v568_el))))))));
          float v580_el = v555_acc[2];
          float v582_el = v555_acc[6];
          float v583_sw = tensorforge::swap<32>(v582_el);
          float v585_el = v555_acc[10];
          float v588_el = v555_acc[14];
          float v589_sw = tensorforge::swap<32>(v588_el);
          r6[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v589_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v585_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v583_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v580_el, v580_el))))))));
          float v592_el = v555_acc[3];
          float v594_el = v555_acc[7];
          float v595_sw = tensorforge::swap<32>(v594_el);
          float v597_el = v555_acc[11];
          float v600_el = v555_acc[15];
          float v601_sw = tensorforge::swap<32>(v600_el);
          r6[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v601_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v597_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v595_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v592_el, v592_el))))))));
          float v605_sw = tensorforge::swap<32>(v556_el);
          float v610_sw = tensorforge::swap<32>(v561_el);
          r6[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v564_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v610_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v558_el, (tensorforge::dppUpdate<228, 1, 15, false>(v605_sw, v605_sw))))))));
          float v617_sw = tensorforge::swap<32>(v568_el);
          r6[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v576_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v573_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v570_el, (tensorforge::dppUpdate<228, 1, 15, false>(v617_sw, v617_sw))))))));
          float v629_sw = tensorforge::swap<32>(v580_el);
          r6[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v588_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v585_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v582_el, (tensorforge::dppUpdate<228, 1, 15, false>(v629_sw, v629_sw))))))));
          float v641_sw = tensorforge::swap<32>(v592_el);
          r6[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v600_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v597_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v594_el, (tensorforge::dppUpdate<228, 1, 15, false>(v641_sw, v641_sw))))))));
          float v653_sw = tensorforge::swap<64>(v556_el);
          r6[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v565_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v561_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v559_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v653_sw, v653_sw))))))));
          float v665_sw = tensorforge::swap<64>(v568_el);
          r6[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v577_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v573_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v571_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v665_sw, v665_sw))))))));
          float v677_sw = tensorforge::swap<64>(v580_el);
          r6[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v589_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v585_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v583_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v677_sw, v677_sw))))))));
          float v689_sw = tensorforge::swap<64>(v592_el);
          r6[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v601_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v597_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v595_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v689_sw, v689_sw))))))));
          float v702_sw = tensorforge::swap<64>(v605_sw);
          r6[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v564_el, (tensorforge::dppUpdate<228, 4, 15, false>(v610_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v558_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v702_sw, v702_sw))))))));
          // glb_m0 = store{r>g}(r6);
          if (v466_g) {
            #pragma unroll
            for (int32_t v712_i1 = 0; v712_i1 < 13; ++v712_i1) {
              float v714_data = r6[v712_i1];
              int32_t v718_a = v29_lead + (v712_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v718_a], v714_data);
            }
          }
          float r9[32]{};
          // r9 = load{g>r}(glb_m6);
          if (v466_g) {
            #pragma unroll
            for (int32_t v929_i1 = 0; v929_i1 < 32; ++v929_i1) {
              float v934_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v929_i1 * 16))]);
              r9[v929_i1] = v934_data;
            }
          }
          float r8[13]{};
          // r8 = +(r0 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v728_data = r7[0];
          float v729_data = r7[1];
          float v730_data = r7[2];
          float v731_data = r7[3];
          float v732_data = r7[4];
          float v733_data = r7[5];
          float v734_data = r7[6];
          float v735_data = r7[7];
          float v736_data = r7[8];
          float v737_data = r7[9];
          float v738_data = r7[10];
          float v739_data = r7[11];
          float v740_data = r7[12];
          float v741_pad{};
          float v742_pad{};
          float v743_pad{};
          tensorforge::transpose16x16b32(v728_data, v729_data, v730_data, v731_data, v732_data, v733_data, v734_data, v735_data, v736_data, v737_data, v738_data, v739_data, v740_data, v741_pad, v742_pad, v743_pad);
          tensorforge::VectorT<float, 16> v744_acc{};
          tensorforge::VectorT<float, 16> v759_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v728_data, v65_data, v744_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v760_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v729_data, v66_data, v759_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v761_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v730_data, v67_data, v760_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v762_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v731_data, v68_data, v761_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v763_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v732_data, v69_data, v762_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v764_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v733_data, v70_data, v763_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v765_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v734_data, v71_data, v764_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v766_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v735_data, v72_data, v765_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v767_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v736_data, v73_data, v766_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v768_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v737_data, v74_data, v767_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v769_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v738_data, v75_data, v768_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v770_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v739_data, v76_data, v769_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v771_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v740_data, v77_data, v770_acc, 1, 0, 0);
          float v772_el = v771_acc[0];
          float v774_el = v771_acc[4];
          float v775_sw = tensorforge::swap<32>(v774_el);
          float v777_el = v771_acc[8];
          float v780_el = v771_acc[12];
          float v781_sw = tensorforge::swap<32>(v780_el);
          r8[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v781_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v777_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v775_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v772_el, v772_el))))))));
          float v784_el = v771_acc[1];
          float v786_el = v771_acc[5];
          float v787_sw = tensorforge::swap<32>(v786_el);
          float v789_el = v771_acc[9];
          float v792_el = v771_acc[13];
          float v793_sw = tensorforge::swap<32>(v792_el);
          r8[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v793_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v789_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v787_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v784_el, v784_el))))))));
          float v796_el = v771_acc[2];
          float v798_el = v771_acc[6];
          float v799_sw = tensorforge::swap<32>(v798_el);
          float v801_el = v771_acc[10];
          float v804_el = v771_acc[14];
          float v805_sw = tensorforge::swap<32>(v804_el);
          r8[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v805_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v801_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v799_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v796_el, v796_el))))))));
          float v808_el = v771_acc[3];
          float v810_el = v771_acc[7];
          float v811_sw = tensorforge::swap<32>(v810_el);
          float v813_el = v771_acc[11];
          float v816_el = v771_acc[15];
          float v817_sw = tensorforge::swap<32>(v816_el);
          r8[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v817_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v813_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v811_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v808_el, v808_el))))))));
          float v821_sw = tensorforge::swap<32>(v772_el);
          float v826_sw = tensorforge::swap<32>(v777_el);
          r8[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v780_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v826_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v774_el, (tensorforge::dppUpdate<228, 1, 15, false>(v821_sw, v821_sw))))))));
          float v833_sw = tensorforge::swap<32>(v784_el);
          r8[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v792_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v789_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v786_el, (tensorforge::dppUpdate<228, 1, 15, false>(v833_sw, v833_sw))))))));
          float v845_sw = tensorforge::swap<32>(v796_el);
          r8[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v804_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v801_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v798_el, (tensorforge::dppUpdate<228, 1, 15, false>(v845_sw, v845_sw))))))));
          float v857_sw = tensorforge::swap<32>(v808_el);
          r8[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v816_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v813_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v810_el, (tensorforge::dppUpdate<228, 1, 15, false>(v857_sw, v857_sw))))))));
          float v869_sw = tensorforge::swap<64>(v772_el);
          r8[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v781_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v777_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v775_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v869_sw, v869_sw))))))));
          float v881_sw = tensorforge::swap<64>(v784_el);
          r8[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v793_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v789_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v787_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v881_sw, v881_sw))))))));
          float v893_sw = tensorforge::swap<64>(v796_el);
          r8[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v805_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v801_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v799_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v893_sw, v893_sw))))))));
          float v905_sw = tensorforge::swap<64>(v808_el);
          r8[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v817_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v813_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v811_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v905_sw, v905_sw))))))));
          float v918_sw = tensorforge::swap<64>(v821_sw);
          r8[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v780_el, (tensorforge::dppUpdate<228, 4, 15, false>(v826_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v774_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v918_sw, v918_sw))))))));
          float r11[13]{};
          // r11 = load{g>r}(glb_m7);
          if (v39_g) {
            #pragma unroll
            for (int32_t v1182_i1 = 0; v1182_i1 < 13; ++v1182_i1) {
              float v1187_data = __builtin_nontemporal_load(&glb_m7[(v29_lead + (v1182_i1 * 13))]);
              r11[v1182_i1] = v1187_data;
            }
          }
          float r10[13]{};
          // r10 = +(r9 * r8) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v937_data = r8[0];
          float v938_data = r8[1];
          float v939_data = r8[2];
          float v940_data = r8[3];
          float v941_data = r8[4];
          float v942_data = r8[5];
          float v943_data = r8[6];
          float v944_data = r8[7];
          float v945_data = r8[8];
          float v946_data = r8[9];
          float v947_data = r8[10];
          float v948_data = r8[11];
          float v949_data = r8[12];
          float v950_pad{};
          float v951_pad{};
          float v952_pad{};
          tensorforge::transpose16x16b32(v937_data, v938_data, v939_data, v940_data, v941_data, v942_data, v943_data, v944_data, v945_data, v946_data, v947_data, v948_data, v949_data, v950_pad, v951_pad, v952_pad);
          tensorforge::VectorT<float, 16> v953_acc{};
          float v954_data = r9[0];
          float v955_data = r9[1];
          float v956_data = r9[2];
          float v957_data = r9[3];
          float v958_data = r9[4];
          float v959_data = r9[5];
          float v960_data = r9[6];
          float v961_data = r9[7];
          float v962_data = r9[8];
          float v963_data = r9[9];
          float v964_data = r9[10];
          float v965_data = r9[11];
          float v966_data = r9[12];
          float v967_data = r9[13];
          float v968_data = r9[14];
          float v969_data = r9[15];
          tensorforge::VectorT<float, 16> v970_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v937_data, v954_data, v953_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v971_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v938_data, v955_data, v970_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v972_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v939_data, v956_data, v971_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v973_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v940_data, v957_data, v972_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v974_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v941_data, v958_data, v973_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v975_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v942_data, v959_data, v974_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v976_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v943_data, v960_data, v975_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v977_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v944_data, v961_data, v976_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v978_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v945_data, v962_data, v977_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v979_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v946_data, v963_data, v978_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v980_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v947_data, v964_data, v979_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v981_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v948_data, v965_data, v980_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v982_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v949_data, v966_data, v981_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v983_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v950_pad, v967_data, v982_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v984_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v951_pad, v968_data, v983_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v985_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v952_pad, v969_data, v984_acc, 1, 0, 0);
          float v986_data = r9[16];
          float v987_data = r9[17];
          float v988_data = r9[18];
          float v989_data = r9[19];
          float v990_data = r9[20];
          float v991_data = r9[21];
          float v992_data = r9[22];
          float v993_data = r9[23];
          float v994_data = r9[24];
          float v995_data = r9[25];
          float v996_data = r9[26];
          float v997_data = r9[27];
          float v998_data = r9[28];
          float v999_data = r9[29];
          float v1000_data = r9[30];
          float v1001_data = r9[31];
          tensorforge::VectorT<float, 16> v1002_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v937_data, v986_data, v985_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1003_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v938_data, v987_data, v1002_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1004_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v939_data, v988_data, v1003_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1005_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v940_data, v989_data, v1004_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1006_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v941_data, v990_data, v1005_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1007_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v942_data, v991_data, v1006_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1008_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v943_data, v992_data, v1007_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1009_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v944_data, v993_data, v1008_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1010_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v945_data, v994_data, v1009_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1011_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v946_data, v995_data, v1010_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1012_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v947_data, v996_data, v1011_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1013_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v948_data, v997_data, v1012_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1014_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v949_data, v998_data, v1013_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1015_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v950_pad, v999_data, v1014_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1016_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v951_pad, v1000_data, v1015_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1017_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v952_pad, v1001_data, v1016_acc, 1, 1, 0);
          float v1018_el = v1017_acc[0];
          float v1020_el = v1017_acc[4];
          float v1021_sw = tensorforge::swap<32>(v1020_el);
          float v1023_el = v1017_acc[8];
          float v1026_el = v1017_acc[12];
          float v1027_sw = tensorforge::swap<32>(v1026_el);
          r10[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1027_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1023_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1021_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1018_el, v1018_el))))))));
          float v1030_el = v1017_acc[1];
          float v1032_el = v1017_acc[5];
          float v1033_sw = tensorforge::swap<32>(v1032_el);
          float v1035_el = v1017_acc[9];
          float v1038_el = v1017_acc[13];
          float v1039_sw = tensorforge::swap<32>(v1038_el);
          r10[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1039_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1035_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1033_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1030_el, v1030_el))))))));
          float v1042_el = v1017_acc[2];
          float v1044_el = v1017_acc[6];
          float v1045_sw = tensorforge::swap<32>(v1044_el);
          float v1047_el = v1017_acc[10];
          float v1050_el = v1017_acc[14];
          float v1051_sw = tensorforge::swap<32>(v1050_el);
          r10[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1051_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1047_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1045_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1042_el, v1042_el))))))));
          float v1054_el = v1017_acc[3];
          float v1056_el = v1017_acc[7];
          float v1057_sw = tensorforge::swap<32>(v1056_el);
          float v1059_el = v1017_acc[11];
          float v1062_el = v1017_acc[15];
          float v1063_sw = tensorforge::swap<32>(v1062_el);
          r10[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1063_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1059_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1057_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1054_el, v1054_el))))))));
          float v1067_sw = tensorforge::swap<32>(v1018_el);
          float v1072_sw = tensorforge::swap<32>(v1023_el);
          r10[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1026_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1072_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1020_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1067_sw, v1067_sw))))))));
          float v1079_sw = tensorforge::swap<32>(v1030_el);
          r10[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1038_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1035_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1032_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1079_sw, v1079_sw))))))));
          float v1091_sw = tensorforge::swap<32>(v1042_el);
          r10[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1050_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1047_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1044_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1091_sw, v1091_sw))))))));
          float v1103_sw = tensorforge::swap<32>(v1054_el);
          r10[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1062_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1059_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1056_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1103_sw, v1103_sw))))))));
          float v1115_sw = tensorforge::swap<64>(v1018_el);
          r10[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1027_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1023_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1021_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1115_sw, v1115_sw))))))));
          float v1127_sw = tensorforge::swap<64>(v1030_el);
          r10[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1039_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1035_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1033_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1127_sw, v1127_sw))))))));
          float v1139_sw = tensorforge::swap<64>(v1042_el);
          r10[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1051_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1047_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1045_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1139_sw, v1139_sw))))))));
          float v1151_sw = tensorforge::swap<64>(v1054_el);
          r10[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1063_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1059_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1057_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1151_sw, v1151_sw))))))));
          float v1164_sw = tensorforge::swap<64>(v1067_sw);
          r10[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1026_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1072_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1020_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1164_sw, v1164_sw))))))));
          // glb_m0 = store{r>g}(r10);
          if (v466_g) {
            #pragma unroll
            for (int32_t v1174_i1 = 0; v1174_i1 < 13; ++v1174_i1) {
              float v1176_data = r10[v1174_i1];
              int32_t v1180_a = v29_lead + (v1174_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1180_a], v1176_data);
            }
          }
          float r13[32]{};
          // r13 = load{g>r}(glb_m8);
          if (v466_g) {
            #pragma unroll
            for (int32_t v1391_i1 = 0; v1391_i1 < 32; ++v1391_i1) {
              float v1396_data = __builtin_nontemporal_load(&glb_m8[(v29_lead + (v1391_i1 * 16))]);
              r13[v1391_i1] = v1396_data;
            }
          }
          float r12[13]{};
          // r12 = +(r0 * r11) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1190_data = r11[0];
          float v1191_data = r11[1];
          float v1192_data = r11[2];
          float v1193_data = r11[3];
          float v1194_data = r11[4];
          float v1195_data = r11[5];
          float v1196_data = r11[6];
          float v1197_data = r11[7];
          float v1198_data = r11[8];
          float v1199_data = r11[9];
          float v1200_data = r11[10];
          float v1201_data = r11[11];
          float v1202_data = r11[12];
          float v1203_pad{};
          float v1204_pad{};
          float v1205_pad{};
          tensorforge::transpose16x16b32(v1190_data, v1191_data, v1192_data, v1193_data, v1194_data, v1195_data, v1196_data, v1197_data, v1198_data, v1199_data, v1200_data, v1201_data, v1202_data, v1203_pad, v1204_pad, v1205_pad);
          tensorforge::VectorT<float, 16> v1206_acc{};
          tensorforge::VectorT<float, 16> v1221_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1190_data, v65_data, v1206_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1222_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1191_data, v66_data, v1221_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1223_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1192_data, v67_data, v1222_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1224_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1193_data, v68_data, v1223_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1225_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1194_data, v69_data, v1224_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1226_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1195_data, v70_data, v1225_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1227_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1196_data, v71_data, v1226_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1228_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1197_data, v72_data, v1227_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1229_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1198_data, v73_data, v1228_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1230_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1199_data, v74_data, v1229_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1231_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1200_data, v75_data, v1230_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1232_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1201_data, v76_data, v1231_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1233_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1202_data, v77_data, v1232_acc, 1, 0, 0);
          float v1234_el = v1233_acc[0];
          float v1236_el = v1233_acc[4];
          float v1237_sw = tensorforge::swap<32>(v1236_el);
          float v1239_el = v1233_acc[8];
          float v1242_el = v1233_acc[12];
          float v1243_sw = tensorforge::swap<32>(v1242_el);
          r12[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1243_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1239_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1237_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1234_el, v1234_el))))))));
          float v1246_el = v1233_acc[1];
          float v1248_el = v1233_acc[5];
          float v1249_sw = tensorforge::swap<32>(v1248_el);
          float v1251_el = v1233_acc[9];
          float v1254_el = v1233_acc[13];
          float v1255_sw = tensorforge::swap<32>(v1254_el);
          r12[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1255_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1251_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1249_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1246_el, v1246_el))))))));
          float v1258_el = v1233_acc[2];
          float v1260_el = v1233_acc[6];
          float v1261_sw = tensorforge::swap<32>(v1260_el);
          float v1263_el = v1233_acc[10];
          float v1266_el = v1233_acc[14];
          float v1267_sw = tensorforge::swap<32>(v1266_el);
          r12[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1267_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1263_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1261_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1258_el, v1258_el))))))));
          float v1270_el = v1233_acc[3];
          float v1272_el = v1233_acc[7];
          float v1273_sw = tensorforge::swap<32>(v1272_el);
          float v1275_el = v1233_acc[11];
          float v1278_el = v1233_acc[15];
          float v1279_sw = tensorforge::swap<32>(v1278_el);
          r12[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1279_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1275_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1273_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1270_el, v1270_el))))))));
          float v1283_sw = tensorforge::swap<32>(v1234_el);
          float v1288_sw = tensorforge::swap<32>(v1239_el);
          r12[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1242_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1288_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1236_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1283_sw, v1283_sw))))))));
          float v1295_sw = tensorforge::swap<32>(v1246_el);
          r12[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1254_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1251_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1248_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1295_sw, v1295_sw))))))));
          float v1307_sw = tensorforge::swap<32>(v1258_el);
          r12[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1266_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1263_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1260_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1307_sw, v1307_sw))))))));
          float v1319_sw = tensorforge::swap<32>(v1270_el);
          r12[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1278_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1275_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1272_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1319_sw, v1319_sw))))))));
          float v1331_sw = tensorforge::swap<64>(v1234_el);
          r12[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1243_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1239_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1237_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1331_sw, v1331_sw))))))));
          float v1343_sw = tensorforge::swap<64>(v1246_el);
          r12[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1255_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1251_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1249_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1343_sw, v1343_sw))))))));
          float v1355_sw = tensorforge::swap<64>(v1258_el);
          r12[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1267_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1263_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1261_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1355_sw, v1355_sw))))))));
          float v1367_sw = tensorforge::swap<64>(v1270_el);
          r12[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1279_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1275_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1273_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1367_sw, v1367_sw))))))));
          float v1380_sw = tensorforge::swap<64>(v1283_sw);
          r12[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1242_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1288_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1236_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1380_sw, v1380_sw))))))));
          float r14[13]{};
          // r14 = +(r13 * r12) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v1399_data = r12[0];
          float v1400_data = r12[1];
          float v1401_data = r12[2];
          float v1402_data = r12[3];
          float v1403_data = r12[4];
          float v1404_data = r12[5];
          float v1405_data = r12[6];
          float v1406_data = r12[7];
          float v1407_data = r12[8];
          float v1408_data = r12[9];
          float v1409_data = r12[10];
          float v1410_data = r12[11];
          float v1411_data = r12[12];
          float v1412_pad{};
          float v1413_pad{};
          float v1414_pad{};
          tensorforge::transpose16x16b32(v1399_data, v1400_data, v1401_data, v1402_data, v1403_data, v1404_data, v1405_data, v1406_data, v1407_data, v1408_data, v1409_data, v1410_data, v1411_data, v1412_pad, v1413_pad, v1414_pad);
          tensorforge::VectorT<float, 16> v1415_acc{};
          float v1416_data = r13[0];
          float v1417_data = r13[1];
          float v1418_data = r13[2];
          float v1419_data = r13[3];
          float v1420_data = r13[4];
          float v1421_data = r13[5];
          float v1422_data = r13[6];
          float v1423_data = r13[7];
          float v1424_data = r13[8];
          float v1425_data = r13[9];
          float v1426_data = r13[10];
          float v1427_data = r13[11];
          float v1428_data = r13[12];
          float v1429_data = r13[13];
          float v1430_data = r13[14];
          float v1431_data = r13[15];
          tensorforge::VectorT<float, 16> v1432_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1399_data, v1416_data, v1415_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1433_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1400_data, v1417_data, v1432_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1434_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1401_data, v1418_data, v1433_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1435_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1402_data, v1419_data, v1434_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1436_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1403_data, v1420_data, v1435_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1437_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1404_data, v1421_data, v1436_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1438_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1405_data, v1422_data, v1437_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1439_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1406_data, v1423_data, v1438_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1440_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1407_data, v1424_data, v1439_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1441_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1408_data, v1425_data, v1440_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1442_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1409_data, v1426_data, v1441_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1443_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1410_data, v1427_data, v1442_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1444_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1411_data, v1428_data, v1443_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1445_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1412_pad, v1429_data, v1444_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1446_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1413_pad, v1430_data, v1445_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1447_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1414_pad, v1431_data, v1446_acc, 1, 0, 0);
          float v1448_data = r13[16];
          float v1449_data = r13[17];
          float v1450_data = r13[18];
          float v1451_data = r13[19];
          float v1452_data = r13[20];
          float v1453_data = r13[21];
          float v1454_data = r13[22];
          float v1455_data = r13[23];
          float v1456_data = r13[24];
          float v1457_data = r13[25];
          float v1458_data = r13[26];
          float v1459_data = r13[27];
          float v1460_data = r13[28];
          float v1461_data = r13[29];
          float v1462_data = r13[30];
          float v1463_data = r13[31];
          tensorforge::VectorT<float, 16> v1464_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1399_data, v1448_data, v1447_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1465_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1400_data, v1449_data, v1464_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1466_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1401_data, v1450_data, v1465_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1467_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1402_data, v1451_data, v1466_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1468_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1403_data, v1452_data, v1467_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1469_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1404_data, v1453_data, v1468_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1470_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1405_data, v1454_data, v1469_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1471_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1406_data, v1455_data, v1470_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1472_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1407_data, v1456_data, v1471_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1473_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1408_data, v1457_data, v1472_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1474_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1409_data, v1458_data, v1473_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1475_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1410_data, v1459_data, v1474_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1476_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1411_data, v1460_data, v1475_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1477_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1412_pad, v1461_data, v1476_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1478_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1413_pad, v1462_data, v1477_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1479_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1414_pad, v1463_data, v1478_acc, 1, 1, 0);
          float v1480_el = v1479_acc[0];
          float v1482_el = v1479_acc[4];
          float v1483_sw = tensorforge::swap<32>(v1482_el);
          float v1485_el = v1479_acc[8];
          float v1488_el = v1479_acc[12];
          float v1489_sw = tensorforge::swap<32>(v1488_el);
          r14[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1489_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1485_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1483_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1480_el, v1480_el))))))));
          float v1492_el = v1479_acc[1];
          float v1494_el = v1479_acc[5];
          float v1495_sw = tensorforge::swap<32>(v1494_el);
          float v1497_el = v1479_acc[9];
          float v1500_el = v1479_acc[13];
          float v1501_sw = tensorforge::swap<32>(v1500_el);
          r14[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1501_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1497_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1495_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1492_el, v1492_el))))))));
          float v1504_el = v1479_acc[2];
          float v1506_el = v1479_acc[6];
          float v1507_sw = tensorforge::swap<32>(v1506_el);
          float v1509_el = v1479_acc[10];
          float v1512_el = v1479_acc[14];
          float v1513_sw = tensorforge::swap<32>(v1512_el);
          r14[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1513_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1509_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1507_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1504_el, v1504_el))))))));
          float v1516_el = v1479_acc[3];
          float v1518_el = v1479_acc[7];
          float v1519_sw = tensorforge::swap<32>(v1518_el);
          float v1521_el = v1479_acc[11];
          float v1524_el = v1479_acc[15];
          float v1525_sw = tensorforge::swap<32>(v1524_el);
          r14[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1525_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1521_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1519_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1516_el, v1516_el))))))));
          float v1529_sw = tensorforge::swap<32>(v1480_el);
          float v1534_sw = tensorforge::swap<32>(v1485_el);
          r14[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1488_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1534_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1482_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1529_sw, v1529_sw))))))));
          float v1541_sw = tensorforge::swap<32>(v1492_el);
          r14[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1500_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1497_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1494_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1541_sw, v1541_sw))))))));
          float v1553_sw = tensorforge::swap<32>(v1504_el);
          r14[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1512_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1509_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1506_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1553_sw, v1553_sw))))))));
          float v1565_sw = tensorforge::swap<32>(v1516_el);
          r14[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1524_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1521_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1518_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1565_sw, v1565_sw))))))));
          float v1577_sw = tensorforge::swap<64>(v1480_el);
          r14[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1489_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1485_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1483_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1577_sw, v1577_sw))))))));
          float v1589_sw = tensorforge::swap<64>(v1492_el);
          r14[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1501_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1497_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1495_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1589_sw, v1589_sw))))))));
          float v1601_sw = tensorforge::swap<64>(v1504_el);
          r14[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1513_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1509_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1507_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1601_sw, v1601_sw))))))));
          float v1613_sw = tensorforge::swap<64>(v1516_el);
          r14[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1525_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1521_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1519_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1613_sw, v1613_sw))))))));
          float v1626_sw = tensorforge::swap<64>(v1529_sw);
          r14[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1488_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1534_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1482_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1626_sw, v1626_sw))))))));
          // glb_m0 = store{r>g}(r14);
          if (v466_g) {
            #pragma unroll
            for (int32_t v1636_i1 = 0; v1636_i1 < 13; ++v1636_i1) {
              float v1638_data = r14[v1636_i1];
              int32_t v1642_a = v29_lead + (v1636_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1642_a], v1638_data);
            }
          }
          float r15[13]{};
          // r15 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1644_i0 = 0; v1644_i0 < 1; ++v1644_i0) {
            int32_t v1647_lead = v29_lead + (v1644_i0 * 32);
            #pragma unroll
            for (int32_t v1645_i1 = 0; v1645_i1 < 13; ++v1645_i1) {
              float v1650_data = glb_m0[(v1647_lead + (v1645_i1 * 32))];
              r15[(v1644_i0 + v1645_i1)] = v1650_data;
            }
          }
          float r16[13]{};
          // r16 = load{g>r}(glb_m10);
          if (v39_g) {
            #pragma unroll
            for (int32_t v1653_i1 = 0; v1653_i1 < 13; ++v1653_i1) {
              float v1658_data = __builtin_nontemporal_load(&glb_m10[(v29_lead + (v1653_i1 * 13))]);
              r16[v1653_i1] = v1658_data;
            }
          }
          float r17[13]{};
          // r17 = +(r15 * r16) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1661_data = r16[0];
          float v1662_data = r16[1];
          float v1663_data = r16[2];
          float v1664_data = r16[3];
          float v1665_data = r16[4];
          float v1666_data = r16[5];
          float v1667_data = r16[6];
          float v1668_data = r16[7];
          float v1669_data = r16[8];
          float v1670_data = r16[9];
          float v1671_data = r16[10];
          float v1672_data = r16[11];
          float v1673_data = r16[12];
          float v1674_pad{};
          float v1675_pad{};
          float v1676_pad{};
          tensorforge::transpose16x16b32(v1661_data, v1662_data, v1663_data, v1664_data, v1665_data, v1666_data, v1667_data, v1668_data, v1669_data, v1670_data, v1671_data, v1672_data, v1673_data, v1674_pad, v1675_pad, v1676_pad);
          tensorforge::VectorT<float, 16> v1677_acc{};
          float v1678_data = r15[0];
          float v1679_data = r15[1];
          float v1680_data = r15[2];
          float v1681_data = r15[3];
          float v1682_data = r15[4];
          float v1683_data = r15[5];
          float v1684_data = r15[6];
          float v1685_data = r15[7];
          float v1686_data = r15[8];
          float v1687_data = r15[9];
          float v1688_data = r15[10];
          float v1689_data = r15[11];
          float v1690_data = r15[12];
          tensorforge::VectorT<float, 16> v1692_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1661_data, v1678_data, v1677_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1693_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1662_data, v1679_data, v1692_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1694_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1663_data, v1680_data, v1693_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1695_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1664_data, v1681_data, v1694_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1696_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1665_data, v1682_data, v1695_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1697_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1666_data, v1683_data, v1696_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1698_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1667_data, v1684_data, v1697_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1699_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1668_data, v1685_data, v1698_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1700_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1669_data, v1686_data, v1699_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1701_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1670_data, v1687_data, v1700_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1702_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1671_data, v1688_data, v1701_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1703_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1672_data, v1689_data, v1702_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1704_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1673_data, v1690_data, v1703_acc, 1, 0, 0);
          float v1705_el = v1704_acc[0];
          float v1707_el = v1704_acc[4];
          float v1708_sw = tensorforge::swap<32>(v1707_el);
          float v1710_el = v1704_acc[8];
          float v1713_el = v1704_acc[12];
          float v1714_sw = tensorforge::swap<32>(v1713_el);
          r17[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1714_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1710_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1708_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1705_el, v1705_el))))))));
          float v1717_el = v1704_acc[1];
          float v1719_el = v1704_acc[5];
          float v1720_sw = tensorforge::swap<32>(v1719_el);
          float v1722_el = v1704_acc[9];
          float v1725_el = v1704_acc[13];
          float v1726_sw = tensorforge::swap<32>(v1725_el);
          r17[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1726_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1722_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1720_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1717_el, v1717_el))))))));
          float v1729_el = v1704_acc[2];
          float v1731_el = v1704_acc[6];
          float v1732_sw = tensorforge::swap<32>(v1731_el);
          float v1734_el = v1704_acc[10];
          float v1737_el = v1704_acc[14];
          float v1738_sw = tensorforge::swap<32>(v1737_el);
          r17[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1738_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1734_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1732_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1729_el, v1729_el))))))));
          float v1741_el = v1704_acc[3];
          float v1743_el = v1704_acc[7];
          float v1744_sw = tensorforge::swap<32>(v1743_el);
          float v1746_el = v1704_acc[11];
          float v1749_el = v1704_acc[15];
          float v1750_sw = tensorforge::swap<32>(v1749_el);
          r17[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1750_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1746_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1744_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1741_el, v1741_el))))))));
          float v1754_sw = tensorforge::swap<32>(v1705_el);
          float v1759_sw = tensorforge::swap<32>(v1710_el);
          r17[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1713_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1759_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1707_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1754_sw, v1754_sw))))))));
          float v1766_sw = tensorforge::swap<32>(v1717_el);
          r17[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1725_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1722_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1719_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1766_sw, v1766_sw))))))));
          float v1778_sw = tensorforge::swap<32>(v1729_el);
          r17[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1737_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1734_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1731_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1778_sw, v1778_sw))))))));
          float v1790_sw = tensorforge::swap<32>(v1741_el);
          r17[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1749_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1746_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1743_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1790_sw, v1790_sw))))))));
          float v1802_sw = tensorforge::swap<64>(v1705_el);
          r17[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1714_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1710_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1708_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1802_sw, v1802_sw))))))));
          float v1814_sw = tensorforge::swap<64>(v1717_el);
          r17[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1726_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1722_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1720_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1814_sw, v1814_sw))))))));
          float v1826_sw = tensorforge::swap<64>(v1729_el);
          r17[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1738_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1734_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1732_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1826_sw, v1826_sw))))))));
          float v1838_sw = tensorforge::swap<64>(v1741_el);
          r17[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1750_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1746_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1744_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1838_sw, v1838_sw))))))));
          float v1851_sw = tensorforge::swap<64>(v1754_sw);
          r17[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1713_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1759_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1707_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1851_sw, v1851_sw))))))));
          // glb_m9 = store{r>g}(r17);
          #pragma unroll
          for (int32_t v1861_i0 = 0; v1861_i0 < 1; ++v1861_i0) {
            int32_t v1866_lead = v29_lead + (v1861_i0 * 32);
            #pragma unroll
            for (int32_t v1862_i1 = 0; v1862_i1 < 13; ++v1862_i1) {
              float v1864_data = r17[(v1861_i0 + v1862_i1)];
              glb_m9[(v1866_lead + (v1862_i1 * 32))] = v1864_data;
            }
          }
        }
      }
    }
  }
}

