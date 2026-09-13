// === base name ===
kernel_9cc2ca778476b84f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9cc2ca778476b84f = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9cc2ca778476b84f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9cc2ca778476b84f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, float * m9, size_t m9_extraOffset, const float * m10, size_t m10_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9cc2ca778476b84f(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9cc2ca778476b84f, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_9cc2ca778476b84f, block.x * block.y * block.z, 0));
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
void launcher_kernel_9cc2ca778476b84f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, float * m9, size_t m9_extraOffset, const float * m10, size_t m10_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9cc2ca778476b84f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9cc2ca778476b84f), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_9cc2ca778476b84f, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, m9Arg, m9_extraOffset, m10Arg, m10_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9cc2ca778476b84f(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m9, size_t m9_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m10, size_t m10_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[13,13]],"name":"m3","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"W0","bbox":[[0,0],[16,32]],"name":"m4","ordered":false,"parts":1,"shape":[16,32],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[13,13]],"name":"m5","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"W1","bbox":[[0,0],[16,32]],"name":"m6","ordered":false,"parts":1,"shape":[16,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[13,13]],"name":"m7","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"W2","bbox":[[0,0],[16,32]],"name":"m8","ordered":false,"parts":1,"shape":[16,32],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m9","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m10","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[16,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,32]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,32]},{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[16,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,32]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[16,32]},{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[16,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,32]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[16,32]},{"addressing":"pointer_based","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m9","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m10","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 416 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 416 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 169 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 169 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 512 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v1_batchId0 * 169 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v1_batchId0 * 512 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v1_batchId0 * 169 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v1_batchId0 * 512 + 0 + m8_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m9 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m9[v1_batchId0 * 416 + 0 + m9_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m10 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m10[v1_batchId0 * 169 + 0 + m10_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v23_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 32);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 13; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m1[(v27_lead + (v25_i1 * 32))]);
              r0[(v24_i0 + v25_i1)] = v30_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          bool v33_g = v23_lead < 13;
          if (v33_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 13; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v34_i1 * 13))]);
              r1[v34_i1] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[13]{};
          // r3 = load{g>r}(glb_m3);
          if (v33_g) {
            #pragma unroll
            for (int32_t v42_i1 = 0; v42_i1 < 13; ++v42_i1) {
              float v47_data = __builtin_nontemporal_load(&glb_m3[(v23_lead + (v42_i1 * 13))]);
              r3[v42_i1] = v47_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[13]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v50_data = r1[0];
          float v51_data = r1[1];
          float v52_data = r1[2];
          float v53_data = r1[3];
          float v54_data = r1[4];
          float v55_data = r1[5];
          float v56_data = r1[6];
          float v57_data = r1[7];
          float v58_data = r1[8];
          float v59_data = r1[9];
          float v60_data = r1[10];
          float v61_data = r1[11];
          float v62_data = r1[12];
          float v63_pad{};
          float v64_pad{};
          float v65_pad{};
          tensorforge::transpose16x16b32(v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_pad, v64_pad, v65_pad);
          tensorforge::VectorT<float, 16> v66_acc{};
          float v67_data = r0[0];
          float v68_data = r0[1];
          float v69_data = r0[2];
          float v70_data = r0[3];
          float v71_data = r0[4];
          float v72_data = r0[5];
          float v73_data = r0[6];
          float v74_data = r0[7];
          float v75_data = r0[8];
          float v76_data = r0[9];
          float v77_data = r0[10];
          float v78_data = r0[11];
          float v79_data = r0[12];
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v66_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v91_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v92_acc, 1, 0, 0);
          float v94_el = v93_acc[0];
          float v96_el = v93_acc[4];
          float v97_sw = tensorforge::swap<32>(v96_el);
          float v99_el = v93_acc[8];
          float v102_el = v93_acc[12];
          float v103_sw = tensorforge::swap<32>(v102_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v103_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v99_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v97_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v94_el, v94_el))))))));
          float v106_el = v93_acc[1];
          float v108_el = v93_acc[5];
          float v109_sw = tensorforge::swap<32>(v108_el);
          float v111_el = v93_acc[9];
          float v114_el = v93_acc[13];
          float v115_sw = tensorforge::swap<32>(v114_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v115_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v109_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v106_el, v106_el))))))));
          float v118_el = v93_acc[2];
          float v120_el = v93_acc[6];
          float v121_sw = tensorforge::swap<32>(v120_el);
          float v123_el = v93_acc[10];
          float v126_el = v93_acc[14];
          float v127_sw = tensorforge::swap<32>(v126_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v121_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v118_el, v118_el))))))));
          float v130_el = v93_acc[3];
          float v132_el = v93_acc[7];
          float v133_sw = tensorforge::swap<32>(v132_el);
          float v135_el = v93_acc[11];
          float v138_el = v93_acc[15];
          float v139_sw = tensorforge::swap<32>(v138_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v130_el, v130_el))))))));
          float v143_sw = tensorforge::swap<32>(v94_el);
          float v148_sw = tensorforge::swap<32>(v99_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v102_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v148_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v96_el, (tensorforge::dppUpdate<228, 1, 15, false>(v143_sw, v143_sw))))))));
          float v155_sw = tensorforge::swap<32>(v106_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v114_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v111_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v108_el, (tensorforge::dppUpdate<228, 1, 15, false>(v155_sw, v155_sw))))))));
          float v167_sw = tensorforge::swap<32>(v118_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v123_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v120_el, (tensorforge::dppUpdate<228, 1, 15, false>(v167_sw, v167_sw))))))));
          float v179_sw = tensorforge::swap<32>(v130_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v135_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v132_el, (tensorforge::dppUpdate<228, 1, 15, false>(v179_sw, v179_sw))))))));
          float v191_sw = tensorforge::swap<64>(v94_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v103_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v99_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v97_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v191_sw, v191_sw))))))));
          float v203_sw = tensorforge::swap<64>(v106_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v115_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v111_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v109_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v203_sw, v203_sw))))))));
          float v215_sw = tensorforge::swap<64>(v118_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v123_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v121_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v215_sw, v215_sw))))))));
          float v227_sw = tensorforge::swap<64>(v130_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v135_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v227_sw, v227_sw))))))));
          float v240_sw = tensorforge::swap<64>(v143_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v102_el, (tensorforge::dppUpdate<228, 4, 15, false>(v148_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v96_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v240_sw, v240_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v250_i0 = 0; v250_i0 < 1; ++v250_i0) {
            int32_t v255_lead = v23_lead + (v250_i0 * 32);
            #pragma unroll
            for (int32_t v251_i1 = 0; v251_i1 < 13; ++v251_i1) {
              float v253_data = r2[(v250_i0 + v251_i1)];
              glb_m0[(v255_lead + (v251_i1 * 32))] = v253_data;
            }
          }
          float r5[32]{};
          // r5 = load{g>r}(glb_m4);
          bool v259_g = v23_lead < 16;
          if (v259_g) {
            #pragma unroll
            for (int32_t v260_i1 = 0; v260_i1 < 32; ++v260_i1) {
              float v265_data = __builtin_nontemporal_load(&glb_m4[(v23_lead + (v260_i1 * 16))]);
              r5[v260_i1] = v265_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r4[13]{};
          // r4 = +(r0 * r3) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v268_data = r3[0];
          float v269_data = r3[1];
          float v270_data = r3[2];
          float v271_data = r3[3];
          float v272_data = r3[4];
          float v273_data = r3[5];
          float v274_data = r3[6];
          float v275_data = r3[7];
          float v276_data = r3[8];
          float v277_data = r3[9];
          float v278_data = r3[10];
          float v279_data = r3[11];
          float v280_data = r3[12];
          float v281_pad{};
          float v282_pad{};
          float v283_pad{};
          tensorforge::transpose16x16b32(v268_data, v269_data, v270_data, v271_data, v272_data, v273_data, v274_data, v275_data, v276_data, v277_data, v278_data, v279_data, v280_data, v281_pad, v282_pad, v283_pad);
          tensorforge::VectorT<float, 16> v284_acc{};
          tensorforge::VectorT<float, 16> v299_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v268_data, v67_data, v284_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v300_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v269_data, v68_data, v299_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v301_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v270_data, v69_data, v300_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v302_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v271_data, v70_data, v301_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v303_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v272_data, v71_data, v302_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v304_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v273_data, v72_data, v303_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v305_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v274_data, v73_data, v304_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v306_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v275_data, v74_data, v305_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v307_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v276_data, v75_data, v306_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v308_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v277_data, v76_data, v307_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v309_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v278_data, v77_data, v308_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v310_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v279_data, v78_data, v309_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v311_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v280_data, v79_data, v310_acc, 1, 0, 0);
          float v312_el = v311_acc[0];
          float v314_el = v311_acc[4];
          float v315_sw = tensorforge::swap<32>(v314_el);
          float v317_el = v311_acc[8];
          float v320_el = v311_acc[12];
          float v321_sw = tensorforge::swap<32>(v320_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v321_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v317_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v315_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v312_el, v312_el))))))));
          float v324_el = v311_acc[1];
          float v326_el = v311_acc[5];
          float v327_sw = tensorforge::swap<32>(v326_el);
          float v329_el = v311_acc[9];
          float v332_el = v311_acc[13];
          float v333_sw = tensorforge::swap<32>(v332_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v333_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v329_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v327_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v324_el, v324_el))))))));
          float v336_el = v311_acc[2];
          float v338_el = v311_acc[6];
          float v339_sw = tensorforge::swap<32>(v338_el);
          float v341_el = v311_acc[10];
          float v344_el = v311_acc[14];
          float v345_sw = tensorforge::swap<32>(v344_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v345_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v341_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v339_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v336_el, v336_el))))))));
          float v348_el = v311_acc[3];
          float v350_el = v311_acc[7];
          float v351_sw = tensorforge::swap<32>(v350_el);
          float v353_el = v311_acc[11];
          float v356_el = v311_acc[15];
          float v357_sw = tensorforge::swap<32>(v356_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v357_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v353_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v351_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v348_el, v348_el))))))));
          float v361_sw = tensorforge::swap<32>(v312_el);
          float v366_sw = tensorforge::swap<32>(v317_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v320_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v366_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v314_el, (tensorforge::dppUpdate<228, 1, 15, false>(v361_sw, v361_sw))))))));
          float v373_sw = tensorforge::swap<32>(v324_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v332_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v329_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v326_el, (tensorforge::dppUpdate<228, 1, 15, false>(v373_sw, v373_sw))))))));
          float v385_sw = tensorforge::swap<32>(v336_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v344_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v341_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v338_el, (tensorforge::dppUpdate<228, 1, 15, false>(v385_sw, v385_sw))))))));
          float v397_sw = tensorforge::swap<32>(v348_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v356_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v353_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v350_el, (tensorforge::dppUpdate<228, 1, 15, false>(v397_sw, v397_sw))))))));
          float v409_sw = tensorforge::swap<64>(v312_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v321_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v317_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v315_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v409_sw, v409_sw))))))));
          float v421_sw = tensorforge::swap<64>(v324_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v333_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v329_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v327_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v421_sw, v421_sw))))))));
          float v433_sw = tensorforge::swap<64>(v336_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v345_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v341_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v339_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v433_sw, v433_sw))))))));
          float v445_sw = tensorforge::swap<64>(v348_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v357_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v353_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v351_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v445_sw, v445_sw))))))));
          float v458_sw = tensorforge::swap<64>(v361_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v320_el, (tensorforge::dppUpdate<228, 4, 15, false>(v366_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v314_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v458_sw, v458_sw))))))));
          float r7[13]{};
          // r7 = load{g>r}(glb_m5);
          if (v33_g) {
            #pragma unroll
            for (int32_t v469_i1 = 0; v469_i1 < 13; ++v469_i1) {
              float v474_data = __builtin_nontemporal_load(&glb_m5[(v23_lead + (v469_i1 * 13))]);
              r7[v469_i1] = v474_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[13]{};
          // r6 = +(r5 * r4) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v477_data = r4[0];
          float v478_data = r4[1];
          float v479_data = r4[2];
          float v480_data = r4[3];
          float v481_data = r4[4];
          float v482_data = r4[5];
          float v483_data = r4[6];
          float v484_data = r4[7];
          float v485_data = r4[8];
          float v486_data = r4[9];
          float v487_data = r4[10];
          float v488_data = r4[11];
          float v489_data = r4[12];
          float v490_pad{};
          float v491_pad{};
          float v492_pad{};
          tensorforge::transpose16x16b32(v477_data, v478_data, v479_data, v480_data, v481_data, v482_data, v483_data, v484_data, v485_data, v486_data, v487_data, v488_data, v489_data, v490_pad, v491_pad, v492_pad);
          tensorforge::VectorT<float, 16> v493_acc{};
          float v494_data = r5[0];
          float v495_data = r5[1];
          float v496_data = r5[2];
          float v497_data = r5[3];
          float v498_data = r5[4];
          float v499_data = r5[5];
          float v500_data = r5[6];
          float v501_data = r5[7];
          float v502_data = r5[8];
          float v503_data = r5[9];
          float v504_data = r5[10];
          float v505_data = r5[11];
          float v506_data = r5[12];
          float v507_data = r5[13];
          float v508_data = r5[14];
          float v509_data = r5[15];
          tensorforge::VectorT<float, 16> v510_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v477_data, v494_data, v493_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v521_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v488_data, v505_data, v520_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v522_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v489_data, v506_data, v521_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v523_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v490_pad, v507_data, v522_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v524_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v491_pad, v508_data, v523_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v525_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v492_pad, v509_data, v524_acc, 1, 0, 0);
          float v526_data = r5[16];
          float v527_data = r5[17];
          float v528_data = r5[18];
          float v529_data = r5[19];
          float v530_data = r5[20];
          float v531_data = r5[21];
          float v532_data = r5[22];
          float v533_data = r5[23];
          float v534_data = r5[24];
          float v535_data = r5[25];
          float v536_data = r5[26];
          float v537_data = r5[27];
          float v538_data = r5[28];
          float v539_data = r5[29];
          float v540_data = r5[30];
          float v541_data = r5[31];
          tensorforge::VectorT<float, 16> v542_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v477_data, v526_data, v525_acc, 1, 1, 0);
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
          tensorforge::VectorT<float, 16> v553_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v488_data, v537_data, v552_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v554_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v489_data, v538_data, v553_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v555_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v490_pad, v539_data, v554_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v556_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v491_pad, v540_data, v555_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v557_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v492_pad, v541_data, v556_acc, 1, 1, 0);
          float v558_el = v557_acc[0];
          float v560_el = v557_acc[4];
          float v561_sw = tensorforge::swap<32>(v560_el);
          float v563_el = v557_acc[8];
          float v566_el = v557_acc[12];
          float v567_sw = tensorforge::swap<32>(v566_el);
          r6[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v567_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v563_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v561_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v558_el, v558_el))))))));
          float v570_el = v557_acc[1];
          float v572_el = v557_acc[5];
          float v573_sw = tensorforge::swap<32>(v572_el);
          float v575_el = v557_acc[9];
          float v578_el = v557_acc[13];
          float v579_sw = tensorforge::swap<32>(v578_el);
          r6[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v579_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v575_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v573_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v570_el, v570_el))))))));
          float v582_el = v557_acc[2];
          float v584_el = v557_acc[6];
          float v585_sw = tensorforge::swap<32>(v584_el);
          float v587_el = v557_acc[10];
          float v590_el = v557_acc[14];
          float v591_sw = tensorforge::swap<32>(v590_el);
          r6[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v591_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v587_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v585_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v582_el, v582_el))))))));
          float v594_el = v557_acc[3];
          float v596_el = v557_acc[7];
          float v597_sw = tensorforge::swap<32>(v596_el);
          float v599_el = v557_acc[11];
          float v602_el = v557_acc[15];
          float v603_sw = tensorforge::swap<32>(v602_el);
          r6[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v603_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v599_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v597_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v594_el, v594_el))))))));
          float v607_sw = tensorforge::swap<32>(v558_el);
          float v612_sw = tensorforge::swap<32>(v563_el);
          r6[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v566_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v612_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v560_el, (tensorforge::dppUpdate<228, 1, 15, false>(v607_sw, v607_sw))))))));
          float v619_sw = tensorforge::swap<32>(v570_el);
          r6[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v578_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v575_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v572_el, (tensorforge::dppUpdate<228, 1, 15, false>(v619_sw, v619_sw))))))));
          float v631_sw = tensorforge::swap<32>(v582_el);
          r6[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v590_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v587_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v584_el, (tensorforge::dppUpdate<228, 1, 15, false>(v631_sw, v631_sw))))))));
          float v643_sw = tensorforge::swap<32>(v594_el);
          r6[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v602_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v599_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v596_el, (tensorforge::dppUpdate<228, 1, 15, false>(v643_sw, v643_sw))))))));
          float v655_sw = tensorforge::swap<64>(v558_el);
          r6[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v567_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v563_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v561_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v655_sw, v655_sw))))))));
          float v667_sw = tensorforge::swap<64>(v570_el);
          r6[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v579_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v575_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v573_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v667_sw, v667_sw))))))));
          float v679_sw = tensorforge::swap<64>(v582_el);
          r6[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v591_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v587_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v585_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v679_sw, v679_sw))))))));
          float v691_sw = tensorforge::swap<64>(v594_el);
          r6[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v603_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v599_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v597_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v691_sw, v691_sw))))))));
          float v704_sw = tensorforge::swap<64>(v607_sw);
          r6[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v566_el, (tensorforge::dppUpdate<228, 4, 15, false>(v612_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v560_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v704_sw, v704_sw))))))));
          // glb_m0 = store{r>g}(r6);
          if (v259_g) {
            #pragma unroll
            for (int32_t v714_i1 = 0; v714_i1 < 13; ++v714_i1) {
              float v716_data = r6[v714_i1];
              int32_t v720_a = v23_lead + (v714_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v720_a], v716_data);
            }
          }
          float r9[32]{};
          // r9 = load{g>r}(glb_m6);
          if (v259_g) {
            #pragma unroll
            for (int32_t v722_i1 = 0; v722_i1 < 32; ++v722_i1) {
              float v727_data = __builtin_nontemporal_load(&glb_m6[(v23_lead + (v722_i1 * 16))]);
              r9[v722_i1] = v727_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r8[13]{};
          // r8 = +(r0 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v730_data = r7[0];
          float v731_data = r7[1];
          float v732_data = r7[2];
          float v733_data = r7[3];
          float v734_data = r7[4];
          float v735_data = r7[5];
          float v736_data = r7[6];
          float v737_data = r7[7];
          float v738_data = r7[8];
          float v739_data = r7[9];
          float v740_data = r7[10];
          float v741_data = r7[11];
          float v742_data = r7[12];
          float v743_pad{};
          float v744_pad{};
          float v745_pad{};
          tensorforge::transpose16x16b32(v730_data, v731_data, v732_data, v733_data, v734_data, v735_data, v736_data, v737_data, v738_data, v739_data, v740_data, v741_data, v742_data, v743_pad, v744_pad, v745_pad);
          tensorforge::VectorT<float, 16> v746_acc{};
          tensorforge::VectorT<float, 16> v761_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v730_data, v67_data, v746_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v772_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v741_data, v78_data, v771_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v773_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v742_data, v79_data, v772_acc, 1, 0, 0);
          float v774_el = v773_acc[0];
          float v776_el = v773_acc[4];
          float v777_sw = tensorforge::swap<32>(v776_el);
          float v779_el = v773_acc[8];
          float v782_el = v773_acc[12];
          float v783_sw = tensorforge::swap<32>(v782_el);
          r8[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v783_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v779_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v777_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v774_el, v774_el))))))));
          float v786_el = v773_acc[1];
          float v788_el = v773_acc[5];
          float v789_sw = tensorforge::swap<32>(v788_el);
          float v791_el = v773_acc[9];
          float v794_el = v773_acc[13];
          float v795_sw = tensorforge::swap<32>(v794_el);
          r8[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v795_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v791_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v789_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v786_el, v786_el))))))));
          float v798_el = v773_acc[2];
          float v800_el = v773_acc[6];
          float v801_sw = tensorforge::swap<32>(v800_el);
          float v803_el = v773_acc[10];
          float v806_el = v773_acc[14];
          float v807_sw = tensorforge::swap<32>(v806_el);
          r8[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v807_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v803_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v801_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v798_el, v798_el))))))));
          float v810_el = v773_acc[3];
          float v812_el = v773_acc[7];
          float v813_sw = tensorforge::swap<32>(v812_el);
          float v815_el = v773_acc[11];
          float v818_el = v773_acc[15];
          float v819_sw = tensorforge::swap<32>(v818_el);
          r8[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v819_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v815_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v813_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v810_el, v810_el))))))));
          float v823_sw = tensorforge::swap<32>(v774_el);
          float v828_sw = tensorforge::swap<32>(v779_el);
          r8[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v782_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v828_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v776_el, (tensorforge::dppUpdate<228, 1, 15, false>(v823_sw, v823_sw))))))));
          float v835_sw = tensorforge::swap<32>(v786_el);
          r8[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v794_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v791_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v788_el, (tensorforge::dppUpdate<228, 1, 15, false>(v835_sw, v835_sw))))))));
          float v847_sw = tensorforge::swap<32>(v798_el);
          r8[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v806_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v803_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v800_el, (tensorforge::dppUpdate<228, 1, 15, false>(v847_sw, v847_sw))))))));
          float v859_sw = tensorforge::swap<32>(v810_el);
          r8[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v818_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v815_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v812_el, (tensorforge::dppUpdate<228, 1, 15, false>(v859_sw, v859_sw))))))));
          float v871_sw = tensorforge::swap<64>(v774_el);
          r8[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v783_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v779_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v777_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v871_sw, v871_sw))))))));
          float v883_sw = tensorforge::swap<64>(v786_el);
          r8[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v795_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v791_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v789_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v883_sw, v883_sw))))))));
          float v895_sw = tensorforge::swap<64>(v798_el);
          r8[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v807_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v803_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v801_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v895_sw, v895_sw))))))));
          float v907_sw = tensorforge::swap<64>(v810_el);
          r8[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v819_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v815_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v813_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v907_sw, v907_sw))))))));
          float v920_sw = tensorforge::swap<64>(v823_sw);
          r8[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v782_el, (tensorforge::dppUpdate<228, 4, 15, false>(v828_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v776_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v920_sw, v920_sw))))))));
          float r11[13]{};
          // r11 = load{g>r}(glb_m7);
          if (v33_g) {
            #pragma unroll
            for (int32_t v931_i1 = 0; v931_i1 < 13; ++v931_i1) {
              float v936_data = __builtin_nontemporal_load(&glb_m7[(v23_lead + (v931_i1 * 13))]);
              r11[v931_i1] = v936_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[13]{};
          // r10 = +(r9 * r8) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v939_data = r8[0];
          float v940_data = r8[1];
          float v941_data = r8[2];
          float v942_data = r8[3];
          float v943_data = r8[4];
          float v944_data = r8[5];
          float v945_data = r8[6];
          float v946_data = r8[7];
          float v947_data = r8[8];
          float v948_data = r8[9];
          float v949_data = r8[10];
          float v950_data = r8[11];
          float v951_data = r8[12];
          float v952_pad{};
          float v953_pad{};
          float v954_pad{};
          tensorforge::transpose16x16b32(v939_data, v940_data, v941_data, v942_data, v943_data, v944_data, v945_data, v946_data, v947_data, v948_data, v949_data, v950_data, v951_data, v952_pad, v953_pad, v954_pad);
          tensorforge::VectorT<float, 16> v955_acc{};
          float v956_data = r9[0];
          float v957_data = r9[1];
          float v958_data = r9[2];
          float v959_data = r9[3];
          float v960_data = r9[4];
          float v961_data = r9[5];
          float v962_data = r9[6];
          float v963_data = r9[7];
          float v964_data = r9[8];
          float v965_data = r9[9];
          float v966_data = r9[10];
          float v967_data = r9[11];
          float v968_data = r9[12];
          float v969_data = r9[13];
          float v970_data = r9[14];
          float v971_data = r9[15];
          tensorforge::VectorT<float, 16> v972_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v939_data, v956_data, v955_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v983_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v950_data, v967_data, v982_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v984_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v951_data, v968_data, v983_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v985_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v952_pad, v969_data, v984_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v986_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v953_pad, v970_data, v985_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v987_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v954_pad, v971_data, v986_acc, 1, 0, 0);
          float v988_data = r9[16];
          float v989_data = r9[17];
          float v990_data = r9[18];
          float v991_data = r9[19];
          float v992_data = r9[20];
          float v993_data = r9[21];
          float v994_data = r9[22];
          float v995_data = r9[23];
          float v996_data = r9[24];
          float v997_data = r9[25];
          float v998_data = r9[26];
          float v999_data = r9[27];
          float v1000_data = r9[28];
          float v1001_data = r9[29];
          float v1002_data = r9[30];
          float v1003_data = r9[31];
          tensorforge::VectorT<float, 16> v1004_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v939_data, v988_data, v987_acc, 1, 1, 0);
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
          tensorforge::VectorT<float, 16> v1015_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v950_data, v999_data, v1014_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1016_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v951_data, v1000_data, v1015_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1017_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v952_pad, v1001_data, v1016_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1018_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v953_pad, v1002_data, v1017_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1019_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v954_pad, v1003_data, v1018_acc, 1, 1, 0);
          float v1020_el = v1019_acc[0];
          float v1022_el = v1019_acc[4];
          float v1023_sw = tensorforge::swap<32>(v1022_el);
          float v1025_el = v1019_acc[8];
          float v1028_el = v1019_acc[12];
          float v1029_sw = tensorforge::swap<32>(v1028_el);
          r10[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1029_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1025_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1023_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1020_el, v1020_el))))))));
          float v1032_el = v1019_acc[1];
          float v1034_el = v1019_acc[5];
          float v1035_sw = tensorforge::swap<32>(v1034_el);
          float v1037_el = v1019_acc[9];
          float v1040_el = v1019_acc[13];
          float v1041_sw = tensorforge::swap<32>(v1040_el);
          r10[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1041_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1037_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1035_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1032_el, v1032_el))))))));
          float v1044_el = v1019_acc[2];
          float v1046_el = v1019_acc[6];
          float v1047_sw = tensorforge::swap<32>(v1046_el);
          float v1049_el = v1019_acc[10];
          float v1052_el = v1019_acc[14];
          float v1053_sw = tensorforge::swap<32>(v1052_el);
          r10[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1053_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1049_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1047_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1044_el, v1044_el))))))));
          float v1056_el = v1019_acc[3];
          float v1058_el = v1019_acc[7];
          float v1059_sw = tensorforge::swap<32>(v1058_el);
          float v1061_el = v1019_acc[11];
          float v1064_el = v1019_acc[15];
          float v1065_sw = tensorforge::swap<32>(v1064_el);
          r10[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1065_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1061_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1059_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1056_el, v1056_el))))))));
          float v1069_sw = tensorforge::swap<32>(v1020_el);
          float v1074_sw = tensorforge::swap<32>(v1025_el);
          r10[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1028_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1074_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1022_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1069_sw, v1069_sw))))))));
          float v1081_sw = tensorforge::swap<32>(v1032_el);
          r10[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1040_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1037_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1034_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1081_sw, v1081_sw))))))));
          float v1093_sw = tensorforge::swap<32>(v1044_el);
          r10[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1052_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1049_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1046_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1093_sw, v1093_sw))))))));
          float v1105_sw = tensorforge::swap<32>(v1056_el);
          r10[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1064_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1061_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1058_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1105_sw, v1105_sw))))))));
          float v1117_sw = tensorforge::swap<64>(v1020_el);
          r10[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1029_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1025_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1023_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1117_sw, v1117_sw))))))));
          float v1129_sw = tensorforge::swap<64>(v1032_el);
          r10[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1041_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1037_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1035_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1129_sw, v1129_sw))))))));
          float v1141_sw = tensorforge::swap<64>(v1044_el);
          r10[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1053_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1049_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1047_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1141_sw, v1141_sw))))))));
          float v1153_sw = tensorforge::swap<64>(v1056_el);
          r10[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1065_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1061_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1059_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1153_sw, v1153_sw))))))));
          float v1166_sw = tensorforge::swap<64>(v1069_sw);
          r10[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1028_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1074_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1022_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1166_sw, v1166_sw))))))));
          // glb_m0 = store{r>g}(r10);
          if (v259_g) {
            #pragma unroll
            for (int32_t v1176_i1 = 0; v1176_i1 < 13; ++v1176_i1) {
              float v1178_data = r10[v1176_i1];
              int32_t v1182_a = v23_lead + (v1176_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1182_a], v1178_data);
            }
          }
          float r13[32]{};
          // r13 = load{g>r}(glb_m8);
          if (v259_g) {
            #pragma unroll
            for (int32_t v1184_i1 = 0; v1184_i1 < 32; ++v1184_i1) {
              float v1189_data = __builtin_nontemporal_load(&glb_m8[(v23_lead + (v1184_i1 * 16))]);
              r13[v1184_i1] = v1189_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m7););
          float r12[13]{};
          // r12 = +(r0 * r11) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1192_data = r11[0];
          float v1193_data = r11[1];
          float v1194_data = r11[2];
          float v1195_data = r11[3];
          float v1196_data = r11[4];
          float v1197_data = r11[5];
          float v1198_data = r11[6];
          float v1199_data = r11[7];
          float v1200_data = r11[8];
          float v1201_data = r11[9];
          float v1202_data = r11[10];
          float v1203_data = r11[11];
          float v1204_data = r11[12];
          float v1205_pad{};
          float v1206_pad{};
          float v1207_pad{};
          tensorforge::transpose16x16b32(v1192_data, v1193_data, v1194_data, v1195_data, v1196_data, v1197_data, v1198_data, v1199_data, v1200_data, v1201_data, v1202_data, v1203_data, v1204_data, v1205_pad, v1206_pad, v1207_pad);
          tensorforge::VectorT<float, 16> v1208_acc{};
          tensorforge::VectorT<float, 16> v1223_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1192_data, v67_data, v1208_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v1234_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1203_data, v78_data, v1233_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1235_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1204_data, v79_data, v1234_acc, 1, 0, 0);
          float v1236_el = v1235_acc[0];
          float v1238_el = v1235_acc[4];
          float v1239_sw = tensorforge::swap<32>(v1238_el);
          float v1241_el = v1235_acc[8];
          float v1244_el = v1235_acc[12];
          float v1245_sw = tensorforge::swap<32>(v1244_el);
          r12[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1245_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1241_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1239_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1236_el, v1236_el))))))));
          float v1248_el = v1235_acc[1];
          float v1250_el = v1235_acc[5];
          float v1251_sw = tensorforge::swap<32>(v1250_el);
          float v1253_el = v1235_acc[9];
          float v1256_el = v1235_acc[13];
          float v1257_sw = tensorforge::swap<32>(v1256_el);
          r12[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1257_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1253_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1251_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1248_el, v1248_el))))))));
          float v1260_el = v1235_acc[2];
          float v1262_el = v1235_acc[6];
          float v1263_sw = tensorforge::swap<32>(v1262_el);
          float v1265_el = v1235_acc[10];
          float v1268_el = v1235_acc[14];
          float v1269_sw = tensorforge::swap<32>(v1268_el);
          r12[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1269_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1265_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1263_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1260_el, v1260_el))))))));
          float v1272_el = v1235_acc[3];
          float v1274_el = v1235_acc[7];
          float v1275_sw = tensorforge::swap<32>(v1274_el);
          float v1277_el = v1235_acc[11];
          float v1280_el = v1235_acc[15];
          float v1281_sw = tensorforge::swap<32>(v1280_el);
          r12[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1281_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1277_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1275_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1272_el, v1272_el))))))));
          float v1285_sw = tensorforge::swap<32>(v1236_el);
          float v1290_sw = tensorforge::swap<32>(v1241_el);
          r12[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1244_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1290_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1238_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1285_sw, v1285_sw))))))));
          float v1297_sw = tensorforge::swap<32>(v1248_el);
          r12[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1256_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1253_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1250_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1297_sw, v1297_sw))))))));
          float v1309_sw = tensorforge::swap<32>(v1260_el);
          r12[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1268_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1265_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1262_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1309_sw, v1309_sw))))))));
          float v1321_sw = tensorforge::swap<32>(v1272_el);
          r12[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1280_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1277_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1274_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1321_sw, v1321_sw))))))));
          float v1333_sw = tensorforge::swap<64>(v1236_el);
          r12[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1245_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1241_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1239_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1333_sw, v1333_sw))))))));
          float v1345_sw = tensorforge::swap<64>(v1248_el);
          r12[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1257_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1253_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1251_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1345_sw, v1345_sw))))))));
          float v1357_sw = tensorforge::swap<64>(v1260_el);
          r12[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1269_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1265_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1263_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1357_sw, v1357_sw))))))));
          float v1369_sw = tensorforge::swap<64>(v1272_el);
          r12[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1281_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1277_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1275_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1369_sw, v1369_sw))))))));
          float v1382_sw = tensorforge::swap<64>(v1285_sw);
          r12[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1244_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1290_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1238_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1382_sw, v1382_sw))))))));
          // wait(r13 = load{g>r}(glb_m8););
          float r14[13]{};
          // r14 = +(r13 * r12) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v1393_data = r12[0];
          float v1394_data = r12[1];
          float v1395_data = r12[2];
          float v1396_data = r12[3];
          float v1397_data = r12[4];
          float v1398_data = r12[5];
          float v1399_data = r12[6];
          float v1400_data = r12[7];
          float v1401_data = r12[8];
          float v1402_data = r12[9];
          float v1403_data = r12[10];
          float v1404_data = r12[11];
          float v1405_data = r12[12];
          float v1406_pad{};
          float v1407_pad{};
          float v1408_pad{};
          tensorforge::transpose16x16b32(v1393_data, v1394_data, v1395_data, v1396_data, v1397_data, v1398_data, v1399_data, v1400_data, v1401_data, v1402_data, v1403_data, v1404_data, v1405_data, v1406_pad, v1407_pad, v1408_pad);
          tensorforge::VectorT<float, 16> v1409_acc{};
          float v1410_data = r13[0];
          float v1411_data = r13[1];
          float v1412_data = r13[2];
          float v1413_data = r13[3];
          float v1414_data = r13[4];
          float v1415_data = r13[5];
          float v1416_data = r13[6];
          float v1417_data = r13[7];
          float v1418_data = r13[8];
          float v1419_data = r13[9];
          float v1420_data = r13[10];
          float v1421_data = r13[11];
          float v1422_data = r13[12];
          float v1423_data = r13[13];
          float v1424_data = r13[14];
          float v1425_data = r13[15];
          tensorforge::VectorT<float, 16> v1426_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1393_data, v1410_data, v1409_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1427_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1394_data, v1411_data, v1426_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1428_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1395_data, v1412_data, v1427_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1429_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1396_data, v1413_data, v1428_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1430_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1397_data, v1414_data, v1429_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1431_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1398_data, v1415_data, v1430_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1432_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1399_data, v1416_data, v1431_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1433_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1400_data, v1417_data, v1432_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1434_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1401_data, v1418_data, v1433_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1435_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1402_data, v1419_data, v1434_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1436_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1403_data, v1420_data, v1435_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1437_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1404_data, v1421_data, v1436_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1438_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1405_data, v1422_data, v1437_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1439_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1406_pad, v1423_data, v1438_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1440_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1407_pad, v1424_data, v1439_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1441_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1408_pad, v1425_data, v1440_acc, 1, 0, 0);
          float v1442_data = r13[16];
          float v1443_data = r13[17];
          float v1444_data = r13[18];
          float v1445_data = r13[19];
          float v1446_data = r13[20];
          float v1447_data = r13[21];
          float v1448_data = r13[22];
          float v1449_data = r13[23];
          float v1450_data = r13[24];
          float v1451_data = r13[25];
          float v1452_data = r13[26];
          float v1453_data = r13[27];
          float v1454_data = r13[28];
          float v1455_data = r13[29];
          float v1456_data = r13[30];
          float v1457_data = r13[31];
          tensorforge::VectorT<float, 16> v1458_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1393_data, v1442_data, v1441_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1459_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1394_data, v1443_data, v1458_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1460_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1395_data, v1444_data, v1459_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1461_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1396_data, v1445_data, v1460_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1462_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1397_data, v1446_data, v1461_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1463_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1398_data, v1447_data, v1462_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1464_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1399_data, v1448_data, v1463_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1465_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1400_data, v1449_data, v1464_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1466_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1401_data, v1450_data, v1465_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1467_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1402_data, v1451_data, v1466_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1468_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1403_data, v1452_data, v1467_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1469_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1404_data, v1453_data, v1468_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1470_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1405_data, v1454_data, v1469_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1471_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1406_pad, v1455_data, v1470_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1472_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1407_pad, v1456_data, v1471_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1473_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1408_pad, v1457_data, v1472_acc, 1, 1, 0);
          float v1474_el = v1473_acc[0];
          float v1476_el = v1473_acc[4];
          float v1477_sw = tensorforge::swap<32>(v1476_el);
          float v1479_el = v1473_acc[8];
          float v1482_el = v1473_acc[12];
          float v1483_sw = tensorforge::swap<32>(v1482_el);
          r14[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1483_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1479_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1477_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1474_el, v1474_el))))))));
          float v1486_el = v1473_acc[1];
          float v1488_el = v1473_acc[5];
          float v1489_sw = tensorforge::swap<32>(v1488_el);
          float v1491_el = v1473_acc[9];
          float v1494_el = v1473_acc[13];
          float v1495_sw = tensorforge::swap<32>(v1494_el);
          r14[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1495_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1491_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1489_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1486_el, v1486_el))))))));
          float v1498_el = v1473_acc[2];
          float v1500_el = v1473_acc[6];
          float v1501_sw = tensorforge::swap<32>(v1500_el);
          float v1503_el = v1473_acc[10];
          float v1506_el = v1473_acc[14];
          float v1507_sw = tensorforge::swap<32>(v1506_el);
          r14[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1507_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1503_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1501_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1498_el, v1498_el))))))));
          float v1510_el = v1473_acc[3];
          float v1512_el = v1473_acc[7];
          float v1513_sw = tensorforge::swap<32>(v1512_el);
          float v1515_el = v1473_acc[11];
          float v1518_el = v1473_acc[15];
          float v1519_sw = tensorforge::swap<32>(v1518_el);
          r14[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1519_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1515_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1513_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1510_el, v1510_el))))))));
          float v1523_sw = tensorforge::swap<32>(v1474_el);
          float v1528_sw = tensorforge::swap<32>(v1479_el);
          r14[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1482_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1528_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1476_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1523_sw, v1523_sw))))))));
          float v1535_sw = tensorforge::swap<32>(v1486_el);
          r14[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1494_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1491_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1488_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1535_sw, v1535_sw))))))));
          float v1547_sw = tensorforge::swap<32>(v1498_el);
          r14[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1506_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1503_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1500_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1547_sw, v1547_sw))))))));
          float v1559_sw = tensorforge::swap<32>(v1510_el);
          r14[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1518_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1515_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1512_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1559_sw, v1559_sw))))))));
          float v1571_sw = tensorforge::swap<64>(v1474_el);
          r14[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1483_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1479_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1477_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1571_sw, v1571_sw))))))));
          float v1583_sw = tensorforge::swap<64>(v1486_el);
          r14[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1495_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1491_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1489_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1583_sw, v1583_sw))))))));
          float v1595_sw = tensorforge::swap<64>(v1498_el);
          r14[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1507_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1503_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1501_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1595_sw, v1595_sw))))))));
          float v1607_sw = tensorforge::swap<64>(v1510_el);
          r14[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1519_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1515_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1513_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1607_sw, v1607_sw))))))));
          float v1620_sw = tensorforge::swap<64>(v1523_sw);
          r14[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1482_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1528_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1476_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1620_sw, v1620_sw))))))));
          // glb_m0 = store{r>g}(r14);
          if (v259_g) {
            #pragma unroll
            for (int32_t v1630_i1 = 0; v1630_i1 < 13; ++v1630_i1) {
              float v1632_data = r14[v1630_i1];
              int32_t v1636_a = v23_lead + (v1630_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1636_a], v1632_data);
            }
          }
          float r15[13]{};
          // r15 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1638_i0 = 0; v1638_i0 < 1; ++v1638_i0) {
            int32_t v1641_lead = v23_lead + (v1638_i0 * 32);
            #pragma unroll
            for (int32_t v1639_i1 = 0; v1639_i1 < 13; ++v1639_i1) {
              float v1644_data = glb_m0[(v1641_lead + (v1639_i1 * 32))];
              r15[(v1638_i0 + v1639_i1)] = v1644_data;
            }
          }
          float r16[13]{};
          // r16 = load{g>r}(glb_m10);
          if (v33_g) {
            #pragma unroll
            for (int32_t v1647_i1 = 0; v1647_i1 < 13; ++v1647_i1) {
              float v1652_data = __builtin_nontemporal_load(&glb_m10[(v23_lead + (v1647_i1 * 13))]);
              r16[v1647_i1] = v1652_data;
            }
          }
          // wait(r15 = load{g>r}(glb_m0););
          // wait(r16 = load{g>r}(glb_m10););
          float r17[13]{};
          // r17 = +(r15 * r16) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1655_data = r16[0];
          float v1656_data = r16[1];
          float v1657_data = r16[2];
          float v1658_data = r16[3];
          float v1659_data = r16[4];
          float v1660_data = r16[5];
          float v1661_data = r16[6];
          float v1662_data = r16[7];
          float v1663_data = r16[8];
          float v1664_data = r16[9];
          float v1665_data = r16[10];
          float v1666_data = r16[11];
          float v1667_data = r16[12];
          float v1668_pad{};
          float v1669_pad{};
          float v1670_pad{};
          tensorforge::transpose16x16b32(v1655_data, v1656_data, v1657_data, v1658_data, v1659_data, v1660_data, v1661_data, v1662_data, v1663_data, v1664_data, v1665_data, v1666_data, v1667_data, v1668_pad, v1669_pad, v1670_pad);
          tensorforge::VectorT<float, 16> v1671_acc{};
          float v1672_data = r15[0];
          float v1673_data = r15[1];
          float v1674_data = r15[2];
          float v1675_data = r15[3];
          float v1676_data = r15[4];
          float v1677_data = r15[5];
          float v1678_data = r15[6];
          float v1679_data = r15[7];
          float v1680_data = r15[8];
          float v1681_data = r15[9];
          float v1682_data = r15[10];
          float v1683_data = r15[11];
          float v1684_data = r15[12];
          tensorforge::VectorT<float, 16> v1686_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1655_data, v1672_data, v1671_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1687_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1656_data, v1673_data, v1686_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1688_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1657_data, v1674_data, v1687_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1689_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1658_data, v1675_data, v1688_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1690_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1659_data, v1676_data, v1689_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1691_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1660_data, v1677_data, v1690_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1692_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1661_data, v1678_data, v1691_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1693_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1662_data, v1679_data, v1692_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1694_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1663_data, v1680_data, v1693_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1695_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1664_data, v1681_data, v1694_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1696_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1665_data, v1682_data, v1695_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1697_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1666_data, v1683_data, v1696_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1698_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1667_data, v1684_data, v1697_acc, 1, 0, 0);
          float v1699_el = v1698_acc[0];
          float v1701_el = v1698_acc[4];
          float v1702_sw = tensorforge::swap<32>(v1701_el);
          float v1704_el = v1698_acc[8];
          float v1707_el = v1698_acc[12];
          float v1708_sw = tensorforge::swap<32>(v1707_el);
          r17[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1708_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1704_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1702_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1699_el, v1699_el))))))));
          float v1711_el = v1698_acc[1];
          float v1713_el = v1698_acc[5];
          float v1714_sw = tensorforge::swap<32>(v1713_el);
          float v1716_el = v1698_acc[9];
          float v1719_el = v1698_acc[13];
          float v1720_sw = tensorforge::swap<32>(v1719_el);
          r17[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1720_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1716_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1714_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1711_el, v1711_el))))))));
          float v1723_el = v1698_acc[2];
          float v1725_el = v1698_acc[6];
          float v1726_sw = tensorforge::swap<32>(v1725_el);
          float v1728_el = v1698_acc[10];
          float v1731_el = v1698_acc[14];
          float v1732_sw = tensorforge::swap<32>(v1731_el);
          r17[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1732_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1728_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1726_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1723_el, v1723_el))))))));
          float v1735_el = v1698_acc[3];
          float v1737_el = v1698_acc[7];
          float v1738_sw = tensorforge::swap<32>(v1737_el);
          float v1740_el = v1698_acc[11];
          float v1743_el = v1698_acc[15];
          float v1744_sw = tensorforge::swap<32>(v1743_el);
          r17[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1744_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1740_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1738_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1735_el, v1735_el))))))));
          float v1748_sw = tensorforge::swap<32>(v1699_el);
          float v1753_sw = tensorforge::swap<32>(v1704_el);
          r17[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1707_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1753_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1701_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1748_sw, v1748_sw))))))));
          float v1760_sw = tensorforge::swap<32>(v1711_el);
          r17[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1719_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1716_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1713_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1760_sw, v1760_sw))))))));
          float v1772_sw = tensorforge::swap<32>(v1723_el);
          r17[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1731_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1728_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1725_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1772_sw, v1772_sw))))))));
          float v1784_sw = tensorforge::swap<32>(v1735_el);
          r17[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1743_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1740_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1737_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1784_sw, v1784_sw))))))));
          float v1796_sw = tensorforge::swap<64>(v1699_el);
          r17[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1708_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1704_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1702_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1796_sw, v1796_sw))))))));
          float v1808_sw = tensorforge::swap<64>(v1711_el);
          r17[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1720_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1716_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1714_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1808_sw, v1808_sw))))))));
          float v1820_sw = tensorforge::swap<64>(v1723_el);
          r17[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1732_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1728_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1726_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1820_sw, v1820_sw))))))));
          float v1832_sw = tensorforge::swap<64>(v1735_el);
          r17[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1744_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1740_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1738_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1832_sw, v1832_sw))))))));
          float v1845_sw = tensorforge::swap<64>(v1748_sw);
          r17[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1707_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1753_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1701_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1845_sw, v1845_sw))))))));
          // glb_m9 = store{r>g}(r17);
          #pragma unroll
          for (int32_t v1855_i0 = 0; v1855_i0 < 1; ++v1855_i0) {
            int32_t v1860_lead = v23_lead + (v1855_i0 * 32);
            #pragma unroll
            for (int32_t v1856_i1 = 0; v1856_i1 < 13; ++v1856_i1) {
              float v1858_data = r17[(v1855_i0 + v1856_i1)];
              glb_m9[(v1860_lead + (v1856_i1 * 32))] = v1858_data;
            }
          }
        }
      }
    }
  }
}

