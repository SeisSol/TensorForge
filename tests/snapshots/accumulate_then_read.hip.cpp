// === base name ===
kernel_46db751defe1ba51

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_46db751defe1ba51 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_46db751defe1ba51(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_46db751defe1ba51(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, float * m9, size_t m9_extraOffset, const float * m10, size_t m10_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_46db751defe1ba51(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_46db751defe1ba51, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_46db751defe1ba51, block.x * block.y * block.z, 0));
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
void launcher_kernel_46db751defe1ba51(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, float * m9, size_t m9_extraOffset, const float * m10, size_t m10_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_46db751defe1ba51(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_46db751defe1ba51), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_46db751defe1ba51, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, m9Arg, m9_extraOffset, m10Arg, m10_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_46db751defe1ba51(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m9, size_t m9_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m10, size_t m10_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m1););
          float r3[13]{};
          // r3 = load{g>r}(glb_m3);
          if (v39_g) {
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 13; ++v48_i1) {
              float v53_data = __builtin_nontemporal_load(&glb_m3[(v29_lead + (v48_i1 * 13))]);
              r3[v48_i1] = v53_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[13]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v56_data = r1[0];
          float v57_data = r1[1];
          float v58_data = r1[2];
          float v59_data = r1[3];
          float v60_data = r1[4];
          float v61_data = r1[5];
          float v62_data = r1[6];
          float v63_data = r1[7];
          float v64_data = r1[8];
          float v65_data = r1[9];
          float v66_data = r1[10];
          float v67_data = r1[11];
          float v68_data = r1[12];
          float v69_pad{};
          float v70_pad{};
          float v71_pad{};
          tensorforge::transpose16x16b32(v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data, v67_data, v68_data, v69_pad, v70_pad, v71_pad);
          tensorforge::VectorT<float, 16> v72_acc{};
          float v73_data = r0[0];
          float v74_data = r0[1];
          float v75_data = r0[2];
          float v76_data = r0[3];
          float v77_data = r0[4];
          float v78_data = r0[5];
          float v79_data = r0[6];
          float v80_data = r0[7];
          float v81_data = r0[8];
          float v82_data = r0[9];
          float v83_data = r0[10];
          float v84_data = r0[11];
          float v85_data = r0[12];
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v72_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v90_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v91_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v92_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v93_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v81_data, v94_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v82_data, v95_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v83_data, v96_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v98_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v67_data, v84_data, v97_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v68_data, v85_data, v98_acc, 1, 0, 0);
          float v100_el = v99_acc[0];
          float v102_el = v99_acc[4];
          float v103_sw = tensorforge::swap<32>(v102_el);
          float v105_el = v99_acc[8];
          float v108_el = v99_acc[12];
          float v109_sw = tensorforge::swap<32>(v108_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v109_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v103_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v100_el, v100_el))))))));
          float v112_el = v99_acc[1];
          float v114_el = v99_acc[5];
          float v115_sw = tensorforge::swap<32>(v114_el);
          float v117_el = v99_acc[9];
          float v120_el = v99_acc[13];
          float v121_sw = tensorforge::swap<32>(v120_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v121_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v115_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v112_el, v112_el))))))));
          float v124_el = v99_acc[2];
          float v126_el = v99_acc[6];
          float v127_sw = tensorforge::swap<32>(v126_el);
          float v129_el = v99_acc[10];
          float v132_el = v99_acc[14];
          float v133_sw = tensorforge::swap<32>(v132_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v124_el, v124_el))))))));
          float v136_el = v99_acc[3];
          float v138_el = v99_acc[7];
          float v139_sw = tensorforge::swap<32>(v138_el);
          float v141_el = v99_acc[11];
          float v144_el = v99_acc[15];
          float v145_sw = tensorforge::swap<32>(v144_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v136_el, v136_el))))))));
          float v149_sw = tensorforge::swap<32>(v100_el);
          float v154_sw = tensorforge::swap<32>(v105_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v108_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v154_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v102_el, (tensorforge::dppUpdate<228, 1, 15, false>(v149_sw, v149_sw))))))));
          float v161_sw = tensorforge::swap<32>(v112_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v120_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v117_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v114_el, (tensorforge::dppUpdate<228, 1, 15, false>(v161_sw, v161_sw))))))));
          float v173_sw = tensorforge::swap<32>(v124_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v129_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v126_el, (tensorforge::dppUpdate<228, 1, 15, false>(v173_sw, v173_sw))))))));
          float v185_sw = tensorforge::swap<32>(v136_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v141_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v138_el, (tensorforge::dppUpdate<228, 1, 15, false>(v185_sw, v185_sw))))))));
          float v197_sw = tensorforge::swap<64>(v100_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v109_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v105_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v103_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v197_sw, v197_sw))))))));
          float v209_sw = tensorforge::swap<64>(v112_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v121_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v117_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v115_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v209_sw, v209_sw))))))));
          float v221_sw = tensorforge::swap<64>(v124_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v129_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v221_sw, v221_sw))))))));
          float v233_sw = tensorforge::swap<64>(v136_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v141_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
          float v246_sw = tensorforge::swap<64>(v149_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v108_el, (tensorforge::dppUpdate<228, 4, 15, false>(v154_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v102_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v246_sw, v246_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v256_i0 = 0; v256_i0 < 1; ++v256_i0) {
            int32_t v261_lead = v29_lead + (v256_i0 * 32);
            #pragma unroll
            for (int32_t v257_i1 = 0; v257_i1 < 13; ++v257_i1) {
              float v259_data = r2[(v256_i0 + v257_i1)];
              glb_m0[(v261_lead + (v257_i1 * 32))] = v259_data;
            }
          }
          float r5[32]{};
          // r5 = load{g>r}(glb_m4);
          bool v265_g = v29_lead < 16;
          if (v265_g) {
            #pragma unroll
            for (int32_t v266_i1 = 0; v266_i1 < 32; ++v266_i1) {
              float v271_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v266_i1 * 16))]);
              r5[v266_i1] = v271_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r4[13]{};
          // r4 = +(r0 * r3) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v274_data = r3[0];
          float v275_data = r3[1];
          float v276_data = r3[2];
          float v277_data = r3[3];
          float v278_data = r3[4];
          float v279_data = r3[5];
          float v280_data = r3[6];
          float v281_data = r3[7];
          float v282_data = r3[8];
          float v283_data = r3[9];
          float v284_data = r3[10];
          float v285_data = r3[11];
          float v286_data = r3[12];
          float v287_pad{};
          float v288_pad{};
          float v289_pad{};
          tensorforge::transpose16x16b32(v274_data, v275_data, v276_data, v277_data, v278_data, v279_data, v280_data, v281_data, v282_data, v283_data, v284_data, v285_data, v286_data, v287_pad, v288_pad, v289_pad);
          tensorforge::VectorT<float, 16> v290_acc{};
          tensorforge::VectorT<float, 16> v305_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v274_data, v73_data, v290_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v306_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v275_data, v74_data, v305_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v307_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v276_data, v75_data, v306_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v308_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v277_data, v76_data, v307_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v309_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v278_data, v77_data, v308_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v310_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v279_data, v78_data, v309_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v311_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v280_data, v79_data, v310_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v312_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v281_data, v80_data, v311_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v313_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v282_data, v81_data, v312_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v314_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v283_data, v82_data, v313_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v315_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v284_data, v83_data, v314_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v316_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v285_data, v84_data, v315_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v317_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v286_data, v85_data, v316_acc, 1, 0, 0);
          float v318_el = v317_acc[0];
          float v320_el = v317_acc[4];
          float v321_sw = tensorforge::swap<32>(v320_el);
          float v323_el = v317_acc[8];
          float v326_el = v317_acc[12];
          float v327_sw = tensorforge::swap<32>(v326_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v327_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v323_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v321_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v318_el, v318_el))))))));
          float v330_el = v317_acc[1];
          float v332_el = v317_acc[5];
          float v333_sw = tensorforge::swap<32>(v332_el);
          float v335_el = v317_acc[9];
          float v338_el = v317_acc[13];
          float v339_sw = tensorforge::swap<32>(v338_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v339_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v335_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v333_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v330_el, v330_el))))))));
          float v342_el = v317_acc[2];
          float v344_el = v317_acc[6];
          float v345_sw = tensorforge::swap<32>(v344_el);
          float v347_el = v317_acc[10];
          float v350_el = v317_acc[14];
          float v351_sw = tensorforge::swap<32>(v350_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v351_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v347_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v345_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v342_el, v342_el))))))));
          float v354_el = v317_acc[3];
          float v356_el = v317_acc[7];
          float v357_sw = tensorforge::swap<32>(v356_el);
          float v359_el = v317_acc[11];
          float v362_el = v317_acc[15];
          float v363_sw = tensorforge::swap<32>(v362_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v363_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v359_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v357_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v354_el, v354_el))))))));
          float v367_sw = tensorforge::swap<32>(v318_el);
          float v372_sw = tensorforge::swap<32>(v323_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v326_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v372_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v320_el, (tensorforge::dppUpdate<228, 1, 15, false>(v367_sw, v367_sw))))))));
          float v379_sw = tensorforge::swap<32>(v330_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v338_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v335_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v332_el, (tensorforge::dppUpdate<228, 1, 15, false>(v379_sw, v379_sw))))))));
          float v391_sw = tensorforge::swap<32>(v342_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v350_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v347_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v344_el, (tensorforge::dppUpdate<228, 1, 15, false>(v391_sw, v391_sw))))))));
          float v403_sw = tensorforge::swap<32>(v354_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v362_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v359_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v356_el, (tensorforge::dppUpdate<228, 1, 15, false>(v403_sw, v403_sw))))))));
          float v415_sw = tensorforge::swap<64>(v318_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v327_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v323_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v321_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v415_sw, v415_sw))))))));
          float v427_sw = tensorforge::swap<64>(v330_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v339_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v335_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v333_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v427_sw, v427_sw))))))));
          float v439_sw = tensorforge::swap<64>(v342_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v351_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v347_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v345_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v439_sw, v439_sw))))))));
          float v451_sw = tensorforge::swap<64>(v354_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v363_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v359_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v357_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v451_sw, v451_sw))))))));
          float v464_sw = tensorforge::swap<64>(v367_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v326_el, (tensorforge::dppUpdate<228, 4, 15, false>(v372_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v320_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v464_sw, v464_sw))))))));
          float r7[13]{};
          // r7 = load{g>r}(glb_m5);
          if (v39_g) {
            #pragma unroll
            for (int32_t v475_i1 = 0; v475_i1 < 13; ++v475_i1) {
              float v480_data = __builtin_nontemporal_load(&glb_m5[(v29_lead + (v475_i1 * 13))]);
              r7[v475_i1] = v480_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[13]{};
          // r6 = +(r5 * r4) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v483_data = r4[0];
          float v484_data = r4[1];
          float v485_data = r4[2];
          float v486_data = r4[3];
          float v487_data = r4[4];
          float v488_data = r4[5];
          float v489_data = r4[6];
          float v490_data = r4[7];
          float v491_data = r4[8];
          float v492_data = r4[9];
          float v493_data = r4[10];
          float v494_data = r4[11];
          float v495_data = r4[12];
          float v496_pad{};
          float v497_pad{};
          float v498_pad{};
          tensorforge::transpose16x16b32(v483_data, v484_data, v485_data, v486_data, v487_data, v488_data, v489_data, v490_data, v491_data, v492_data, v493_data, v494_data, v495_data, v496_pad, v497_pad, v498_pad);
          tensorforge::VectorT<float, 16> v499_acc{};
          float v500_data = r5[0];
          float v501_data = r5[1];
          float v502_data = r5[2];
          float v503_data = r5[3];
          float v504_data = r5[4];
          float v505_data = r5[5];
          float v506_data = r5[6];
          float v507_data = r5[7];
          float v508_data = r5[8];
          float v509_data = r5[9];
          float v510_data = r5[10];
          float v511_data = r5[11];
          float v512_data = r5[12];
          float v513_data = r5[13];
          float v514_data = r5[14];
          float v515_data = r5[15];
          tensorforge::VectorT<float, 16> v516_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v483_data, v500_data, v499_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v517_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v484_data, v501_data, v516_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v518_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v485_data, v502_data, v517_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v519_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v486_data, v503_data, v518_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v520_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v487_data, v504_data, v519_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v521_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v488_data, v505_data, v520_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v522_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v489_data, v506_data, v521_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v523_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v490_data, v507_data, v522_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v524_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v491_data, v508_data, v523_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v525_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v492_data, v509_data, v524_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v526_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v493_data, v510_data, v525_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v527_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v494_data, v511_data, v526_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v528_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v495_data, v512_data, v527_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v529_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v496_pad, v513_data, v528_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v530_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v497_pad, v514_data, v529_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v531_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v498_pad, v515_data, v530_acc, 1, 0, 0);
          float v532_data = r5[16];
          float v533_data = r5[17];
          float v534_data = r5[18];
          float v535_data = r5[19];
          float v536_data = r5[20];
          float v537_data = r5[21];
          float v538_data = r5[22];
          float v539_data = r5[23];
          float v540_data = r5[24];
          float v541_data = r5[25];
          float v542_data = r5[26];
          float v543_data = r5[27];
          float v544_data = r5[28];
          float v545_data = r5[29];
          float v546_data = r5[30];
          float v547_data = r5[31];
          tensorforge::VectorT<float, 16> v548_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v483_data, v532_data, v531_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v549_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v484_data, v533_data, v548_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v550_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v485_data, v534_data, v549_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v551_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v486_data, v535_data, v550_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v552_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v487_data, v536_data, v551_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v553_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v488_data, v537_data, v552_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v554_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v489_data, v538_data, v553_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v555_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v490_data, v539_data, v554_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v556_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v491_data, v540_data, v555_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v557_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v492_data, v541_data, v556_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v558_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v493_data, v542_data, v557_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v559_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v494_data, v543_data, v558_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v560_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v495_data, v544_data, v559_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v561_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v496_pad, v545_data, v560_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v562_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v497_pad, v546_data, v561_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v563_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v498_pad, v547_data, v562_acc, 1, 1, 0);
          float v564_el = v563_acc[0];
          float v566_el = v563_acc[4];
          float v567_sw = tensorforge::swap<32>(v566_el);
          float v569_el = v563_acc[8];
          float v572_el = v563_acc[12];
          float v573_sw = tensorforge::swap<32>(v572_el);
          r6[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v573_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v569_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v567_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v564_el, v564_el))))))));
          float v576_el = v563_acc[1];
          float v578_el = v563_acc[5];
          float v579_sw = tensorforge::swap<32>(v578_el);
          float v581_el = v563_acc[9];
          float v584_el = v563_acc[13];
          float v585_sw = tensorforge::swap<32>(v584_el);
          r6[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v585_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v581_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v579_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v576_el, v576_el))))))));
          float v588_el = v563_acc[2];
          float v590_el = v563_acc[6];
          float v591_sw = tensorforge::swap<32>(v590_el);
          float v593_el = v563_acc[10];
          float v596_el = v563_acc[14];
          float v597_sw = tensorforge::swap<32>(v596_el);
          r6[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v597_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v593_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v591_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v588_el, v588_el))))))));
          float v600_el = v563_acc[3];
          float v602_el = v563_acc[7];
          float v603_sw = tensorforge::swap<32>(v602_el);
          float v605_el = v563_acc[11];
          float v608_el = v563_acc[15];
          float v609_sw = tensorforge::swap<32>(v608_el);
          r6[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v609_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v605_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v603_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v600_el, v600_el))))))));
          float v613_sw = tensorforge::swap<32>(v564_el);
          float v618_sw = tensorforge::swap<32>(v569_el);
          r6[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v572_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v618_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v566_el, (tensorforge::dppUpdate<228, 1, 15, false>(v613_sw, v613_sw))))))));
          float v625_sw = tensorforge::swap<32>(v576_el);
          r6[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v584_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v581_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v578_el, (tensorforge::dppUpdate<228, 1, 15, false>(v625_sw, v625_sw))))))));
          float v637_sw = tensorforge::swap<32>(v588_el);
          r6[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v596_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v593_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v590_el, (tensorforge::dppUpdate<228, 1, 15, false>(v637_sw, v637_sw))))))));
          float v649_sw = tensorforge::swap<32>(v600_el);
          r6[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v608_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v605_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v602_el, (tensorforge::dppUpdate<228, 1, 15, false>(v649_sw, v649_sw))))))));
          float v661_sw = tensorforge::swap<64>(v564_el);
          r6[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v573_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v569_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v567_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v661_sw, v661_sw))))))));
          float v673_sw = tensorforge::swap<64>(v576_el);
          r6[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v585_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v581_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v579_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v673_sw, v673_sw))))))));
          float v685_sw = tensorforge::swap<64>(v588_el);
          r6[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v597_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v593_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v591_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v685_sw, v685_sw))))))));
          float v697_sw = tensorforge::swap<64>(v600_el);
          r6[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v609_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v605_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v603_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v697_sw, v697_sw))))))));
          float v710_sw = tensorforge::swap<64>(v613_sw);
          r6[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v572_el, (tensorforge::dppUpdate<228, 4, 15, false>(v618_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v566_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v710_sw, v710_sw))))))));
          // glb_m0 = store{r>g}(r6);
          if (v265_g) {
            #pragma unroll
            for (int32_t v720_i1 = 0; v720_i1 < 13; ++v720_i1) {
              float v722_data = r6[v720_i1];
              int32_t v726_a = v29_lead + (v720_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v726_a], v722_data);
            }
          }
          float r9[32]{};
          // r9 = load{g>r}(glb_m6);
          if (v265_g) {
            #pragma unroll
            for (int32_t v728_i1 = 0; v728_i1 < 32; ++v728_i1) {
              float v733_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v728_i1 * 16))]);
              r9[v728_i1] = v733_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r8[13]{};
          // r8 = +(r0 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v736_data = r7[0];
          float v737_data = r7[1];
          float v738_data = r7[2];
          float v739_data = r7[3];
          float v740_data = r7[4];
          float v741_data = r7[5];
          float v742_data = r7[6];
          float v743_data = r7[7];
          float v744_data = r7[8];
          float v745_data = r7[9];
          float v746_data = r7[10];
          float v747_data = r7[11];
          float v748_data = r7[12];
          float v749_pad{};
          float v750_pad{};
          float v751_pad{};
          tensorforge::transpose16x16b32(v736_data, v737_data, v738_data, v739_data, v740_data, v741_data, v742_data, v743_data, v744_data, v745_data, v746_data, v747_data, v748_data, v749_pad, v750_pad, v751_pad);
          tensorforge::VectorT<float, 16> v752_acc{};
          tensorforge::VectorT<float, 16> v767_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v736_data, v73_data, v752_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v768_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v737_data, v74_data, v767_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v769_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v738_data, v75_data, v768_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v770_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v739_data, v76_data, v769_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v771_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v740_data, v77_data, v770_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v772_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v741_data, v78_data, v771_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v773_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v742_data, v79_data, v772_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v774_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v743_data, v80_data, v773_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v775_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v744_data, v81_data, v774_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v776_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v745_data, v82_data, v775_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v777_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v746_data, v83_data, v776_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v778_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v747_data, v84_data, v777_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v779_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v748_data, v85_data, v778_acc, 1, 0, 0);
          float v780_el = v779_acc[0];
          float v782_el = v779_acc[4];
          float v783_sw = tensorforge::swap<32>(v782_el);
          float v785_el = v779_acc[8];
          float v788_el = v779_acc[12];
          float v789_sw = tensorforge::swap<32>(v788_el);
          r8[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v789_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v785_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v783_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v780_el, v780_el))))))));
          float v792_el = v779_acc[1];
          float v794_el = v779_acc[5];
          float v795_sw = tensorforge::swap<32>(v794_el);
          float v797_el = v779_acc[9];
          float v800_el = v779_acc[13];
          float v801_sw = tensorforge::swap<32>(v800_el);
          r8[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v801_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v797_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v795_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v792_el, v792_el))))))));
          float v804_el = v779_acc[2];
          float v806_el = v779_acc[6];
          float v807_sw = tensorforge::swap<32>(v806_el);
          float v809_el = v779_acc[10];
          float v812_el = v779_acc[14];
          float v813_sw = tensorforge::swap<32>(v812_el);
          r8[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v813_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v809_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v807_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v804_el, v804_el))))))));
          float v816_el = v779_acc[3];
          float v818_el = v779_acc[7];
          float v819_sw = tensorforge::swap<32>(v818_el);
          float v821_el = v779_acc[11];
          float v824_el = v779_acc[15];
          float v825_sw = tensorforge::swap<32>(v824_el);
          r8[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v825_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v821_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v819_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v816_el, v816_el))))))));
          float v829_sw = tensorforge::swap<32>(v780_el);
          float v834_sw = tensorforge::swap<32>(v785_el);
          r8[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v788_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v834_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v782_el, (tensorforge::dppUpdate<228, 1, 15, false>(v829_sw, v829_sw))))))));
          float v841_sw = tensorforge::swap<32>(v792_el);
          r8[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v800_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v797_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v794_el, (tensorforge::dppUpdate<228, 1, 15, false>(v841_sw, v841_sw))))))));
          float v853_sw = tensorforge::swap<32>(v804_el);
          r8[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v812_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v809_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v806_el, (tensorforge::dppUpdate<228, 1, 15, false>(v853_sw, v853_sw))))))));
          float v865_sw = tensorforge::swap<32>(v816_el);
          r8[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v824_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v821_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v818_el, (tensorforge::dppUpdate<228, 1, 15, false>(v865_sw, v865_sw))))))));
          float v877_sw = tensorforge::swap<64>(v780_el);
          r8[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v789_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v785_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v783_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v877_sw, v877_sw))))))));
          float v889_sw = tensorforge::swap<64>(v792_el);
          r8[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v801_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v797_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v795_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v889_sw, v889_sw))))))));
          float v901_sw = tensorforge::swap<64>(v804_el);
          r8[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v813_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v809_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v807_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v901_sw, v901_sw))))))));
          float v913_sw = tensorforge::swap<64>(v816_el);
          r8[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v825_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v821_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v819_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v913_sw, v913_sw))))))));
          float v926_sw = tensorforge::swap<64>(v829_sw);
          r8[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v788_el, (tensorforge::dppUpdate<228, 4, 15, false>(v834_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v782_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v926_sw, v926_sw))))))));
          float r11[13]{};
          // r11 = load{g>r}(glb_m7);
          if (v39_g) {
            #pragma unroll
            for (int32_t v937_i1 = 0; v937_i1 < 13; ++v937_i1) {
              float v942_data = __builtin_nontemporal_load(&glb_m7[(v29_lead + (v937_i1 * 13))]);
              r11[v937_i1] = v942_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[13]{};
          // r10 = +(r9 * r8) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v945_data = r8[0];
          float v946_data = r8[1];
          float v947_data = r8[2];
          float v948_data = r8[3];
          float v949_data = r8[4];
          float v950_data = r8[5];
          float v951_data = r8[6];
          float v952_data = r8[7];
          float v953_data = r8[8];
          float v954_data = r8[9];
          float v955_data = r8[10];
          float v956_data = r8[11];
          float v957_data = r8[12];
          float v958_pad{};
          float v959_pad{};
          float v960_pad{};
          tensorforge::transpose16x16b32(v945_data, v946_data, v947_data, v948_data, v949_data, v950_data, v951_data, v952_data, v953_data, v954_data, v955_data, v956_data, v957_data, v958_pad, v959_pad, v960_pad);
          tensorforge::VectorT<float, 16> v961_acc{};
          float v962_data = r9[0];
          float v963_data = r9[1];
          float v964_data = r9[2];
          float v965_data = r9[3];
          float v966_data = r9[4];
          float v967_data = r9[5];
          float v968_data = r9[6];
          float v969_data = r9[7];
          float v970_data = r9[8];
          float v971_data = r9[9];
          float v972_data = r9[10];
          float v973_data = r9[11];
          float v974_data = r9[12];
          float v975_data = r9[13];
          float v976_data = r9[14];
          float v977_data = r9[15];
          tensorforge::VectorT<float, 16> v978_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v945_data, v962_data, v961_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v979_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v946_data, v963_data, v978_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v980_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v947_data, v964_data, v979_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v981_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v948_data, v965_data, v980_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v982_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v949_data, v966_data, v981_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v983_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v950_data, v967_data, v982_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v984_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v951_data, v968_data, v983_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v985_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v952_data, v969_data, v984_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v986_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v953_data, v970_data, v985_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v987_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v954_data, v971_data, v986_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v988_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v955_data, v972_data, v987_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v989_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v956_data, v973_data, v988_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v990_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v957_data, v974_data, v989_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v991_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v958_pad, v975_data, v990_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v992_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v959_pad, v976_data, v991_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v993_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v960_pad, v977_data, v992_acc, 1, 0, 0);
          float v994_data = r9[16];
          float v995_data = r9[17];
          float v996_data = r9[18];
          float v997_data = r9[19];
          float v998_data = r9[20];
          float v999_data = r9[21];
          float v1000_data = r9[22];
          float v1001_data = r9[23];
          float v1002_data = r9[24];
          float v1003_data = r9[25];
          float v1004_data = r9[26];
          float v1005_data = r9[27];
          float v1006_data = r9[28];
          float v1007_data = r9[29];
          float v1008_data = r9[30];
          float v1009_data = r9[31];
          tensorforge::VectorT<float, 16> v1010_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v945_data, v994_data, v993_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1011_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v946_data, v995_data, v1010_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1012_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v947_data, v996_data, v1011_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1013_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v948_data, v997_data, v1012_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1014_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v949_data, v998_data, v1013_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1015_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v950_data, v999_data, v1014_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1016_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v951_data, v1000_data, v1015_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1017_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v952_data, v1001_data, v1016_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1018_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v953_data, v1002_data, v1017_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1019_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v954_data, v1003_data, v1018_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1020_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v955_data, v1004_data, v1019_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1021_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v956_data, v1005_data, v1020_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1022_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v957_data, v1006_data, v1021_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1023_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v958_pad, v1007_data, v1022_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1024_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v959_pad, v1008_data, v1023_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v1025_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v960_pad, v1009_data, v1024_acc, 1, 1, 0);
          float v1026_el = v1025_acc[0];
          float v1028_el = v1025_acc[4];
          float v1029_sw = tensorforge::swap<32>(v1028_el);
          float v1031_el = v1025_acc[8];
          float v1034_el = v1025_acc[12];
          float v1035_sw = tensorforge::swap<32>(v1034_el);
          r10[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1035_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1031_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1029_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1026_el, v1026_el))))))));
          float v1038_el = v1025_acc[1];
          float v1040_el = v1025_acc[5];
          float v1041_sw = tensorforge::swap<32>(v1040_el);
          float v1043_el = v1025_acc[9];
          float v1046_el = v1025_acc[13];
          float v1047_sw = tensorforge::swap<32>(v1046_el);
          r10[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1047_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1043_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1041_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1038_el, v1038_el))))))));
          float v1050_el = v1025_acc[2];
          float v1052_el = v1025_acc[6];
          float v1053_sw = tensorforge::swap<32>(v1052_el);
          float v1055_el = v1025_acc[10];
          float v1058_el = v1025_acc[14];
          float v1059_sw = tensorforge::swap<32>(v1058_el);
          r10[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1059_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1055_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1053_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1050_el, v1050_el))))))));
          float v1062_el = v1025_acc[3];
          float v1064_el = v1025_acc[7];
          float v1065_sw = tensorforge::swap<32>(v1064_el);
          float v1067_el = v1025_acc[11];
          float v1070_el = v1025_acc[15];
          float v1071_sw = tensorforge::swap<32>(v1070_el);
          r10[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1071_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1067_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1065_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1062_el, v1062_el))))))));
          float v1075_sw = tensorforge::swap<32>(v1026_el);
          float v1080_sw = tensorforge::swap<32>(v1031_el);
          r10[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1034_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1080_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1028_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1075_sw, v1075_sw))))))));
          float v1087_sw = tensorforge::swap<32>(v1038_el);
          r10[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1046_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1043_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1040_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1087_sw, v1087_sw))))))));
          float v1099_sw = tensorforge::swap<32>(v1050_el);
          r10[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1058_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1055_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1052_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1099_sw, v1099_sw))))))));
          float v1111_sw = tensorforge::swap<32>(v1062_el);
          r10[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1070_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1067_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1064_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1111_sw, v1111_sw))))))));
          float v1123_sw = tensorforge::swap<64>(v1026_el);
          r10[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1035_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1031_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1029_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1123_sw, v1123_sw))))))));
          float v1135_sw = tensorforge::swap<64>(v1038_el);
          r10[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1047_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1043_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1041_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1135_sw, v1135_sw))))))));
          float v1147_sw = tensorforge::swap<64>(v1050_el);
          r10[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1059_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1055_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1053_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1147_sw, v1147_sw))))))));
          float v1159_sw = tensorforge::swap<64>(v1062_el);
          r10[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1071_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1067_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1065_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1159_sw, v1159_sw))))))));
          float v1172_sw = tensorforge::swap<64>(v1075_sw);
          r10[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1034_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1080_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1028_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1172_sw, v1172_sw))))))));
          // glb_m0 = store{r>g}(r10);
          if (v265_g) {
            #pragma unroll
            for (int32_t v1182_i1 = 0; v1182_i1 < 13; ++v1182_i1) {
              float v1184_data = r10[v1182_i1];
              int32_t v1188_a = v29_lead + (v1182_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1188_a], v1184_data);
            }
          }
          float r13[32]{};
          // r13 = load{g>r}(glb_m8);
          if (v265_g) {
            #pragma unroll
            for (int32_t v1190_i1 = 0; v1190_i1 < 32; ++v1190_i1) {
              float v1195_data = __builtin_nontemporal_load(&glb_m8[(v29_lead + (v1190_i1 * 16))]);
              r13[v1190_i1] = v1195_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m7););
          float r12[13]{};
          // r12 = +(r0 * r11) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1198_data = r11[0];
          float v1199_data = r11[1];
          float v1200_data = r11[2];
          float v1201_data = r11[3];
          float v1202_data = r11[4];
          float v1203_data = r11[5];
          float v1204_data = r11[6];
          float v1205_data = r11[7];
          float v1206_data = r11[8];
          float v1207_data = r11[9];
          float v1208_data = r11[10];
          float v1209_data = r11[11];
          float v1210_data = r11[12];
          float v1211_pad{};
          float v1212_pad{};
          float v1213_pad{};
          tensorforge::transpose16x16b32(v1198_data, v1199_data, v1200_data, v1201_data, v1202_data, v1203_data, v1204_data, v1205_data, v1206_data, v1207_data, v1208_data, v1209_data, v1210_data, v1211_pad, v1212_pad, v1213_pad);
          tensorforge::VectorT<float, 16> v1214_acc{};
          tensorforge::VectorT<float, 16> v1229_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1198_data, v73_data, v1214_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1230_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1199_data, v74_data, v1229_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1231_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1200_data, v75_data, v1230_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1232_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1201_data, v76_data, v1231_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1233_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1202_data, v77_data, v1232_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1234_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1203_data, v78_data, v1233_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1235_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1204_data, v79_data, v1234_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1236_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1205_data, v80_data, v1235_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1237_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1206_data, v81_data, v1236_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1238_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1207_data, v82_data, v1237_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1239_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1208_data, v83_data, v1238_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1240_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1209_data, v84_data, v1239_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v1241_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1210_data, v85_data, v1240_acc, 1, 0, 0);
          float v1242_el = v1241_acc[0];
          float v1244_el = v1241_acc[4];
          float v1245_sw = tensorforge::swap<32>(v1244_el);
          float v1247_el = v1241_acc[8];
          float v1250_el = v1241_acc[12];
          float v1251_sw = tensorforge::swap<32>(v1250_el);
          r12[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1251_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1247_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1245_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1242_el, v1242_el))))))));
          float v1254_el = v1241_acc[1];
          float v1256_el = v1241_acc[5];
          float v1257_sw = tensorforge::swap<32>(v1256_el);
          float v1259_el = v1241_acc[9];
          float v1262_el = v1241_acc[13];
          float v1263_sw = tensorforge::swap<32>(v1262_el);
          r12[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1263_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1259_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1257_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1254_el, v1254_el))))))));
          float v1266_el = v1241_acc[2];
          float v1268_el = v1241_acc[6];
          float v1269_sw = tensorforge::swap<32>(v1268_el);
          float v1271_el = v1241_acc[10];
          float v1274_el = v1241_acc[14];
          float v1275_sw = tensorforge::swap<32>(v1274_el);
          r12[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1275_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1271_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1269_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1266_el, v1266_el))))))));
          float v1278_el = v1241_acc[3];
          float v1280_el = v1241_acc[7];
          float v1281_sw = tensorforge::swap<32>(v1280_el);
          float v1283_el = v1241_acc[11];
          float v1286_el = v1241_acc[15];
          float v1287_sw = tensorforge::swap<32>(v1286_el);
          r12[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1287_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1283_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1281_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1278_el, v1278_el))))))));
          float v1291_sw = tensorforge::swap<32>(v1242_el);
          float v1296_sw = tensorforge::swap<32>(v1247_el);
          r12[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1250_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1296_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1244_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1291_sw, v1291_sw))))))));
          float v1303_sw = tensorforge::swap<32>(v1254_el);
          r12[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1262_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1259_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1256_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1303_sw, v1303_sw))))))));
          float v1315_sw = tensorforge::swap<32>(v1266_el);
          r12[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1274_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1271_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1268_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1315_sw, v1315_sw))))))));
          float v1327_sw = tensorforge::swap<32>(v1278_el);
          r12[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1286_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1283_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1280_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1327_sw, v1327_sw))))))));
          float v1339_sw = tensorforge::swap<64>(v1242_el);
          r12[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v1251_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1247_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1245_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1339_sw, v1339_sw))))))));
          float v1351_sw = tensorforge::swap<64>(v1254_el);
          r12[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v1263_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1259_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1257_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1351_sw, v1351_sw))))))));
          float v1363_sw = tensorforge::swap<64>(v1266_el);
          r12[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v1275_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1271_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1269_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1363_sw, v1363_sw))))))));
          float v1375_sw = tensorforge::swap<64>(v1278_el);
          r12[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v1287_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1283_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1281_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1375_sw, v1375_sw))))))));
          float v1388_sw = tensorforge::swap<64>(v1291_sw);
          r12[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v1250_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1296_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1244_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1388_sw, v1388_sw))))))));
          // wait(r13 = load{g>r}(glb_m8););
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
          if (v265_g) {
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
          // wait(r15 = load{g>r}(glb_m0););
          // wait(r16 = load{g>r}(glb_m10););
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

