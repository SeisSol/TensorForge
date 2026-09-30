// === base name ===
kernel_96ceed1fe3ecd983

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_96ceed1fe3ecd983 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_96ceed1fe3ecd983(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_96ceed1fe3ecd983(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_96ceed1fe3ecd983(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_96ceed1fe3ecd983, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_96ceed1fe3ecd983, block.x * block.y * block.z, 0));
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
void launcher_kernel_96ceed1fe3ecd983(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_96ceed1fe3ecd983(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_96ceed1fe3ecd983), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_96ceed1fe3ecd983, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_96ceed1fe3ecd983(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(12×12) {0..12}×{0..12} strided
    //   m3 32×32(4×12) {0..4}×{0..12} strided
    //   m4 32×32(12×12) {0..12}×{0..12} strided
    //   m5 32×32(12×12) {0..12}×{0..12} strided
    //   m6 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = t0[i,k] × m2[k,j]
    //   t0[i,j] = m3[i,k] × m1[k,j]
    //   t0[i,j] += t1[i,k] × m4[k,j]
    //   m5[i,j] = m6[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[4,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m6","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v5_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v5_batchId0 * 48 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 144 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v5_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v5_batchId0 * 144 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 16;
          bool v24_g = v23_lead < 12;
          if (v24_g) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v23_lead + (v25_i1 * 12))]);
              r0[v25_i1] = v30_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v24_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v33_i1 * 12))]);
              r1[v33_i1] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v24_g) {
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
              float v46_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v41_i1 * 12))]);
              r3[v41_i1] = v46_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
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
          if (v24_g) {
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 12; ++v160_i1) {
              float v162_data = r2[v160_i1];
              int32_t v166_a = v23_lead + (v160_i1 * 12);
              s0[(v166_a ^ ((v166_a >> 4) & 15))] = v162_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          bool v171_g = v23_lead < 4;
          if (v171_g) {
            #pragma unroll
            for (int32_t v172_i1 = 0; v172_i1 < 12; ++v172_i1) {
              float v177_data = __builtin_nontemporal_load(&glb_m3[(v23_lead + (v172_i1 * 4))]);
              r5[v172_i1] = v177_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v180_data = r3[0];
          float v181_data = r3[1];
          float v182_data = r3[2];
          float v183_data = r3[3];
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          tensorforge::transpose4x4b32(v184_tp, v185_tp, v186_tp, v187_tp, v180_data, v181_data, v182_data, v183_data);
          tensorforge::VectorT<float, 4> v188_acc{};
          int32_t v193_sw = (v23_lead >> 4) & 15;
          float v195_data = s0[(v23_lead ^ v193_sw)];
          int32_t v196_a = v23_lead + 12;
          int32_t v197_sw = v196_a >> 4;
          float v200_data = s0[(v196_a ^ (v197_sw & 15))];
          int32_t v201_a = v23_lead + 24;
          int32_t v202_sw = v201_a >> 4;
          float v205_data = s0[(v201_a ^ (v202_sw & 15))];
          int32_t v206_a = v23_lead + 36;
          int32_t v207_sw = v206_a >> 4;
          float v210_data = s0[(v206_a ^ (v207_sw & 15))];
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v195_data, v188_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v200_data, v211_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v205_data, v212_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v210_data, v213_acc, 2, 0, 0);
          int32_t v215_a = v23_lead + 48;
          int32_t v216_sw = v215_a >> 4;
          float v219_data = s0[(v215_a ^ (v216_sw & 15))];
          int32_t v220_a = v23_lead + 60;
          int32_t v221_sw = v220_a >> 4;
          float v224_data = s0[(v220_a ^ (v221_sw & 15))];
          int32_t v225_a = v23_lead + 72;
          int32_t v226_sw = v225_a >> 4;
          float v229_data = s0[(v225_a ^ (v226_sw & 15))];
          int32_t v230_a = v23_lead + 84;
          int32_t v231_sw = v230_a >> 4;
          float v234_data = s0[(v230_a ^ (v231_sw & 15))];
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v219_data, v214_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v224_data, v235_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v229_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v234_data, v237_acc, 2, 1, 0);
          int32_t v239_a = v23_lead + 96;
          int32_t v240_sw = v239_a >> 4;
          float v243_data = s0[(v239_a ^ (v240_sw & 15))];
          int32_t v244_a = v23_lead + 108;
          int32_t v245_sw = v244_a >> 4;
          float v248_data = s0[(v244_a ^ (v245_sw & 15))];
          int32_t v249_a = v23_lead + 120;
          int32_t v250_sw = v249_a >> 4;
          float v253_data = s0[(v249_a ^ (v250_sw & 15))];
          int32_t v254_a = v23_lead + 132;
          int32_t v255_sw = v254_a >> 4;
          float v258_data = s0[(v254_a ^ (v255_sw & 15))];
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v243_data, v238_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v248_data, v259_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v253_data, v260_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v258_data, v261_acc, 2, 2, 0);
          r4[0] = (v262_acc[0]);
          r4[1] = (v262_acc[1]);
          r4[2] = (v262_acc[2]);
          r4[3] = (v262_acc[3]);
          float v267_data = r3[4];
          float v268_data = r3[5];
          float v269_data = r3[6];
          float v270_data = r3[7];
          float v271_tp{};
          float v272_tp{};
          float v273_tp{};
          float v274_tp{};
          tensorforge::transpose4x4b32(v271_tp, v272_tp, v273_tp, v274_tp, v267_data, v268_data, v269_data, v270_data);
          tensorforge::VectorT<float, 4> v275_acc{};
          float v282_data = s0[(v23_lead ^ v193_sw)];
          float v287_data = s0[(v196_a ^ (v197_sw & 15))];
          float v292_data = s0[(v201_a ^ (v202_sw & 15))];
          float v297_data = s0[(v206_a ^ (v207_sw & 15))];
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v282_data, v275_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v287_data, v298_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v292_data, v299_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v297_data, v300_acc, 2, 0, 0);
          float v306_data = s0[(v215_a ^ (v216_sw & 15))];
          float v311_data = s0[(v220_a ^ (v221_sw & 15))];
          float v316_data = s0[(v225_a ^ (v226_sw & 15))];
          float v321_data = s0[(v230_a ^ (v231_sw & 15))];
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v306_data, v301_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v311_data, v322_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v316_data, v323_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v321_data, v324_acc, 2, 1, 0);
          float v330_data = s0[(v239_a ^ (v240_sw & 15))];
          float v335_data = s0[(v244_a ^ (v245_sw & 15))];
          float v340_data = s0[(v249_a ^ (v250_sw & 15))];
          float v345_data = s0[(v254_a ^ (v255_sw & 15))];
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v330_data, v325_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v335_data, v346_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v340_data, v347_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v345_data, v348_acc, 2, 2, 0);
          r4[4] = (v349_acc[0]);
          r4[5] = (v349_acc[1]);
          r4[6] = (v349_acc[2]);
          r4[7] = (v349_acc[3]);
          float v354_data = r3[8];
          float v355_data = r3[9];
          float v356_data = r3[10];
          float v357_data = r3[11];
          float v358_tp{};
          float v359_tp{};
          float v360_tp{};
          float v361_tp{};
          tensorforge::transpose4x4b32(v358_tp, v359_tp, v360_tp, v361_tp, v354_data, v355_data, v356_data, v357_data);
          tensorforge::VectorT<float, 4> v362_acc{};
          float v369_data = s0[(v23_lead ^ v193_sw)];
          float v374_data = s0[(v196_a ^ (v197_sw & 15))];
          float v379_data = s0[(v201_a ^ (v202_sw & 15))];
          float v384_data = s0[(v206_a ^ (v207_sw & 15))];
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v369_data, v362_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v374_data, v385_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v379_data, v386_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v384_data, v387_acc, 2, 0, 0);
          float v393_data = s0[(v215_a ^ (v216_sw & 15))];
          float v398_data = s0[(v220_a ^ (v221_sw & 15))];
          float v403_data = s0[(v225_a ^ (v226_sw & 15))];
          float v408_data = s0[(v230_a ^ (v231_sw & 15))];
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v393_data, v388_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v398_data, v409_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v403_data, v410_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v408_data, v411_acc, 2, 1, 0);
          float v417_data = s0[(v239_a ^ (v240_sw & 15))];
          float v422_data = s0[(v244_a ^ (v245_sw & 15))];
          float v427_data = s0[(v249_a ^ (v250_sw & 15))];
          float v432_data = s0[(v254_a ^ (v255_sw & 15))];
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v417_data, v412_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v422_data, v433_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v427_data, v434_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v432_data, v435_acc, 2, 2, 0);
          r4[8] = (v436_acc[0]);
          r4[9] = (v436_acc[1]);
          r4[10] = (v436_acc[2]);
          r4[11] = (v436_acc[3]);
          float r7[12]{};
          // r7 = load{g>r}(glb_m4);
          if (v24_g) {
            #pragma unroll
            for (int32_t v442_i1 = 0; v442_i1 < 12; ++v442_i1) {
              float v447_data = __builtin_nontemporal_load(&glb_m4[(v23_lead + (v442_i1 * 12))]);
              r7[v442_i1] = v447_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = +(r5 * r1) + None
          // [(0, 4), (0, 12)] [(0, 12)]
          float v454_tp{};
          float v455_tp{};
          float v456_tp{};
          float v457_tp{};
          tensorforge::transpose4x4b32(v454_tp, v455_tp, v456_tp, v457_tp, v49_data, v50_data, v51_data, v52_data);
          tensorforge::VectorT<float, 4> v458_acc{};
          float v459_data = r5[0];
          float v460_data = r5[1];
          float v461_data = r5[2];
          float v462_data = r5[3];
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v459_data, v458_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v460_data, v463_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v461_data, v464_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v462_data, v465_acc, 2, 0, 0);
          float v467_data = r5[4];
          float v468_data = r5[5];
          float v469_data = r5[6];
          float v470_data = r5[7];
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v467_data, v466_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v468_data, v471_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v469_data, v472_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v470_data, v473_acc, 2, 1, 0);
          float v475_data = r5[8];
          float v476_data = r5[9];
          float v477_data = r5[10];
          float v478_data = r5[11];
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v475_data, v474_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v476_data, v479_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v477_data, v480_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v478_data, v481_acc, 2, 2, 0);
          r6[0] = (v482_acc[0]);
          r6[1] = (v482_acc[1]);
          r6[2] = (v482_acc[2]);
          r6[3] = (v482_acc[3]);
          float v491_tp{};
          float v492_tp{};
          float v493_tp{};
          float v494_tp{};
          tensorforge::transpose4x4b32(v491_tp, v492_tp, v493_tp, v494_tp, v86_data, v87_data, v88_data, v89_data);
          tensorforge::VectorT<float, 4> v495_acc{};
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v459_data, v495_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v460_data, v500_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v493_tp, v461_data, v501_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v462_data, v502_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v467_data, v503_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v468_data, v508_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v493_tp, v469_data, v509_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v470_data, v510_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v491_tp, v475_data, v511_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v492_tp, v476_data, v516_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v493_tp, v477_data, v517_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v478_data, v518_acc, 2, 2, 0);
          r6[4] = (v519_acc[0]);
          r6[5] = (v519_acc[1]);
          r6[6] = (v519_acc[2]);
          r6[7] = (v519_acc[3]);
          float v528_tp{};
          float v529_tp{};
          float v530_tp{};
          float v531_tp{};
          tensorforge::transpose4x4b32(v528_tp, v529_tp, v530_tp, v531_tp, v123_data, v124_data, v125_data, v126_data);
          tensorforge::VectorT<float, 4> v532_acc{};
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v528_tp, v459_data, v532_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v529_tp, v460_data, v537_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v530_tp, v461_data, v538_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v531_tp, v462_data, v539_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v528_tp, v467_data, v540_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v529_tp, v468_data, v545_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v530_tp, v469_data, v546_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v531_tp, v470_data, v547_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v528_tp, v475_data, v548_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v529_tp, v476_data, v553_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v530_tp, v477_data, v554_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v531_tp, v478_data, v555_acc, 2, 2, 0);
          r6[8] = (v556_acc[0]);
          r6[9] = (v556_acc[1]);
          r6[10] = (v556_acc[2]);
          r6[11] = (v556_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r6);
          if ((v23_lead >= 4) && v24_g) {
            #pragma unroll
            for (int32_t v563_z1 = 0; v563_z1 < 12; ++v563_z1) {
              int32_t v568_a = v23_lead + (v563_z1 * 12);
              s0[(v568_a ^ ((v568_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v171_g) {
            #pragma unroll
            for (int32_t v572_i1 = 0; v572_i1 < 12; ++v572_i1) {
              float v574_data = r6[v572_i1];
              int32_t v578_a = v23_lead + (v572_i1 * 12);
              s0[(v578_a ^ ((v578_a >> 4) & 15))] = v574_data;
            }
          }
          float r9[12]{};
          // r9 = load{g>r}(glb_m6);
          if (v24_g) {
            #pragma unroll
            for (int32_t v583_i1 = 0; v583_i1 < 12; ++v583_i1) {
              float v588_data = __builtin_nontemporal_load(&glb_m6[(v23_lead + (v583_i1 * 12))]);
              r9[v583_i1] = v588_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m4););
          float r8[12]{};
          // ir8 = +(r4 * r7)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir8[12]{};
          float v592_data = r7[0];
          float v593_data = r7[1];
          float v594_data = r7[2];
          float v595_data = r7[3];
          float v596_tp{};
          float v597_tp{};
          float v598_tp{};
          float v599_tp{};
          tensorforge::transpose4x4b32(v596_tp, v597_tp, v598_tp, v599_tp, v592_data, v593_data, v594_data, v595_data);
          tensorforge::VectorT<float, 4> v600_acc{};
          float v601_data = r4[0];
          float v602_data = r4[1];
          float v603_data = r4[2];
          float v604_data = r4[3];
          tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v596_tp, v601_data, v600_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v597_tp, v602_data, v605_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v607_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v598_tp, v603_data, v606_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v604_data, v607_acc, 2, 0, 0);
          float v609_data = r4[4];
          float v610_data = r4[5];
          float v611_data = r4[6];
          float v612_data = r4[7];
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v596_tp, v609_data, v608_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v597_tp, v610_data, v613_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v598_tp, v611_data, v614_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v612_data, v615_acc, 2, 1, 0);
          float v617_data = r4[8];
          float v618_data = r4[9];
          float v619_data = r4[10];
          float v620_data = r4[11];
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v596_tp, v617_data, v616_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v597_tp, v618_data, v621_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v598_tp, v619_data, v622_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v620_data, v623_acc, 2, 2, 0);
          ir8[0] = (v624_acc[0]);
          ir8[1] = (v624_acc[1]);
          ir8[2] = (v624_acc[2]);
          ir8[3] = (v624_acc[3]);
          float v629_data = r7[4];
          float v630_data = r7[5];
          float v631_data = r7[6];
          float v632_data = r7[7];
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
          ir8[4] = (v661_acc[0]);
          ir8[5] = (v661_acc[1]);
          ir8[6] = (v661_acc[2]);
          ir8[7] = (v661_acc[3]);
          float v666_data = r7[8];
          float v667_data = r7[9];
          float v668_data = r7[10];
          float v669_data = r7[11];
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
          ir8[8] = (v698_acc[0]);
          ir8[9] = (v698_acc[1]);
          ir8[10] = (v698_acc[2]);
          ir8[11] = (v698_acc[3]);
          // r8 = ir8 + s0
          if (v24_g) {
            #pragma unroll
            for (int32_t v703_n1 = 0; v703_n1 < 12; ++v703_n1) {
              float v705_data = ir8[v703_n1];
              int32_t v709_a = v23_lead + (v703_n1 * 12);
              float v713_data = s0[(v709_a ^ ((v709_a >> 4) & 15))];
              r8[v703_n1] = (v713_data + v705_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r8);
          if (v24_g) {
            #pragma unroll
            for (int32_t v715_i1 = 0; v715_i1 < 12; ++v715_i1) {
              float v717_data = r8[v715_i1];
              int32_t v721_a = v23_lead + (v715_i1 * 12);
              s0[(v721_a ^ ((v721_a >> 4) & 15))] = v717_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[12]{};
          // r10 = +(r9 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v732_data = s0[(v23_lead ^ v193_sw)];
          float v737_data = s0[(v196_a ^ (v197_sw & 15))];
          float v742_data = s0[(v201_a ^ (v202_sw & 15))];
          float v747_data = s0[(v206_a ^ (v207_sw & 15))];
          float v748_tp{};
          float v749_tp{};
          float v750_tp{};
          float v751_tp{};
          tensorforge::transpose4x4b32(v748_tp, v749_tp, v750_tp, v751_tp, v732_data, v737_data, v742_data, v747_data);
          tensorforge::VectorT<float, 4> v752_acc{};
          float v753_data = r9[0];
          float v754_data = r9[1];
          float v755_data = r9[2];
          float v756_data = r9[3];
          tensorforge::VectorT<float, 4> v757_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v748_tp, v753_data, v752_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v749_tp, v754_data, v757_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v759_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v750_tp, v755_data, v758_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v751_tp, v756_data, v759_acc, 2, 0, 0);
          float v761_data = r9[4];
          float v762_data = r9[5];
          float v763_data = r9[6];
          float v764_data = r9[7];
          tensorforge::VectorT<float, 4> v765_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v748_tp, v761_data, v760_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v766_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v749_tp, v762_data, v765_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v767_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v750_tp, v763_data, v766_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v751_tp, v764_data, v767_acc, 2, 1, 0);
          float v769_data = r9[8];
          float v770_data = r9[9];
          float v771_data = r9[10];
          float v772_data = r9[11];
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v748_tp, v769_data, v768_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v749_tp, v770_data, v773_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v750_tp, v771_data, v774_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v751_tp, v772_data, v775_acc, 2, 2, 0);
          r10[0] = (v776_acc[0]);
          r10[1] = (v776_acc[1]);
          r10[2] = (v776_acc[2]);
          r10[3] = (v776_acc[3]);
          float v787_data = s0[(v215_a ^ (v216_sw & 15))];
          float v792_data = s0[(v220_a ^ (v221_sw & 15))];
          float v797_data = s0[(v225_a ^ (v226_sw & 15))];
          float v802_data = s0[(v230_a ^ (v231_sw & 15))];
          float v803_tp{};
          float v804_tp{};
          float v805_tp{};
          float v806_tp{};
          tensorforge::transpose4x4b32(v803_tp, v804_tp, v805_tp, v806_tp, v787_data, v792_data, v797_data, v802_data);
          tensorforge::VectorT<float, 4> v807_acc{};
          tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v753_data, v807_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v813_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v754_data, v812_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v814_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v805_tp, v755_data, v813_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v815_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v756_data, v814_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v761_data, v815_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v821_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v762_data, v820_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v805_tp, v763_data, v821_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v823_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v764_data, v822_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v803_tp, v769_data, v823_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v829_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v804_tp, v770_data, v828_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v805_tp, v771_data, v829_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v772_data, v830_acc, 2, 2, 0);
          r10[4] = (v831_acc[0]);
          r10[5] = (v831_acc[1]);
          r10[6] = (v831_acc[2]);
          r10[7] = (v831_acc[3]);
          float v842_data = s0[(v239_a ^ (v240_sw & 15))];
          float v847_data = s0[(v244_a ^ (v245_sw & 15))];
          float v852_data = s0[(v249_a ^ (v250_sw & 15))];
          float v857_data = s0[(v254_a ^ (v255_sw & 15))];
          float v858_tp{};
          float v859_tp{};
          float v860_tp{};
          float v861_tp{};
          tensorforge::transpose4x4b32(v858_tp, v859_tp, v860_tp, v861_tp, v842_data, v847_data, v852_data, v857_data);
          tensorforge::VectorT<float, 4> v862_acc{};
          tensorforge::VectorT<float, 4> v867_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v858_tp, v753_data, v862_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v868_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v859_tp, v754_data, v867_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v869_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v860_tp, v755_data, v868_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v870_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v861_tp, v756_data, v869_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v875_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v858_tp, v761_data, v870_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v876_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v859_tp, v762_data, v875_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v877_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v860_tp, v763_data, v876_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v861_tp, v764_data, v877_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v858_tp, v769_data, v878_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v859_tp, v770_data, v883_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v860_tp, v771_data, v884_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v861_tp, v772_data, v885_acc, 2, 2, 0);
          r10[8] = (v886_acc[0]);
          r10[9] = (v886_acc[1]);
          r10[10] = (v886_acc[2]);
          r10[11] = (v886_acc[3]);
          // glb_m5 = store{r>g}(r10);
          if (v24_g) {
            #pragma unroll
            for (int32_t v891_i1 = 0; v891_i1 < 12; ++v891_i1) {
              float v893_data = r10[v891_i1];
              glb_m5[(v23_lead + (v891_i1 * 12))] = v893_data;
            }
          }
        }
      }
    }
  }
}

