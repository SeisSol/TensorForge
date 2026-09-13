// === base name ===
kernel_d5f9fffe05b5a7c3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d5f9fffe05b5a7c3 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d5f9fffe05b5a7c3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d5f9fffe05b5a7c3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d5f9fffe05b5a7c3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d5f9fffe05b5a7c3, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d5f9fffe05b5a7c3, block.x * block.y * block.z, 0));
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
void launcher_kernel_d5f9fffe05b5a7c3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d5f9fffe05b5a7c3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d5f9fffe05b5a7c3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_d5f9fffe05b5a7c3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d5f9fffe05b5a7c3(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = m4[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 16;
          bool v22_g = v21_lead < 6;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 12; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v21_lead + (v23_i1 * 6))]);
              r0[v23_i1] = v28_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v31_g = v21_lead < 12;
          if (v31_g) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v32_i1 * 12))]);
              r1[v32_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v40_i1 * 6))]);
              r3[v40_i1] = v45_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
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
          if (v22_g) {
            #pragma unroll
            for (int32_t v159_i1 = 0; v159_i1 < 12; ++v159_i1) {
              float v161_data = r2[v159_i1];
              int32_t v165_a = v21_lead + (v159_i1 * 12);
              s0[(v165_a ^ ((v165_a >> 4) & 15))] = v161_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v31_g) {
            #pragma unroll
            for (int32_t v170_i1 = 0; v170_i1 < 12; ++v170_i1) {
              float v175_data = __builtin_nontemporal_load(&glb_m4[(v21_lead + (v170_i1 * 12))]);
              r5[v170_i1] = v175_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
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
          if (v22_g) {
            int32_t v294_off = v21_lead + 6;
            #pragma unroll
            for (int32_t v289_i1 = 0; v289_i1 < 12; ++v289_i1) {
              float v291_data = r4[v289_i1];
              int32_t v296_a = v294_off + (v289_i1 * 12);
              s0[(v296_a ^ ((v296_a >> 4) & 15))] = v291_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v307_data = s0[(v21_lead ^ ((v21_lead >> 4) & 15))];
          int32_t v308_a = v21_lead + 12;
          float v312_data = s0[(v308_a ^ ((v308_a >> 4) & 15))];
          int32_t v313_a = v21_lead + 24;
          float v317_data = s0[(v313_a ^ ((v313_a >> 4) & 15))];
          int32_t v318_a = v21_lead + 36;
          float v322_data = s0[(v318_a ^ ((v318_a >> 4) & 15))];
          float v323_tp{};
          float v324_tp{};
          float v325_tp{};
          float v326_tp{};
          tensorforge::transpose4x4b32(v323_tp, v324_tp, v325_tp, v326_tp, v307_data, v312_data, v317_data, v322_data);
          tensorforge::VectorT<float, 4> v327_acc{};
          float v328_data = r5[0];
          float v329_data = r5[1];
          float v330_data = r5[2];
          float v331_data = r5[3];
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v328_data, v327_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v329_data, v332_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v330_data, v333_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v331_data, v334_acc, 2, 0, 0);
          float v336_data = r5[4];
          float v337_data = r5[5];
          float v338_data = r5[6];
          float v339_data = r5[7];
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v336_data, v335_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v337_data, v340_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v338_data, v341_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v339_data, v342_acc, 2, 1, 0);
          float v344_data = r5[8];
          float v345_data = r5[9];
          float v346_data = r5[10];
          float v347_data = r5[11];
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v344_data, v343_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v345_data, v348_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v346_data, v349_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v347_data, v350_acc, 2, 2, 0);
          r6[0] = (v351_acc[0]);
          r6[1] = (v351_acc[1]);
          r6[2] = (v351_acc[2]);
          r6[3] = (v351_acc[3]);
          int32_t v358_a = v21_lead + 48;
          float v362_data = s0[(v358_a ^ ((v358_a >> 4) & 15))];
          int32_t v363_a = v21_lead + 60;
          float v367_data = s0[(v363_a ^ ((v363_a >> 4) & 15))];
          int32_t v368_a = v21_lead + 72;
          float v372_data = s0[(v368_a ^ ((v368_a >> 4) & 15))];
          int32_t v373_a = v21_lead + 84;
          float v377_data = s0[(v373_a ^ ((v373_a >> 4) & 15))];
          float v378_tp{};
          float v379_tp{};
          float v380_tp{};
          float v381_tp{};
          tensorforge::transpose4x4b32(v378_tp, v379_tp, v380_tp, v381_tp, v362_data, v367_data, v372_data, v377_data);
          tensorforge::VectorT<float, 4> v382_acc{};
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v328_data, v382_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v329_data, v387_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v330_data, v388_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v331_data, v389_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v336_data, v390_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v337_data, v395_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v338_data, v396_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v339_data, v397_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v344_data, v398_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v345_data, v403_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v346_data, v404_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v347_data, v405_acc, 2, 2, 0);
          r6[4] = (v406_acc[0]);
          r6[5] = (v406_acc[1]);
          r6[6] = (v406_acc[2]);
          r6[7] = (v406_acc[3]);
          int32_t v413_a = v21_lead + 96;
          float v417_data = s0[(v413_a ^ ((v413_a >> 4) & 15))];
          int32_t v418_a = v21_lead + 108;
          float v422_data = s0[(v418_a ^ ((v418_a >> 4) & 15))];
          int32_t v423_a = v21_lead + 120;
          float v427_data = s0[(v423_a ^ ((v423_a >> 4) & 15))];
          int32_t v428_a = v21_lead + 132;
          float v432_data = s0[(v428_a ^ ((v428_a >> 4) & 15))];
          float v433_tp{};
          float v434_tp{};
          float v435_tp{};
          float v436_tp{};
          tensorforge::transpose4x4b32(v433_tp, v434_tp, v435_tp, v436_tp, v417_data, v422_data, v427_data, v432_data);
          tensorforge::VectorT<float, 4> v437_acc{};
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v328_data, v437_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v329_data, v442_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v330_data, v443_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v331_data, v444_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v336_data, v445_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v337_data, v450_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v338_data, v451_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v339_data, v452_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v344_data, v453_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v345_data, v458_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v435_tp, v346_data, v459_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v347_data, v460_acc, 2, 2, 0);
          r6[8] = (v461_acc[0]);
          r6[9] = (v461_acc[1]);
          r6[10] = (v461_acc[2]);
          r6[11] = (v461_acc[3]);
          // glb_m3 = store{r>g}(r6);
          if (v31_g) {
            #pragma unroll
            for (int32_t v466_i1 = 0; v466_i1 < 12; ++v466_i1) {
              float v468_data = r6[v466_i1];
              glb_m3[(v21_lead + (v466_i1 * 12))] = v468_data;
            }
          }
        }
      }
    }
  }
}

