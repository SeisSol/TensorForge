// === base name ===
kernel_d18ea70f912811c4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d18ea70f912811c4 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d18ea70f912811c4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d18ea70f912811c4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d18ea70f912811c4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d18ea70f912811c4, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d18ea70f912811c4, block.x * block.y * block.z, 0));
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
void launcher_kernel_d18ea70f912811c4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d18ea70f912811c4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d18ea70f912811c4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_d18ea70f912811c4, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d18ea70f912811c4(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 6×12(6×12) {0..6}×{0..12} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 6×12(6×12) {0..6}×{0..12} strided
    //   m3 6×12(6×12) {0..6}×{0..12} strided
    //   m4 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j] = m2[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
    //   m4[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v5_batchId0 * 72 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 144 + 0 + m4_extraOffset];
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
            int32_t v164_off = v21_lead + 6;
            #pragma unroll
            for (int32_t v159_i1 = 0; v159_i1 < 12; ++v159_i1) {
              float v161_data = r2[v159_i1];
              int32_t v166_a = v164_off + (v159_i1 * 12);
              s0[(v166_a ^ ((v166_a >> 4) & 15))] = v161_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          if (v22_g) {
            #pragma unroll
            for (int32_t v171_i1 = 0; v171_i1 < 12; ++v171_i1) {
              float v176_data = __builtin_nontemporal_load(&glb_m3[(v21_lead + (v171_i1 * 6))]);
              r5[v171_i1] = v176_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          tensorforge::transpose4x4b32(v183_tp, v184_tp, v185_tp, v186_tp, v48_data, v49_data, v50_data, v51_data);
          tensorforge::VectorT<float, 4> v187_acc{};
          float v188_data = r3[0];
          float v189_data = r3[1];
          float v190_data = r3[2];
          float v191_data = r3[3];
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v188_data, v187_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v189_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v190_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v191_data, v194_acc, 2, 0, 0);
          float v196_data = r3[4];
          float v197_data = r3[5];
          float v198_data = r3[6];
          float v199_data = r3[7];
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v196_data, v195_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v197_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v198_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v199_data, v202_acc, 2, 1, 0);
          float v204_data = r3[8];
          float v205_data = r3[9];
          float v206_data = r3[10];
          float v207_data = r3[11];
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v204_data, v203_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v205_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v206_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v207_data, v210_acc, 2, 2, 0);
          r4[0] = (v211_acc[0]);
          r4[1] = (v211_acc[1]);
          r4[2] = (v211_acc[2]);
          r4[3] = (v211_acc[3]);
          float v220_tp{};
          float v221_tp{};
          float v222_tp{};
          float v223_tp{};
          tensorforge::transpose4x4b32(v220_tp, v221_tp, v222_tp, v223_tp, v85_data, v86_data, v87_data, v88_data);
          tensorforge::VectorT<float, 4> v224_acc{};
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v188_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v189_data, v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v190_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v191_data, v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v196_data, v232_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v197_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v198_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v199_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v204_data, v240_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v205_data, v245_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v206_data, v246_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v207_data, v247_acc, 2, 2, 0);
          r4[4] = (v248_acc[0]);
          r4[5] = (v248_acc[1]);
          r4[6] = (v248_acc[2]);
          r4[7] = (v248_acc[3]);
          float v257_tp{};
          float v258_tp{};
          float v259_tp{};
          float v260_tp{};
          tensorforge::transpose4x4b32(v257_tp, v258_tp, v259_tp, v260_tp, v122_data, v123_data, v124_data, v125_data);
          tensorforge::VectorT<float, 4> v261_acc{};
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v188_data, v261_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v189_data, v266_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v190_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v191_data, v268_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v196_data, v269_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v197_data, v274_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v198_data, v275_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v199_data, v276_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v204_data, v277_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v205_data, v282_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v206_data, v283_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v207_data, v284_acc, 2, 2, 0);
          r4[8] = (v285_acc[0]);
          r4[9] = (v285_acc[1]);
          r4[10] = (v285_acc[2]);
          r4[11] = (v285_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r4);
          if ((v21_lead >= 6) && v31_g) {
            #pragma unroll
            for (int32_t v292_z1 = 0; v292_z1 < 12; ++v292_z1) {
              int32_t v297_a = v21_lead + (v292_z1 * 12);
              s0[(v297_a ^ ((v297_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v22_g) {
            #pragma unroll
            for (int32_t v301_i1 = 0; v301_i1 < 12; ++v301_i1) {
              float v303_data = r4[v301_i1];
              int32_t v307_a = v21_lead + (v301_i1 * 12);
              s0[(v307_a ^ ((v307_a >> 4) & 15))] = v303_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = +(r5) + None
          // [(0, 6), (0, 12)] []
          float v312_data = r5[0];
          float v313_data = r6[0];
          r6[0] = (v313_data + v312_data);
          float v315_data = r5[1];
          float v316_data = r6[1];
          r6[1] = (v316_data + v315_data);
          float v318_data = r5[2];
          float v319_data = r6[2];
          r6[2] = (v319_data + v318_data);
          float v321_data = r5[3];
          float v322_data = r6[3];
          r6[3] = (v322_data + v321_data);
          float v324_data = r5[4];
          float v325_data = r6[4];
          r6[4] = (v325_data + v324_data);
          float v327_data = r5[5];
          float v328_data = r6[5];
          r6[5] = (v328_data + v327_data);
          float v330_data = r5[6];
          float v331_data = r6[6];
          r6[6] = (v331_data + v330_data);
          float v333_data = r5[7];
          float v334_data = r6[7];
          r6[7] = (v334_data + v333_data);
          float v336_data = r5[8];
          float v337_data = r6[8];
          r6[8] = (v337_data + v336_data);
          float v339_data = r5[9];
          float v340_data = r6[9];
          r6[9] = (v340_data + v339_data);
          float v342_data = r5[10];
          float v343_data = r6[10];
          r6[10] = (v343_data + v342_data);
          float v345_data = r5[11];
          float v346_data = r6[11];
          r6[11] = (v346_data + v345_data);
          // s0 = store{r>s}(localShrMem0, r6);
          if (v22_g) {
            int32_t v353_off = v21_lead + 6;
            #pragma unroll
            for (int32_t v348_i1 = 0; v348_i1 < 12; ++v348_i1) {
              float v350_data = r6[v348_i1];
              int32_t v355_a = v353_off + (v348_i1 * 12);
              s0[(v355_a ^ ((v355_a >> 4) & 15))] = v350_data;
            }
          }
          float r7[12]{};
          // r7 = +(s0) + None
          // [(0, 12), (0, 12)] []
          float v366_data = v31_g ? (s0[(v21_lead ^ ((v21_lead >> 4) & 15))]) : (0.0f);
          float v367_data = r7[0];
          r7[0] = (v367_data + v366_data);
          int32_t v369_a = v21_lead + 12;
          float v373_data = v31_g ? (s0[(v369_a ^ ((v369_a >> 4) & 15))]) : (0.0f);
          float v374_data = r7[1];
          r7[1] = (v374_data + v373_data);
          int32_t v376_a = v21_lead + 24;
          float v380_data = v31_g ? (s0[(v376_a ^ ((v376_a >> 4) & 15))]) : (0.0f);
          float v381_data = r7[2];
          r7[2] = (v381_data + v380_data);
          int32_t v383_a = v21_lead + 36;
          float v387_data = v31_g ? (s0[(v383_a ^ ((v383_a >> 4) & 15))]) : (0.0f);
          float v388_data = r7[3];
          r7[3] = (v388_data + v387_data);
          int32_t v390_a = v21_lead + 48;
          float v394_data = v31_g ? (s0[(v390_a ^ ((v390_a >> 4) & 15))]) : (0.0f);
          float v395_data = r7[4];
          r7[4] = (v395_data + v394_data);
          int32_t v397_a = v21_lead + 60;
          float v401_data = v31_g ? (s0[(v397_a ^ ((v397_a >> 4) & 15))]) : (0.0f);
          float v402_data = r7[5];
          r7[5] = (v402_data + v401_data);
          int32_t v404_a = v21_lead + 72;
          float v408_data = v31_g ? (s0[(v404_a ^ ((v404_a >> 4) & 15))]) : (0.0f);
          float v409_data = r7[6];
          r7[6] = (v409_data + v408_data);
          int32_t v411_a = v21_lead + 84;
          float v415_data = v31_g ? (s0[(v411_a ^ ((v411_a >> 4) & 15))]) : (0.0f);
          float v416_data = r7[7];
          r7[7] = (v416_data + v415_data);
          int32_t v418_a = v21_lead + 96;
          float v422_data = v31_g ? (s0[(v418_a ^ ((v418_a >> 4) & 15))]) : (0.0f);
          float v423_data = r7[8];
          r7[8] = (v423_data + v422_data);
          int32_t v425_a = v21_lead + 108;
          float v429_data = v31_g ? (s0[(v425_a ^ ((v425_a >> 4) & 15))]) : (0.0f);
          float v430_data = r7[9];
          r7[9] = (v430_data + v429_data);
          int32_t v432_a = v21_lead + 120;
          float v436_data = v31_g ? (s0[(v432_a ^ ((v432_a >> 4) & 15))]) : (0.0f);
          float v437_data = r7[10];
          r7[10] = (v437_data + v436_data);
          int32_t v439_a = v21_lead + 132;
          float v443_data = v31_g ? (s0[(v439_a ^ ((v439_a >> 4) & 15))]) : (0.0f);
          float v444_data = r7[11];
          r7[11] = (v444_data + v443_data);
          // glb_m4 = store{r>g}(r7);
          if (v31_g) {
            #pragma unroll
            for (int32_t v446_i1 = 0; v446_i1 < 12; ++v446_i1) {
              float v448_data = r7[v446_i1];
              glb_m4[(v21_lead + (v446_i1 * 12))] = v448_data;
            }
          }
        }
      }
    }
  }
}

