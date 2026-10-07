// === base name ===
kernel_e48d46f1e8e575b3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e48d46f1e8e575b3 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e48d46f1e8e575b3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e48d46f1e8e575b3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e48d46f1e8e575b3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e48d46f1e8e575b3, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e48d46f1e8e575b3, block.x * block.y * block.z, 0));
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
void launcher_kernel_e48d46f1e8e575b3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e48d46f1e8e575b3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e48d46f1e8e575b3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_e48d46f1e8e575b3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e48d46f1e8e575b3(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 72 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v8_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 16;
          bool v25_g = v24_lead < 6;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m0[(v24_lead + (v26_i1 * 6))]);
              r0[v26_i1] = v31_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v34_g = v24_lead < 12;
          if (v34_g) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v35_i1 * 12))]);
              r1[v35_i1] = v40_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v166_i1 = 0; v166_i1 < 12; ++v166_i1) {
              float v171_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + (v166_i1 * 6))]);
              r3[v166_i1] = v171_data;
            }
          }
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v43_data = r1[0];
          float v44_data = r1[1];
          float v45_data = r1[2];
          float v46_data = r1[3];
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          tensorforge::transpose4x4b32(v47_tp, v48_tp, v49_tp, v50_tp, v43_data, v44_data, v45_data, v46_data);
          tensorforge::VectorT<float, 4> v51_acc{};
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v51_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 2, 0, 0);
          float v60_data = r0[4];
          float v61_data = r0[5];
          float v62_data = r0[6];
          float v63_data = r0[7];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v59_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 1, 0);
          float v68_data = r0[8];
          float v69_data = r0[9];
          float v70_data = r0[10];
          float v71_data = r0[11];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v67_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 2, 0);
          r2[0] = (v75_acc[0]);
          r2[1] = (v75_acc[1]);
          r2[2] = (v75_acc[2]);
          r2[3] = (v75_acc[3]);
          float v80_data = r1[4];
          float v81_data = r1[5];
          float v82_data = r1[6];
          float v83_data = r1[7];
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          float v87_tp{};
          tensorforge::transpose4x4b32(v84_tp, v85_tp, v86_tp, v87_tp, v80_data, v81_data, v82_data, v83_data);
          tensorforge::VectorT<float, 4> v88_acc{};
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v88_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v94_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v95_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v102_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v103_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v104_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v109_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v110_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v111_acc, 2, 2, 0);
          r2[4] = (v112_acc[0]);
          r2[5] = (v112_acc[1]);
          r2[6] = (v112_acc[2]);
          r2[7] = (v112_acc[3]);
          float v117_data = r1[8];
          float v118_data = r1[9];
          float v119_data = r1[10];
          float v120_data = r1[11];
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          tensorforge::transpose4x4b32(v121_tp, v122_tp, v123_tp, v124_tp, v117_data, v118_data, v119_data, v120_data);
          tensorforge::VectorT<float, 4> v125_acc{};
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v52_data, v125_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v53_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v54_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v55_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v60_data, v133_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v61_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v62_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v63_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v68_data, v141_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v69_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v70_data, v147_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v71_data, v148_acc, 2, 2, 0);
          r2[8] = (v149_acc[0]);
          r2[9] = (v149_acc[1]);
          r2[10] = (v149_acc[2]);
          r2[11] = (v149_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v25_g) {
            int32_t v159_off = v24_lead + 6;
            #pragma unroll
            for (int32_t v154_i1 = 0; v154_i1 < 12; ++v154_i1) {
              float v156_data = r2[v154_i1];
              int32_t v161_a = v159_off + (v154_i1 * 12);
              s0[(v161_a ^ ((v161_a >> 4) & 15))] = v156_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          if (v25_g) {
            #pragma unroll
            for (int32_t v307_i1 = 0; v307_i1 < 12; ++v307_i1) {
              float v312_data = __builtin_nontemporal_load(&glb_m3[(v24_lead + (v307_i1 * 6))]);
              r5[v307_i1] = v312_data;
            }
          }
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          float v181_tp{};
          tensorforge::transpose4x4b32(v178_tp, v179_tp, v180_tp, v181_tp, v43_data, v44_data, v45_data, v46_data);
          tensorforge::VectorT<float, 4> v182_acc{};
          float v183_data = r3[0];
          float v184_data = r3[1];
          float v185_data = r3[2];
          float v186_data = r3[3];
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v183_data, v182_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v184_data, v187_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v185_data, v188_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v186_data, v189_acc, 2, 0, 0);
          float v191_data = r3[4];
          float v192_data = r3[5];
          float v193_data = r3[6];
          float v194_data = r3[7];
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v191_data, v190_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v192_data, v195_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v193_data, v196_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v194_data, v197_acc, 2, 1, 0);
          float v199_data = r3[8];
          float v200_data = r3[9];
          float v201_data = r3[10];
          float v202_data = r3[11];
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v199_data, v198_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v200_data, v203_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v201_data, v204_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v202_data, v205_acc, 2, 2, 0);
          r4[0] = (v206_acc[0]);
          r4[1] = (v206_acc[1]);
          r4[2] = (v206_acc[2]);
          r4[3] = (v206_acc[3]);
          float v215_tp{};
          float v216_tp{};
          float v217_tp{};
          float v218_tp{};
          tensorforge::transpose4x4b32(v215_tp, v216_tp, v217_tp, v218_tp, v80_data, v81_data, v82_data, v83_data);
          tensorforge::VectorT<float, 4> v219_acc{};
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v183_data, v219_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v184_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v185_data, v225_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v186_data, v226_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v191_data, v227_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v192_data, v232_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v193_data, v233_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v194_data, v234_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v199_data, v235_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v200_data, v240_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v201_data, v241_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v202_data, v242_acc, 2, 2, 0);
          r4[4] = (v243_acc[0]);
          r4[5] = (v243_acc[1]);
          r4[6] = (v243_acc[2]);
          r4[7] = (v243_acc[3]);
          float v252_tp{};
          float v253_tp{};
          float v254_tp{};
          float v255_tp{};
          tensorforge::transpose4x4b32(v252_tp, v253_tp, v254_tp, v255_tp, v117_data, v118_data, v119_data, v120_data);
          tensorforge::VectorT<float, 4> v256_acc{};
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v183_data, v256_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v184_data, v261_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v185_data, v262_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v186_data, v263_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v191_data, v264_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v192_data, v269_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v193_data, v270_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v194_data, v271_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v199_data, v272_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v200_data, v277_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v201_data, v278_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v202_data, v279_acc, 2, 2, 0);
          r4[8] = (v280_acc[0]);
          r4[9] = (v280_acc[1]);
          r4[10] = (v280_acc[2]);
          r4[11] = (v280_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r4);
          if ((v24_lead >= 6) && v34_g) {
            #pragma unroll
            for (int32_t v287_z1 = 0; v287_z1 < 12; ++v287_z1) {
              int32_t v292_a = v24_lead + (v287_z1 * 12);
              s0[(v292_a ^ ((v292_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v25_g) {
            #pragma unroll
            for (int32_t v296_i1 = 0; v296_i1 < 12; ++v296_i1) {
              float v298_data = r4[v296_i1];
              int32_t v302_a = v24_lead + (v296_i1 * 12);
              s0[(v302_a ^ ((v302_a >> 4) & 15))] = v298_data;
            }
          }
          float r6[12]{};
          // r6 = +(r5) + None
          // [(0, 6), (0, 12)] []
          float v315_data = r5[0];
          float v316_data = r6[0];
          r6[0] = (v316_data + v315_data);
          float v318_data = r5[1];
          float v319_data = r6[1];
          r6[1] = (v319_data + v318_data);
          float v321_data = r5[2];
          float v322_data = r6[2];
          r6[2] = (v322_data + v321_data);
          float v324_data = r5[3];
          float v325_data = r6[3];
          r6[3] = (v325_data + v324_data);
          float v327_data = r5[4];
          float v328_data = r6[4];
          r6[4] = (v328_data + v327_data);
          float v330_data = r5[5];
          float v331_data = r6[5];
          r6[5] = (v331_data + v330_data);
          float v333_data = r5[6];
          float v334_data = r6[6];
          r6[6] = (v334_data + v333_data);
          float v336_data = r5[7];
          float v337_data = r6[7];
          r6[7] = (v337_data + v336_data);
          float v339_data = r5[8];
          float v340_data = r6[8];
          r6[8] = (v340_data + v339_data);
          float v342_data = r5[9];
          float v343_data = r6[9];
          r6[9] = (v343_data + v342_data);
          float v345_data = r5[10];
          float v346_data = r6[10];
          r6[10] = (v346_data + v345_data);
          float v348_data = r5[11];
          float v349_data = r6[11];
          r6[11] = (v349_data + v348_data);
          // s0 = store{r>s}(localShrMem0, r6);
          if (v25_g) {
            int32_t v356_off = v24_lead + 6;
            #pragma unroll
            for (int32_t v351_i1 = 0; v351_i1 < 12; ++v351_i1) {
              float v353_data = r6[v351_i1];
              int32_t v358_a = v356_off + (v351_i1 * 12);
              s0[(v358_a ^ ((v358_a >> 4) & 15))] = v353_data;
            }
          }
          float r7[12]{};
          // r7 = +(s0) + None
          // [(0, 12), (0, 12)] []
          int32_t v368_sw = v24_lead ^ ((v24_lead >> 4) & 15);
          float v369_data = v34_g ? (s0[v368_sw]) : (0.0f);
          float v370_data = r7[0];
          r7[0] = (v370_data + v369_data);
          int32_t v372_a = v24_lead + 12;
          float v376_data = v34_g ? (s0[(v372_a ^ ((v372_a >> 4) & 15))]) : (0.0f);
          float v377_data = r7[1];
          r7[1] = (v377_data + v376_data);
          int32_t v379_a = v24_lead + 24;
          float v383_data = v34_g ? (s0[(v379_a ^ ((v379_a >> 4) & 15))]) : (0.0f);
          float v384_data = r7[2];
          r7[2] = (v384_data + v383_data);
          int32_t v386_a = v24_lead + 36;
          float v390_data = v34_g ? (s0[(v386_a ^ ((v386_a >> 4) & 15))]) : (0.0f);
          float v391_data = r7[3];
          r7[3] = (v391_data + v390_data);
          int32_t v393_a = v24_lead + 48;
          float v397_data = v34_g ? (s0[(v393_a ^ ((v393_a >> 4) & 15))]) : (0.0f);
          float v398_data = r7[4];
          r7[4] = (v398_data + v397_data);
          int32_t v400_a = v24_lead + 60;
          float v404_data = v34_g ? (s0[(v400_a ^ ((v400_a >> 4) & 15))]) : (0.0f);
          float v405_data = r7[5];
          r7[5] = (v405_data + v404_data);
          int32_t v407_a = v24_lead + 72;
          float v411_data = v34_g ? (s0[(v407_a ^ ((v407_a >> 4) & 15))]) : (0.0f);
          float v412_data = r7[6];
          r7[6] = (v412_data + v411_data);
          int32_t v414_a = v24_lead + 84;
          float v418_data = v34_g ? (s0[(v414_a ^ ((v414_a >> 4) & 15))]) : (0.0f);
          float v419_data = r7[7];
          r7[7] = (v419_data + v418_data);
          int32_t v421_a = v24_lead + 96;
          float v425_data = v34_g ? (s0[(v421_a ^ ((v421_a >> 4) & 15))]) : (0.0f);
          float v426_data = r7[8];
          r7[8] = (v426_data + v425_data);
          int32_t v428_a = v24_lead + 108;
          float v432_data = v34_g ? (s0[(v428_a ^ ((v428_a >> 4) & 15))]) : (0.0f);
          float v433_data = r7[9];
          r7[9] = (v433_data + v432_data);
          int32_t v435_a = v24_lead + 120;
          float v439_data = v34_g ? (s0[(v435_a ^ ((v435_a >> 4) & 15))]) : (0.0f);
          float v440_data = r7[10];
          r7[10] = (v440_data + v439_data);
          int32_t v442_a = v24_lead + 132;
          float v446_data = v34_g ? (s0[(v442_a ^ ((v442_a >> 4) & 15))]) : (0.0f);
          float v447_data = r7[11];
          r7[11] = (v447_data + v446_data);
          // glb_m4 = store{r>g}(r7);
          if (v34_g) {
            #pragma unroll
            for (int32_t v449_i1 = 0; v449_i1 < 12; ++v449_i1) {
              float v451_data = r7[v449_i1];
              glb_m4[(v24_lead + (v449_i1 * 12))] = v451_data;
            }
          }
        }
      }
    }
  }
}

