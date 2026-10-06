// === base name ===
kernel_97fd594be17469d9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_97fd594be17469d9 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_97fd594be17469d9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_97fd594be17469d9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_97fd594be17469d9(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_97fd594be17469d9, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_97fd594be17469d9, block.x * block.y * block.z, 0));
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
void launcher_kernel_97fd594be17469d9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_97fd594be17469d9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_97fd594be17469d9), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_97fd594be17469d9, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_97fd594be17469d9(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
    //   m3 12×12(12×12) {0..12}×{0..12} strided
    //   m4 2×12(2×12) {0..2}×{0..12} strided
    //   m5 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = t0[i,j]
    //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
    //   m5[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v11_batchId0 * 24 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v11_batchId0 * 144 + 0 + m5_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v28_lead = threadIdx.x % 16;
          bool v29_g = v28_lead < 6;
          if (v29_g) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 12; ++v30_i1) {
              float v35_data = __builtin_nontemporal_load(&glb_m0[(v28_lead + (v30_i1 * 6))]);
              r0[v30_i1] = v35_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v38_g = v28_lead < 12;
          if (v38_g) {
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m1[(v28_lead + (v39_i1 * 12))]);
              r1[v39_i1] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v29_g) {
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 12; ++v47_i1) {
              float v52_data = __builtin_nontemporal_load(&glb_m2[(v28_lead + (v47_i1 * 6))]);
              r3[v47_i1] = v52_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v55_data = r1[0];
          float v56_data = r1[1];
          float v57_data = r1[2];
          float v58_data = r1[3];
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          tensorforge::transpose4x4b32(v59_tp, v60_tp, v61_tp, v62_tp, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 4> v63_acc{};
          float v64_data = r0[0];
          float v65_data = r0[1];
          float v66_data = r0[2];
          float v67_data = r0[3];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v63_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 2, 0, 0);
          float v72_data = r0[4];
          float v73_data = r0[5];
          float v74_data = r0[6];
          float v75_data = r0[7];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v71_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 2, 1, 0);
          float v80_data = r0[8];
          float v81_data = r0[9];
          float v82_data = r0[10];
          float v83_data = r0[11];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v79_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v86_acc, 2, 2, 0);
          r2[0] = (v87_acc[0]);
          r2[1] = (v87_acc[1]);
          r2[2] = (v87_acc[2]);
          r2[3] = (v87_acc[3]);
          float v92_data = r1[4];
          float v93_data = r1[5];
          float v94_data = r1[6];
          float v95_data = r1[7];
          float v96_tp{};
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          tensorforge::transpose4x4b32(v96_tp, v97_tp, v98_tp, v99_tp, v92_data, v93_data, v94_data, v95_data);
          tensorforge::VectorT<float, 4> v100_acc{};
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v100_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v108_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v113_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v116_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v121_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v123_acc, 2, 2, 0);
          r2[4] = (v124_acc[0]);
          r2[5] = (v124_acc[1]);
          r2[6] = (v124_acc[2]);
          r2[7] = (v124_acc[3]);
          float v129_data = r1[8];
          float v130_data = r1[9];
          float v131_data = r1[10];
          float v132_data = r1[11];
          float v133_tp{};
          float v134_tp{};
          float v135_tp{};
          float v136_tp{};
          tensorforge::transpose4x4b32(v133_tp, v134_tp, v135_tp, v136_tp, v129_data, v130_data, v131_data, v132_data);
          tensorforge::VectorT<float, 4> v137_acc{};
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v137_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v142_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v143_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v144_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v145_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v150_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v151_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v152_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v80_data, v153_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v81_data, v158_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v82_data, v159_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v83_data, v160_acc, 2, 2, 0);
          r2[8] = (v161_acc[0]);
          r2[9] = (v161_acc[1]);
          r2[10] = (v161_acc[2]);
          r2[11] = (v161_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v29_g) {
            #pragma unroll
            for (int32_t v166_i1 = 0; v166_i1 < 12; ++v166_i1) {
              float v168_data = r2[v166_i1];
              int32_t v172_a = v28_lead + (v166_i1 * 12);
              s0[(v172_a ^ ((v172_a >> 4) & 15))] = v168_data;
            }
          }
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          bool v177_g = v28_lead < 2;
          if (v177_g) {
            #pragma unroll
            for (int32_t v178_i1 = 0; v178_i1 < 12; ++v178_i1) {
              float v183_data = __builtin_nontemporal_load(&glb_m4[(v28_lead + (v178_i1 * 2))]);
              r6[v178_i1] = v183_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          float v193_tp{};
          tensorforge::transpose4x4b32(v190_tp, v191_tp, v192_tp, v193_tp, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 4> v194_acc{};
          float v195_data = r3[0];
          float v196_data = r3[1];
          float v197_data = r3[2];
          float v198_data = r3[3];
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v195_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v196_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v197_data, v200_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v198_data, v201_acc, 2, 0, 0);
          float v203_data = r3[4];
          float v204_data = r3[5];
          float v205_data = r3[6];
          float v206_data = r3[7];
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v203_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v204_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v205_data, v208_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v206_data, v209_acc, 2, 1, 0);
          float v211_data = r3[8];
          float v212_data = r3[9];
          float v213_data = r3[10];
          float v214_data = r3[11];
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v211_data, v210_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v212_data, v215_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v213_data, v216_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v214_data, v217_acc, 2, 2, 0);
          r4[0] = (v218_acc[0]);
          r4[1] = (v218_acc[1]);
          r4[2] = (v218_acc[2]);
          r4[3] = (v218_acc[3]);
          float v227_tp{};
          float v228_tp{};
          float v229_tp{};
          float v230_tp{};
          tensorforge::transpose4x4b32(v227_tp, v228_tp, v229_tp, v230_tp, v92_data, v93_data, v94_data, v95_data);
          tensorforge::VectorT<float, 4> v231_acc{};
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v195_data, v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v196_data, v236_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v197_data, v237_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v198_data, v238_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v203_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v204_data, v244_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v205_data, v245_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v206_data, v246_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v211_data, v247_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v212_data, v252_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v213_data, v253_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v214_data, v254_acc, 2, 2, 0);
          r4[4] = (v255_acc[0]);
          r4[5] = (v255_acc[1]);
          r4[6] = (v255_acc[2]);
          r4[7] = (v255_acc[3]);
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          float v267_tp{};
          tensorforge::transpose4x4b32(v264_tp, v265_tp, v266_tp, v267_tp, v129_data, v130_data, v131_data, v132_data);
          tensorforge::VectorT<float, 4> v268_acc{};
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v195_data, v268_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v196_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v197_data, v274_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v198_data, v275_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v203_data, v276_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v204_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v205_data, v282_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v206_data, v283_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v211_data, v284_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v212_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v213_data, v290_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v214_data, v291_acc, 2, 2, 0);
          r4[8] = (v292_acc[0]);
          r4[9] = (v292_acc[1]);
          r4[10] = (v292_acc[2]);
          r4[11] = (v292_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v29_g) {
            int32_t v302_off = v28_lead + 6;
            #pragma unroll
            for (int32_t v297_i1 = 0; v297_i1 < 12; ++v297_i1) {
              float v299_data = r4[v297_i1];
              int32_t v304_a = v302_off + (v297_i1 * 12);
              s0[(v304_a ^ ((v304_a >> 4) & 15))] = v299_data;
            }
          }
          float r5[12]{};
          // r5 = +(s0) + None
          // [(0, 12), (0, 12)] []
          int32_t v313_sw = (v28_lead >> 4) & 15;
          float v315_data = v38_g ? (s0[(v28_lead ^ v313_sw)]) : (0.0f);
          float v316_data = r5[0];
          r5[0] = (v316_data + v315_data);
          int32_t v318_a = v28_lead + 12;
          int32_t v319_sw = v318_a >> 4;
          float v322_data = v38_g ? (s0[(v318_a ^ (v319_sw & 15))]) : (0.0f);
          float v323_data = r5[1];
          r5[1] = (v323_data + v322_data);
          int32_t v325_a = v28_lead + 24;
          int32_t v326_sw = v325_a >> 4;
          float v329_data = v38_g ? (s0[(v325_a ^ (v326_sw & 15))]) : (0.0f);
          float v330_data = r5[2];
          r5[2] = (v330_data + v329_data);
          int32_t v332_a = v28_lead + 36;
          int32_t v333_sw = v332_a >> 4;
          float v336_data = v38_g ? (s0[(v332_a ^ (v333_sw & 15))]) : (0.0f);
          float v337_data = r5[3];
          r5[3] = (v337_data + v336_data);
          int32_t v339_a = v28_lead + 48;
          int32_t v340_sw = v339_a >> 4;
          float v343_data = v38_g ? (s0[(v339_a ^ (v340_sw & 15))]) : (0.0f);
          float v344_data = r5[4];
          r5[4] = (v344_data + v343_data);
          int32_t v346_a = v28_lead + 60;
          int32_t v347_sw = v346_a >> 4;
          float v350_data = v38_g ? (s0[(v346_a ^ (v347_sw & 15))]) : (0.0f);
          float v351_data = r5[5];
          r5[5] = (v351_data + v350_data);
          int32_t v353_a = v28_lead + 72;
          int32_t v354_sw = v353_a >> 4;
          float v357_data = v38_g ? (s0[(v353_a ^ (v354_sw & 15))]) : (0.0f);
          float v358_data = r5[6];
          r5[6] = (v358_data + v357_data);
          int32_t v360_a = v28_lead + 84;
          int32_t v361_sw = v360_a >> 4;
          float v364_data = v38_g ? (s0[(v360_a ^ (v361_sw & 15))]) : (0.0f);
          float v365_data = r5[7];
          r5[7] = (v365_data + v364_data);
          int32_t v367_a = v28_lead + 96;
          int32_t v368_sw = v367_a >> 4;
          float v371_data = v38_g ? (s0[(v367_a ^ (v368_sw & 15))]) : (0.0f);
          float v372_data = r5[8];
          r5[8] = (v372_data + v371_data);
          int32_t v374_a = v28_lead + 108;
          int32_t v375_sw = v374_a >> 4;
          float v378_data = v38_g ? (s0[(v374_a ^ (v375_sw & 15))]) : (0.0f);
          float v379_data = r5[9];
          r5[9] = (v379_data + v378_data);
          int32_t v381_a = v28_lead + 120;
          int32_t v382_sw = v381_a >> 4;
          float v385_data = v38_g ? (s0[(v381_a ^ (v382_sw & 15))]) : (0.0f);
          float v386_data = r5[10];
          r5[10] = (v386_data + v385_data);
          int32_t v388_a = v28_lead + 132;
          int32_t v389_sw = v388_a >> 4;
          float v392_data = v38_g ? (s0[(v388_a ^ (v389_sw & 15))]) : (0.0f);
          float v393_data = r5[11];
          r5[11] = (v393_data + v392_data);
          // glb_m3 = store{r>g}(r5);
          if (v38_g) {
            #pragma unroll
            for (int32_t v395_i1 = 0; v395_i1 < 12; ++v395_i1) {
              float v397_data = r5[v395_i1];
              glb_m3[(v28_lead + (v395_i1 * 12))] = v397_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[12]{};
          // r7 = +(r6 * r1) + None
          // [(0, 2), (0, 12)] [(0, 12)]
          float v407_tp{};
          float v408_tp{};
          float v409_tp{};
          float v410_tp{};
          tensorforge::transpose4x4b32(v407_tp, v408_tp, v409_tp, v410_tp, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 4> v411_acc{};
          float v412_data = r6[0];
          float v413_data = r6[1];
          float v414_data = r6[2];
          float v415_data = r6[3];
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v412_data, v411_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v413_data, v416_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v414_data, v417_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v415_data, v418_acc, 2, 0, 0);
          float v420_data = r6[4];
          float v421_data = r6[5];
          float v422_data = r6[6];
          float v423_data = r6[7];
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v420_data, v419_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v421_data, v424_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v422_data, v425_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v423_data, v426_acc, 2, 1, 0);
          float v428_data = r6[8];
          float v429_data = r6[9];
          float v430_data = r6[10];
          float v431_data = r6[11];
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v428_data, v427_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v429_data, v432_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v430_data, v433_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v431_data, v434_acc, 2, 2, 0);
          r7[0] = (v435_acc[0]);
          r7[1] = (v435_acc[1]);
          r7[2] = (v435_acc[2]);
          r7[3] = (v435_acc[3]);
          float v444_tp{};
          float v445_tp{};
          float v446_tp{};
          float v447_tp{};
          tensorforge::transpose4x4b32(v444_tp, v445_tp, v446_tp, v447_tp, v92_data, v93_data, v94_data, v95_data);
          tensorforge::VectorT<float, 4> v448_acc{};
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v412_data, v448_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v413_data, v453_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v414_data, v454_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v415_data, v455_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v420_data, v456_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v421_data, v461_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v422_data, v462_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v423_data, v463_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v428_data, v464_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v429_data, v469_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v430_data, v470_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v431_data, v471_acc, 2, 2, 0);
          r7[4] = (v472_acc[0]);
          r7[5] = (v472_acc[1]);
          r7[6] = (v472_acc[2]);
          r7[7] = (v472_acc[3]);
          float v481_tp{};
          float v482_tp{};
          float v483_tp{};
          float v484_tp{};
          tensorforge::transpose4x4b32(v481_tp, v482_tp, v483_tp, v484_tp, v129_data, v130_data, v131_data, v132_data);
          tensorforge::VectorT<float, 4> v485_acc{};
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v412_data, v485_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v413_data, v490_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v414_data, v491_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v415_data, v492_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v420_data, v493_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v421_data, v498_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v422_data, v499_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v423_data, v500_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v428_data, v501_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v429_data, v506_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v430_data, v507_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v431_data, v508_acc, 2, 2, 0);
          r7[8] = (v509_acc[0]);
          r7[9] = (v509_acc[1]);
          r7[10] = (v509_acc[2]);
          r7[11] = (v509_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r7);
          if ((v28_lead >= 8) && v38_g) {
            #pragma unroll
            for (int32_t v516_z1 = 0; v516_z1 < 12; ++v516_z1) {
              int32_t v521_a = v28_lead + (v516_z1 * 12);
              s0[(v521_a ^ ((v521_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v177_g) {
            int32_t v530_off = v28_lead + 6;
            #pragma unroll
            for (int32_t v525_i1 = 0; v525_i1 < 12; ++v525_i1) {
              float v527_data = r7[v525_i1];
              int32_t v532_a = v530_off + (v525_i1 * 12);
              s0[(v532_a ^ ((v532_a >> 4) & 15))] = v527_data;
            }
          }
          float r8[12]{};
          // r8 = +(s0) + None
          // [(0, 12), (0, 12)] []
          float v543_data = v38_g ? (s0[(v28_lead ^ v313_sw)]) : (0.0f);
          float v544_data = r8[0];
          r8[0] = (v544_data + v543_data);
          float v550_data = v38_g ? (s0[(v318_a ^ (v319_sw & 15))]) : (0.0f);
          float v551_data = r8[1];
          r8[1] = (v551_data + v550_data);
          float v557_data = v38_g ? (s0[(v325_a ^ (v326_sw & 15))]) : (0.0f);
          float v558_data = r8[2];
          r8[2] = (v558_data + v557_data);
          float v564_data = v38_g ? (s0[(v332_a ^ (v333_sw & 15))]) : (0.0f);
          float v565_data = r8[3];
          r8[3] = (v565_data + v564_data);
          float v571_data = v38_g ? (s0[(v339_a ^ (v340_sw & 15))]) : (0.0f);
          float v572_data = r8[4];
          r8[4] = (v572_data + v571_data);
          float v578_data = v38_g ? (s0[(v346_a ^ (v347_sw & 15))]) : (0.0f);
          float v579_data = r8[5];
          r8[5] = (v579_data + v578_data);
          float v585_data = v38_g ? (s0[(v353_a ^ (v354_sw & 15))]) : (0.0f);
          float v586_data = r8[6];
          r8[6] = (v586_data + v585_data);
          float v592_data = v38_g ? (s0[(v360_a ^ (v361_sw & 15))]) : (0.0f);
          float v593_data = r8[7];
          r8[7] = (v593_data + v592_data);
          float v599_data = v38_g ? (s0[(v367_a ^ (v368_sw & 15))]) : (0.0f);
          float v600_data = r8[8];
          r8[8] = (v600_data + v599_data);
          float v606_data = v38_g ? (s0[(v374_a ^ (v375_sw & 15))]) : (0.0f);
          float v607_data = r8[9];
          r8[9] = (v607_data + v606_data);
          float v613_data = v38_g ? (s0[(v381_a ^ (v382_sw & 15))]) : (0.0f);
          float v614_data = r8[10];
          r8[10] = (v614_data + v613_data);
          float v620_data = v38_g ? (s0[(v388_a ^ (v389_sw & 15))]) : (0.0f);
          float v621_data = r8[11];
          r8[11] = (v621_data + v620_data);
          // glb_m5 = store{r>g}(r8);
          if (v38_g) {
            #pragma unroll
            for (int32_t v623_i1 = 0; v623_i1 < 12; ++v623_i1) {
              float v625_data = r8[v623_i1];
              glb_m5[(v28_lead + (v623_i1 * 12))] = v625_data;
            }
          }
        }
      }
    }
  }
}

