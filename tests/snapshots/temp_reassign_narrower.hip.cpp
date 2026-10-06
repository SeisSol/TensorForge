// === base name ===
kernel_c150bf0963d82257

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c150bf0963d82257 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c150bf0963d82257(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c150bf0963d82257(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c150bf0963d82257(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c150bf0963d82257, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c150bf0963d82257, block.x * block.y * block.z, 0));
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
void launcher_kernel_c150bf0963d82257(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c150bf0963d82257(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c150bf0963d82257), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_c150bf0963d82257, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c150bf0963d82257(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 48 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v11_batchId0 * 144 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v11_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v11_batchId0 * 144 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v29_lead = threadIdx.x % 16;
          bool v30_g = v29_lead < 12;
          if (v30_g) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m0[(v29_lead + (v31_i1 * 12))]);
              r0[v31_i1] = v36_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v30_g) {
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v39_i1 * 12))]);
              r1[v39_i1] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v30_g) {
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 12; ++v47_i1) {
              float v52_data = __builtin_nontemporal_load(&glb_m2[(v29_lead + (v47_i1 * 12))]);
              r3[v47_i1] = v52_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
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
          if (v30_g) {
            #pragma unroll
            for (int32_t v166_i1 = 0; v166_i1 < 12; ++v166_i1) {
              float v168_data = r2[v166_i1];
              int32_t v172_a = v29_lead + (v166_i1 * 12);
              s0[(v172_a ^ ((v172_a >> 4) & 15))] = v168_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          bool v177_g = v29_lead < 4;
          if (v177_g) {
            #pragma unroll
            for (int32_t v178_i1 = 0; v178_i1 < 12; ++v178_i1) {
              float v183_data = __builtin_nontemporal_load(&glb_m3[(v29_lead + (v178_i1 * 4))]);
              r5[v178_i1] = v183_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v186_data = r3[0];
          float v187_data = r3[1];
          float v188_data = r3[2];
          float v189_data = r3[3];
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          float v193_tp{};
          tensorforge::transpose4x4b32(v190_tp, v191_tp, v192_tp, v193_tp, v186_data, v187_data, v188_data, v189_data);
          tensorforge::VectorT<float, 4> v194_acc{};
          int32_t v199_sw = (v29_lead >> 4) & 15;
          float v201_data = s0[(v29_lead ^ v199_sw)];
          int32_t v202_a = v29_lead + 12;
          int32_t v203_sw = v202_a >> 4;
          float v206_data = s0[(v202_a ^ (v203_sw & 15))];
          int32_t v207_a = v29_lead + 24;
          int32_t v208_sw = v207_a >> 4;
          float v211_data = s0[(v207_a ^ (v208_sw & 15))];
          int32_t v212_a = v29_lead + 36;
          int32_t v213_sw = v212_a >> 4;
          float v216_data = s0[(v212_a ^ (v213_sw & 15))];
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v201_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v206_data, v217_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v211_data, v218_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v216_data, v219_acc, 2, 0, 0);
          int32_t v221_a = v29_lead + 48;
          int32_t v222_sw = v221_a >> 4;
          float v225_data = s0[(v221_a ^ (v222_sw & 15))];
          int32_t v226_a = v29_lead + 60;
          int32_t v227_sw = v226_a >> 4;
          float v230_data = s0[(v226_a ^ (v227_sw & 15))];
          int32_t v231_a = v29_lead + 72;
          int32_t v232_sw = v231_a >> 4;
          float v235_data = s0[(v231_a ^ (v232_sw & 15))];
          int32_t v236_a = v29_lead + 84;
          int32_t v237_sw = v236_a >> 4;
          float v240_data = s0[(v236_a ^ (v237_sw & 15))];
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v225_data, v220_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v230_data, v241_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v235_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v240_data, v243_acc, 2, 1, 0);
          int32_t v245_a = v29_lead + 96;
          int32_t v246_sw = v245_a >> 4;
          float v249_data = s0[(v245_a ^ (v246_sw & 15))];
          int32_t v250_a = v29_lead + 108;
          int32_t v251_sw = v250_a >> 4;
          float v254_data = s0[(v250_a ^ (v251_sw & 15))];
          int32_t v255_a = v29_lead + 120;
          int32_t v256_sw = v255_a >> 4;
          float v259_data = s0[(v255_a ^ (v256_sw & 15))];
          int32_t v260_a = v29_lead + 132;
          int32_t v261_sw = v260_a >> 4;
          float v264_data = s0[(v260_a ^ (v261_sw & 15))];
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v249_data, v244_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v254_data, v265_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v259_data, v266_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v264_data, v267_acc, 2, 2, 0);
          r4[0] = (v268_acc[0]);
          r4[1] = (v268_acc[1]);
          r4[2] = (v268_acc[2]);
          r4[3] = (v268_acc[3]);
          float v273_data = r3[4];
          float v274_data = r3[5];
          float v275_data = r3[6];
          float v276_data = r3[7];
          float v277_tp{};
          float v278_tp{};
          float v279_tp{};
          float v280_tp{};
          tensorforge::transpose4x4b32(v277_tp, v278_tp, v279_tp, v280_tp, v273_data, v274_data, v275_data, v276_data);
          tensorforge::VectorT<float, 4> v281_acc{};
          float v288_data = s0[(v29_lead ^ v199_sw)];
          float v293_data = s0[(v202_a ^ (v203_sw & 15))];
          float v298_data = s0[(v207_a ^ (v208_sw & 15))];
          float v303_data = s0[(v212_a ^ (v213_sw & 15))];
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v288_data, v281_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v293_data, v304_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v298_data, v305_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v303_data, v306_acc, 2, 0, 0);
          float v312_data = s0[(v221_a ^ (v222_sw & 15))];
          float v317_data = s0[(v226_a ^ (v227_sw & 15))];
          float v322_data = s0[(v231_a ^ (v232_sw & 15))];
          float v327_data = s0[(v236_a ^ (v237_sw & 15))];
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v312_data, v307_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v317_data, v328_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v322_data, v329_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v327_data, v330_acc, 2, 1, 0);
          float v336_data = s0[(v245_a ^ (v246_sw & 15))];
          float v341_data = s0[(v250_a ^ (v251_sw & 15))];
          float v346_data = s0[(v255_a ^ (v256_sw & 15))];
          float v351_data = s0[(v260_a ^ (v261_sw & 15))];
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v336_data, v331_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v341_data, v352_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v346_data, v353_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v351_data, v354_acc, 2, 2, 0);
          r4[4] = (v355_acc[0]);
          r4[5] = (v355_acc[1]);
          r4[6] = (v355_acc[2]);
          r4[7] = (v355_acc[3]);
          float v360_data = r3[8];
          float v361_data = r3[9];
          float v362_data = r3[10];
          float v363_data = r3[11];
          float v364_tp{};
          float v365_tp{};
          float v366_tp{};
          float v367_tp{};
          tensorforge::transpose4x4b32(v364_tp, v365_tp, v366_tp, v367_tp, v360_data, v361_data, v362_data, v363_data);
          tensorforge::VectorT<float, 4> v368_acc{};
          float v375_data = s0[(v29_lead ^ v199_sw)];
          float v380_data = s0[(v202_a ^ (v203_sw & 15))];
          float v385_data = s0[(v207_a ^ (v208_sw & 15))];
          float v390_data = s0[(v212_a ^ (v213_sw & 15))];
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v375_data, v368_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v380_data, v391_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v385_data, v392_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v390_data, v393_acc, 2, 0, 0);
          float v399_data = s0[(v221_a ^ (v222_sw & 15))];
          float v404_data = s0[(v226_a ^ (v227_sw & 15))];
          float v409_data = s0[(v231_a ^ (v232_sw & 15))];
          float v414_data = s0[(v236_a ^ (v237_sw & 15))];
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v399_data, v394_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v404_data, v415_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v409_data, v416_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v414_data, v417_acc, 2, 1, 0);
          float v423_data = s0[(v245_a ^ (v246_sw & 15))];
          float v428_data = s0[(v250_a ^ (v251_sw & 15))];
          float v433_data = s0[(v255_a ^ (v256_sw & 15))];
          float v438_data = s0[(v260_a ^ (v261_sw & 15))];
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v423_data, v418_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v428_data, v439_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v433_data, v440_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v438_data, v441_acc, 2, 2, 0);
          r4[8] = (v442_acc[0]);
          r4[9] = (v442_acc[1]);
          r4[10] = (v442_acc[2]);
          r4[11] = (v442_acc[3]);
          float r7[12]{};
          // r7 = load{g>r}(glb_m4);
          if (v30_g) {
            #pragma unroll
            for (int32_t v448_i1 = 0; v448_i1 < 12; ++v448_i1) {
              float v453_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v448_i1 * 12))]);
              r7[v448_i1] = v453_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = +(r5 * r1) + None
          // [(0, 4), (0, 12)] [(0, 12)]
          float v460_tp{};
          float v461_tp{};
          float v462_tp{};
          float v463_tp{};
          tensorforge::transpose4x4b32(v460_tp, v461_tp, v462_tp, v463_tp, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 4> v464_acc{};
          float v465_data = r5[0];
          float v466_data = r5[1];
          float v467_data = r5[2];
          float v468_data = r5[3];
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v465_data, v464_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v466_data, v469_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v467_data, v470_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v468_data, v471_acc, 2, 0, 0);
          float v473_data = r5[4];
          float v474_data = r5[5];
          float v475_data = r5[6];
          float v476_data = r5[7];
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v473_data, v472_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v474_data, v477_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v475_data, v478_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v476_data, v479_acc, 2, 1, 0);
          float v481_data = r5[8];
          float v482_data = r5[9];
          float v483_data = r5[10];
          float v484_data = r5[11];
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v481_data, v480_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v482_data, v485_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v483_data, v486_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v484_data, v487_acc, 2, 2, 0);
          r6[0] = (v488_acc[0]);
          r6[1] = (v488_acc[1]);
          r6[2] = (v488_acc[2]);
          r6[3] = (v488_acc[3]);
          float v497_tp{};
          float v498_tp{};
          float v499_tp{};
          float v500_tp{};
          tensorforge::transpose4x4b32(v497_tp, v498_tp, v499_tp, v500_tp, v92_data, v93_data, v94_data, v95_data);
          tensorforge::VectorT<float, 4> v501_acc{};
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v465_data, v501_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v466_data, v506_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v467_data, v507_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v468_data, v508_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v473_data, v509_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v474_data, v514_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v475_data, v515_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v476_data, v516_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v481_data, v517_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v482_data, v522_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v483_data, v523_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v484_data, v524_acc, 2, 2, 0);
          r6[4] = (v525_acc[0]);
          r6[5] = (v525_acc[1]);
          r6[6] = (v525_acc[2]);
          r6[7] = (v525_acc[3]);
          float v534_tp{};
          float v535_tp{};
          float v536_tp{};
          float v537_tp{};
          tensorforge::transpose4x4b32(v534_tp, v535_tp, v536_tp, v537_tp, v129_data, v130_data, v131_data, v132_data);
          tensorforge::VectorT<float, 4> v538_acc{};
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v534_tp, v465_data, v538_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v535_tp, v466_data, v543_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v536_tp, v467_data, v544_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v537_tp, v468_data, v545_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v534_tp, v473_data, v546_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v535_tp, v474_data, v551_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v536_tp, v475_data, v552_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v537_tp, v476_data, v553_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v534_tp, v481_data, v554_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v535_tp, v482_data, v559_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v536_tp, v483_data, v560_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v537_tp, v484_data, v561_acc, 2, 2, 0);
          r6[8] = (v562_acc[0]);
          r6[9] = (v562_acc[1]);
          r6[10] = (v562_acc[2]);
          r6[11] = (v562_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r6);
          if ((v29_lead >= 4) && v30_g) {
            #pragma unroll
            for (int32_t v569_z1 = 0; v569_z1 < 12; ++v569_z1) {
              int32_t v574_a = v29_lead + (v569_z1 * 12);
              s0[(v574_a ^ ((v574_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v177_g) {
            #pragma unroll
            for (int32_t v578_i1 = 0; v578_i1 < 12; ++v578_i1) {
              float v580_data = r6[v578_i1];
              int32_t v584_a = v29_lead + (v578_i1 * 12);
              s0[(v584_a ^ ((v584_a >> 4) & 15))] = v580_data;
            }
          }
          float r9[12]{};
          // r9 = load{g>r}(glb_m6);
          if (v30_g) {
            #pragma unroll
            for (int32_t v589_i1 = 0; v589_i1 < 12; ++v589_i1) {
              float v594_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v589_i1 * 12))]);
              r9[v589_i1] = v594_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m4););
          float r8[12]{};
          // ir8 = +(r4 * r7)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir8[12]{};
          float v598_data = r7[0];
          float v599_data = r7[1];
          float v600_data = r7[2];
          float v601_data = r7[3];
          float v602_tp{};
          float v603_tp{};
          float v604_tp{};
          float v605_tp{};
          tensorforge::transpose4x4b32(v602_tp, v603_tp, v604_tp, v605_tp, v598_data, v599_data, v600_data, v601_data);
          tensorforge::VectorT<float, 4> v606_acc{};
          float v607_data = r4[0];
          float v608_data = r4[1];
          float v609_data = r4[2];
          float v610_data = r4[3];
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v607_data, v606_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v603_tp, v608_data, v611_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v604_tp, v609_data, v612_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v605_tp, v610_data, v613_acc, 2, 0, 0);
          float v615_data = r4[4];
          float v616_data = r4[5];
          float v617_data = r4[6];
          float v618_data = r4[7];
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v615_data, v614_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v603_tp, v616_data, v619_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v604_tp, v617_data, v620_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v605_tp, v618_data, v621_acc, 2, 1, 0);
          float v623_data = r4[8];
          float v624_data = r4[9];
          float v625_data = r4[10];
          float v626_data = r4[11];
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v623_data, v622_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v603_tp, v624_data, v627_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v604_tp, v625_data, v628_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v605_tp, v626_data, v629_acc, 2, 2, 0);
          ir8[0] = (v630_acc[0]);
          ir8[1] = (v630_acc[1]);
          ir8[2] = (v630_acc[2]);
          ir8[3] = (v630_acc[3]);
          float v635_data = r7[4];
          float v636_data = r7[5];
          float v637_data = r7[6];
          float v638_data = r7[7];
          float v639_tp{};
          float v640_tp{};
          float v641_tp{};
          float v642_tp{};
          tensorforge::transpose4x4b32(v639_tp, v640_tp, v641_tp, v642_tp, v635_data, v636_data, v637_data, v638_data);
          tensorforge::VectorT<float, 4> v643_acc{};
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v607_data, v643_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v608_data, v648_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v609_data, v649_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v610_data, v650_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v615_data, v651_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v616_data, v656_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v617_data, v657_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v618_data, v658_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v623_data, v659_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v624_data, v664_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v625_data, v665_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v626_data, v666_acc, 2, 2, 0);
          ir8[4] = (v667_acc[0]);
          ir8[5] = (v667_acc[1]);
          ir8[6] = (v667_acc[2]);
          ir8[7] = (v667_acc[3]);
          float v672_data = r7[8];
          float v673_data = r7[9];
          float v674_data = r7[10];
          float v675_data = r7[11];
          float v676_tp{};
          float v677_tp{};
          float v678_tp{};
          float v679_tp{};
          tensorforge::transpose4x4b32(v676_tp, v677_tp, v678_tp, v679_tp, v672_data, v673_data, v674_data, v675_data);
          tensorforge::VectorT<float, 4> v680_acc{};
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v607_data, v680_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v686_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v677_tp, v608_data, v685_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v678_tp, v609_data, v686_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v679_tp, v610_data, v687_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v615_data, v688_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v694_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v677_tp, v616_data, v693_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v695_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v678_tp, v617_data, v694_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v679_tp, v618_data, v695_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v623_data, v696_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v677_tp, v624_data, v701_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v678_tp, v625_data, v702_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v679_tp, v626_data, v703_acc, 2, 2, 0);
          ir8[8] = (v704_acc[0]);
          ir8[9] = (v704_acc[1]);
          ir8[10] = (v704_acc[2]);
          ir8[11] = (v704_acc[3]);
          // r8 = ir8 + s0
          if (v30_g) {
            #pragma unroll
            for (int32_t v709_n1 = 0; v709_n1 < 12; ++v709_n1) {
              float v711_data = ir8[v709_n1];
              int32_t v715_a = v29_lead + (v709_n1 * 12);
              float v719_data = s0[(v715_a ^ ((v715_a >> 4) & 15))];
              r8[v709_n1] = (v719_data + v711_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r8);
          if (v30_g) {
            #pragma unroll
            for (int32_t v721_i1 = 0; v721_i1 < 12; ++v721_i1) {
              float v723_data = r8[v721_i1];
              int32_t v727_a = v29_lead + (v721_i1 * 12);
              s0[(v727_a ^ ((v727_a >> 4) & 15))] = v723_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[12]{};
          // r10 = +(r9 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v738_data = s0[(v29_lead ^ v199_sw)];
          float v743_data = s0[(v202_a ^ (v203_sw & 15))];
          float v748_data = s0[(v207_a ^ (v208_sw & 15))];
          float v753_data = s0[(v212_a ^ (v213_sw & 15))];
          float v754_tp{};
          float v755_tp{};
          float v756_tp{};
          float v757_tp{};
          tensorforge::transpose4x4b32(v754_tp, v755_tp, v756_tp, v757_tp, v738_data, v743_data, v748_data, v753_data);
          tensorforge::VectorT<float, 4> v758_acc{};
          float v759_data = r9[0];
          float v760_data = r9[1];
          float v761_data = r9[2];
          float v762_data = r9[3];
          tensorforge::VectorT<float, 4> v763_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v754_tp, v759_data, v758_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v764_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v755_tp, v760_data, v763_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v765_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v756_tp, v761_data, v764_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v766_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v757_tp, v762_data, v765_acc, 2, 0, 0);
          float v767_data = r9[4];
          float v768_data = r9[5];
          float v769_data = r9[6];
          float v770_data = r9[7];
          tensorforge::VectorT<float, 4> v771_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v754_tp, v767_data, v766_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v755_tp, v768_data, v771_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v756_tp, v769_data, v772_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v757_tp, v770_data, v773_acc, 2, 1, 0);
          float v775_data = r9[8];
          float v776_data = r9[9];
          float v777_data = r9[10];
          float v778_data = r9[11];
          tensorforge::VectorT<float, 4> v779_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v754_tp, v775_data, v774_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v755_tp, v776_data, v779_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v756_tp, v777_data, v780_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v757_tp, v778_data, v781_acc, 2, 2, 0);
          r10[0] = (v782_acc[0]);
          r10[1] = (v782_acc[1]);
          r10[2] = (v782_acc[2]);
          r10[3] = (v782_acc[3]);
          float v793_data = s0[(v221_a ^ (v222_sw & 15))];
          float v798_data = s0[(v226_a ^ (v227_sw & 15))];
          float v803_data = s0[(v231_a ^ (v232_sw & 15))];
          float v808_data = s0[(v236_a ^ (v237_sw & 15))];
          float v809_tp{};
          float v810_tp{};
          float v811_tp{};
          float v812_tp{};
          tensorforge::transpose4x4b32(v809_tp, v810_tp, v811_tp, v812_tp, v793_data, v798_data, v803_data, v808_data);
          tensorforge::VectorT<float, 4> v813_acc{};
          tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v759_data, v813_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v819_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v810_tp, v760_data, v818_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v811_tp, v761_data, v819_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v821_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v812_tp, v762_data, v820_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v767_data, v821_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v827_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v810_tp, v768_data, v826_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v811_tp, v769_data, v827_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v829_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v812_tp, v770_data, v828_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v834_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v775_data, v829_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v835_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v810_tp, v776_data, v834_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v836_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v811_tp, v777_data, v835_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v837_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v812_tp, v778_data, v836_acc, 2, 2, 0);
          r10[4] = (v837_acc[0]);
          r10[5] = (v837_acc[1]);
          r10[6] = (v837_acc[2]);
          r10[7] = (v837_acc[3]);
          float v848_data = s0[(v245_a ^ (v246_sw & 15))];
          float v853_data = s0[(v250_a ^ (v251_sw & 15))];
          float v858_data = s0[(v255_a ^ (v256_sw & 15))];
          float v863_data = s0[(v260_a ^ (v261_sw & 15))];
          float v864_tp{};
          float v865_tp{};
          float v866_tp{};
          float v867_tp{};
          tensorforge::transpose4x4b32(v864_tp, v865_tp, v866_tp, v867_tp, v848_data, v853_data, v858_data, v863_data);
          tensorforge::VectorT<float, 4> v868_acc{};
          tensorforge::VectorT<float, 4> v873_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v864_tp, v759_data, v868_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v874_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v865_tp, v760_data, v873_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v875_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v866_tp, v761_data, v874_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v876_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v867_tp, v762_data, v875_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v881_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v864_tp, v767_data, v876_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v882_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v865_tp, v768_data, v881_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v866_tp, v769_data, v882_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v867_tp, v770_data, v883_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v889_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v864_tp, v775_data, v884_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v890_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v865_tp, v776_data, v889_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v891_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v866_tp, v777_data, v890_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v892_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v867_tp, v778_data, v891_acc, 2, 2, 0);
          r10[8] = (v892_acc[0]);
          r10[9] = (v892_acc[1]);
          r10[10] = (v892_acc[2]);
          r10[11] = (v892_acc[3]);
          // glb_m5 = store{r>g}(r10);
          if (v30_g) {
            #pragma unroll
            for (int32_t v897_i1 = 0; v897_i1 < 12; ++v897_i1) {
              float v899_data = r10[v897_i1];
              glb_m5[(v29_lead + (v897_i1 * 12))] = v899_data;
            }
          }
        }
      }
    }
  }
}

