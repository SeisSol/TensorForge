// === base name ===
kernel_4412aa190ce94c80

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4412aa190ce94c80 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4412aa190ce94c80(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4412aa190ce94c80(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4412aa190ce94c80(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4412aa190ce94c80, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4412aa190ce94c80, block.x * block.y * block.z, 0));
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
void launcher_kernel_4412aa190ce94c80(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4412aa190ce94c80(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4412aa190ce94c80), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_4412aa190ce94c80, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4412aa190ce94c80(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(12×12) {0..12}×{0..12} strided
    //   m3 32×32(4×12) {4..8}×{0..12} strided
    //   m4 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = t0[i,k] × m2[k,j]
    //   t0 12×12(12×12) {0..12}×{0..12} pointer_based({4..8}×{0..12}) = abs(N)
    //   m4[i,j] = t1[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[4,0],[8,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 16;
          bool v22_g = v21_lead < 12;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 12; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v21_lead + (v23_i1 * 12))]);
              r0[v23_i1] = v28_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v22_g) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v31_i1 * 12))]);
              r1[v31_i1] = v36_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v39_i1 * 12))]);
              r3[v39_i1] = v44_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v47_data = r1[0];
          float v48_data = r1[1];
          float v49_data = r1[2];
          float v50_data = r1[3];
          float v51_tp{};
          float v52_tp{};
          float v53_tp{};
          float v54_tp{};
          tensorforge::transpose4x4b32(v51_tp, v52_tp, v53_tp, v54_tp, v47_data, v48_data, v49_data, v50_data);
          tensorforge::VectorT<float, 4> v55_acc{};
          float v56_data = r0[0];
          float v57_data = r0[1];
          float v58_data = r0[2];
          float v59_data = r0[3];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v55_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v60_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v62_acc, 2, 0, 0);
          float v64_data = r0[4];
          float v65_data = r0[5];
          float v66_data = r0[6];
          float v67_data = r0[7];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v63_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v68_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 2, 1, 0);
          float v72_data = r0[8];
          float v73_data = r0[9];
          float v74_data = r0[10];
          float v75_data = r0[11];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v71_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v76_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v78_acc, 2, 2, 0);
          r2[0] = (v79_acc[0]);
          r2[1] = (v79_acc[1]);
          r2[2] = (v79_acc[2]);
          r2[3] = (v79_acc[3]);
          float v84_data = r1[4];
          float v85_data = r1[5];
          float v86_data = r1[6];
          float v87_data = r1[7];
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v84_data, v85_data, v86_data, v87_data);
          tensorforge::VectorT<float, 4> v92_acc{};
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v107_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v108_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v115_acc, 2, 2, 0);
          r2[4] = (v116_acc[0]);
          r2[5] = (v116_acc[1]);
          r2[6] = (v116_acc[2]);
          r2[7] = (v116_acc[3]);
          float v121_data = r1[8];
          float v122_data = r1[9];
          float v123_data = r1[10];
          float v124_data = r1[11];
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          float v128_tp{};
          tensorforge::transpose4x4b32(v125_tp, v126_tp, v127_tp, v128_tp, v121_data, v122_data, v123_data, v124_data);
          tensorforge::VectorT<float, 4> v129_acc{};
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v56_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v57_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v58_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v59_data, v136_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v64_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v65_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v66_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v67_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v72_data, v145_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v73_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v74_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v75_data, v152_acc, 2, 2, 0);
          r2[8] = (v153_acc[0]);
          r2[9] = (v153_acc[1]);
          r2[10] = (v153_acc[2]);
          r2[11] = (v153_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v158_i1 = 0; v158_i1 < 12; ++v158_i1) {
              float v160_data = r2[v158_i1];
              int32_t v164_a = v21_lead + (v158_i1 * 12);
              s0[(v164_a ^ ((v164_a >> 4) & 15))] = v160_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v169_data = r3[0];
          float v170_data = r3[1];
          float v171_data = r3[2];
          float v172_data = r3[3];
          float v173_tp{};
          float v174_tp{};
          float v175_tp{};
          float v176_tp{};
          tensorforge::transpose4x4b32(v173_tp, v174_tp, v175_tp, v176_tp, v169_data, v170_data, v171_data, v172_data);
          tensorforge::VectorT<float, 4> v177_acc{};
          int32_t v182_sw = (v21_lead >> 4) & 15;
          float v184_data = s0[(v21_lead ^ v182_sw)];
          int32_t v185_a = v21_lead + 12;
          int32_t v186_sw = v185_a >> 4;
          float v189_data = s0[(v185_a ^ (v186_sw & 15))];
          int32_t v190_a = v21_lead + 24;
          int32_t v191_sw = v190_a >> 4;
          float v194_data = s0[(v190_a ^ (v191_sw & 15))];
          int32_t v195_a = v21_lead + 36;
          int32_t v196_sw = v195_a >> 4;
          float v199_data = s0[(v195_a ^ (v196_sw & 15))];
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v184_data, v177_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v189_data, v200_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v194_data, v201_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v199_data, v202_acc, 2, 0, 0);
          int32_t v204_a = v21_lead + 48;
          int32_t v205_sw = v204_a >> 4;
          float v208_data = s0[(v204_a ^ (v205_sw & 15))];
          int32_t v209_a = v21_lead + 60;
          int32_t v210_sw = v209_a >> 4;
          float v213_data = s0[(v209_a ^ (v210_sw & 15))];
          int32_t v214_a = v21_lead + 72;
          int32_t v215_sw = v214_a >> 4;
          float v218_data = s0[(v214_a ^ (v215_sw & 15))];
          int32_t v219_a = v21_lead + 84;
          int32_t v220_sw = v219_a >> 4;
          float v223_data = s0[(v219_a ^ (v220_sw & 15))];
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v208_data, v203_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v213_data, v224_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v218_data, v225_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v223_data, v226_acc, 2, 1, 0);
          int32_t v228_a = v21_lead + 96;
          int32_t v229_sw = v228_a >> 4;
          float v232_data = s0[(v228_a ^ (v229_sw & 15))];
          int32_t v233_a = v21_lead + 108;
          int32_t v234_sw = v233_a >> 4;
          float v237_data = s0[(v233_a ^ (v234_sw & 15))];
          int32_t v238_a = v21_lead + 120;
          int32_t v239_sw = v238_a >> 4;
          float v242_data = s0[(v238_a ^ (v239_sw & 15))];
          int32_t v243_a = v21_lead + 132;
          int32_t v244_sw = v243_a >> 4;
          float v247_data = s0[(v243_a ^ (v244_sw & 15))];
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v232_data, v227_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v237_data, v248_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v242_data, v249_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v247_data, v250_acc, 2, 2, 0);
          r4[0] = (v251_acc[0]);
          r4[1] = (v251_acc[1]);
          r4[2] = (v251_acc[2]);
          r4[3] = (v251_acc[3]);
          float v256_data = r3[4];
          float v257_data = r3[5];
          float v258_data = r3[6];
          float v259_data = r3[7];
          float v260_tp{};
          float v261_tp{};
          float v262_tp{};
          float v263_tp{};
          tensorforge::transpose4x4b32(v260_tp, v261_tp, v262_tp, v263_tp, v256_data, v257_data, v258_data, v259_data);
          tensorforge::VectorT<float, 4> v264_acc{};
          float v271_data = s0[(v21_lead ^ v182_sw)];
          float v276_data = s0[(v185_a ^ (v186_sw & 15))];
          float v281_data = s0[(v190_a ^ (v191_sw & 15))];
          float v286_data = s0[(v195_a ^ (v196_sw & 15))];
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v271_data, v264_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v276_data, v287_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v281_data, v288_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v286_data, v289_acc, 2, 0, 0);
          float v295_data = s0[(v204_a ^ (v205_sw & 15))];
          float v300_data = s0[(v209_a ^ (v210_sw & 15))];
          float v305_data = s0[(v214_a ^ (v215_sw & 15))];
          float v310_data = s0[(v219_a ^ (v220_sw & 15))];
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v295_data, v290_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v300_data, v311_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v305_data, v312_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v310_data, v313_acc, 2, 1, 0);
          float v319_data = s0[(v228_a ^ (v229_sw & 15))];
          float v324_data = s0[(v233_a ^ (v234_sw & 15))];
          float v329_data = s0[(v238_a ^ (v239_sw & 15))];
          float v334_data = s0[(v243_a ^ (v244_sw & 15))];
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v319_data, v314_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v324_data, v335_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v329_data, v336_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v334_data, v337_acc, 2, 2, 0);
          r4[4] = (v338_acc[0]);
          r4[5] = (v338_acc[1]);
          r4[6] = (v338_acc[2]);
          r4[7] = (v338_acc[3]);
          float v343_data = r3[8];
          float v344_data = r3[9];
          float v345_data = r3[10];
          float v346_data = r3[11];
          float v347_tp{};
          float v348_tp{};
          float v349_tp{};
          float v350_tp{};
          tensorforge::transpose4x4b32(v347_tp, v348_tp, v349_tp, v350_tp, v343_data, v344_data, v345_data, v346_data);
          tensorforge::VectorT<float, 4> v351_acc{};
          float v358_data = s0[(v21_lead ^ v182_sw)];
          float v363_data = s0[(v185_a ^ (v186_sw & 15))];
          float v368_data = s0[(v190_a ^ (v191_sw & 15))];
          float v373_data = s0[(v195_a ^ (v196_sw & 15))];
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v358_data, v351_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v363_data, v374_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v368_data, v375_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v373_data, v376_acc, 2, 0, 0);
          float v382_data = s0[(v204_a ^ (v205_sw & 15))];
          float v387_data = s0[(v209_a ^ (v210_sw & 15))];
          float v392_data = s0[(v214_a ^ (v215_sw & 15))];
          float v397_data = s0[(v219_a ^ (v220_sw & 15))];
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v382_data, v377_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v387_data, v398_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v392_data, v399_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v397_data, v400_acc, 2, 1, 0);
          float v406_data = s0[(v228_a ^ (v229_sw & 15))];
          float v411_data = s0[(v233_a ^ (v234_sw & 15))];
          float v416_data = s0[(v238_a ^ (v239_sw & 15))];
          float v421_data = s0[(v243_a ^ (v244_sw & 15))];
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v406_data, v401_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v411_data, v422_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v416_data, v423_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v421_data, v424_acc, 2, 2, 0);
          r4[8] = (v425_acc[0]);
          r4[9] = (v425_acc[1]);
          r4[10] = (v425_acc[2]);
          r4[11] = (v425_acc[3]);
          float r5[12]{};
          // r5 = abs(glb_m3)
          bool v431_g = v21_lead < 4;
          if (v431_g) {
            int32_t v436_a = (v21_lead + 4) - 4;
            #pragma unroll
            for (int32_t v432_k1 = 0; v432_k1 < 12; ++v432_k1) {
              float v439_data = glb_m3[(v436_a + (v432_k1 * 4))];
              r5[v432_k1] = (fabsf(v439_data));
            }
          }
          // s0 = store{r>s, clear}(localShrMem0, r5);
          if (v431_g) {
            #pragma unroll
            for (int32_t v443_z1 = 0; v443_z1 < 12; ++v443_z1) {
              int32_t v448_a = v21_lead + (v443_z1 * 12);
              s0[(v448_a ^ ((v448_a >> 4) & 15))] = 0.0f;
            }
          }
          if ((v21_lead >= 8) && v22_g) {
            #pragma unroll
            for (int32_t v454_z1 = 0; v454_z1 < 12; ++v454_z1) {
              int32_t v459_a = v21_lead + (v454_z1 * 12);
              s0[(v459_a ^ ((v459_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v431_g) {
            int32_t v468_off = v21_lead + 4;
            #pragma unroll
            for (int32_t v463_i1 = 0; v463_i1 < 12; ++v463_i1) {
              float v465_data = r5[v463_i1];
              int32_t v470_a = v468_off + (v463_i1 * 12);
              s0[(v470_a ^ ((v470_a >> 4) & 15))] = v465_data;
            }
          }
          float r6[12]{};
          // r6 = +(r4 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v481_data = s0[(v21_lead ^ v182_sw)];
          float v486_data = s0[(v185_a ^ (v186_sw & 15))];
          float v491_data = s0[(v190_a ^ (v191_sw & 15))];
          float v496_data = s0[(v195_a ^ (v196_sw & 15))];
          float v497_tp{};
          float v498_tp{};
          float v499_tp{};
          float v500_tp{};
          tensorforge::transpose4x4b32(v497_tp, v498_tp, v499_tp, v500_tp, v481_data, v486_data, v491_data, v496_data);
          tensorforge::VectorT<float, 4> v501_acc{};
          float v502_data = r4[0];
          float v503_data = r4[1];
          float v504_data = r4[2];
          float v505_data = r4[3];
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v502_data, v501_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v503_data, v506_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v504_data, v507_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v505_data, v508_acc, 2, 0, 0);
          float v510_data = r4[4];
          float v511_data = r4[5];
          float v512_data = r4[6];
          float v513_data = r4[7];
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v510_data, v509_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v511_data, v514_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v512_data, v515_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v513_data, v516_acc, 2, 1, 0);
          float v518_data = r4[8];
          float v519_data = r4[9];
          float v520_data = r4[10];
          float v521_data = r4[11];
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v518_data, v517_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v519_data, v522_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v520_data, v523_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v521_data, v524_acc, 2, 2, 0);
          r6[0] = (v525_acc[0]);
          r6[1] = (v525_acc[1]);
          r6[2] = (v525_acc[2]);
          r6[3] = (v525_acc[3]);
          float v536_data = s0[(v204_a ^ (v205_sw & 15))];
          float v541_data = s0[(v209_a ^ (v210_sw & 15))];
          float v546_data = s0[(v214_a ^ (v215_sw & 15))];
          float v551_data = s0[(v219_a ^ (v220_sw & 15))];
          float v552_tp{};
          float v553_tp{};
          float v554_tp{};
          float v555_tp{};
          tensorforge::transpose4x4b32(v552_tp, v553_tp, v554_tp, v555_tp, v536_data, v541_data, v546_data, v551_data);
          tensorforge::VectorT<float, 4> v556_acc{};
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v502_data, v556_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v503_data, v561_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v504_data, v562_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v505_data, v563_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v510_data, v564_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v511_data, v569_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v512_data, v570_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v513_data, v571_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v518_data, v572_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v519_data, v577_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v554_tp, v520_data, v578_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v521_data, v579_acc, 2, 2, 0);
          r6[4] = (v580_acc[0]);
          r6[5] = (v580_acc[1]);
          r6[6] = (v580_acc[2]);
          r6[7] = (v580_acc[3]);
          float v591_data = s0[(v228_a ^ (v229_sw & 15))];
          float v596_data = s0[(v233_a ^ (v234_sw & 15))];
          float v601_data = s0[(v238_a ^ (v239_sw & 15))];
          float v606_data = s0[(v243_a ^ (v244_sw & 15))];
          float v607_tp{};
          float v608_tp{};
          float v609_tp{};
          float v610_tp{};
          tensorforge::transpose4x4b32(v607_tp, v608_tp, v609_tp, v610_tp, v591_data, v596_data, v601_data, v606_data);
          tensorforge::VectorT<float, 4> v611_acc{};
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v607_tp, v502_data, v611_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v608_tp, v503_data, v616_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v609_tp, v504_data, v617_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v610_tp, v505_data, v618_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v607_tp, v510_data, v619_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v608_tp, v511_data, v624_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v609_tp, v512_data, v625_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v610_tp, v513_data, v626_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v607_tp, v518_data, v627_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v608_tp, v519_data, v632_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v609_tp, v520_data, v633_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v610_tp, v521_data, v634_acc, 2, 2, 0);
          r6[8] = (v635_acc[0]);
          r6[9] = (v635_acc[1]);
          r6[10] = (v635_acc[2]);
          r6[11] = (v635_acc[3]);
          // glb_m4 = store{r>g}(r6);
          if (v22_g) {
            #pragma unroll
            for (int32_t v640_i1 = 0; v640_i1 < 12; ++v640_i1) {
              float v642_data = r6[v640_i1];
              glb_m4[(v21_lead + (v640_i1 * 12))] = v642_data;
            }
          }
        }
      }
    }
  }
}

