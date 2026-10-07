// === base name ===
kernel_4a3b9dc2d9b07c1f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4a3b9dc2d9b07c1f = {{16, 16, 1}, 16, 12, 1, 16, 25600, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4a3b9dc2d9b07c1f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4a3b9dc2d9b07c1f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4a3b9dc2d9b07c1f(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4a3b9dc2d9b07c1f, block.x * block.y * block.z, 6400 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6400 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4a3b9dc2d9b07c1f, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (6400 * sizeof(float)));
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
  config.sharedMemBytes = 6400 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4a3b9dc2d9b07c1f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4a3b9dc2d9b07c1f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4a3b9dc2d9b07c1f), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_4a3b9dc2d9b07c1f, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4a3b9dc2d9b07c1f(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 25600 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(6×12) {0..6}×{0..12} strided
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    //   m4 32×32(6×12) {0..6}×{0..12} strided
    //   m5 32×32(12×12) {0..12}×{0..12} strided
    //   m6 32×32(12×12) {0..12}×{0..12} strided
    //   m7 32×32(4×12) {0..4}×{0..12} strided
    //   m8 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j]@{0..6}×{0..12} = m2[i,k] × m3[k,j]
    //   t1[i,j]@{6..12}×{0..12} = m4[i,k] × m3[k,j]
    //   m5[i,j] = t1[i,k] × m6[k,j]
    //   t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..4}×{0..12}) = abs(K)
    //   m5[i,j] += m8[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":6400}],"shared_bytes":25600,"shared_elements":6400,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"G","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[6,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"H","bbox":[[0,0],[12,12]],"name":"m6","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"K","bbox":[[0,0],[4,12]],"name":"m7","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m8","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[4,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[4,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[400 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[192];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v9_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v9_batchId0 * 72 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v9_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v9_batchId0 * 144 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v9_batchId0 * 48 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v9_batchId0 * 144 + 0 + m8_extraOffset];
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
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          bool v169_g = v29_lead < 6;
          if (v169_g) {
            #pragma unroll
            for (int32_t v170_i1 = 0; v170_i1 < 12; ++v170_i1) {
              float v175_data = __builtin_nontemporal_load(&glb_m2[(v29_lead + (v170_i1 * 6))]);
              r3[v170_i1] = v175_data;
            }
          }
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
          if (v30_g) {
            #pragma unroll
            for (int32_t v158_i1 = 0; v158_i1 < 12; ++v158_i1) {
              float v160_data = r2[v158_i1];
              int32_t v164_a = v29_lead + (v158_i1 * 12);
              s0[(v164_a ^ ((v164_a >> 4) & 15))] = v160_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m3);
          if (v30_g) {
            #pragma unroll
            for (int32_t v178_i1 = 0; v178_i1 < 12; ++v178_i1) {
              float v183_data = __builtin_nontemporal_load(&glb_m3[(v29_lead + (v178_i1 * 12))]);
              r4[v178_i1] = v183_data;
            }
          }
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          if (v169_g) {
            #pragma unroll
            for (int32_t v308_i1 = 0; v308_i1 < 12; ++v308_i1) {
              float v313_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v308_i1 * 6))]);
              r6[v308_i1] = v313_data;
            }
          }
          float r5[12]{};
          // r5 = +(r3 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v186_data = r4[0];
          float v187_data = r4[1];
          float v188_data = r4[2];
          float v189_data = r4[3];
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          float v193_tp{};
          tensorforge::transpose4x4b32(v190_tp, v191_tp, v192_tp, v193_tp, v186_data, v187_data, v188_data, v189_data);
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
          r5[0] = (v218_acc[0]);
          r5[1] = (v218_acc[1]);
          r5[2] = (v218_acc[2]);
          r5[3] = (v218_acc[3]);
          float v223_data = r4[4];
          float v224_data = r4[5];
          float v225_data = r4[6];
          float v226_data = r4[7];
          float v227_tp{};
          float v228_tp{};
          float v229_tp{};
          float v230_tp{};
          tensorforge::transpose4x4b32(v227_tp, v228_tp, v229_tp, v230_tp, v223_data, v224_data, v225_data, v226_data);
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
          r5[4] = (v255_acc[0]);
          r5[5] = (v255_acc[1]);
          r5[6] = (v255_acc[2]);
          r5[7] = (v255_acc[3]);
          float v260_data = r4[8];
          float v261_data = r4[9];
          float v262_data = r4[10];
          float v263_data = r4[11];
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          float v267_tp{};
          tensorforge::transpose4x4b32(v264_tp, v265_tp, v266_tp, v267_tp, v260_data, v261_data, v262_data, v263_data);
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
          r5[8] = (v292_acc[0]);
          r5[9] = (v292_acc[1]);
          r5[10] = (v292_acc[2]);
          r5[11] = (v292_acc[3]);
          // s1 = store{r>s}(localShrMem0, r5);
          if (v169_g) {
            #pragma unroll
            for (int32_t v297_i1 = 0; v297_i1 < 12; ++v297_i1) {
              float v299_data = r5[v297_i1];
              int32_t v303_a = v29_lead + (v297_i1 * 12);
              s1[(v303_a ^ ((v303_a >> 4) & 15))] = v299_data;
            }
          }
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v30_g) {
            #pragma unroll
            for (int32_t v439_i1 = 0; v439_i1 < 12; ++v439_i1) {
              float v444_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v439_i1 * 12))]);
              r8[v439_i1] = v444_data;
            }
          }
          float r7[12]{};
          // r7 = +(r6 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v320_tp{};
          float v321_tp{};
          float v322_tp{};
          float v323_tp{};
          tensorforge::transpose4x4b32(v320_tp, v321_tp, v322_tp, v323_tp, v186_data, v187_data, v188_data, v189_data);
          tensorforge::VectorT<float, 4> v324_acc{};
          float v325_data = r6[0];
          float v326_data = r6[1];
          float v327_data = r6[2];
          float v328_data = r6[3];
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v325_data, v324_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v326_data, v329_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v327_data, v330_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v328_data, v331_acc, 2, 0, 0);
          float v333_data = r6[4];
          float v334_data = r6[5];
          float v335_data = r6[6];
          float v336_data = r6[7];
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v333_data, v332_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v334_data, v337_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v335_data, v338_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v336_data, v339_acc, 2, 1, 0);
          float v341_data = r6[8];
          float v342_data = r6[9];
          float v343_data = r6[10];
          float v344_data = r6[11];
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v341_data, v340_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v342_data, v345_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v343_data, v346_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v344_data, v347_acc, 2, 2, 0);
          r7[0] = (v348_acc[0]);
          r7[1] = (v348_acc[1]);
          r7[2] = (v348_acc[2]);
          r7[3] = (v348_acc[3]);
          float v357_tp{};
          float v358_tp{};
          float v359_tp{};
          float v360_tp{};
          tensorforge::transpose4x4b32(v357_tp, v358_tp, v359_tp, v360_tp, v223_data, v224_data, v225_data, v226_data);
          tensorforge::VectorT<float, 4> v361_acc{};
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v325_data, v361_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v326_data, v366_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v327_data, v367_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v328_data, v368_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v333_data, v369_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v334_data, v374_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v335_data, v375_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v336_data, v376_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v341_data, v377_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v342_data, v382_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v343_data, v383_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v344_data, v384_acc, 2, 2, 0);
          r7[4] = (v385_acc[0]);
          r7[5] = (v385_acc[1]);
          r7[6] = (v385_acc[2]);
          r7[7] = (v385_acc[3]);
          float v394_tp{};
          float v395_tp{};
          float v396_tp{};
          float v397_tp{};
          tensorforge::transpose4x4b32(v394_tp, v395_tp, v396_tp, v397_tp, v260_data, v261_data, v262_data, v263_data);
          tensorforge::VectorT<float, 4> v398_acc{};
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v394_tp, v325_data, v398_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v326_data, v403_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v327_data, v404_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v328_data, v405_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v394_tp, v333_data, v406_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v334_data, v411_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v335_data, v412_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v336_data, v413_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v394_tp, v341_data, v414_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v342_data, v419_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v343_data, v420_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v344_data, v421_acc, 2, 2, 0);
          r7[8] = (v422_acc[0]);
          r7[9] = (v422_acc[1]);
          r7[10] = (v422_acc[2]);
          r7[11] = (v422_acc[3]);
          // s1 = store{r>s}(localShrMem0, r7);
          if (v169_g) {
            int32_t v432_off = v29_lead + 6;
            #pragma unroll
            for (int32_t v427_i1 = 0; v427_i1 < 12; ++v427_i1) {
              float v429_data = r7[v427_i1];
              int32_t v434_a = v432_off + (v427_i1 * 12);
              s1[(v434_a ^ ((v434_a >> 4) & 15))] = v429_data;
            }
          }
          float r11[12]{};
          // r11 = load{g>r}(glb_m8);
          if (v30_g) {
            #pragma unroll
            for (int32_t v730_i1 = 0; v730_i1 < 12; ++v730_i1) {
              float v735_data = __builtin_nontemporal_load(&glb_m8[(v29_lead + (v730_i1 * 12))]);
              r11[v730_i1] = v735_data;
            }
          }
          float r9[12]{};
          // r9 = +(s1 * r8) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v447_data = r8[0];
          float v448_data = r8[1];
          float v449_data = r8[2];
          float v450_data = r8[3];
          float v451_tp{};
          float v452_tp{};
          float v453_tp{};
          float v454_tp{};
          tensorforge::transpose4x4b32(v451_tp, v452_tp, v453_tp, v454_tp, v447_data, v448_data, v449_data, v450_data);
          tensorforge::VectorT<float, 4> v455_acc{};
          int32_t v460_sw = (v29_lead >> 4) & 15;
          int32_t v461_sw = v29_lead ^ v460_sw;
          float v462_data = s1[v461_sw];
          int32_t v463_a = v29_lead + 12;
          int32_t v464_sw = v463_a >> 4;
          float v467_data = s1[(v463_a ^ (v464_sw & 15))];
          int32_t v468_a = v29_lead + 24;
          int32_t v469_sw = v468_a >> 4;
          float v472_data = s1[(v468_a ^ (v469_sw & 15))];
          int32_t v473_a = v29_lead + 36;
          int32_t v474_sw = v473_a >> 4;
          float v477_data = s1[(v473_a ^ (v474_sw & 15))];
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v462_data, v455_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v467_data, v478_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v472_data, v479_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v477_data, v480_acc, 2, 0, 0);
          int32_t v482_a = v29_lead + 48;
          int32_t v483_sw = v482_a >> 4;
          float v486_data = s1[(v482_a ^ (v483_sw & 15))];
          int32_t v487_a = v29_lead + 60;
          int32_t v488_sw = v487_a >> 4;
          float v491_data = s1[(v487_a ^ (v488_sw & 15))];
          int32_t v492_a = v29_lead + 72;
          int32_t v493_sw = v492_a >> 4;
          float v496_data = s1[(v492_a ^ (v493_sw & 15))];
          int32_t v497_a = v29_lead + 84;
          int32_t v498_sw = v497_a >> 4;
          float v501_data = s1[(v497_a ^ (v498_sw & 15))];
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v486_data, v481_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v491_data, v502_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v496_data, v503_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v501_data, v504_acc, 2, 1, 0);
          int32_t v506_a = v29_lead + 96;
          int32_t v507_sw = v506_a >> 4;
          float v510_data = s1[(v506_a ^ (v507_sw & 15))];
          int32_t v511_a = v29_lead + 108;
          int32_t v512_sw = v511_a >> 4;
          float v515_data = s1[(v511_a ^ (v512_sw & 15))];
          int32_t v516_a = v29_lead + 120;
          int32_t v517_sw = v516_a >> 4;
          float v520_data = s1[(v516_a ^ (v517_sw & 15))];
          int32_t v521_a = v29_lead + 132;
          int32_t v522_sw = v521_a >> 4;
          float v525_data = s1[(v521_a ^ (v522_sw & 15))];
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v510_data, v505_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v515_data, v526_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v520_data, v527_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v525_data, v528_acc, 2, 2, 0);
          r9[0] = (v529_acc[0]);
          r9[1] = (v529_acc[1]);
          r9[2] = (v529_acc[2]);
          r9[3] = (v529_acc[3]);
          float v534_data = r8[4];
          float v535_data = r8[5];
          float v536_data = r8[6];
          float v537_data = r8[7];
          float v538_tp{};
          float v539_tp{};
          float v540_tp{};
          float v541_tp{};
          tensorforge::transpose4x4b32(v538_tp, v539_tp, v540_tp, v541_tp, v534_data, v535_data, v536_data, v537_data);
          tensorforge::VectorT<float, 4> v542_acc{};
          float v549_data = s1[(v29_lead ^ v460_sw)];
          float v554_data = s1[(v463_a ^ (v464_sw & 15))];
          float v559_data = s1[(v468_a ^ (v469_sw & 15))];
          float v564_data = s1[(v473_a ^ (v474_sw & 15))];
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v538_tp, v549_data, v542_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v539_tp, v554_data, v565_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v540_tp, v559_data, v566_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v541_tp, v564_data, v567_acc, 2, 0, 0);
          float v573_data = s1[(v482_a ^ (v483_sw & 15))];
          float v578_data = s1[(v487_a ^ (v488_sw & 15))];
          float v583_data = s1[(v492_a ^ (v493_sw & 15))];
          float v588_data = s1[(v497_a ^ (v498_sw & 15))];
          tensorforge::VectorT<float, 4> v589_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v538_tp, v573_data, v568_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v590_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v539_tp, v578_data, v589_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v591_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v540_tp, v583_data, v590_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v541_tp, v588_data, v591_acc, 2, 1, 0);
          float v597_data = s1[(v506_a ^ (v507_sw & 15))];
          float v602_data = s1[(v511_a ^ (v512_sw & 15))];
          float v607_data = s1[(v516_a ^ (v517_sw & 15))];
          float v612_data = s1[(v521_a ^ (v522_sw & 15))];
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v538_tp, v597_data, v592_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v539_tp, v602_data, v613_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v540_tp, v607_data, v614_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v541_tp, v612_data, v615_acc, 2, 2, 0);
          r9[4] = (v616_acc[0]);
          r9[5] = (v616_acc[1]);
          r9[6] = (v616_acc[2]);
          r9[7] = (v616_acc[3]);
          float v621_data = r8[8];
          float v622_data = r8[9];
          float v623_data = r8[10];
          float v624_data = r8[11];
          float v625_tp{};
          float v626_tp{};
          float v627_tp{};
          float v628_tp{};
          tensorforge::transpose4x4b32(v625_tp, v626_tp, v627_tp, v628_tp, v621_data, v622_data, v623_data, v624_data);
          tensorforge::VectorT<float, 4> v629_acc{};
          float v636_data = s1[(v29_lead ^ v460_sw)];
          float v641_data = s1[(v463_a ^ (v464_sw & 15))];
          float v646_data = s1[(v468_a ^ (v469_sw & 15))];
          float v651_data = s1[(v473_a ^ (v474_sw & 15))];
          tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v636_data, v629_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v641_data, v652_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v646_data, v653_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v651_data, v654_acc, 2, 0, 0);
          float v660_data = s1[(v482_a ^ (v483_sw & 15))];
          float v665_data = s1[(v487_a ^ (v488_sw & 15))];
          float v670_data = s1[(v492_a ^ (v493_sw & 15))];
          float v675_data = s1[(v497_a ^ (v498_sw & 15))];
          tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v660_data, v655_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v665_data, v676_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v678_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v670_data, v677_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v679_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v675_data, v678_acc, 2, 1, 0);
          float v684_data = s1[(v506_a ^ (v507_sw & 15))];
          float v689_data = s1[(v511_a ^ (v512_sw & 15))];
          float v694_data = s1[(v516_a ^ (v517_sw & 15))];
          float v699_data = s1[(v521_a ^ (v522_sw & 15))];
          tensorforge::VectorT<float, 4> v700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v684_data, v679_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v689_data, v700_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v694_data, v701_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v699_data, v702_acc, 2, 2, 0);
          r9[8] = (v703_acc[0]);
          r9[9] = (v703_acc[1]);
          r9[10] = (v703_acc[2]);
          r9[11] = (v703_acc[3]);
          float r10[12]{};
          // r10 = abs(glb_m7)
          bool v709_g = v29_lead < 4;
          if (v709_g) {
            #pragma unroll
            for (int32_t v710_k1 = 0; v710_k1 < 12; ++v710_k1) {
              float v715_data = glb_m7[(v29_lead + (v710_k1 * 4))];
              r10[v710_k1] = (fabsf(v715_data));
            }
          }
          // s0 = store{r>s}(localShrMem0, r10);
          if (v709_g) {
            #pragma unroll
            for (int32_t v719_i1 = 0; v719_i1 < 12; ++v719_i1) {
              float v721_data = r10[v719_i1];
              int32_t v725_a = v29_lead + (v719_i1 * 12);
              s0[(v725_a ^ ((v725_a >> 4) & 15))] = v721_data;
            }
          }
          float r12[12]{};
          // ir12 = +(r11 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir12[12]{};
          int32_t v744_sw = v29_lead ^ v460_sw;
          float v745_data = s0[v744_sw];
          float v750_data = s0[(v463_a ^ (v464_sw & 15))];
          float v755_data = s0[(v468_a ^ (v469_sw & 15))];
          float v760_data = s0[(v473_a ^ (v474_sw & 15))];
          float v761_tp{};
          float v762_tp{};
          float v763_tp{};
          float v764_tp{};
          tensorforge::transpose4x4b32(v761_tp, v762_tp, v763_tp, v764_tp, v745_data, v750_data, v755_data, v760_data);
          tensorforge::VectorT<float, 4> v765_acc{};
          float v766_data = r11[0];
          float v767_data = r11[1];
          float v768_data = r11[2];
          float v769_data = r11[3];
          tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v766_data, v765_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v771_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v762_tp, v767_data, v770_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v763_tp, v768_data, v771_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v769_data, v772_acc, 2, 0, 0);
          float v774_data = r11[4];
          float v775_data = r11[5];
          float v776_data = r11[6];
          float v777_data = r11[7];
          tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v774_data, v773_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v779_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v762_tp, v775_data, v778_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v763_tp, v776_data, v779_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v777_data, v780_acc, 2, 1, 0);
          float v782_data = r11[8];
          float v783_data = r11[9];
          float v784_data = r11[10];
          float v785_data = r11[11];
          tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v761_tp, v782_data, v781_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v787_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v762_tp, v783_data, v786_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v788_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v763_tp, v784_data, v787_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v785_data, v788_acc, 2, 2, 0);
          ir12[0] = (v789_acc[0]);
          ir12[1] = (v789_acc[1]);
          ir12[2] = (v789_acc[2]);
          ir12[3] = (v789_acc[3]);
          float v800_data = s0[(v482_a ^ (v483_sw & 15))];
          float v805_data = s0[(v487_a ^ (v488_sw & 15))];
          float v810_data = s0[(v492_a ^ (v493_sw & 15))];
          float v815_data = s0[(v497_a ^ (v498_sw & 15))];
          float v816_tp{};
          float v817_tp{};
          float v818_tp{};
          float v819_tp{};
          tensorforge::transpose4x4b32(v816_tp, v817_tp, v818_tp, v819_tp, v800_data, v805_data, v810_data, v815_data);
          tensorforge::VectorT<float, 4> v820_acc{};
          tensorforge::VectorT<float, 4> v825_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v816_tp, v766_data, v820_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v817_tp, v767_data, v825_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v827_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v818_tp, v768_data, v826_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v819_tp, v769_data, v827_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v833_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v816_tp, v774_data, v828_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v834_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v817_tp, v775_data, v833_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v835_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v818_tp, v776_data, v834_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v836_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v819_tp, v777_data, v835_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v841_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v816_tp, v782_data, v836_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v842_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v817_tp, v783_data, v841_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v843_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v818_tp, v784_data, v842_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v844_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v819_tp, v785_data, v843_acc, 2, 2, 0);
          ir12[4] = (v844_acc[0]);
          ir12[5] = (v844_acc[1]);
          ir12[6] = (v844_acc[2]);
          ir12[7] = (v844_acc[3]);
          float v855_data = s0[(v506_a ^ (v507_sw & 15))];
          float v860_data = s0[(v511_a ^ (v512_sw & 15))];
          float v865_data = s0[(v516_a ^ (v517_sw & 15))];
          float v870_data = s0[(v521_a ^ (v522_sw & 15))];
          float v871_tp{};
          float v872_tp{};
          float v873_tp{};
          float v874_tp{};
          tensorforge::transpose4x4b32(v871_tp, v872_tp, v873_tp, v874_tp, v855_data, v860_data, v865_data, v870_data);
          tensorforge::VectorT<float, 4> v875_acc{};
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v766_data, v875_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v881_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v872_tp, v767_data, v880_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v882_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v873_tp, v768_data, v881_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v874_tp, v769_data, v882_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v774_data, v883_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v889_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v872_tp, v775_data, v888_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v890_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v873_tp, v776_data, v889_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v891_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v874_tp, v777_data, v890_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v896_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v871_tp, v782_data, v891_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v897_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v872_tp, v783_data, v896_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v898_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v873_tp, v784_data, v897_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v899_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v874_tp, v785_data, v898_acc, 2, 2, 0);
          ir12[8] = (v899_acc[0]);
          ir12[9] = (v899_acc[1]);
          ir12[10] = (v899_acc[2]);
          ir12[11] = (v899_acc[3]);
          // r12 = ir12 + r9
          if (v30_g) {
            #pragma unroll
            for (int32_t v904_n1 = 0; v904_n1 < 12; ++v904_n1) {
              float v906_data = ir12[v904_n1];
              float v907_data = r9[v904_n1];
              r12[v904_n1] = (v907_data + v906_data);
            }
          }
          // glb_m5 = store{r>g}(r12);
          if (v30_g) {
            #pragma unroll
            for (int32_t v909_i1 = 0; v909_i1 < 12; ++v909_i1) {
              float v911_data = r12[v909_i1];
              glb_m5[(v29_lead + (v909_i1 * 12))] = v911_data;
            }
          }
        }
      }
    }
  }
}

