// === base name ===
kernel_aad6e84e0d3d554b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_aad6e84e0d3d554b = {{16, 16, 1}, 16, 12, 1, 16, 25600, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_aad6e84e0d3d554b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_aad6e84e0d3d554b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_aad6e84e0d3d554b(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_aad6e84e0d3d554b, block.x * block.y * block.z, 6400 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6400 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_aad6e84e0d3d554b, block.x * block.y * block.z, 0));
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
void launcher_kernel_aad6e84e0d3d554b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_aad6e84e0d3d554b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_aad6e84e0d3d554b), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_aad6e84e0d3d554b, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_aad6e84e0d3d554b(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          bool v47_g = v29_lead < 6;
          if (v47_g) {
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 12; ++v48_i1) {
              float v53_data = __builtin_nontemporal_load(&glb_m2[(v29_lead + (v48_i1 * 6))]);
              r3[v48_i1] = v53_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v56_data = r1[0];
          float v57_data = r1[1];
          float v58_data = r1[2];
          float v59_data = r1[3];
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v56_data, v57_data, v58_data, v59_data);
          tensorforge::VectorT<float, 4> v64_acc{};
          float v65_data = r0[0];
          float v66_data = r0[1];
          float v67_data = r0[2];
          float v68_data = r0[3];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 2, 0, 0);
          float v73_data = r0[4];
          float v74_data = r0[5];
          float v75_data = r0[6];
          float v76_data = r0[7];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v72_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v79_acc, 2, 1, 0);
          float v81_data = r0[8];
          float v82_data = r0[9];
          float v83_data = r0[10];
          float v84_data = r0[11];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v80_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v87_acc, 2, 2, 0);
          r2[0] = (v88_acc[0]);
          r2[1] = (v88_acc[1]);
          r2[2] = (v88_acc[2]);
          r2[3] = (v88_acc[3]);
          float v93_data = r1[4];
          float v94_data = r1[5];
          float v95_data = r1[6];
          float v96_data = r1[7];
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          float v100_tp{};
          tensorforge::transpose4x4b32(v97_tp, v98_tp, v99_tp, v100_tp, v93_data, v94_data, v95_data, v96_data);
          tensorforge::VectorT<float, 4> v101_acc{};
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v116_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v123_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v124_acc, 2, 2, 0);
          r2[4] = (v125_acc[0]);
          r2[5] = (v125_acc[1]);
          r2[6] = (v125_acc[2]);
          r2[7] = (v125_acc[3]);
          float v130_data = r1[8];
          float v131_data = r1[9];
          float v132_data = r1[10];
          float v133_data = r1[11];
          float v134_tp{};
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          tensorforge::transpose4x4b32(v134_tp, v135_tp, v136_tp, v137_tp, v130_data, v131_data, v132_data, v133_data);
          tensorforge::VectorT<float, 4> v138_acc{};
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v143_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v144_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v146_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v151_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v152_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v81_data, v154_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v82_data, v159_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v83_data, v160_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v84_data, v161_acc, 2, 2, 0);
          r2[8] = (v162_acc[0]);
          r2[9] = (v162_acc[1]);
          r2[10] = (v162_acc[2]);
          r2[11] = (v162_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v30_g) {
            #pragma unroll
            for (int32_t v167_i1 = 0; v167_i1 < 12; ++v167_i1) {
              float v169_data = r2[v167_i1];
              int32_t v173_a = v29_lead + (v167_i1 * 12);
              s0[(v173_a ^ ((v173_a >> 4) & 15))] = v169_data;
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
          // wait(r3 = load{g>r}(glb_m2););
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          if (v47_g) {
            #pragma unroll
            for (int32_t v186_i1 = 0; v186_i1 < 12; ++v186_i1) {
              float v191_data = __builtin_nontemporal_load(&glb_m4[(v29_lead + (v186_i1 * 6))]);
              r6[v186_i1] = v191_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[12]{};
          // r5 = +(r3 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v194_data = r4[0];
          float v195_data = r4[1];
          float v196_data = r4[2];
          float v197_data = r4[3];
          float v198_tp{};
          float v199_tp{};
          float v200_tp{};
          float v201_tp{};
          tensorforge::transpose4x4b32(v198_tp, v199_tp, v200_tp, v201_tp, v194_data, v195_data, v196_data, v197_data);
          tensorforge::VectorT<float, 4> v202_acc{};
          float v203_data = r3[0];
          float v204_data = r3[1];
          float v205_data = r3[2];
          float v206_data = r3[3];
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v203_data, v202_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v204_data, v207_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v205_data, v208_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v206_data, v209_acc, 2, 0, 0);
          float v211_data = r3[4];
          float v212_data = r3[5];
          float v213_data = r3[6];
          float v214_data = r3[7];
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v211_data, v210_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v212_data, v215_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v213_data, v216_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v214_data, v217_acc, 2, 1, 0);
          float v219_data = r3[8];
          float v220_data = r3[9];
          float v221_data = r3[10];
          float v222_data = r3[11];
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v219_data, v218_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v220_data, v223_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v221_data, v224_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v222_data, v225_acc, 2, 2, 0);
          r5[0] = (v226_acc[0]);
          r5[1] = (v226_acc[1]);
          r5[2] = (v226_acc[2]);
          r5[3] = (v226_acc[3]);
          float v231_data = r4[4];
          float v232_data = r4[5];
          float v233_data = r4[6];
          float v234_data = r4[7];
          float v235_tp{};
          float v236_tp{};
          float v237_tp{};
          float v238_tp{};
          tensorforge::transpose4x4b32(v235_tp, v236_tp, v237_tp, v238_tp, v231_data, v232_data, v233_data, v234_data);
          tensorforge::VectorT<float, 4> v239_acc{};
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v203_data, v239_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v204_data, v244_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v205_data, v245_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v206_data, v246_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v211_data, v247_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v212_data, v252_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v213_data, v253_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v214_data, v254_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v219_data, v255_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v220_data, v260_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v221_data, v261_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v222_data, v262_acc, 2, 2, 0);
          r5[4] = (v263_acc[0]);
          r5[5] = (v263_acc[1]);
          r5[6] = (v263_acc[2]);
          r5[7] = (v263_acc[3]);
          float v268_data = r4[8];
          float v269_data = r4[9];
          float v270_data = r4[10];
          float v271_data = r4[11];
          float v272_tp{};
          float v273_tp{};
          float v274_tp{};
          float v275_tp{};
          tensorforge::transpose4x4b32(v272_tp, v273_tp, v274_tp, v275_tp, v268_data, v269_data, v270_data, v271_data);
          tensorforge::VectorT<float, 4> v276_acc{};
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v203_data, v276_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v204_data, v281_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v205_data, v282_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v206_data, v283_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v211_data, v284_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v212_data, v289_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v213_data, v290_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v214_data, v291_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v219_data, v292_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v220_data, v297_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v221_data, v298_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v222_data, v299_acc, 2, 2, 0);
          r5[8] = (v300_acc[0]);
          r5[9] = (v300_acc[1]);
          r5[10] = (v300_acc[2]);
          r5[11] = (v300_acc[3]);
          // s1 = store{r>s}(localShrMem0, r5);
          if (v47_g) {
            #pragma unroll
            for (int32_t v305_i1 = 0; v305_i1 < 12; ++v305_i1) {
              float v307_data = r5[v305_i1];
              int32_t v311_a = v29_lead + (v305_i1 * 12);
              s1[(v311_a ^ ((v311_a >> 4) & 15))] = v307_data;
            }
          }
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v30_g) {
            #pragma unroll
            for (int32_t v316_i1 = 0; v316_i1 < 12; ++v316_i1) {
              float v321_data = __builtin_nontemporal_load(&glb_m6[(v29_lead + (v316_i1 * 12))]);
              r8[v316_i1] = v321_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[12]{};
          // r7 = +(r6 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v328_tp{};
          float v329_tp{};
          float v330_tp{};
          float v331_tp{};
          tensorforge::transpose4x4b32(v328_tp, v329_tp, v330_tp, v331_tp, v194_data, v195_data, v196_data, v197_data);
          tensorforge::VectorT<float, 4> v332_acc{};
          float v333_data = r6[0];
          float v334_data = r6[1];
          float v335_data = r6[2];
          float v336_data = r6[3];
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v333_data, v332_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v334_data, v337_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v335_data, v338_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v336_data, v339_acc, 2, 0, 0);
          float v341_data = r6[4];
          float v342_data = r6[5];
          float v343_data = r6[6];
          float v344_data = r6[7];
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v341_data, v340_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v342_data, v345_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v343_data, v346_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v344_data, v347_acc, 2, 1, 0);
          float v349_data = r6[8];
          float v350_data = r6[9];
          float v351_data = r6[10];
          float v352_data = r6[11];
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v349_data, v348_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v350_data, v353_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v351_data, v354_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v352_data, v355_acc, 2, 2, 0);
          r7[0] = (v356_acc[0]);
          r7[1] = (v356_acc[1]);
          r7[2] = (v356_acc[2]);
          r7[3] = (v356_acc[3]);
          float v365_tp{};
          float v366_tp{};
          float v367_tp{};
          float v368_tp{};
          tensorforge::transpose4x4b32(v365_tp, v366_tp, v367_tp, v368_tp, v231_data, v232_data, v233_data, v234_data);
          tensorforge::VectorT<float, 4> v369_acc{};
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v333_data, v369_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v334_data, v374_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v335_data, v375_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v336_data, v376_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v341_data, v377_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v342_data, v382_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v343_data, v383_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v344_data, v384_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v365_tp, v349_data, v385_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v366_tp, v350_data, v390_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v367_tp, v351_data, v391_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v352_data, v392_acc, 2, 2, 0);
          r7[4] = (v393_acc[0]);
          r7[5] = (v393_acc[1]);
          r7[6] = (v393_acc[2]);
          r7[7] = (v393_acc[3]);
          float v402_tp{};
          float v403_tp{};
          float v404_tp{};
          float v405_tp{};
          tensorforge::transpose4x4b32(v402_tp, v403_tp, v404_tp, v405_tp, v268_data, v269_data, v270_data, v271_data);
          tensorforge::VectorT<float, 4> v406_acc{};
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v333_data, v406_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v334_data, v411_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v335_data, v412_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v336_data, v413_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v341_data, v414_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v342_data, v419_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v343_data, v420_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v344_data, v421_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v349_data, v422_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v350_data, v427_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v351_data, v428_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v352_data, v429_acc, 2, 2, 0);
          r7[8] = (v430_acc[0]);
          r7[9] = (v430_acc[1]);
          r7[10] = (v430_acc[2]);
          r7[11] = (v430_acc[3]);
          // s1 = store{r>s}(localShrMem0, r7);
          if (v47_g) {
            int32_t v440_off = v29_lead + 6;
            #pragma unroll
            for (int32_t v435_i1 = 0; v435_i1 < 12; ++v435_i1) {
              float v437_data = r7[v435_i1];
              int32_t v442_a = v440_off + (v435_i1 * 12);
              s1[(v442_a ^ ((v442_a >> 4) & 15))] = v437_data;
            }
          }
          float r11[12]{};
          // r11 = load{g>r}(glb_m8);
          if (v30_g) {
            #pragma unroll
            for (int32_t v447_i1 = 0; v447_i1 < 12; ++v447_i1) {
              float v452_data = __builtin_nontemporal_load(&glb_m8[(v29_lead + (v447_i1 * 12))]);
              r11[v447_i1] = v452_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(s1 * r8) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v455_data = r8[0];
          float v456_data = r8[1];
          float v457_data = r8[2];
          float v458_data = r8[3];
          float v459_tp{};
          float v460_tp{};
          float v461_tp{};
          float v462_tp{};
          tensorforge::transpose4x4b32(v459_tp, v460_tp, v461_tp, v462_tp, v455_data, v456_data, v457_data, v458_data);
          tensorforge::VectorT<float, 4> v463_acc{};
          int32_t v468_sw = (v29_lead >> 4) & 15;
          int32_t v469_sw = v29_lead ^ v468_sw;
          float v470_data = s1[v469_sw];
          int32_t v471_a = v29_lead + 12;
          int32_t v472_sw = v471_a >> 4;
          float v475_data = s1[(v471_a ^ (v472_sw & 15))];
          int32_t v476_a = v29_lead + 24;
          int32_t v477_sw = v476_a >> 4;
          float v480_data = s1[(v476_a ^ (v477_sw & 15))];
          int32_t v481_a = v29_lead + 36;
          int32_t v482_sw = v481_a >> 4;
          float v485_data = s1[(v481_a ^ (v482_sw & 15))];
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v470_data, v463_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v475_data, v486_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v480_data, v487_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v485_data, v488_acc, 2, 0, 0);
          int32_t v490_a = v29_lead + 48;
          int32_t v491_sw = v490_a >> 4;
          float v494_data = s1[(v490_a ^ (v491_sw & 15))];
          int32_t v495_a = v29_lead + 60;
          int32_t v496_sw = v495_a >> 4;
          float v499_data = s1[(v495_a ^ (v496_sw & 15))];
          int32_t v500_a = v29_lead + 72;
          int32_t v501_sw = v500_a >> 4;
          float v504_data = s1[(v500_a ^ (v501_sw & 15))];
          int32_t v505_a = v29_lead + 84;
          int32_t v506_sw = v505_a >> 4;
          float v509_data = s1[(v505_a ^ (v506_sw & 15))];
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v494_data, v489_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v499_data, v510_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v504_data, v511_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v509_data, v512_acc, 2, 1, 0);
          int32_t v514_a = v29_lead + 96;
          int32_t v515_sw = v514_a >> 4;
          float v518_data = s1[(v514_a ^ (v515_sw & 15))];
          int32_t v519_a = v29_lead + 108;
          int32_t v520_sw = v519_a >> 4;
          float v523_data = s1[(v519_a ^ (v520_sw & 15))];
          int32_t v524_a = v29_lead + 120;
          int32_t v525_sw = v524_a >> 4;
          float v528_data = s1[(v524_a ^ (v525_sw & 15))];
          int32_t v529_a = v29_lead + 132;
          int32_t v530_sw = v529_a >> 4;
          float v533_data = s1[(v529_a ^ (v530_sw & 15))];
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v518_data, v513_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v523_data, v534_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v528_data, v535_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v533_data, v536_acc, 2, 2, 0);
          r9[0] = (v537_acc[0]);
          r9[1] = (v537_acc[1]);
          r9[2] = (v537_acc[2]);
          r9[3] = (v537_acc[3]);
          float v542_data = r8[4];
          float v543_data = r8[5];
          float v544_data = r8[6];
          float v545_data = r8[7];
          float v546_tp{};
          float v547_tp{};
          float v548_tp{};
          float v549_tp{};
          tensorforge::transpose4x4b32(v546_tp, v547_tp, v548_tp, v549_tp, v542_data, v543_data, v544_data, v545_data);
          tensorforge::VectorT<float, 4> v550_acc{};
          float v557_data = s1[(v29_lead ^ v468_sw)];
          float v562_data = s1[(v471_a ^ (v472_sw & 15))];
          float v567_data = s1[(v476_a ^ (v477_sw & 15))];
          float v572_data = s1[(v481_a ^ (v482_sw & 15))];
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v557_data, v550_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v562_data, v573_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v548_tp, v567_data, v574_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v572_data, v575_acc, 2, 0, 0);
          float v581_data = s1[(v490_a ^ (v491_sw & 15))];
          float v586_data = s1[(v495_a ^ (v496_sw & 15))];
          float v591_data = s1[(v500_a ^ (v501_sw & 15))];
          float v596_data = s1[(v505_a ^ (v506_sw & 15))];
          tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v581_data, v576_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v586_data, v597_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v599_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v548_tp, v591_data, v598_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v596_data, v599_acc, 2, 1, 0);
          float v605_data = s1[(v514_a ^ (v515_sw & 15))];
          float v610_data = s1[(v519_a ^ (v520_sw & 15))];
          float v615_data = s1[(v524_a ^ (v525_sw & 15))];
          float v620_data = s1[(v529_a ^ (v530_sw & 15))];
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v605_data, v600_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v610_data, v621_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v548_tp, v615_data, v622_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v620_data, v623_acc, 2, 2, 0);
          r9[4] = (v624_acc[0]);
          r9[5] = (v624_acc[1]);
          r9[6] = (v624_acc[2]);
          r9[7] = (v624_acc[3]);
          float v629_data = r8[8];
          float v630_data = r8[9];
          float v631_data = r8[10];
          float v632_data = r8[11];
          float v633_tp{};
          float v634_tp{};
          float v635_tp{};
          float v636_tp{};
          tensorforge::transpose4x4b32(v633_tp, v634_tp, v635_tp, v636_tp, v629_data, v630_data, v631_data, v632_data);
          tensorforge::VectorT<float, 4> v637_acc{};
          float v644_data = s1[(v29_lead ^ v468_sw)];
          float v649_data = s1[(v471_a ^ (v472_sw & 15))];
          float v654_data = s1[(v476_a ^ (v477_sw & 15))];
          float v659_data = s1[(v481_a ^ (v482_sw & 15))];
          tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v644_data, v637_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v649_data, v660_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v635_tp, v654_data, v661_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v659_data, v662_acc, 2, 0, 0);
          float v668_data = s1[(v490_a ^ (v491_sw & 15))];
          float v673_data = s1[(v495_a ^ (v496_sw & 15))];
          float v678_data = s1[(v500_a ^ (v501_sw & 15))];
          float v683_data = s1[(v505_a ^ (v506_sw & 15))];
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v668_data, v663_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v673_data, v684_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v686_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v635_tp, v678_data, v685_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v683_data, v686_acc, 2, 1, 0);
          float v692_data = s1[(v514_a ^ (v515_sw & 15))];
          float v697_data = s1[(v519_a ^ (v520_sw & 15))];
          float v702_data = s1[(v524_a ^ (v525_sw & 15))];
          float v707_data = s1[(v529_a ^ (v530_sw & 15))];
          tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v692_data, v687_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v634_tp, v697_data, v708_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v635_tp, v702_data, v709_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v707_data, v710_acc, 2, 2, 0);
          r9[8] = (v711_acc[0]);
          r9[9] = (v711_acc[1]);
          r9[10] = (v711_acc[2]);
          r9[11] = (v711_acc[3]);
          float r10[12]{};
          // r10 = abs(glb_m7)
          bool v717_g = v29_lead < 4;
          if (v717_g) {
            #pragma unroll
            for (int32_t v718_k1 = 0; v718_k1 < 12; ++v718_k1) {
              float v723_data = glb_m7[(v29_lead + (v718_k1 * 4))];
              r10[v718_k1] = (fabsf(v723_data));
            }
          }
          // s0 = store{r>s}(localShrMem0, r10);
          if (v717_g) {
            #pragma unroll
            for (int32_t v727_i1 = 0; v727_i1 < 12; ++v727_i1) {
              float v729_data = r10[v727_i1];
              int32_t v733_a = v29_lead + (v727_i1 * 12);
              s0[(v733_a ^ ((v733_a >> 4) & 15))] = v729_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m8););
          float r12[12]{};
          // ir12 = +(r11 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir12[12]{};
          int32_t v744_sw = v29_lead ^ v468_sw;
          float v745_data = s0[v744_sw];
          float v750_data = s0[(v471_a ^ (v472_sw & 15))];
          float v755_data = s0[(v476_a ^ (v477_sw & 15))];
          float v760_data = s0[(v481_a ^ (v482_sw & 15))];
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
          float v800_data = s0[(v490_a ^ (v491_sw & 15))];
          float v805_data = s0[(v495_a ^ (v496_sw & 15))];
          float v810_data = s0[(v500_a ^ (v501_sw & 15))];
          float v815_data = s0[(v505_a ^ (v506_sw & 15))];
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
          float v855_data = s0[(v514_a ^ (v515_sw & 15))];
          float v860_data = s0[(v519_a ^ (v520_sw & 15))];
          float v865_data = s0[(v524_a ^ (v525_sw & 15))];
          float v870_data = s0[(v529_a ^ (v530_sw & 15))];
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

