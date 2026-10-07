// === base name ===
kernel_edf5f02322add436

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_edf5f02322add436 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_edf5f02322add436(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_edf5f02322add436(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_edf5f02322add436(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_edf5f02322add436, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_edf5f02322add436, block.x * block.y * block.z, 0));
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
void launcher_kernel_edf5f02322add436(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_edf5f02322add436(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_edf5f02322add436), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_edf5f02322add436, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_edf5f02322add436(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 48 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v8_batchId0 * 144 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v8_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v8_batchId0 * 144 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 16;
          bool v27_g = v26_lead < 12;
          if (v27_g) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m0[(v26_lead + (v28_i1 * 12))]);
              r0[v28_i1] = v33_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v27_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v26_lead + (v36_i1 * 12))]);
              r1[v36_i1] = v41_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v166_i1 = 0; v166_i1 < 12; ++v166_i1) {
              float v171_data = __builtin_nontemporal_load(&glb_m2[(v26_lead + (v166_i1 * 12))]);
              r3[v166_i1] = v171_data;
            }
          }
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v44_data = r1[0];
          float v45_data = r1[1];
          float v46_data = r1[2];
          float v47_data = r1[3];
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          tensorforge::transpose4x4b32(v48_tp, v49_tp, v50_tp, v51_tp, v44_data, v45_data, v46_data, v47_data);
          tensorforge::VectorT<float, 4> v52_acc{};
          float v53_data = r0[0];
          float v54_data = r0[1];
          float v55_data = r0[2];
          float v56_data = r0[3];
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v52_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 2, 0, 0);
          float v61_data = r0[4];
          float v62_data = r0[5];
          float v63_data = r0[6];
          float v64_data = r0[7];
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v60_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 2, 1, 0);
          float v69_data = r0[8];
          float v70_data = r0[9];
          float v71_data = r0[10];
          float v72_data = r0[11];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v68_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 2, 2, 0);
          r2[0] = (v76_acc[0]);
          r2[1] = (v76_acc[1]);
          r2[2] = (v76_acc[2]);
          r2[3] = (v76_acc[3]);
          float v81_data = r1[4];
          float v82_data = r1[5];
          float v83_data = r1[6];
          float v84_data = r1[7];
          float v85_tp{};
          float v86_tp{};
          float v87_tp{};
          float v88_tp{};
          tensorforge::transpose4x4b32(v85_tp, v86_tp, v87_tp, v88_tp, v81_data, v82_data, v83_data, v84_data);
          tensorforge::VectorT<float, 4> v89_acc{};
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v89_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v94_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v95_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v102_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v103_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v105_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v110_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v111_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v112_acc, 2, 2, 0);
          r2[4] = (v113_acc[0]);
          r2[5] = (v113_acc[1]);
          r2[6] = (v113_acc[2]);
          r2[7] = (v113_acc[3]);
          float v118_data = r1[8];
          float v119_data = r1[9];
          float v120_data = r1[10];
          float v121_data = r1[11];
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          tensorforge::transpose4x4b32(v122_tp, v123_tp, v124_tp, v125_tp, v118_data, v119_data, v120_data, v121_data);
          tensorforge::VectorT<float, 4> v126_acc{};
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v53_data, v126_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v54_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v55_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v56_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v61_data, v134_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v62_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v63_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v64_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v69_data, v142_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v70_data, v147_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v71_data, v148_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v72_data, v149_acc, 2, 2, 0);
          r2[8] = (v150_acc[0]);
          r2[9] = (v150_acc[1]);
          r2[10] = (v150_acc[2]);
          r2[11] = (v150_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v155_i1 = 0; v155_i1 < 12; ++v155_i1) {
              float v157_data = r2[v155_i1];
              int32_t v161_a = v26_lead + (v155_i1 * 12);
              s0[(v161_a ^ ((v161_a >> 4) & 15))] = v157_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          bool v436_g = v26_lead < 4;
          if (v436_g) {
            #pragma unroll
            for (int32_t v437_i1 = 0; v437_i1 < 12; ++v437_i1) {
              float v442_data = __builtin_nontemporal_load(&glb_m3[(v26_lead + (v437_i1 * 4))]);
              r5[v437_i1] = v442_data;
            }
          }
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v174_data = r3[0];
          float v175_data = r3[1];
          float v176_data = r3[2];
          float v177_data = r3[3];
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          float v181_tp{};
          tensorforge::transpose4x4b32(v178_tp, v179_tp, v180_tp, v181_tp, v174_data, v175_data, v176_data, v177_data);
          tensorforge::VectorT<float, 4> v182_acc{};
          int32_t v187_sw = (v26_lead >> 4) & 15;
          int32_t v188_sw = v26_lead ^ v187_sw;
          float v189_data = s0[v188_sw];
          int32_t v190_a = v26_lead + 12;
          int32_t v191_sw = v190_a >> 4;
          float v194_data = s0[(v190_a ^ (v191_sw & 15))];
          int32_t v195_a = v26_lead + 24;
          int32_t v196_sw = v195_a >> 4;
          float v199_data = s0[(v195_a ^ (v196_sw & 15))];
          int32_t v200_a = v26_lead + 36;
          int32_t v201_sw = v200_a >> 4;
          float v204_data = s0[(v200_a ^ (v201_sw & 15))];
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v189_data, v182_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v194_data, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v199_data, v206_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v204_data, v207_acc, 2, 0, 0);
          int32_t v209_a = v26_lead + 48;
          int32_t v210_sw = v209_a >> 4;
          float v213_data = s0[(v209_a ^ (v210_sw & 15))];
          int32_t v214_a = v26_lead + 60;
          int32_t v215_sw = v214_a >> 4;
          float v218_data = s0[(v214_a ^ (v215_sw & 15))];
          int32_t v219_a = v26_lead + 72;
          int32_t v220_sw = v219_a >> 4;
          float v223_data = s0[(v219_a ^ (v220_sw & 15))];
          int32_t v224_a = v26_lead + 84;
          int32_t v225_sw = v224_a >> 4;
          float v228_data = s0[(v224_a ^ (v225_sw & 15))];
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v213_data, v208_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v218_data, v229_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v223_data, v230_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v228_data, v231_acc, 2, 1, 0);
          int32_t v233_a = v26_lead + 96;
          int32_t v234_sw = v233_a >> 4;
          float v237_data = s0[(v233_a ^ (v234_sw & 15))];
          int32_t v238_a = v26_lead + 108;
          int32_t v239_sw = v238_a >> 4;
          float v242_data = s0[(v238_a ^ (v239_sw & 15))];
          int32_t v243_a = v26_lead + 120;
          int32_t v244_sw = v243_a >> 4;
          float v247_data = s0[(v243_a ^ (v244_sw & 15))];
          int32_t v248_a = v26_lead + 132;
          int32_t v249_sw = v248_a >> 4;
          float v252_data = s0[(v248_a ^ (v249_sw & 15))];
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v237_data, v232_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v242_data, v253_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v247_data, v254_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v252_data, v255_acc, 2, 2, 0);
          r4[0] = (v256_acc[0]);
          r4[1] = (v256_acc[1]);
          r4[2] = (v256_acc[2]);
          r4[3] = (v256_acc[3]);
          float v261_data = r3[4];
          float v262_data = r3[5];
          float v263_data = r3[6];
          float v264_data = r3[7];
          float v265_tp{};
          float v266_tp{};
          float v267_tp{};
          float v268_tp{};
          tensorforge::transpose4x4b32(v265_tp, v266_tp, v267_tp, v268_tp, v261_data, v262_data, v263_data, v264_data);
          tensorforge::VectorT<float, 4> v269_acc{};
          float v276_data = s0[(v26_lead ^ v187_sw)];
          float v281_data = s0[(v190_a ^ (v191_sw & 15))];
          float v286_data = s0[(v195_a ^ (v196_sw & 15))];
          float v291_data = s0[(v200_a ^ (v201_sw & 15))];
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v276_data, v269_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v281_data, v292_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v286_data, v293_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v291_data, v294_acc, 2, 0, 0);
          float v300_data = s0[(v209_a ^ (v210_sw & 15))];
          float v305_data = s0[(v214_a ^ (v215_sw & 15))];
          float v310_data = s0[(v219_a ^ (v220_sw & 15))];
          float v315_data = s0[(v224_a ^ (v225_sw & 15))];
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v300_data, v295_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v305_data, v316_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v310_data, v317_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v315_data, v318_acc, 2, 1, 0);
          float v324_data = s0[(v233_a ^ (v234_sw & 15))];
          float v329_data = s0[(v238_a ^ (v239_sw & 15))];
          float v334_data = s0[(v243_a ^ (v244_sw & 15))];
          float v339_data = s0[(v248_a ^ (v249_sw & 15))];
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v324_data, v319_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v329_data, v340_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v334_data, v341_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v339_data, v342_acc, 2, 2, 0);
          r4[4] = (v343_acc[0]);
          r4[5] = (v343_acc[1]);
          r4[6] = (v343_acc[2]);
          r4[7] = (v343_acc[3]);
          float v348_data = r3[8];
          float v349_data = r3[9];
          float v350_data = r3[10];
          float v351_data = r3[11];
          float v352_tp{};
          float v353_tp{};
          float v354_tp{};
          float v355_tp{};
          tensorforge::transpose4x4b32(v352_tp, v353_tp, v354_tp, v355_tp, v348_data, v349_data, v350_data, v351_data);
          tensorforge::VectorT<float, 4> v356_acc{};
          float v363_data = s0[(v26_lead ^ v187_sw)];
          float v368_data = s0[(v190_a ^ (v191_sw & 15))];
          float v373_data = s0[(v195_a ^ (v196_sw & 15))];
          float v378_data = s0[(v200_a ^ (v201_sw & 15))];
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v363_data, v356_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v368_data, v379_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v373_data, v380_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v378_data, v381_acc, 2, 0, 0);
          float v387_data = s0[(v209_a ^ (v210_sw & 15))];
          float v392_data = s0[(v214_a ^ (v215_sw & 15))];
          float v397_data = s0[(v219_a ^ (v220_sw & 15))];
          float v402_data = s0[(v224_a ^ (v225_sw & 15))];
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v387_data, v382_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v392_data, v403_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v397_data, v404_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v402_data, v405_acc, 2, 1, 0);
          float v411_data = s0[(v233_a ^ (v234_sw & 15))];
          float v416_data = s0[(v238_a ^ (v239_sw & 15))];
          float v421_data = s0[(v243_a ^ (v244_sw & 15))];
          float v426_data = s0[(v248_a ^ (v249_sw & 15))];
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v411_data, v406_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v416_data, v427_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v421_data, v428_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v426_data, v429_acc, 2, 2, 0);
          r4[8] = (v430_acc[0]);
          r4[9] = (v430_acc[1]);
          r4[10] = (v430_acc[2]);
          r4[11] = (v430_acc[3]);
          float r7[12]{};
          // r7 = load{g>r}(glb_m4);
          if (v27_g) {
            #pragma unroll
            for (int32_t v578_i1 = 0; v578_i1 < 12; ++v578_i1) {
              float v583_data = __builtin_nontemporal_load(&glb_m4[(v26_lead + (v578_i1 * 12))]);
              r7[v578_i1] = v583_data;
            }
          }
          float r6[12]{};
          // r6 = +(r5 * r1) + None
          // [(0, 4), (0, 12)] [(0, 12)]
          float v449_tp{};
          float v450_tp{};
          float v451_tp{};
          float v452_tp{};
          tensorforge::transpose4x4b32(v449_tp, v450_tp, v451_tp, v452_tp, v44_data, v45_data, v46_data, v47_data);
          tensorforge::VectorT<float, 4> v453_acc{};
          float v454_data = r5[0];
          float v455_data = r5[1];
          float v456_data = r5[2];
          float v457_data = r5[3];
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v449_tp, v454_data, v453_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v450_tp, v455_data, v458_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v456_data, v459_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v457_data, v460_acc, 2, 0, 0);
          float v462_data = r5[4];
          float v463_data = r5[5];
          float v464_data = r5[6];
          float v465_data = r5[7];
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v449_tp, v462_data, v461_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v450_tp, v463_data, v466_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v464_data, v467_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v465_data, v468_acc, 2, 1, 0);
          float v470_data = r5[8];
          float v471_data = r5[9];
          float v472_data = r5[10];
          float v473_data = r5[11];
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v449_tp, v470_data, v469_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v450_tp, v471_data, v474_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v451_tp, v472_data, v475_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v473_data, v476_acc, 2, 2, 0);
          r6[0] = (v477_acc[0]);
          r6[1] = (v477_acc[1]);
          r6[2] = (v477_acc[2]);
          r6[3] = (v477_acc[3]);
          float v486_tp{};
          float v487_tp{};
          float v488_tp{};
          float v489_tp{};
          tensorforge::transpose4x4b32(v486_tp, v487_tp, v488_tp, v489_tp, v81_data, v82_data, v83_data, v84_data);
          tensorforge::VectorT<float, 4> v490_acc{};
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v454_data, v490_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v455_data, v495_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v456_data, v496_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v457_data, v497_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v462_data, v498_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v463_data, v503_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v464_data, v504_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v465_data, v505_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v470_data, v506_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v471_data, v511_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v472_data, v512_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v489_tp, v473_data, v513_acc, 2, 2, 0);
          r6[4] = (v514_acc[0]);
          r6[5] = (v514_acc[1]);
          r6[6] = (v514_acc[2]);
          r6[7] = (v514_acc[3]);
          float v523_tp{};
          float v524_tp{};
          float v525_tp{};
          float v526_tp{};
          tensorforge::transpose4x4b32(v523_tp, v524_tp, v525_tp, v526_tp, v118_data, v119_data, v120_data, v121_data);
          tensorforge::VectorT<float, 4> v527_acc{};
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v523_tp, v454_data, v527_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v524_tp, v455_data, v532_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v525_tp, v456_data, v533_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v526_tp, v457_data, v534_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v523_tp, v462_data, v535_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v524_tp, v463_data, v540_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v525_tp, v464_data, v541_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v526_tp, v465_data, v542_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v523_tp, v470_data, v543_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v524_tp, v471_data, v548_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v525_tp, v472_data, v549_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v526_tp, v473_data, v550_acc, 2, 2, 0);
          r6[8] = (v551_acc[0]);
          r6[9] = (v551_acc[1]);
          r6[10] = (v551_acc[2]);
          r6[11] = (v551_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r6);
          bool v557_g = (v26_lead >= 4) && v27_g;
          if (v557_g) {
            #pragma unroll
            for (int32_t v558_z1 = 0; v558_z1 < 12; ++v558_z1) {
              int32_t v563_a = v26_lead + (v558_z1 * 12);
              s0[(v563_a ^ ((v563_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v436_g) {
            #pragma unroll
            for (int32_t v567_i1 = 0; v567_i1 < 12; ++v567_i1) {
              float v569_data = r6[v567_i1];
              int32_t v573_a = v26_lead + (v567_i1 * 12);
              s0[(v573_a ^ ((v573_a >> 4) & 15))] = v569_data;
            }
          }
          float r9[12]{};
          // r9 = load{g>r}(glb_m6);
          if (v27_g) {
            #pragma unroll
            for (int32_t v721_i1 = 0; v721_i1 < 12; ++v721_i1) {
              float v726_data = __builtin_nontemporal_load(&glb_m6[(v26_lead + (v721_i1 * 12))]);
              r9[v721_i1] = v726_data;
            }
          }
          float r8[12]{};
          // ir8 = +(r4 * r7)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir8[12]{};
          float v587_data = r7[0];
          float v588_data = r7[1];
          float v589_data = r7[2];
          float v590_data = r7[3];
          float v591_tp{};
          float v592_tp{};
          float v593_tp{};
          float v594_tp{};
          tensorforge::transpose4x4b32(v591_tp, v592_tp, v593_tp, v594_tp, v587_data, v588_data, v589_data, v590_data);
          tensorforge::VectorT<float, 4> v595_acc{};
          float v596_data = r4[0];
          float v597_data = r4[1];
          float v598_data = r4[2];
          float v599_data = r4[3];
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v591_tp, v596_data, v595_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v592_tp, v597_data, v600_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v593_tp, v598_data, v601_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v594_tp, v599_data, v602_acc, 2, 0, 0);
          float v604_data = r4[4];
          float v605_data = r4[5];
          float v606_data = r4[6];
          float v607_data = r4[7];
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v591_tp, v604_data, v603_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v592_tp, v605_data, v608_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v593_tp, v606_data, v609_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v594_tp, v607_data, v610_acc, 2, 1, 0);
          float v612_data = r4[8];
          float v613_data = r4[9];
          float v614_data = r4[10];
          float v615_data = r4[11];
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v591_tp, v612_data, v611_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v592_tp, v613_data, v616_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v593_tp, v614_data, v617_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v594_tp, v615_data, v618_acc, 2, 2, 0);
          ir8[0] = (v619_acc[0]);
          ir8[1] = (v619_acc[1]);
          ir8[2] = (v619_acc[2]);
          ir8[3] = (v619_acc[3]);
          float v624_data = r7[4];
          float v625_data = r7[5];
          float v626_data = r7[6];
          float v627_data = r7[7];
          float v628_tp{};
          float v629_tp{};
          float v630_tp{};
          float v631_tp{};
          tensorforge::transpose4x4b32(v628_tp, v629_tp, v630_tp, v631_tp, v624_data, v625_data, v626_data, v627_data);
          tensorforge::VectorT<float, 4> v632_acc{};
          tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v596_data, v632_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v629_tp, v597_data, v637_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v598_data, v638_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v599_data, v639_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v604_data, v640_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v629_tp, v605_data, v645_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v606_data, v646_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v607_data, v647_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v628_tp, v612_data, v648_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v629_tp, v613_data, v653_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v614_data, v654_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v615_data, v655_acc, 2, 2, 0);
          ir8[4] = (v656_acc[0]);
          ir8[5] = (v656_acc[1]);
          ir8[6] = (v656_acc[2]);
          ir8[7] = (v656_acc[3]);
          float v661_data = r7[8];
          float v662_data = r7[9];
          float v663_data = r7[10];
          float v664_data = r7[11];
          float v665_tp{};
          float v666_tp{};
          float v667_tp{};
          float v668_tp{};
          tensorforge::transpose4x4b32(v665_tp, v666_tp, v667_tp, v668_tp, v661_data, v662_data, v663_data, v664_data);
          tensorforge::VectorT<float, 4> v669_acc{};
          tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v596_data, v669_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v597_data, v674_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v598_data, v675_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v599_data, v676_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v604_data, v677_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v605_data, v682_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v606_data, v683_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v607_data, v684_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v612_data, v685_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v613_data, v690_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v614_data, v691_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v615_data, v692_acc, 2, 2, 0);
          ir8[8] = (v693_acc[0]);
          ir8[9] = (v693_acc[1]);
          ir8[10] = (v693_acc[2]);
          ir8[11] = (v693_acc[3]);
          // r8 = ir8 + s0
          if (v27_g) {
            #pragma unroll
            for (int32_t v698_n1 = 0; v698_n1 < 12; ++v698_n1) {
              float v700_data = ir8[v698_n1];
              int32_t v704_a = v26_lead + (v698_n1 * 12);
              float v708_data = s0[(v704_a ^ ((v704_a >> 4) & 15))];
              r8[v698_n1] = (v708_data + v700_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r8);
          if (v27_g) {
            #pragma unroll
            for (int32_t v710_i1 = 0; v710_i1 < 12; ++v710_i1) {
              float v712_data = r8[v710_i1];
              int32_t v716_a = v26_lead + (v710_i1 * 12);
              s0[(v716_a ^ ((v716_a >> 4) & 15))] = v712_data;
            }
          }
          float r10[12]{};
          // r10 = +(r9 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          int32_t v734_sw = v26_lead ^ v187_sw;
          float v735_data = s0[v734_sw];
          float v740_data = s0[(v190_a ^ (v191_sw & 15))];
          float v745_data = s0[(v195_a ^ (v196_sw & 15))];
          float v750_data = s0[(v200_a ^ (v201_sw & 15))];
          float v751_tp{};
          float v752_tp{};
          float v753_tp{};
          float v754_tp{};
          tensorforge::transpose4x4b32(v751_tp, v752_tp, v753_tp, v754_tp, v735_data, v740_data, v745_data, v750_data);
          tensorforge::VectorT<float, 4> v755_acc{};
          float v756_data = r9[0];
          float v757_data = r9[1];
          float v758_data = r9[2];
          float v759_data = r9[3];
          tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v751_tp, v756_data, v755_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v761_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v752_tp, v757_data, v760_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v762_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v753_tp, v758_data, v761_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v763_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v754_tp, v759_data, v762_acc, 2, 0, 0);
          float v764_data = r9[4];
          float v765_data = r9[5];
          float v766_data = r9[6];
          float v767_data = r9[7];
          tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v751_tp, v764_data, v763_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v769_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v752_tp, v765_data, v768_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v753_tp, v766_data, v769_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v771_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v754_tp, v767_data, v770_acc, 2, 1, 0);
          float v772_data = r9[8];
          float v773_data = r9[9];
          float v774_data = r9[10];
          float v775_data = r9[11];
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v751_tp, v772_data, v771_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v777_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v752_tp, v773_data, v776_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v753_tp, v774_data, v777_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v779_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v754_tp, v775_data, v778_acc, 2, 2, 0);
          r10[0] = (v779_acc[0]);
          r10[1] = (v779_acc[1]);
          r10[2] = (v779_acc[2]);
          r10[3] = (v779_acc[3]);
          float v790_data = s0[(v209_a ^ (v210_sw & 15))];
          float v795_data = s0[(v214_a ^ (v215_sw & 15))];
          float v800_data = s0[(v219_a ^ (v220_sw & 15))];
          float v805_data = s0[(v224_a ^ (v225_sw & 15))];
          float v806_tp{};
          float v807_tp{};
          float v808_tp{};
          float v809_tp{};
          tensorforge::transpose4x4b32(v806_tp, v807_tp, v808_tp, v809_tp, v790_data, v795_data, v800_data, v805_data);
          tensorforge::VectorT<float, 4> v810_acc{};
          tensorforge::VectorT<float, 4> v815_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v756_data, v810_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v816_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v807_tp, v757_data, v815_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v817_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v808_tp, v758_data, v816_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v759_data, v817_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v823_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v764_data, v818_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v824_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v807_tp, v765_data, v823_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v825_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v808_tp, v766_data, v824_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v767_data, v825_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v772_data, v826_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v832_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v807_tp, v773_data, v831_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v833_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v808_tp, v774_data, v832_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v834_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v775_data, v833_acc, 2, 2, 0);
          r10[4] = (v834_acc[0]);
          r10[5] = (v834_acc[1]);
          r10[6] = (v834_acc[2]);
          r10[7] = (v834_acc[3]);
          float v845_data = s0[(v233_a ^ (v234_sw & 15))];
          float v850_data = s0[(v238_a ^ (v239_sw & 15))];
          float v855_data = s0[(v243_a ^ (v244_sw & 15))];
          float v860_data = s0[(v248_a ^ (v249_sw & 15))];
          float v861_tp{};
          float v862_tp{};
          float v863_tp{};
          float v864_tp{};
          tensorforge::transpose4x4b32(v861_tp, v862_tp, v863_tp, v864_tp, v845_data, v850_data, v855_data, v860_data);
          tensorforge::VectorT<float, 4> v865_acc{};
          tensorforge::VectorT<float, 4> v870_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v861_tp, v756_data, v865_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v871_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v862_tp, v757_data, v870_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v872_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v863_tp, v758_data, v871_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v873_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v864_tp, v759_data, v872_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v861_tp, v764_data, v873_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v879_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v862_tp, v765_data, v878_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v863_tp, v766_data, v879_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v881_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v864_tp, v767_data, v880_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v861_tp, v772_data, v881_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v862_tp, v773_data, v886_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v863_tp, v774_data, v887_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v889_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v864_tp, v775_data, v888_acc, 2, 2, 0);
          r10[8] = (v889_acc[0]);
          r10[9] = (v889_acc[1]);
          r10[10] = (v889_acc[2]);
          r10[11] = (v889_acc[3]);
          // glb_m5 = store{r>g}(r10);
          if (v27_g) {
            #pragma unroll
            for (int32_t v894_i1 = 0; v894_i1 < 12; ++v894_i1) {
              float v896_data = r10[v894_i1];
              glb_m5[(v26_lead + (v894_i1 * 12))] = v896_data;
            }
          }
        }
      }
    }
  }
}

