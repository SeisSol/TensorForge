// === base name ===
kernel_8a79723af69a8125

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8a79723af69a8125 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8a79723af69a8125(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8a79723af69a8125(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8a79723af69a8125(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8a79723af69a8125, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_8a79723af69a8125, block.x * block.y * block.z, 0));
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
void launcher_kernel_8a79723af69a8125(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8a79723af69a8125(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8a79723af69a8125), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_8a79723af69a8125, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8a79723af69a8125(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 12; ++v44_i1) {
              float v49_data = __builtin_nontemporal_load(&glb_m2[(v26_lead + (v44_i1 * 12))]);
              r3[v44_i1] = v49_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v52_data = r1[0];
          float v53_data = r1[1];
          float v54_data = r1[2];
          float v55_data = r1[3];
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          float v59_tp{};
          tensorforge::transpose4x4b32(v56_tp, v57_tp, v58_tp, v59_tp, v52_data, v53_data, v54_data, v55_data);
          tensorforge::VectorT<float, 4> v60_acc{};
          float v61_data = r0[0];
          float v62_data = r0[1];
          float v63_data = r0[2];
          float v64_data = r0[3];
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v60_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v67_acc, 2, 0, 0);
          float v69_data = r0[4];
          float v70_data = r0[5];
          float v71_data = r0[6];
          float v72_data = r0[7];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v68_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 1, 0);
          float v77_data = r0[8];
          float v78_data = r0[9];
          float v79_data = r0[10];
          float v80_data = r0[11];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v76_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 2, 0);
          r2[0] = (v84_acc[0]);
          r2[1] = (v84_acc[1]);
          r2[2] = (v84_acc[2]);
          r2[3] = (v84_acc[3]);
          float v89_data = r1[4];
          float v90_data = r1[5];
          float v91_data = r1[6];
          float v92_data = r1[7];
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          tensorforge::transpose4x4b32(v93_tp, v94_tp, v95_tp, v96_tp, v89_data, v90_data, v91_data, v92_data);
          tensorforge::VectorT<float, 4> v97_acc{};
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v110_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v118_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v119_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v120_acc, 2, 2, 0);
          r2[4] = (v121_acc[0]);
          r2[5] = (v121_acc[1]);
          r2[6] = (v121_acc[2]);
          r2[7] = (v121_acc[3]);
          float v126_data = r1[8];
          float v127_data = r1[9];
          float v128_data = r1[10];
          float v129_data = r1[11];
          float v130_tp{};
          float v131_tp{};
          float v132_tp{};
          float v133_tp{};
          tensorforge::transpose4x4b32(v130_tp, v131_tp, v132_tp, v133_tp, v126_data, v127_data, v128_data, v129_data);
          tensorforge::VectorT<float, 4> v134_acc{};
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v61_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v139_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v140_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v69_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v147_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v148_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v77_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v78_data, v155_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v79_data, v156_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v80_data, v157_acc, 2, 2, 0);
          r2[8] = (v158_acc[0]);
          r2[9] = (v158_acc[1]);
          r2[10] = (v158_acc[2]);
          r2[11] = (v158_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v163_i1 = 0; v163_i1 < 12; ++v163_i1) {
              float v165_data = r2[v163_i1];
              int32_t v169_a = v26_lead + (v163_i1 * 12);
              s0[(v169_a ^ ((v169_a >> 4) & 15))] = v165_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          bool v174_g = v26_lead < 4;
          if (v174_g) {
            #pragma unroll
            for (int32_t v175_i1 = 0; v175_i1 < 12; ++v175_i1) {
              float v180_data = __builtin_nontemporal_load(&glb_m3[(v26_lead + (v175_i1 * 4))]);
              r5[v175_i1] = v180_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v183_data = r3[0];
          float v184_data = r3[1];
          float v185_data = r3[2];
          float v186_data = r3[3];
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          tensorforge::transpose4x4b32(v187_tp, v188_tp, v189_tp, v190_tp, v183_data, v184_data, v185_data, v186_data);
          tensorforge::VectorT<float, 4> v191_acc{};
          int32_t v196_sw = (v26_lead >> 4) & 15;
          int32_t v197_sw = v26_lead ^ v196_sw;
          float v198_data = s0[v197_sw];
          int32_t v199_a = v26_lead + 12;
          int32_t v200_sw = v199_a >> 4;
          float v203_data = s0[(v199_a ^ (v200_sw & 15))];
          int32_t v204_a = v26_lead + 24;
          int32_t v205_sw = v204_a >> 4;
          float v208_data = s0[(v204_a ^ (v205_sw & 15))];
          int32_t v209_a = v26_lead + 36;
          int32_t v210_sw = v209_a >> 4;
          float v213_data = s0[(v209_a ^ (v210_sw & 15))];
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v198_data, v191_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v203_data, v214_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v208_data, v215_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v213_data, v216_acc, 2, 0, 0);
          int32_t v218_a = v26_lead + 48;
          int32_t v219_sw = v218_a >> 4;
          float v222_data = s0[(v218_a ^ (v219_sw & 15))];
          int32_t v223_a = v26_lead + 60;
          int32_t v224_sw = v223_a >> 4;
          float v227_data = s0[(v223_a ^ (v224_sw & 15))];
          int32_t v228_a = v26_lead + 72;
          int32_t v229_sw = v228_a >> 4;
          float v232_data = s0[(v228_a ^ (v229_sw & 15))];
          int32_t v233_a = v26_lead + 84;
          int32_t v234_sw = v233_a >> 4;
          float v237_data = s0[(v233_a ^ (v234_sw & 15))];
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v222_data, v217_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v227_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v232_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v237_data, v240_acc, 2, 1, 0);
          int32_t v242_a = v26_lead + 96;
          int32_t v243_sw = v242_a >> 4;
          float v246_data = s0[(v242_a ^ (v243_sw & 15))];
          int32_t v247_a = v26_lead + 108;
          int32_t v248_sw = v247_a >> 4;
          float v251_data = s0[(v247_a ^ (v248_sw & 15))];
          int32_t v252_a = v26_lead + 120;
          int32_t v253_sw = v252_a >> 4;
          float v256_data = s0[(v252_a ^ (v253_sw & 15))];
          int32_t v257_a = v26_lead + 132;
          int32_t v258_sw = v257_a >> 4;
          float v261_data = s0[(v257_a ^ (v258_sw & 15))];
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v246_data, v241_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v251_data, v262_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v256_data, v263_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v261_data, v264_acc, 2, 2, 0);
          r4[0] = (v265_acc[0]);
          r4[1] = (v265_acc[1]);
          r4[2] = (v265_acc[2]);
          r4[3] = (v265_acc[3]);
          float v270_data = r3[4];
          float v271_data = r3[5];
          float v272_data = r3[6];
          float v273_data = r3[7];
          float v274_tp{};
          float v275_tp{};
          float v276_tp{};
          float v277_tp{};
          tensorforge::transpose4x4b32(v274_tp, v275_tp, v276_tp, v277_tp, v270_data, v271_data, v272_data, v273_data);
          tensorforge::VectorT<float, 4> v278_acc{};
          float v285_data = s0[(v26_lead ^ v196_sw)];
          float v290_data = s0[(v199_a ^ (v200_sw & 15))];
          float v295_data = s0[(v204_a ^ (v205_sw & 15))];
          float v300_data = s0[(v209_a ^ (v210_sw & 15))];
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v285_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v290_data, v301_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v295_data, v302_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v300_data, v303_acc, 2, 0, 0);
          float v309_data = s0[(v218_a ^ (v219_sw & 15))];
          float v314_data = s0[(v223_a ^ (v224_sw & 15))];
          float v319_data = s0[(v228_a ^ (v229_sw & 15))];
          float v324_data = s0[(v233_a ^ (v234_sw & 15))];
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v309_data, v304_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v314_data, v325_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v319_data, v326_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v324_data, v327_acc, 2, 1, 0);
          float v333_data = s0[(v242_a ^ (v243_sw & 15))];
          float v338_data = s0[(v247_a ^ (v248_sw & 15))];
          float v343_data = s0[(v252_a ^ (v253_sw & 15))];
          float v348_data = s0[(v257_a ^ (v258_sw & 15))];
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v333_data, v328_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v338_data, v349_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v343_data, v350_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v348_data, v351_acc, 2, 2, 0);
          r4[4] = (v352_acc[0]);
          r4[5] = (v352_acc[1]);
          r4[6] = (v352_acc[2]);
          r4[7] = (v352_acc[3]);
          float v357_data = r3[8];
          float v358_data = r3[9];
          float v359_data = r3[10];
          float v360_data = r3[11];
          float v361_tp{};
          float v362_tp{};
          float v363_tp{};
          float v364_tp{};
          tensorforge::transpose4x4b32(v361_tp, v362_tp, v363_tp, v364_tp, v357_data, v358_data, v359_data, v360_data);
          tensorforge::VectorT<float, 4> v365_acc{};
          float v372_data = s0[(v26_lead ^ v196_sw)];
          float v377_data = s0[(v199_a ^ (v200_sw & 15))];
          float v382_data = s0[(v204_a ^ (v205_sw & 15))];
          float v387_data = s0[(v209_a ^ (v210_sw & 15))];
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v372_data, v365_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v377_data, v388_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v382_data, v389_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v387_data, v390_acc, 2, 0, 0);
          float v396_data = s0[(v218_a ^ (v219_sw & 15))];
          float v401_data = s0[(v223_a ^ (v224_sw & 15))];
          float v406_data = s0[(v228_a ^ (v229_sw & 15))];
          float v411_data = s0[(v233_a ^ (v234_sw & 15))];
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v396_data, v391_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v401_data, v412_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v406_data, v413_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v411_data, v414_acc, 2, 1, 0);
          float v420_data = s0[(v242_a ^ (v243_sw & 15))];
          float v425_data = s0[(v247_a ^ (v248_sw & 15))];
          float v430_data = s0[(v252_a ^ (v253_sw & 15))];
          float v435_data = s0[(v257_a ^ (v258_sw & 15))];
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v420_data, v415_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v425_data, v436_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v430_data, v437_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v435_data, v438_acc, 2, 2, 0);
          r4[8] = (v439_acc[0]);
          r4[9] = (v439_acc[1]);
          r4[10] = (v439_acc[2]);
          r4[11] = (v439_acc[3]);
          float r7[12]{};
          // r7 = load{g>r}(glb_m4);
          if (v27_g) {
            #pragma unroll
            for (int32_t v445_i1 = 0; v445_i1 < 12; ++v445_i1) {
              float v450_data = __builtin_nontemporal_load(&glb_m4[(v26_lead + (v445_i1 * 12))]);
              r7[v445_i1] = v450_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = +(r5 * r1) + None
          // [(0, 4), (0, 12)] [(0, 12)]
          float v457_tp{};
          float v458_tp{};
          float v459_tp{};
          float v460_tp{};
          tensorforge::transpose4x4b32(v457_tp, v458_tp, v459_tp, v460_tp, v52_data, v53_data, v54_data, v55_data);
          tensorforge::VectorT<float, 4> v461_acc{};
          float v462_data = r5[0];
          float v463_data = r5[1];
          float v464_data = r5[2];
          float v465_data = r5[3];
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v462_data, v461_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v463_data, v466_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v464_data, v467_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v465_data, v468_acc, 2, 0, 0);
          float v470_data = r5[4];
          float v471_data = r5[5];
          float v472_data = r5[6];
          float v473_data = r5[7];
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v470_data, v469_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v471_data, v474_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v472_data, v475_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v473_data, v476_acc, 2, 1, 0);
          float v478_data = r5[8];
          float v479_data = r5[9];
          float v480_data = r5[10];
          float v481_data = r5[11];
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v478_data, v477_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v479_data, v482_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v480_data, v483_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v481_data, v484_acc, 2, 2, 0);
          r6[0] = (v485_acc[0]);
          r6[1] = (v485_acc[1]);
          r6[2] = (v485_acc[2]);
          r6[3] = (v485_acc[3]);
          float v494_tp{};
          float v495_tp{};
          float v496_tp{};
          float v497_tp{};
          tensorforge::transpose4x4b32(v494_tp, v495_tp, v496_tp, v497_tp, v89_data, v90_data, v91_data, v92_data);
          tensorforge::VectorT<float, 4> v498_acc{};
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v462_data, v498_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v463_data, v503_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v464_data, v504_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v465_data, v505_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v470_data, v506_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v471_data, v511_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v472_data, v512_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v473_data, v513_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v494_tp, v478_data, v514_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v495_tp, v479_data, v519_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v480_data, v520_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v481_data, v521_acc, 2, 2, 0);
          r6[4] = (v522_acc[0]);
          r6[5] = (v522_acc[1]);
          r6[6] = (v522_acc[2]);
          r6[7] = (v522_acc[3]);
          float v531_tp{};
          float v532_tp{};
          float v533_tp{};
          float v534_tp{};
          tensorforge::transpose4x4b32(v531_tp, v532_tp, v533_tp, v534_tp, v126_data, v127_data, v128_data, v129_data);
          tensorforge::VectorT<float, 4> v535_acc{};
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v531_tp, v462_data, v535_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v532_tp, v463_data, v540_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v533_tp, v464_data, v541_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v534_tp, v465_data, v542_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v531_tp, v470_data, v543_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v532_tp, v471_data, v548_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v533_tp, v472_data, v549_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v534_tp, v473_data, v550_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v531_tp, v478_data, v551_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v532_tp, v479_data, v556_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v533_tp, v480_data, v557_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v534_tp, v481_data, v558_acc, 2, 2, 0);
          r6[8] = (v559_acc[0]);
          r6[9] = (v559_acc[1]);
          r6[10] = (v559_acc[2]);
          r6[11] = (v559_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r6);
          bool v565_g = (v26_lead >= 4) && v27_g;
          if (v565_g) {
            #pragma unroll
            for (int32_t v566_z1 = 0; v566_z1 < 12; ++v566_z1) {
              int32_t v571_a = v26_lead + (v566_z1 * 12);
              s0[(v571_a ^ ((v571_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v174_g) {
            #pragma unroll
            for (int32_t v575_i1 = 0; v575_i1 < 12; ++v575_i1) {
              float v577_data = r6[v575_i1];
              int32_t v581_a = v26_lead + (v575_i1 * 12);
              s0[(v581_a ^ ((v581_a >> 4) & 15))] = v577_data;
            }
          }
          float r9[12]{};
          // r9 = load{g>r}(glb_m6);
          if (v27_g) {
            #pragma unroll
            for (int32_t v586_i1 = 0; v586_i1 < 12; ++v586_i1) {
              float v591_data = __builtin_nontemporal_load(&glb_m6[(v26_lead + (v586_i1 * 12))]);
              r9[v586_i1] = v591_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m4););
          float r8[12]{};
          // ir8 = +(r4 * r7)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir8[12]{};
          float v595_data = r7[0];
          float v596_data = r7[1];
          float v597_data = r7[2];
          float v598_data = r7[3];
          float v599_tp{};
          float v600_tp{};
          float v601_tp{};
          float v602_tp{};
          tensorforge::transpose4x4b32(v599_tp, v600_tp, v601_tp, v602_tp, v595_data, v596_data, v597_data, v598_data);
          tensorforge::VectorT<float, 4> v603_acc{};
          float v604_data = r4[0];
          float v605_data = r4[1];
          float v606_data = r4[2];
          float v607_data = r4[3];
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v604_data, v603_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v600_tp, v605_data, v608_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v601_tp, v606_data, v609_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v607_data, v610_acc, 2, 0, 0);
          float v612_data = r4[4];
          float v613_data = r4[5];
          float v614_data = r4[6];
          float v615_data = r4[7];
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v612_data, v611_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v600_tp, v613_data, v616_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v601_tp, v614_data, v617_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v615_data, v618_acc, 2, 1, 0);
          float v620_data = r4[8];
          float v621_data = r4[9];
          float v622_data = r4[10];
          float v623_data = r4[11];
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v599_tp, v620_data, v619_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v600_tp, v621_data, v624_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v601_tp, v622_data, v625_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v623_data, v626_acc, 2, 2, 0);
          ir8[0] = (v627_acc[0]);
          ir8[1] = (v627_acc[1]);
          ir8[2] = (v627_acc[2]);
          ir8[3] = (v627_acc[3]);
          float v632_data = r7[4];
          float v633_data = r7[5];
          float v634_data = r7[6];
          float v635_data = r7[7];
          float v636_tp{};
          float v637_tp{};
          float v638_tp{};
          float v639_tp{};
          tensorforge::transpose4x4b32(v636_tp, v637_tp, v638_tp, v639_tp, v632_data, v633_data, v634_data, v635_data);
          tensorforge::VectorT<float, 4> v640_acc{};
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v604_data, v640_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v605_data, v645_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v606_data, v646_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v607_data, v647_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v612_data, v648_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v613_data, v653_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v614_data, v654_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v615_data, v655_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v620_data, v656_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v621_data, v661_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v622_data, v662_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v623_data, v663_acc, 2, 2, 0);
          ir8[4] = (v664_acc[0]);
          ir8[5] = (v664_acc[1]);
          ir8[6] = (v664_acc[2]);
          ir8[7] = (v664_acc[3]);
          float v669_data = r7[8];
          float v670_data = r7[9];
          float v671_data = r7[10];
          float v672_data = r7[11];
          float v673_tp{};
          float v674_tp{};
          float v675_tp{};
          float v676_tp{};
          tensorforge::transpose4x4b32(v673_tp, v674_tp, v675_tp, v676_tp, v669_data, v670_data, v671_data, v672_data);
          tensorforge::VectorT<float, 4> v677_acc{};
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v604_data, v677_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v674_tp, v605_data, v682_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v675_tp, v606_data, v683_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v607_data, v684_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v612_data, v685_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v674_tp, v613_data, v690_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v675_tp, v614_data, v691_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v615_data, v692_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v673_tp, v620_data, v693_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v674_tp, v621_data, v698_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v675_tp, v622_data, v699_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v676_tp, v623_data, v700_acc, 2, 2, 0);
          ir8[8] = (v701_acc[0]);
          ir8[9] = (v701_acc[1]);
          ir8[10] = (v701_acc[2]);
          ir8[11] = (v701_acc[3]);
          // r8 = ir8 + s0
          if (v27_g) {
            #pragma unroll
            for (int32_t v706_n1 = 0; v706_n1 < 12; ++v706_n1) {
              float v708_data = ir8[v706_n1];
              int32_t v712_a = v26_lead + (v706_n1 * 12);
              float v716_data = s0[(v712_a ^ ((v712_a >> 4) & 15))];
              r8[v706_n1] = (v716_data + v708_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r8);
          if (v27_g) {
            #pragma unroll
            for (int32_t v718_i1 = 0; v718_i1 < 12; ++v718_i1) {
              float v720_data = r8[v718_i1];
              int32_t v724_a = v26_lead + (v718_i1 * 12);
              s0[(v724_a ^ ((v724_a >> 4) & 15))] = v720_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[12]{};
          // r10 = +(r9 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          int32_t v734_sw = v26_lead ^ v196_sw;
          float v735_data = s0[v734_sw];
          float v740_data = s0[(v199_a ^ (v200_sw & 15))];
          float v745_data = s0[(v204_a ^ (v205_sw & 15))];
          float v750_data = s0[(v209_a ^ (v210_sw & 15))];
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
          float v790_data = s0[(v218_a ^ (v219_sw & 15))];
          float v795_data = s0[(v223_a ^ (v224_sw & 15))];
          float v800_data = s0[(v228_a ^ (v229_sw & 15))];
          float v805_data = s0[(v233_a ^ (v234_sw & 15))];
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
          float v845_data = s0[(v242_a ^ (v243_sw & 15))];
          float v850_data = s0[(v247_a ^ (v248_sw & 15))];
          float v855_data = s0[(v252_a ^ (v253_sw & 15))];
          float v860_data = s0[(v257_a ^ (v258_sw & 15))];
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

