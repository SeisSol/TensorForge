// === base name ===
kernel_38e28837989cc229

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_38e28837989cc229 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_38e28837989cc229(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_38e28837989cc229(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_38e28837989cc229(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_38e28837989cc229, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_38e28837989cc229, block.x * block.y * block.z, 0));
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
void launcher_kernel_38e28837989cc229(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_38e28837989cc229(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_38e28837989cc229), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_38e28837989cc229, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_38e28837989cc229(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v8_batchId0 * 24 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v8_batchId0 * 144 + 0 + m5_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 16;
          bool v26_g = v25_lead < 6;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m0[(v25_lead + (v27_i1 * 6))]);
              r0[v27_i1] = v32_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v35_g = v25_lead < 12;
          if (v35_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v25_lead + (v36_i1 * 12))]);
              r1[v36_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v26_g) {
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 12; ++v44_i1) {
              float v49_data = __builtin_nontemporal_load(&glb_m2[(v25_lead + (v44_i1 * 6))]);
              r3[v44_i1] = v49_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
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
          if (v26_g) {
            #pragma unroll
            for (int32_t v163_i1 = 0; v163_i1 < 12; ++v163_i1) {
              float v165_data = r2[v163_i1];
              int32_t v169_a = v25_lead + (v163_i1 * 12);
              s0[(v169_a ^ ((v169_a >> 4) & 15))] = v165_data;
            }
          }
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          bool v174_g = v25_lead < 2;
          if (v174_g) {
            #pragma unroll
            for (int32_t v175_i1 = 0; v175_i1 < 12; ++v175_i1) {
              float v180_data = __builtin_nontemporal_load(&glb_m4[(v25_lead + (v175_i1 * 2))]);
              r6[v175_i1] = v180_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          tensorforge::transpose4x4b32(v187_tp, v188_tp, v189_tp, v190_tp, v52_data, v53_data, v54_data, v55_data);
          tensorforge::VectorT<float, 4> v191_acc{};
          float v192_data = r3[0];
          float v193_data = r3[1];
          float v194_data = r3[2];
          float v195_data = r3[3];
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v192_data, v191_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v193_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v194_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v195_data, v198_acc, 2, 0, 0);
          float v200_data = r3[4];
          float v201_data = r3[5];
          float v202_data = r3[6];
          float v203_data = r3[7];
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v200_data, v199_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v201_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v202_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v203_data, v206_acc, 2, 1, 0);
          float v208_data = r3[8];
          float v209_data = r3[9];
          float v210_data = r3[10];
          float v211_data = r3[11];
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v208_data, v207_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v209_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v210_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v211_data, v214_acc, 2, 2, 0);
          r4[0] = (v215_acc[0]);
          r4[1] = (v215_acc[1]);
          r4[2] = (v215_acc[2]);
          r4[3] = (v215_acc[3]);
          float v224_tp{};
          float v225_tp{};
          float v226_tp{};
          float v227_tp{};
          tensorforge::transpose4x4b32(v224_tp, v225_tp, v226_tp, v227_tp, v89_data, v90_data, v91_data, v92_data);
          tensorforge::VectorT<float, 4> v228_acc{};
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v192_data, v228_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v193_data, v233_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v194_data, v234_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v195_data, v235_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v200_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v201_data, v241_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v202_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v203_data, v243_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v208_data, v244_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v209_data, v249_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v210_data, v250_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v211_data, v251_acc, 2, 2, 0);
          r4[4] = (v252_acc[0]);
          r4[5] = (v252_acc[1]);
          r4[6] = (v252_acc[2]);
          r4[7] = (v252_acc[3]);
          float v261_tp{};
          float v262_tp{};
          float v263_tp{};
          float v264_tp{};
          tensorforge::transpose4x4b32(v261_tp, v262_tp, v263_tp, v264_tp, v126_data, v127_data, v128_data, v129_data);
          tensorforge::VectorT<float, 4> v265_acc{};
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v192_data, v265_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v193_data, v270_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v194_data, v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v195_data, v272_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v200_data, v273_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v201_data, v278_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v202_data, v279_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v203_data, v280_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v208_data, v281_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v209_data, v286_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v210_data, v287_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v211_data, v288_acc, 2, 2, 0);
          r4[8] = (v289_acc[0]);
          r4[9] = (v289_acc[1]);
          r4[10] = (v289_acc[2]);
          r4[11] = (v289_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v26_g) {
            int32_t v299_off = v25_lead + 6;
            #pragma unroll
            for (int32_t v294_i1 = 0; v294_i1 < 12; ++v294_i1) {
              float v296_data = r4[v294_i1];
              int32_t v301_a = v299_off + (v294_i1 * 12);
              s0[(v301_a ^ ((v301_a >> 4) & 15))] = v296_data;
            }
          }
          float r5[12]{};
          // r5 = +(s0) + None
          // [(0, 12), (0, 12)] []
          int32_t v310_sw = (v25_lead >> 4) & 15;
          int32_t v311_sw = v25_lead ^ v310_sw;
          float v312_data = v35_g ? (s0[v311_sw]) : (0.0f);
          float v313_data = r5[0];
          r5[0] = (v313_data + v312_data);
          int32_t v315_a = v25_lead + 12;
          int32_t v316_sw = v315_a >> 4;
          float v319_data = v35_g ? (s0[(v315_a ^ (v316_sw & 15))]) : (0.0f);
          float v320_data = r5[1];
          r5[1] = (v320_data + v319_data);
          int32_t v322_a = v25_lead + 24;
          int32_t v323_sw = v322_a >> 4;
          float v326_data = v35_g ? (s0[(v322_a ^ (v323_sw & 15))]) : (0.0f);
          float v327_data = r5[2];
          r5[2] = (v327_data + v326_data);
          int32_t v329_a = v25_lead + 36;
          int32_t v330_sw = v329_a >> 4;
          float v333_data = v35_g ? (s0[(v329_a ^ (v330_sw & 15))]) : (0.0f);
          float v334_data = r5[3];
          r5[3] = (v334_data + v333_data);
          int32_t v336_a = v25_lead + 48;
          int32_t v337_sw = v336_a >> 4;
          float v340_data = v35_g ? (s0[(v336_a ^ (v337_sw & 15))]) : (0.0f);
          float v341_data = r5[4];
          r5[4] = (v341_data + v340_data);
          int32_t v343_a = v25_lead + 60;
          int32_t v344_sw = v343_a >> 4;
          float v347_data = v35_g ? (s0[(v343_a ^ (v344_sw & 15))]) : (0.0f);
          float v348_data = r5[5];
          r5[5] = (v348_data + v347_data);
          int32_t v350_a = v25_lead + 72;
          int32_t v351_sw = v350_a >> 4;
          float v354_data = v35_g ? (s0[(v350_a ^ (v351_sw & 15))]) : (0.0f);
          float v355_data = r5[6];
          r5[6] = (v355_data + v354_data);
          int32_t v357_a = v25_lead + 84;
          int32_t v358_sw = v357_a >> 4;
          float v361_data = v35_g ? (s0[(v357_a ^ (v358_sw & 15))]) : (0.0f);
          float v362_data = r5[7];
          r5[7] = (v362_data + v361_data);
          int32_t v364_a = v25_lead + 96;
          int32_t v365_sw = v364_a >> 4;
          float v368_data = v35_g ? (s0[(v364_a ^ (v365_sw & 15))]) : (0.0f);
          float v369_data = r5[8];
          r5[8] = (v369_data + v368_data);
          int32_t v371_a = v25_lead + 108;
          int32_t v372_sw = v371_a >> 4;
          float v375_data = v35_g ? (s0[(v371_a ^ (v372_sw & 15))]) : (0.0f);
          float v376_data = r5[9];
          r5[9] = (v376_data + v375_data);
          int32_t v378_a = v25_lead + 120;
          int32_t v379_sw = v378_a >> 4;
          float v382_data = v35_g ? (s0[(v378_a ^ (v379_sw & 15))]) : (0.0f);
          float v383_data = r5[10];
          r5[10] = (v383_data + v382_data);
          int32_t v385_a = v25_lead + 132;
          int32_t v386_sw = v385_a >> 4;
          float v389_data = v35_g ? (s0[(v385_a ^ (v386_sw & 15))]) : (0.0f);
          float v390_data = r5[11];
          r5[11] = (v390_data + v389_data);
          // glb_m3 = store{r>g}(r5);
          if (v35_g) {
            #pragma unroll
            for (int32_t v392_i1 = 0; v392_i1 < 12; ++v392_i1) {
              float v394_data = r5[v392_i1];
              glb_m3[(v25_lead + (v392_i1 * 12))] = v394_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[12]{};
          // r7 = +(r6 * r1) + None
          // [(0, 2), (0, 12)] [(0, 12)]
          float v404_tp{};
          float v405_tp{};
          float v406_tp{};
          float v407_tp{};
          tensorforge::transpose4x4b32(v404_tp, v405_tp, v406_tp, v407_tp, v52_data, v53_data, v54_data, v55_data);
          tensorforge::VectorT<float, 4> v408_acc{};
          float v409_data = r6[0];
          float v410_data = r6[1];
          float v411_data = r6[2];
          float v412_data = r6[3];
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v409_data, v408_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v410_data, v413_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v411_data, v414_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v412_data, v415_acc, 2, 0, 0);
          float v417_data = r6[4];
          float v418_data = r6[5];
          float v419_data = r6[6];
          float v420_data = r6[7];
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v417_data, v416_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v418_data, v421_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v419_data, v422_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v420_data, v423_acc, 2, 1, 0);
          float v425_data = r6[8];
          float v426_data = r6[9];
          float v427_data = r6[10];
          float v428_data = r6[11];
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v425_data, v424_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v426_data, v429_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v427_data, v430_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v428_data, v431_acc, 2, 2, 0);
          r7[0] = (v432_acc[0]);
          r7[1] = (v432_acc[1]);
          r7[2] = (v432_acc[2]);
          r7[3] = (v432_acc[3]);
          float v441_tp{};
          float v442_tp{};
          float v443_tp{};
          float v444_tp{};
          tensorforge::transpose4x4b32(v441_tp, v442_tp, v443_tp, v444_tp, v89_data, v90_data, v91_data, v92_data);
          tensorforge::VectorT<float, 4> v445_acc{};
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v409_data, v445_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v410_data, v450_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v411_data, v451_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v412_data, v452_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v417_data, v453_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v418_data, v458_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v419_data, v459_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v420_data, v460_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v425_data, v461_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v426_data, v466_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v427_data, v467_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v428_data, v468_acc, 2, 2, 0);
          r7[4] = (v469_acc[0]);
          r7[5] = (v469_acc[1]);
          r7[6] = (v469_acc[2]);
          r7[7] = (v469_acc[3]);
          float v478_tp{};
          float v479_tp{};
          float v480_tp{};
          float v481_tp{};
          tensorforge::transpose4x4b32(v478_tp, v479_tp, v480_tp, v481_tp, v126_data, v127_data, v128_data, v129_data);
          tensorforge::VectorT<float, 4> v482_acc{};
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v409_data, v482_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v410_data, v487_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v411_data, v488_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v412_data, v489_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v417_data, v490_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v418_data, v495_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v419_data, v496_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v420_data, v497_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v425_data, v498_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v426_data, v503_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v427_data, v504_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v428_data, v505_acc, 2, 2, 0);
          r7[8] = (v506_acc[0]);
          r7[9] = (v506_acc[1]);
          r7[10] = (v506_acc[2]);
          r7[11] = (v506_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r7);
          bool v512_g = (v25_lead >= 8) && v35_g;
          if (v512_g) {
            #pragma unroll
            for (int32_t v513_z1 = 0; v513_z1 < 12; ++v513_z1) {
              int32_t v518_a = v25_lead + (v513_z1 * 12);
              s0[(v518_a ^ ((v518_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v174_g) {
            int32_t v527_off = v25_lead + 6;
            #pragma unroll
            for (int32_t v522_i1 = 0; v522_i1 < 12; ++v522_i1) {
              float v524_data = r7[v522_i1];
              int32_t v529_a = v527_off + (v522_i1 * 12);
              s0[(v529_a ^ ((v529_a >> 4) & 15))] = v524_data;
            }
          }
          float r8[12]{};
          // r8 = +(s0) + None
          // [(0, 12), (0, 12)] []
          int32_t v539_sw = v25_lead ^ v310_sw;
          float v540_data = v35_g ? (s0[v539_sw]) : (0.0f);
          float v541_data = r8[0];
          r8[0] = (v541_data + v540_data);
          float v547_data = v35_g ? (s0[(v315_a ^ (v316_sw & 15))]) : (0.0f);
          float v548_data = r8[1];
          r8[1] = (v548_data + v547_data);
          float v554_data = v35_g ? (s0[(v322_a ^ (v323_sw & 15))]) : (0.0f);
          float v555_data = r8[2];
          r8[2] = (v555_data + v554_data);
          float v561_data = v35_g ? (s0[(v329_a ^ (v330_sw & 15))]) : (0.0f);
          float v562_data = r8[3];
          r8[3] = (v562_data + v561_data);
          float v568_data = v35_g ? (s0[(v336_a ^ (v337_sw & 15))]) : (0.0f);
          float v569_data = r8[4];
          r8[4] = (v569_data + v568_data);
          float v575_data = v35_g ? (s0[(v343_a ^ (v344_sw & 15))]) : (0.0f);
          float v576_data = r8[5];
          r8[5] = (v576_data + v575_data);
          float v582_data = v35_g ? (s0[(v350_a ^ (v351_sw & 15))]) : (0.0f);
          float v583_data = r8[6];
          r8[6] = (v583_data + v582_data);
          float v589_data = v35_g ? (s0[(v357_a ^ (v358_sw & 15))]) : (0.0f);
          float v590_data = r8[7];
          r8[7] = (v590_data + v589_data);
          float v596_data = v35_g ? (s0[(v364_a ^ (v365_sw & 15))]) : (0.0f);
          float v597_data = r8[8];
          r8[8] = (v597_data + v596_data);
          float v603_data = v35_g ? (s0[(v371_a ^ (v372_sw & 15))]) : (0.0f);
          float v604_data = r8[9];
          r8[9] = (v604_data + v603_data);
          float v610_data = v35_g ? (s0[(v378_a ^ (v379_sw & 15))]) : (0.0f);
          float v611_data = r8[10];
          r8[10] = (v611_data + v610_data);
          float v617_data = v35_g ? (s0[(v385_a ^ (v386_sw & 15))]) : (0.0f);
          float v618_data = r8[11];
          r8[11] = (v618_data + v617_data);
          // glb_m5 = store{r>g}(r8);
          if (v35_g) {
            #pragma unroll
            for (int32_t v620_i1 = 0; v620_i1 < 12; ++v620_i1) {
              float v622_data = r8[v620_i1];
              glb_m5[(v25_lead + (v620_i1 * 12))] = v622_data;
            }
          }
        }
      }
    }
  }
}

