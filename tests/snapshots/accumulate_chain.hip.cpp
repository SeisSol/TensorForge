// === base name ===
kernel_170b2e09817c1086

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_170b2e09817c1086 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_170b2e09817c1086(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_170b2e09817c1086(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_170b2e09817c1086(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_170b2e09817c1086, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_170b2e09817c1086, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_170b2e09817c1086(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_170b2e09817c1086(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_170b2e09817c1086), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_170b2e09817c1086, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_170b2e09817c1086(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 12×8(12×8) {0..12}×{0..8} strided
    //   m3 12×12(12×12) {0..12}×{0..12} strided
    //   m4 12×8(12×8) {0..12}×{0..8} strided
    //   m5 12×12(12×12) {0..12}×{0..12} strided
    //   m6 12×8(12×8) {0..12}×{0..8} strided
    //   m7 12×12(12×12) {0..12}×{0..12} strided
    //   m8 12×8(12×8) {0..12}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    //   m0[i,j] += m5[i,k] × m6[k,j]
    //   m0[i,j] += m7[i,k] × m8[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 96 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 96 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 96 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v7_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v7_batchId0 * 96 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v7_batchId0 * 144 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v7_batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v27_lead = threadIdx.x % 16;
          bool v28_g = v27_lead < 12;
          if (v28_g) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v34_data = __builtin_nontemporal_load(&glb_m1[(v27_lead + (v29_i1 * 12))]);
              r0[v29_i1] = v34_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          if (v28_g) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m2[(v27_lead + (v37_i1 * 12))]);
              r1[v37_i1] = v42_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v28_g) {
            #pragma unroll
            for (int32_t v120_i1 = 0; v120_i1 < 12; ++v120_i1) {
              float v125_data = __builtin_nontemporal_load(&glb_m3[(v27_lead + (v120_i1 * 12))]);
              r3[v120_i1] = v125_data;
            }
          }
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v45_data = r1[0];
          float v46_data = r1[1];
          float v47_data = r1[2];
          float v48_data = r1[3];
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          float v52_tp{};
          tensorforge::transpose4x4b32(v49_tp, v50_tp, v51_tp, v52_tp, v45_data, v46_data, v47_data, v48_data);
          tensorforge::VectorT<float, 4> v53_acc{};
          float v54_data = r0[0];
          float v55_data = r0[1];
          float v56_data = r0[2];
          float v57_data = r0[3];
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v53_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v60_acc, 2, 0, 0);
          float v62_data = r0[4];
          float v63_data = r0[5];
          float v64_data = r0[6];
          float v65_data = r0[7];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v61_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v68_acc, 2, 1, 0);
          float v70_data = r0[8];
          float v71_data = r0[9];
          float v72_data = r0[10];
          float v73_data = r0[11];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v69_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v76_acc, 2, 2, 0);
          r2[0] = (v77_acc[0]);
          r2[1] = (v77_acc[1]);
          r2[2] = (v77_acc[2]);
          r2[3] = (v77_acc[3]);
          float v82_data = r1[4];
          float v83_data = r1[5];
          float v84_data = r1[6];
          float v85_data = r1[7];
          float v86_tp{};
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          tensorforge::transpose4x4b32(v86_tp, v87_tp, v88_tp, v89_tp, v82_data, v83_data, v84_data, v85_data);
          tensorforge::VectorT<float, 4> v90_acc{};
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v95_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v103_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v106_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v111_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v113_acc, 2, 2, 0);
          r2[4] = (v114_acc[0]);
          r2[5] = (v114_acc[1]);
          r2[6] = (v114_acc[2]);
          r2[7] = (v114_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v28_g) {
            #pragma unroll
            for (int32_t v128_i1 = 0; v128_i1 < 8; ++v128_i1) {
              float v133_data = __builtin_nontemporal_load(&glb_m4[(v27_lead + (v128_i1 * 12))]);
              r4[v128_i1] = v133_data;
            }
          }
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v28_g) {
            #pragma unroll
            for (int32_t v217_i1 = 0; v217_i1 < 12; ++v217_i1) {
              float v222_data = __builtin_nontemporal_load(&glb_m5[(v27_lead + (v217_i1 * 12))]);
              r6[v217_i1] = v222_data;
            }
          }
          float r5[8]{};
          // ir5 = +(r3 * r4)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v137_data = r4[0];
          float v138_data = r4[1];
          float v139_data = r4[2];
          float v140_data = r4[3];
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          float v144_tp{};
          tensorforge::transpose4x4b32(v141_tp, v142_tp, v143_tp, v144_tp, v137_data, v138_data, v139_data, v140_data);
          tensorforge::VectorT<float, 4> v145_acc{};
          float v146_data = r3[0];
          float v147_data = r3[1];
          float v148_data = r3[2];
          float v149_data = r3[3];
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v146_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v147_data, v150_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v148_data, v151_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v149_data, v152_acc, 2, 0, 0);
          float v154_data = r3[4];
          float v155_data = r3[5];
          float v156_data = r3[6];
          float v157_data = r3[7];
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v154_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v155_data, v158_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v156_data, v159_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v157_data, v160_acc, 2, 1, 0);
          float v162_data = r3[8];
          float v163_data = r3[9];
          float v164_data = r3[10];
          float v165_data = r3[11];
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v162_data, v161_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v163_data, v166_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v164_data, v167_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v165_data, v168_acc, 2, 2, 0);
          ir5[0] = (v169_acc[0]);
          ir5[1] = (v169_acc[1]);
          ir5[2] = (v169_acc[2]);
          ir5[3] = (v169_acc[3]);
          float v174_data = r4[4];
          float v175_data = r4[5];
          float v176_data = r4[6];
          float v177_data = r4[7];
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          float v181_tp{};
          tensorforge::transpose4x4b32(v178_tp, v179_tp, v180_tp, v181_tp, v174_data, v175_data, v176_data, v177_data);
          tensorforge::VectorT<float, 4> v182_acc{};
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v146_data, v182_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v147_data, v187_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v148_data, v188_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v149_data, v189_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v154_data, v190_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v155_data, v195_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v156_data, v196_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v157_data, v197_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v162_data, v198_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v163_data, v203_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v164_data, v204_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v165_data, v205_acc, 2, 2, 0);
          ir5[4] = (v206_acc[0]);
          ir5[5] = (v206_acc[1]);
          ir5[6] = (v206_acc[2]);
          ir5[7] = (v206_acc[3]);
          // r5 = ir5 + r2
          if (v28_g) {
            #pragma unroll
            for (int32_t v211_n1 = 0; v211_n1 < 8; ++v211_n1) {
              float v213_data = ir5[v211_n1];
              float v214_data = r2[v211_n1];
              r5[v211_n1] = (v214_data + v213_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v28_g) {
            #pragma unroll
            for (int32_t v225_i1 = 0; v225_i1 < 8; ++v225_i1) {
              float v230_data = __builtin_nontemporal_load(&glb_m6[(v27_lead + (v225_i1 * 12))]);
              r7[v225_i1] = v230_data;
            }
          }
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v28_g) {
            #pragma unroll
            for (int32_t v314_i1 = 0; v314_i1 < 12; ++v314_i1) {
              float v319_data = __builtin_nontemporal_load(&glb_m7[(v27_lead + (v314_i1 * 12))]);
              r9[v314_i1] = v319_data;
            }
          }
          float r8[8]{};
          // ir8 = +(r6 * r7)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v234_data = r7[0];
          float v235_data = r7[1];
          float v236_data = r7[2];
          float v237_data = r7[3];
          float v238_tp{};
          float v239_tp{};
          float v240_tp{};
          float v241_tp{};
          tensorforge::transpose4x4b32(v238_tp, v239_tp, v240_tp, v241_tp, v234_data, v235_data, v236_data, v237_data);
          tensorforge::VectorT<float, 4> v242_acc{};
          float v243_data = r6[0];
          float v244_data = r6[1];
          float v245_data = r6[2];
          float v246_data = r6[3];
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v243_data, v242_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v244_data, v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v245_data, v248_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v246_data, v249_acc, 2, 0, 0);
          float v251_data = r6[4];
          float v252_data = r6[5];
          float v253_data = r6[6];
          float v254_data = r6[7];
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v251_data, v250_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v252_data, v255_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v253_data, v256_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v254_data, v257_acc, 2, 1, 0);
          float v259_data = r6[8];
          float v260_data = r6[9];
          float v261_data = r6[10];
          float v262_data = r6[11];
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v259_data, v258_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v260_data, v263_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v261_data, v264_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v262_data, v265_acc, 2, 2, 0);
          ir8[0] = (v266_acc[0]);
          ir8[1] = (v266_acc[1]);
          ir8[2] = (v266_acc[2]);
          ir8[3] = (v266_acc[3]);
          float v271_data = r7[4];
          float v272_data = r7[5];
          float v273_data = r7[6];
          float v274_data = r7[7];
          float v275_tp{};
          float v276_tp{};
          float v277_tp{};
          float v278_tp{};
          tensorforge::transpose4x4b32(v275_tp, v276_tp, v277_tp, v278_tp, v271_data, v272_data, v273_data, v274_data);
          tensorforge::VectorT<float, 4> v279_acc{};
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v243_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v244_data, v284_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v245_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v246_data, v286_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v251_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v252_data, v292_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v253_data, v293_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v254_data, v294_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v259_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v260_data, v300_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v261_data, v301_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v262_data, v302_acc, 2, 2, 0);
          ir8[4] = (v303_acc[0]);
          ir8[5] = (v303_acc[1]);
          ir8[6] = (v303_acc[2]);
          ir8[7] = (v303_acc[3]);
          // r8 = ir8 + r5
          if (v28_g) {
            #pragma unroll
            for (int32_t v308_n1 = 0; v308_n1 < 8; ++v308_n1) {
              float v310_data = ir8[v308_n1];
              float v311_data = r5[v308_n1];
              r8[v308_n1] = (v311_data + v310_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          if (v28_g) {
            #pragma unroll
            for (int32_t v322_i1 = 0; v322_i1 < 8; ++v322_i1) {
              float v327_data = __builtin_nontemporal_load(&glb_m8[(v27_lead + (v322_i1 * 12))]);
              r10[v322_i1] = v327_data;
            }
          }
          float r11[8]{};
          // ir11 = +(r9 * r10)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v331_data = r10[0];
          float v332_data = r10[1];
          float v333_data = r10[2];
          float v334_data = r10[3];
          float v335_tp{};
          float v336_tp{};
          float v337_tp{};
          float v338_tp{};
          tensorforge::transpose4x4b32(v335_tp, v336_tp, v337_tp, v338_tp, v331_data, v332_data, v333_data, v334_data);
          tensorforge::VectorT<float, 4> v339_acc{};
          float v340_data = r9[0];
          float v341_data = r9[1];
          float v342_data = r9[2];
          float v343_data = r9[3];
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v340_data, v339_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v341_data, v344_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v342_data, v345_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v343_data, v346_acc, 2, 0, 0);
          float v348_data = r9[4];
          float v349_data = r9[5];
          float v350_data = r9[6];
          float v351_data = r9[7];
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v348_data, v347_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v349_data, v352_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v350_data, v353_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v351_data, v354_acc, 2, 1, 0);
          float v356_data = r9[8];
          float v357_data = r9[9];
          float v358_data = r9[10];
          float v359_data = r9[11];
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v356_data, v355_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v357_data, v360_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v358_data, v361_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v359_data, v362_acc, 2, 2, 0);
          ir11[0] = (v363_acc[0]);
          ir11[1] = (v363_acc[1]);
          ir11[2] = (v363_acc[2]);
          ir11[3] = (v363_acc[3]);
          float v368_data = r10[4];
          float v369_data = r10[5];
          float v370_data = r10[6];
          float v371_data = r10[7];
          float v372_tp{};
          float v373_tp{};
          float v374_tp{};
          float v375_tp{};
          tensorforge::transpose4x4b32(v372_tp, v373_tp, v374_tp, v375_tp, v368_data, v369_data, v370_data, v371_data);
          tensorforge::VectorT<float, 4> v376_acc{};
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v340_data, v376_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v341_data, v381_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v374_tp, v342_data, v382_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v343_data, v383_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v348_data, v384_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v349_data, v389_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v374_tp, v350_data, v390_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v351_data, v391_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v356_data, v392_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v357_data, v397_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v374_tp, v358_data, v398_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v359_data, v399_acc, 2, 2, 0);
          ir11[4] = (v400_acc[0]);
          ir11[5] = (v400_acc[1]);
          ir11[6] = (v400_acc[2]);
          ir11[7] = (v400_acc[3]);
          // r11 = ir11 + r8
          if (v28_g) {
            #pragma unroll
            for (int32_t v405_n1 = 0; v405_n1 < 8; ++v405_n1) {
              float v407_data = ir11[v405_n1];
              float v408_data = r8[v405_n1];
              r11[v405_n1] = (v408_data + v407_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v28_g) {
            #pragma unroll
            for (int32_t v410_i1 = 0; v410_i1 < 8; ++v410_i1) {
              float v412_data = r11[v410_i1];
              glb_m0[(v27_lead + (v410_i1 * 12))] = v412_data;
            }
          }
        }
      }
    }
  }
}

