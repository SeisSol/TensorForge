// === base name ===
kernel_faef1bf011fd7b94

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_faef1bf011fd7b94 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_faef1bf011fd7b94(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_faef1bf011fd7b94(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_faef1bf011fd7b94(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_faef1bf011fd7b94, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_faef1bf011fd7b94, block.x * block.y * block.z, 0));
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
void launcher_kernel_faef1bf011fd7b94(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_faef1bf011fd7b94(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_faef1bf011fd7b94), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_faef1bf011fd7b94, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_faef1bf011fd7b94(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v28_g) {
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              float v50_data = __builtin_nontemporal_load(&glb_m3[(v27_lead + (v45_i1 * 12))]);
              r3[v45_i1] = v50_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v53_data = r1[0];
          float v54_data = r1[1];
          float v55_data = r1[2];
          float v56_data = r1[3];
          float v57_tp{};
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          tensorforge::transpose4x4b32(v57_tp, v58_tp, v59_tp, v60_tp, v53_data, v54_data, v55_data, v56_data);
          tensorforge::VectorT<float, 4> v61_acc{};
          float v62_data = r0[0];
          float v63_data = r0[1];
          float v64_data = r0[2];
          float v65_data = r0[3];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 2, 0, 0);
          float v70_data = r0[4];
          float v71_data = r0[5];
          float v72_data = r0[6];
          float v73_data = r0[7];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 2, 1, 0);
          float v78_data = r0[8];
          float v79_data = r0[9];
          float v80_data = r0[10];
          float v81_data = r0[11];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v77_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 2, 2, 0);
          r2[0] = (v85_acc[0]);
          r2[1] = (v85_acc[1]);
          r2[2] = (v85_acc[2]);
          r2[3] = (v85_acc[3]);
          float v90_data = r1[4];
          float v91_data = r1[5];
          float v92_data = r1[6];
          float v93_data = r1[7];
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          float v97_tp{};
          tensorforge::transpose4x4b32(v94_tp, v95_tp, v96_tp, v97_tp, v90_data, v91_data, v92_data, v93_data);
          tensorforge::VectorT<float, 4> v98_acc{};
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v113_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v119_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v121_acc, 2, 2, 0);
          r2[4] = (v122_acc[0]);
          r2[5] = (v122_acc[1]);
          r2[6] = (v122_acc[2]);
          r2[7] = (v122_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v28_g) {
            #pragma unroll
            for (int32_t v128_i1 = 0; v128_i1 < 8; ++v128_i1) {
              float v133_data = __builtin_nontemporal_load(&glb_m4[(v27_lead + (v128_i1 * 12))]);
              r4[v128_i1] = v133_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v28_g) {
            #pragma unroll
            for (int32_t v136_i1 = 0; v136_i1 < 12; ++v136_i1) {
              float v141_data = __builtin_nontemporal_load(&glb_m5[(v27_lead + (v136_i1 * 12))]);
              r6[v136_i1] = v141_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // ir5 = +(r3 * r4)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v145_data = r4[0];
          float v146_data = r4[1];
          float v147_data = r4[2];
          float v148_data = r4[3];
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          tensorforge::transpose4x4b32(v149_tp, v150_tp, v151_tp, v152_tp, v145_data, v146_data, v147_data, v148_data);
          tensorforge::VectorT<float, 4> v153_acc{};
          float v154_data = r3[0];
          float v155_data = r3[1];
          float v156_data = r3[2];
          float v157_data = r3[3];
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v154_data, v153_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v155_data, v158_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v156_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v157_data, v160_acc, 2, 0, 0);
          float v162_data = r3[4];
          float v163_data = r3[5];
          float v164_data = r3[6];
          float v165_data = r3[7];
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v162_data, v161_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v163_data, v166_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v164_data, v167_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v165_data, v168_acc, 2, 1, 0);
          float v170_data = r3[8];
          float v171_data = r3[9];
          float v172_data = r3[10];
          float v173_data = r3[11];
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v170_data, v169_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v171_data, v174_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v172_data, v175_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v173_data, v176_acc, 2, 2, 0);
          ir5[0] = (v177_acc[0]);
          ir5[1] = (v177_acc[1]);
          ir5[2] = (v177_acc[2]);
          ir5[3] = (v177_acc[3]);
          float v182_data = r4[4];
          float v183_data = r4[5];
          float v184_data = r4[6];
          float v185_data = r4[7];
          float v186_tp{};
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          tensorforge::transpose4x4b32(v186_tp, v187_tp, v188_tp, v189_tp, v182_data, v183_data, v184_data, v185_data);
          tensorforge::VectorT<float, 4> v190_acc{};
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v154_data, v190_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v155_data, v195_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v156_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v157_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v162_data, v198_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v163_data, v203_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v164_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v165_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v170_data, v206_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v171_data, v211_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v172_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v173_data, v213_acc, 2, 2, 0);
          ir5[4] = (v214_acc[0]);
          ir5[5] = (v214_acc[1]);
          ir5[6] = (v214_acc[2]);
          ir5[7] = (v214_acc[3]);
          // r5 = ir5 + r2
          if (v28_g) {
            #pragma unroll
            for (int32_t v219_n1 = 0; v219_n1 < 8; ++v219_n1) {
              float v221_data = ir5[v219_n1];
              float v222_data = r2[v219_n1];
              r5[v219_n1] = (v222_data + v221_data);
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
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v28_g) {
            #pragma unroll
            for (int32_t v233_i1 = 0; v233_i1 < 12; ++v233_i1) {
              float v238_data = __builtin_nontemporal_load(&glb_m7[(v27_lead + (v233_i1 * 12))]);
              r9[v233_i1] = v238_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // ir8 = +(r6 * r7)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v242_data = r7[0];
          float v243_data = r7[1];
          float v244_data = r7[2];
          float v245_data = r7[3];
          float v246_tp{};
          float v247_tp{};
          float v248_tp{};
          float v249_tp{};
          tensorforge::transpose4x4b32(v246_tp, v247_tp, v248_tp, v249_tp, v242_data, v243_data, v244_data, v245_data);
          tensorforge::VectorT<float, 4> v250_acc{};
          float v251_data = r6[0];
          float v252_data = r6[1];
          float v253_data = r6[2];
          float v254_data = r6[3];
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v251_data, v250_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v252_data, v255_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v253_data, v256_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v254_data, v257_acc, 2, 0, 0);
          float v259_data = r6[4];
          float v260_data = r6[5];
          float v261_data = r6[6];
          float v262_data = r6[7];
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v259_data, v258_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v260_data, v263_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v261_data, v264_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v262_data, v265_acc, 2, 1, 0);
          float v267_data = r6[8];
          float v268_data = r6[9];
          float v269_data = r6[10];
          float v270_data = r6[11];
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v267_data, v266_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v268_data, v271_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v269_data, v272_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v270_data, v273_acc, 2, 2, 0);
          ir8[0] = (v274_acc[0]);
          ir8[1] = (v274_acc[1]);
          ir8[2] = (v274_acc[2]);
          ir8[3] = (v274_acc[3]);
          float v279_data = r7[4];
          float v280_data = r7[5];
          float v281_data = r7[6];
          float v282_data = r7[7];
          float v283_tp{};
          float v284_tp{};
          float v285_tp{};
          float v286_tp{};
          tensorforge::transpose4x4b32(v283_tp, v284_tp, v285_tp, v286_tp, v279_data, v280_data, v281_data, v282_data);
          tensorforge::VectorT<float, 4> v287_acc{};
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v251_data, v287_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v252_data, v292_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v253_data, v293_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v254_data, v294_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v259_data, v295_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v260_data, v300_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v261_data, v301_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v262_data, v302_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v267_data, v303_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v268_data, v308_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v269_data, v309_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v270_data, v310_acc, 2, 2, 0);
          ir8[4] = (v311_acc[0]);
          ir8[5] = (v311_acc[1]);
          ir8[6] = (v311_acc[2]);
          ir8[7] = (v311_acc[3]);
          // r8 = ir8 + r5
          if (v28_g) {
            #pragma unroll
            for (int32_t v316_n1 = 0; v316_n1 < 8; ++v316_n1) {
              float v318_data = ir8[v316_n1];
              float v319_data = r5[v316_n1];
              r8[v316_n1] = (v319_data + v318_data);
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
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
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

