// === base name ===
kernel_cc3e62c935f382bb

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cc3e62c935f382bb = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cc3e62c935f382bb(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cc3e62c935f382bb(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cc3e62c935f382bb(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_cc3e62c935f382bb, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_cc3e62c935f382bb, block.x * block.y * block.z, 0));
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
void launcher_kernel_cc3e62c935f382bb(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cc3e62c935f382bb(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_cc3e62c935f382bb), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_cc3e62c935f382bb, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_cc3e62c935f382bb(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 32×32(6×12) {0..6}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(6×12) {0..6}×{0..12} strided
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    //   m4 32×32(12×12) {0..12}×{0..12} strided
    //   m5 32×32(6×12) {0..6}×{0..12} strided
    //   m6 32×32(12×12) {0..12}×{0..12} strided
    //   m7 32×32(6×12) {0..6}×{0..12} strided
    //   m8 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = t0[i,k] × m4[k,j]
    //   t1[i,j]@{0..6}×{0..12} = m5[i,k] × m6[k,j]
    //   t1[i,j]@{6..12}×{0..12} = m7[i,k] × m6[k,j]
    //   m3[i,j] += t1[i,k] × m8[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"F1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"G","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"H","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m6","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m7","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m8","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t1","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v12_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v12_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v12_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v12_batchId0 * 144 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v12_batchId0 * 72 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v12_batchId0 * 144 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v12_batchId0 * 72 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v12_batchId0 * 144 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v32_lead = threadIdx.x % 16;
          bool v33_g = v32_lead < 6;
          if (v33_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m0[(v32_lead + (v34_i1 * 6))]);
              r0[v34_i1] = v39_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v42_g = v32_lead < 12;
          if (v42_g) {
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m1[(v32_lead + (v43_i1 * 12))]);
              r1[v43_i1] = v48_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v33_g) {
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 12; ++v51_i1) {
              float v56_data = __builtin_nontemporal_load(&glb_m2[(v32_lead + (v51_i1 * 6))]);
              r3[v51_i1] = v56_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v59_data = r1[0];
          float v60_data = r1[1];
          float v61_data = r1[2];
          float v62_data = r1[3];
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          tensorforge::transpose4x4b32(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v67_acc{};
          float v68_data = r0[0];
          float v69_data = r0[1];
          float v70_data = r0[2];
          float v71_data = r0[3];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v73_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v74_acc, 2, 0, 0);
          float v76_data = r0[4];
          float v77_data = r0[5];
          float v78_data = r0[6];
          float v79_data = r0[7];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v81_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v82_acc, 2, 1, 0);
          float v84_data = r0[8];
          float v85_data = r0[9];
          float v86_data = r0[10];
          float v87_data = r0[11];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v85_data, v88_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v86_data, v89_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v90_acc, 2, 2, 0);
          r2[0] = (v91_acc[0]);
          r2[1] = (v91_acc[1]);
          r2[2] = (v91_acc[2]);
          r2[3] = (v91_acc[3]);
          float v96_data = r1[4];
          float v97_data = r1[5];
          float v98_data = r1[6];
          float v99_data = r1[7];
          float v100_tp{};
          float v101_tp{};
          float v102_tp{};
          float v103_tp{};
          tensorforge::transpose4x4b32(v100_tp, v101_tp, v102_tp, v103_tp, v96_data, v97_data, v98_data, v99_data);
          tensorforge::VectorT<float, 4> v104_acc{};
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v69_data, v109_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v70_data, v110_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v71_data, v111_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v77_data, v117_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v78_data, v118_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v79_data, v119_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v85_data, v125_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v86_data, v126_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v87_data, v127_acc, 2, 2, 0);
          r2[4] = (v128_acc[0]);
          r2[5] = (v128_acc[1]);
          r2[6] = (v128_acc[2]);
          r2[7] = (v128_acc[3]);
          float v133_data = r1[8];
          float v134_data = r1[9];
          float v135_data = r1[10];
          float v136_data = r1[11];
          float v137_tp{};
          float v138_tp{};
          float v139_tp{};
          float v140_tp{};
          tensorforge::transpose4x4b32(v137_tp, v138_tp, v139_tp, v140_tp, v133_data, v134_data, v135_data, v136_data);
          tensorforge::VectorT<float, 4> v141_acc{};
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v146_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v147_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v71_data, v148_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v154_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v155_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v79_data, v156_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v84_data, v157_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v85_data, v162_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v86_data, v163_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v87_data, v164_acc, 2, 2, 0);
          r2[8] = (v165_acc[0]);
          r2[9] = (v165_acc[1]);
          r2[10] = (v165_acc[2]);
          r2[11] = (v165_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v33_g) {
            #pragma unroll
            for (int32_t v170_i1 = 0; v170_i1 < 12; ++v170_i1) {
              float v172_data = r2[v170_i1];
              int32_t v176_a = v32_lead + (v170_i1 * 12);
              s0[(v176_a ^ ((v176_a >> 4) & 15))] = v172_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v42_g) {
            #pragma unroll
            for (int32_t v181_i1 = 0; v181_i1 < 12; ++v181_i1) {
              float v186_data = __builtin_nontemporal_load(&glb_m4[(v32_lead + (v181_i1 * 12))]);
              r5[v181_i1] = v186_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v193_tp{};
          float v194_tp{};
          float v195_tp{};
          float v196_tp{};
          tensorforge::transpose4x4b32(v193_tp, v194_tp, v195_tp, v196_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v197_acc{};
          float v198_data = r3[0];
          float v199_data = r3[1];
          float v200_data = r3[2];
          float v201_data = r3[3];
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v198_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v199_data, v202_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v200_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v201_data, v204_acc, 2, 0, 0);
          float v206_data = r3[4];
          float v207_data = r3[5];
          float v208_data = r3[6];
          float v209_data = r3[7];
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v206_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v207_data, v210_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v208_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v209_data, v212_acc, 2, 1, 0);
          float v214_data = r3[8];
          float v215_data = r3[9];
          float v216_data = r3[10];
          float v217_data = r3[11];
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v214_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v215_data, v218_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v216_data, v219_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v217_data, v220_acc, 2, 2, 0);
          r4[0] = (v221_acc[0]);
          r4[1] = (v221_acc[1]);
          r4[2] = (v221_acc[2]);
          r4[3] = (v221_acc[3]);
          float v230_tp{};
          float v231_tp{};
          float v232_tp{};
          float v233_tp{};
          tensorforge::transpose4x4b32(v230_tp, v231_tp, v232_tp, v233_tp, v96_data, v97_data, v98_data, v99_data);
          tensorforge::VectorT<float, 4> v234_acc{};
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v198_data, v234_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v199_data, v239_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v200_data, v240_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v201_data, v241_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v206_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v207_data, v247_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v208_data, v248_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v209_data, v249_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v214_data, v250_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v215_data, v255_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v216_data, v256_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v217_data, v257_acc, 2, 2, 0);
          r4[4] = (v258_acc[0]);
          r4[5] = (v258_acc[1]);
          r4[6] = (v258_acc[2]);
          r4[7] = (v258_acc[3]);
          float v267_tp{};
          float v268_tp{};
          float v269_tp{};
          float v270_tp{};
          tensorforge::transpose4x4b32(v267_tp, v268_tp, v269_tp, v270_tp, v133_data, v134_data, v135_data, v136_data);
          tensorforge::VectorT<float, 4> v271_acc{};
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v198_data, v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v199_data, v276_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v200_data, v277_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v201_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v206_data, v279_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v207_data, v284_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v208_data, v285_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v209_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v214_data, v287_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v215_data, v292_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v216_data, v293_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v217_data, v294_acc, 2, 2, 0);
          r4[8] = (v295_acc[0]);
          r4[9] = (v295_acc[1]);
          r4[10] = (v295_acc[2]);
          r4[11] = (v295_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v33_g) {
            int32_t v305_off = v32_lead + 6;
            #pragma unroll
            for (int32_t v300_i1 = 0; v300_i1 < 12; ++v300_i1) {
              float v302_data = r4[v300_i1];
              int32_t v307_a = v305_off + (v300_i1 * 12);
              s0[(v307_a ^ ((v307_a >> 4) & 15))] = v302_data;
            }
          }
          float r7[12]{};
          // r7 = load{g>r}(glb_m5);
          if (v33_g) {
            #pragma unroll
            for (int32_t v312_i1 = 0; v312_i1 < 12; ++v312_i1) {
              float v317_data = __builtin_nontemporal_load(&glb_m5[(v32_lead + (v312_i1 * 6))]);
              r7[v312_i1] = v317_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(s0 * r5) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v320_data = r5[0];
          float v321_data = r5[1];
          float v322_data = r5[2];
          float v323_data = r5[3];
          float v324_tp{};
          float v325_tp{};
          float v326_tp{};
          float v327_tp{};
          tensorforge::transpose4x4b32(v324_tp, v325_tp, v326_tp, v327_tp, v320_data, v321_data, v322_data, v323_data);
          tensorforge::VectorT<float, 4> v328_acc{};
          int32_t v333_sw = (v32_lead >> 4) & 15;
          float v335_data = s0[(v32_lead ^ v333_sw)];
          int32_t v336_a = v32_lead + 12;
          int32_t v337_sw = v336_a >> 4;
          float v340_data = s0[(v336_a ^ (v337_sw & 15))];
          int32_t v341_a = v32_lead + 24;
          int32_t v342_sw = v341_a >> 4;
          float v345_data = s0[(v341_a ^ (v342_sw & 15))];
          int32_t v346_a = v32_lead + 36;
          int32_t v347_sw = v346_a >> 4;
          float v350_data = s0[(v346_a ^ (v347_sw & 15))];
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v335_data, v328_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v340_data, v351_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v345_data, v352_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v350_data, v353_acc, 2, 0, 0);
          int32_t v355_a = v32_lead + 48;
          int32_t v356_sw = v355_a >> 4;
          float v359_data = s0[(v355_a ^ (v356_sw & 15))];
          int32_t v360_a = v32_lead + 60;
          int32_t v361_sw = v360_a >> 4;
          float v364_data = s0[(v360_a ^ (v361_sw & 15))];
          int32_t v365_a = v32_lead + 72;
          int32_t v366_sw = v365_a >> 4;
          float v369_data = s0[(v365_a ^ (v366_sw & 15))];
          int32_t v370_a = v32_lead + 84;
          int32_t v371_sw = v370_a >> 4;
          float v374_data = s0[(v370_a ^ (v371_sw & 15))];
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v359_data, v354_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v364_data, v375_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v369_data, v376_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v374_data, v377_acc, 2, 1, 0);
          int32_t v379_a = v32_lead + 96;
          int32_t v380_sw = v379_a >> 4;
          float v383_data = s0[(v379_a ^ (v380_sw & 15))];
          int32_t v384_a = v32_lead + 108;
          int32_t v385_sw = v384_a >> 4;
          float v388_data = s0[(v384_a ^ (v385_sw & 15))];
          int32_t v389_a = v32_lead + 120;
          int32_t v390_sw = v389_a >> 4;
          float v393_data = s0[(v389_a ^ (v390_sw & 15))];
          int32_t v394_a = v32_lead + 132;
          int32_t v395_sw = v394_a >> 4;
          float v398_data = s0[(v394_a ^ (v395_sw & 15))];
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v383_data, v378_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v388_data, v399_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v393_data, v400_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v398_data, v401_acc, 2, 2, 0);
          r6[0] = (v402_acc[0]);
          r6[1] = (v402_acc[1]);
          r6[2] = (v402_acc[2]);
          r6[3] = (v402_acc[3]);
          float v407_data = r5[4];
          float v408_data = r5[5];
          float v409_data = r5[6];
          float v410_data = r5[7];
          float v411_tp{};
          float v412_tp{};
          float v413_tp{};
          float v414_tp{};
          tensorforge::transpose4x4b32(v411_tp, v412_tp, v413_tp, v414_tp, v407_data, v408_data, v409_data, v410_data);
          tensorforge::VectorT<float, 4> v415_acc{};
          float v422_data = s0[(v32_lead ^ v333_sw)];
          float v427_data = s0[(v336_a ^ (v337_sw & 15))];
          float v432_data = s0[(v341_a ^ (v342_sw & 15))];
          float v437_data = s0[(v346_a ^ (v347_sw & 15))];
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v422_data, v415_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v427_data, v438_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v432_data, v439_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v437_data, v440_acc, 2, 0, 0);
          float v446_data = s0[(v355_a ^ (v356_sw & 15))];
          float v451_data = s0[(v360_a ^ (v361_sw & 15))];
          float v456_data = s0[(v365_a ^ (v366_sw & 15))];
          float v461_data = s0[(v370_a ^ (v371_sw & 15))];
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v446_data, v441_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v451_data, v462_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v456_data, v463_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v461_data, v464_acc, 2, 1, 0);
          float v470_data = s0[(v379_a ^ (v380_sw & 15))];
          float v475_data = s0[(v384_a ^ (v385_sw & 15))];
          float v480_data = s0[(v389_a ^ (v390_sw & 15))];
          float v485_data = s0[(v394_a ^ (v395_sw & 15))];
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v470_data, v465_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v412_tp, v475_data, v486_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v480_data, v487_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v485_data, v488_acc, 2, 2, 0);
          r6[4] = (v489_acc[0]);
          r6[5] = (v489_acc[1]);
          r6[6] = (v489_acc[2]);
          r6[7] = (v489_acc[3]);
          float v494_data = r5[8];
          float v495_data = r5[9];
          float v496_data = r5[10];
          float v497_data = r5[11];
          float v498_tp{};
          float v499_tp{};
          float v500_tp{};
          float v501_tp{};
          tensorforge::transpose4x4b32(v498_tp, v499_tp, v500_tp, v501_tp, v494_data, v495_data, v496_data, v497_data);
          tensorforge::VectorT<float, 4> v502_acc{};
          float v509_data = s0[(v32_lead ^ v333_sw)];
          float v514_data = s0[(v336_a ^ (v337_sw & 15))];
          float v519_data = s0[(v341_a ^ (v342_sw & 15))];
          float v524_data = s0[(v346_a ^ (v347_sw & 15))];
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v509_data, v502_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v514_data, v525_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v519_data, v526_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v524_data, v527_acc, 2, 0, 0);
          float v533_data = s0[(v355_a ^ (v356_sw & 15))];
          float v538_data = s0[(v360_a ^ (v361_sw & 15))];
          float v543_data = s0[(v365_a ^ (v366_sw & 15))];
          float v548_data = s0[(v370_a ^ (v371_sw & 15))];
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v533_data, v528_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v538_data, v549_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v543_data, v550_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v548_data, v551_acc, 2, 1, 0);
          float v557_data = s0[(v379_a ^ (v380_sw & 15))];
          float v562_data = s0[(v384_a ^ (v385_sw & 15))];
          float v567_data = s0[(v389_a ^ (v390_sw & 15))];
          float v572_data = s0[(v394_a ^ (v395_sw & 15))];
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v557_data, v552_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v562_data, v573_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v567_data, v574_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v572_data, v575_acc, 2, 2, 0);
          r6[8] = (v576_acc[0]);
          r6[9] = (v576_acc[1]);
          r6[10] = (v576_acc[2]);
          r6[11] = (v576_acc[3]);
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v42_g) {
            #pragma unroll
            for (int32_t v582_i1 = 0; v582_i1 < 12; ++v582_i1) {
              float v587_data = __builtin_nontemporal_load(&glb_m6[(v32_lead + (v582_i1 * 12))]);
              r8[v582_i1] = v587_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r10[12]{};
          // r10 = load{g>r}(glb_m7);
          if (v33_g) {
            #pragma unroll
            for (int32_t v590_i1 = 0; v590_i1 < 12; ++v590_i1) {
              float v595_data = __builtin_nontemporal_load(&glb_m7[(v32_lead + (v590_i1 * 6))]);
              r10[v590_i1] = v595_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(r7 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v598_data = r8[0];
          float v599_data = r8[1];
          float v600_data = r8[2];
          float v601_data = r8[3];
          float v602_tp{};
          float v603_tp{};
          float v604_tp{};
          float v605_tp{};
          tensorforge::transpose4x4b32(v602_tp, v603_tp, v604_tp, v605_tp, v598_data, v599_data, v600_data, v601_data);
          tensorforge::VectorT<float, 4> v606_acc{};
          float v607_data = r7[0];
          float v608_data = r7[1];
          float v609_data = r7[2];
          float v610_data = r7[3];
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v607_data, v606_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v603_tp, v608_data, v611_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v604_tp, v609_data, v612_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v605_tp, v610_data, v613_acc, 2, 0, 0);
          float v615_data = r7[4];
          float v616_data = r7[5];
          float v617_data = r7[6];
          float v618_data = r7[7];
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v615_data, v614_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v603_tp, v616_data, v619_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v604_tp, v617_data, v620_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v605_tp, v618_data, v621_acc, 2, 1, 0);
          float v623_data = r7[8];
          float v624_data = r7[9];
          float v625_data = r7[10];
          float v626_data = r7[11];
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v602_tp, v623_data, v622_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v603_tp, v624_data, v627_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v604_tp, v625_data, v628_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v605_tp, v626_data, v629_acc, 2, 2, 0);
          r9[0] = (v630_acc[0]);
          r9[1] = (v630_acc[1]);
          r9[2] = (v630_acc[2]);
          r9[3] = (v630_acc[3]);
          float v635_data = r8[4];
          float v636_data = r8[5];
          float v637_data = r8[6];
          float v638_data = r8[7];
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
          r9[4] = (v667_acc[0]);
          r9[5] = (v667_acc[1]);
          r9[6] = (v667_acc[2]);
          r9[7] = (v667_acc[3]);
          float v672_data = r8[8];
          float v673_data = r8[9];
          float v674_data = r8[10];
          float v675_data = r8[11];
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
          r9[8] = (v704_acc[0]);
          r9[9] = (v704_acc[1]);
          r9[10] = (v704_acc[2]);
          r9[11] = (v704_acc[3]);
          // s1 = store{r>s}(localShrMem0, r9);
          if (v33_g) {
            #pragma unroll
            for (int32_t v709_i1 = 0; v709_i1 < 12; ++v709_i1) {
              float v711_data = r9[v709_i1];
              int32_t v715_a = v32_lead + (v709_i1 * 12);
              s1[(v715_a ^ ((v715_a >> 4) & 15))] = v711_data;
            }
          }
          float r12[12]{};
          // r12 = load{g>r}(glb_m8);
          if (v42_g) {
            #pragma unroll
            for (int32_t v720_i1 = 0; v720_i1 < 12; ++v720_i1) {
              float v725_data = __builtin_nontemporal_load(&glb_m8[(v32_lead + (v720_i1 * 12))]);
              r12[v720_i1] = v725_data;
            }
          }
          // wait(r10 = load{g>r}(glb_m7););
          float r11[12]{};
          // r11 = +(r10 * r8) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v732_tp{};
          float v733_tp{};
          float v734_tp{};
          float v735_tp{};
          tensorforge::transpose4x4b32(v732_tp, v733_tp, v734_tp, v735_tp, v598_data, v599_data, v600_data, v601_data);
          tensorforge::VectorT<float, 4> v736_acc{};
          float v737_data = r10[0];
          float v738_data = r10[1];
          float v739_data = r10[2];
          float v740_data = r10[3];
          tensorforge::VectorT<float, 4> v741_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v732_tp, v737_data, v736_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v742_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v738_data, v741_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v743_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v739_data, v742_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v744_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v740_data, v743_acc, 2, 0, 0);
          float v745_data = r10[4];
          float v746_data = r10[5];
          float v747_data = r10[6];
          float v748_data = r10[7];
          tensorforge::VectorT<float, 4> v749_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v732_tp, v745_data, v744_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v750_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v746_data, v749_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v751_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v747_data, v750_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v748_data, v751_acc, 2, 1, 0);
          float v753_data = r10[8];
          float v754_data = r10[9];
          float v755_data = r10[10];
          float v756_data = r10[11];
          tensorforge::VectorT<float, 4> v757_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v732_tp, v753_data, v752_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v754_data, v757_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v759_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v755_data, v758_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v756_data, v759_acc, 2, 2, 0);
          r11[0] = (v760_acc[0]);
          r11[1] = (v760_acc[1]);
          r11[2] = (v760_acc[2]);
          r11[3] = (v760_acc[3]);
          float v769_tp{};
          float v770_tp{};
          float v771_tp{};
          float v772_tp{};
          tensorforge::transpose4x4b32(v769_tp, v770_tp, v771_tp, v772_tp, v635_data, v636_data, v637_data, v638_data);
          tensorforge::VectorT<float, 4> v773_acc{};
          tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v769_tp, v737_data, v773_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v779_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v770_tp, v738_data, v778_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v771_tp, v739_data, v779_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v772_tp, v740_data, v780_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v769_tp, v745_data, v781_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v787_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v770_tp, v746_data, v786_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v788_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v771_tp, v747_data, v787_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v772_tp, v748_data, v788_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v769_tp, v753_data, v789_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v795_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v770_tp, v754_data, v794_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v771_tp, v755_data, v795_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v797_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v772_tp, v756_data, v796_acc, 2, 2, 0);
          r11[4] = (v797_acc[0]);
          r11[5] = (v797_acc[1]);
          r11[6] = (v797_acc[2]);
          r11[7] = (v797_acc[3]);
          float v806_tp{};
          float v807_tp{};
          float v808_tp{};
          float v809_tp{};
          tensorforge::transpose4x4b32(v806_tp, v807_tp, v808_tp, v809_tp, v672_data, v673_data, v674_data, v675_data);
          tensorforge::VectorT<float, 4> v810_acc{};
          tensorforge::VectorT<float, 4> v815_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v737_data, v810_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v816_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v807_tp, v738_data, v815_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v817_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v808_tp, v739_data, v816_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v740_data, v817_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v823_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v745_data, v818_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v824_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v807_tp, v746_data, v823_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v825_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v808_tp, v747_data, v824_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v748_data, v825_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v806_tp, v753_data, v826_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v832_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v807_tp, v754_data, v831_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v833_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v808_tp, v755_data, v832_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v834_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v809_tp, v756_data, v833_acc, 2, 2, 0);
          r11[8] = (v834_acc[0]);
          r11[9] = (v834_acc[1]);
          r11[10] = (v834_acc[2]);
          r11[11] = (v834_acc[3]);
          // s1 = store{r>s}(localShrMem0, r11);
          if (v33_g) {
            int32_t v844_off = v32_lead + 6;
            #pragma unroll
            for (int32_t v839_i1 = 0; v839_i1 < 12; ++v839_i1) {
              float v841_data = r11[v839_i1];
              int32_t v846_a = v844_off + (v839_i1 * 12);
              s1[(v846_a ^ ((v846_a >> 4) & 15))] = v841_data;
            }
          }
          // wait(r12 = load{g>r}(glb_m8););
          float r13[12]{};
          // ir13 = +(s1 * r12)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir13[12]{};
          float v852_data = r12[0];
          float v853_data = r12[1];
          float v854_data = r12[2];
          float v855_data = r12[3];
          float v856_tp{};
          float v857_tp{};
          float v858_tp{};
          float v859_tp{};
          tensorforge::transpose4x4b32(v856_tp, v857_tp, v858_tp, v859_tp, v852_data, v853_data, v854_data, v855_data);
          tensorforge::VectorT<float, 4> v860_acc{};
          float v867_data = s1[(v32_lead ^ v333_sw)];
          float v872_data = s1[(v336_a ^ (v337_sw & 15))];
          float v877_data = s1[(v341_a ^ (v342_sw & 15))];
          float v882_data = s1[(v346_a ^ (v347_sw & 15))];
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v867_data, v860_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v872_data, v883_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v858_tp, v877_data, v884_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v859_tp, v882_data, v885_acc, 2, 0, 0);
          float v891_data = s1[(v355_a ^ (v356_sw & 15))];
          float v896_data = s1[(v360_a ^ (v361_sw & 15))];
          float v901_data = s1[(v365_a ^ (v366_sw & 15))];
          float v906_data = s1[(v370_a ^ (v371_sw & 15))];
          tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v891_data, v886_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v908_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v896_data, v907_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v909_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v858_tp, v901_data, v908_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v910_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v859_tp, v906_data, v909_acc, 2, 1, 0);
          float v915_data = s1[(v379_a ^ (v380_sw & 15))];
          float v920_data = s1[(v384_a ^ (v385_sw & 15))];
          float v925_data = s1[(v389_a ^ (v390_sw & 15))];
          float v930_data = s1[(v394_a ^ (v395_sw & 15))];
          tensorforge::VectorT<float, 4> v931_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v856_tp, v915_data, v910_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v932_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v857_tp, v920_data, v931_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v933_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v858_tp, v925_data, v932_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v934_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v859_tp, v930_data, v933_acc, 2, 2, 0);
          ir13[0] = (v934_acc[0]);
          ir13[1] = (v934_acc[1]);
          ir13[2] = (v934_acc[2]);
          ir13[3] = (v934_acc[3]);
          float v939_data = r12[4];
          float v940_data = r12[5];
          float v941_data = r12[6];
          float v942_data = r12[7];
          float v943_tp{};
          float v944_tp{};
          float v945_tp{};
          float v946_tp{};
          tensorforge::transpose4x4b32(v943_tp, v944_tp, v945_tp, v946_tp, v939_data, v940_data, v941_data, v942_data);
          tensorforge::VectorT<float, 4> v947_acc{};
          float v954_data = s1[(v32_lead ^ v333_sw)];
          float v959_data = s1[(v336_a ^ (v337_sw & 15))];
          float v964_data = s1[(v341_a ^ (v342_sw & 15))];
          float v969_data = s1[(v346_a ^ (v347_sw & 15))];
          tensorforge::VectorT<float, 4> v970_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v954_data, v947_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v971_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v944_tp, v959_data, v970_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v972_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v945_tp, v964_data, v971_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v973_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v946_tp, v969_data, v972_acc, 2, 0, 0);
          float v978_data = s1[(v355_a ^ (v356_sw & 15))];
          float v983_data = s1[(v360_a ^ (v361_sw & 15))];
          float v988_data = s1[(v365_a ^ (v366_sw & 15))];
          float v993_data = s1[(v370_a ^ (v371_sw & 15))];
          tensorforge::VectorT<float, 4> v994_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v978_data, v973_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v995_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v944_tp, v983_data, v994_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v996_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v945_tp, v988_data, v995_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v997_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v946_tp, v993_data, v996_acc, 2, 1, 0);
          float v1002_data = s1[(v379_a ^ (v380_sw & 15))];
          float v1007_data = s1[(v384_a ^ (v385_sw & 15))];
          float v1012_data = s1[(v389_a ^ (v390_sw & 15))];
          float v1017_data = s1[(v394_a ^ (v395_sw & 15))];
          tensorforge::VectorT<float, 4> v1018_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v943_tp, v1002_data, v997_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1019_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v944_tp, v1007_data, v1018_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1020_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v945_tp, v1012_data, v1019_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1021_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v946_tp, v1017_data, v1020_acc, 2, 2, 0);
          ir13[4] = (v1021_acc[0]);
          ir13[5] = (v1021_acc[1]);
          ir13[6] = (v1021_acc[2]);
          ir13[7] = (v1021_acc[3]);
          float v1026_data = r12[8];
          float v1027_data = r12[9];
          float v1028_data = r12[10];
          float v1029_data = r12[11];
          float v1030_tp{};
          float v1031_tp{};
          float v1032_tp{};
          float v1033_tp{};
          tensorforge::transpose4x4b32(v1030_tp, v1031_tp, v1032_tp, v1033_tp, v1026_data, v1027_data, v1028_data, v1029_data);
          tensorforge::VectorT<float, 4> v1034_acc{};
          float v1041_data = s1[(v32_lead ^ v333_sw)];
          float v1046_data = s1[(v336_a ^ (v337_sw & 15))];
          float v1051_data = s1[(v341_a ^ (v342_sw & 15))];
          float v1056_data = s1[(v346_a ^ (v347_sw & 15))];
          tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1041_data, v1034_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1058_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1046_data, v1057_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1059_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1051_data, v1058_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v1060_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1056_data, v1059_acc, 2, 0, 0);
          float v1065_data = s1[(v355_a ^ (v356_sw & 15))];
          float v1070_data = s1[(v360_a ^ (v361_sw & 15))];
          float v1075_data = s1[(v365_a ^ (v366_sw & 15))];
          float v1080_data = s1[(v370_a ^ (v371_sw & 15))];
          tensorforge::VectorT<float, 4> v1081_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1065_data, v1060_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1082_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1070_data, v1081_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1083_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1075_data, v1082_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v1084_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1080_data, v1083_acc, 2, 1, 0);
          float v1089_data = s1[(v379_a ^ (v380_sw & 15))];
          float v1094_data = s1[(v384_a ^ (v385_sw & 15))];
          float v1099_data = s1[(v389_a ^ (v390_sw & 15))];
          float v1104_data = s1[(v394_a ^ (v395_sw & 15))];
          tensorforge::VectorT<float, 4> v1105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1089_data, v1084_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1094_data, v1105_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1099_data, v1106_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v1108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1104_data, v1107_acc, 2, 2, 0);
          ir13[8] = (v1108_acc[0]);
          ir13[9] = (v1108_acc[1]);
          ir13[10] = (v1108_acc[2]);
          ir13[11] = (v1108_acc[3]);
          // r13 = ir13 + r6
          if (v42_g) {
            #pragma unroll
            for (int32_t v1113_n1 = 0; v1113_n1 < 12; ++v1113_n1) {
              float v1115_data = ir13[v1113_n1];
              float v1116_data = r6[v1113_n1];
              r13[v1113_n1] = (v1116_data + v1115_data);
            }
          }
          // glb_m3 = store{r>g}(r13);
          if (v42_g) {
            #pragma unroll
            for (int32_t v1118_i1 = 0; v1118_i1 < 12; ++v1118_i1) {
              float v1120_data = r13[v1118_i1];
              glb_m3[(v32_lead + (v1118_i1 * 12))] = v1120_data;
            }
          }
        }
      }
    }
  }
}

