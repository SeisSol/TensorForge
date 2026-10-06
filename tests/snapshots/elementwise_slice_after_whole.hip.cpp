// === base name ===
kernel_e3b49c45024c315d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e3b49c45024c315d = {{16, 16, 1}, 16, 12, 1, 16, 25600, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e3b49c45024c315d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e3b49c45024c315d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e3b49c45024c315d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e3b49c45024c315d, block.x * block.y * block.z, 6400 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6400 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e3b49c45024c315d, block.x * block.y * block.z, 0));
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
void launcher_kernel_e3b49c45024c315d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e3b49c45024c315d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e3b49c45024c315d), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_e3b49c45024c315d, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e3b49c45024c315d(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float* tempShrMem = &localShrMem0[384];
      float * __restrict__ s0 = &localShrMem0[192];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v12_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v12_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v12_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v12_batchId0 * 72 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v12_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v12_batchId0 * 144 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v12_batchId0 * 48 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v12_batchId0 * 144 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v32_lead = threadIdx.x % 16;
          bool v33_g = v32_lead < 12;
          if (v33_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m0[(v32_lead + (v34_i1 * 12))]);
              r0[v34_i1] = v39_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v33_g) {
            #pragma unroll
            for (int32_t v42_i1 = 0; v42_i1 < 12; ++v42_i1) {
              float v47_data = __builtin_nontemporal_load(&glb_m1[(v32_lead + (v42_i1 * 12))]);
              r1[v42_i1] = v47_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          bool v50_g = v32_lead < 6;
          if (v50_g) {
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 12; ++v51_i1) {
              float v56_data = __builtin_nontemporal_load(&glb_m2[(v32_lead + (v51_i1 * 6))]);
              r3[v51_i1] = v56_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
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
          float r4[12]{};
          // r4 = load{g>r}(glb_m3);
          if (v33_g) {
            #pragma unroll
            for (int32_t v181_i1 = 0; v181_i1 < 12; ++v181_i1) {
              float v186_data = __builtin_nontemporal_load(&glb_m3[(v32_lead + (v181_i1 * 12))]);
              r4[v181_i1] = v186_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r6[12]{};
          // r6 = load{g>r}(glb_m4);
          if (v50_g) {
            #pragma unroll
            for (int32_t v189_i1 = 0; v189_i1 < 12; ++v189_i1) {
              float v194_data = __builtin_nontemporal_load(&glb_m4[(v32_lead + (v189_i1 * 6))]);
              r6[v189_i1] = v194_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[12]{};
          // r5 = +(r3 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v197_data = r4[0];
          float v198_data = r4[1];
          float v199_data = r4[2];
          float v200_data = r4[3];
          float v201_tp{};
          float v202_tp{};
          float v203_tp{};
          float v204_tp{};
          tensorforge::transpose4x4b32(v201_tp, v202_tp, v203_tp, v204_tp, v197_data, v198_data, v199_data, v200_data);
          tensorforge::VectorT<float, 4> v205_acc{};
          float v206_data = r3[0];
          float v207_data = r3[1];
          float v208_data = r3[2];
          float v209_data = r3[3];
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v206_data, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v207_data, v210_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v208_data, v211_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v209_data, v212_acc, 2, 0, 0);
          float v214_data = r3[4];
          float v215_data = r3[5];
          float v216_data = r3[6];
          float v217_data = r3[7];
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v214_data, v213_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v215_data, v218_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v216_data, v219_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v217_data, v220_acc, 2, 1, 0);
          float v222_data = r3[8];
          float v223_data = r3[9];
          float v224_data = r3[10];
          float v225_data = r3[11];
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v222_data, v221_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v223_data, v226_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v224_data, v227_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v225_data, v228_acc, 2, 2, 0);
          r5[0] = (v229_acc[0]);
          r5[1] = (v229_acc[1]);
          r5[2] = (v229_acc[2]);
          r5[3] = (v229_acc[3]);
          float v234_data = r4[4];
          float v235_data = r4[5];
          float v236_data = r4[6];
          float v237_data = r4[7];
          float v238_tp{};
          float v239_tp{};
          float v240_tp{};
          float v241_tp{};
          tensorforge::transpose4x4b32(v238_tp, v239_tp, v240_tp, v241_tp, v234_data, v235_data, v236_data, v237_data);
          tensorforge::VectorT<float, 4> v242_acc{};
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v206_data, v242_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v207_data, v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v208_data, v248_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v209_data, v249_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v214_data, v250_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v215_data, v255_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v216_data, v256_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v217_data, v257_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v222_data, v258_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v223_data, v263_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v224_data, v264_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v225_data, v265_acc, 2, 2, 0);
          r5[4] = (v266_acc[0]);
          r5[5] = (v266_acc[1]);
          r5[6] = (v266_acc[2]);
          r5[7] = (v266_acc[3]);
          float v271_data = r4[8];
          float v272_data = r4[9];
          float v273_data = r4[10];
          float v274_data = r4[11];
          float v275_tp{};
          float v276_tp{};
          float v277_tp{};
          float v278_tp{};
          tensorforge::transpose4x4b32(v275_tp, v276_tp, v277_tp, v278_tp, v271_data, v272_data, v273_data, v274_data);
          tensorforge::VectorT<float, 4> v279_acc{};
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v206_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v207_data, v284_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v208_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v209_data, v286_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v214_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v215_data, v292_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v216_data, v293_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v217_data, v294_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v222_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v223_data, v300_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v224_data, v301_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v225_data, v302_acc, 2, 2, 0);
          r5[8] = (v303_acc[0]);
          r5[9] = (v303_acc[1]);
          r5[10] = (v303_acc[2]);
          r5[11] = (v303_acc[3]);
          // s1 = store{r>s}(localShrMem0, r5);
          if (v50_g) {
            #pragma unroll
            for (int32_t v308_i1 = 0; v308_i1 < 12; ++v308_i1) {
              float v310_data = r5[v308_i1];
              int32_t v314_a = v32_lead + (v308_i1 * 12);
              s1[(v314_a ^ ((v314_a >> 4) & 15))] = v310_data;
            }
          }
          float r8[12]{};
          // r8 = load{g>r}(glb_m6);
          if (v33_g) {
            #pragma unroll
            for (int32_t v319_i1 = 0; v319_i1 < 12; ++v319_i1) {
              float v324_data = __builtin_nontemporal_load(&glb_m6[(v32_lead + (v319_i1 * 12))]);
              r8[v319_i1] = v324_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[12]{};
          // r7 = +(r6 * r4) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v331_tp{};
          float v332_tp{};
          float v333_tp{};
          float v334_tp{};
          tensorforge::transpose4x4b32(v331_tp, v332_tp, v333_tp, v334_tp, v197_data, v198_data, v199_data, v200_data);
          tensorforge::VectorT<float, 4> v335_acc{};
          float v336_data = r6[0];
          float v337_data = r6[1];
          float v338_data = r6[2];
          float v339_data = r6[3];
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v336_data, v335_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v337_data, v340_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v338_data, v341_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v339_data, v342_acc, 2, 0, 0);
          float v344_data = r6[4];
          float v345_data = r6[5];
          float v346_data = r6[6];
          float v347_data = r6[7];
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v344_data, v343_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v345_data, v348_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v346_data, v349_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v347_data, v350_acc, 2, 1, 0);
          float v352_data = r6[8];
          float v353_data = r6[9];
          float v354_data = r6[10];
          float v355_data = r6[11];
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v352_data, v351_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v353_data, v356_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v354_data, v357_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v355_data, v358_acc, 2, 2, 0);
          r7[0] = (v359_acc[0]);
          r7[1] = (v359_acc[1]);
          r7[2] = (v359_acc[2]);
          r7[3] = (v359_acc[3]);
          float v368_tp{};
          float v369_tp{};
          float v370_tp{};
          float v371_tp{};
          tensorforge::transpose4x4b32(v368_tp, v369_tp, v370_tp, v371_tp, v234_data, v235_data, v236_data, v237_data);
          tensorforge::VectorT<float, 4> v372_acc{};
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v336_data, v372_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v337_data, v377_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v338_data, v378_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v339_data, v379_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v344_data, v380_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v345_data, v385_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v346_data, v386_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v347_data, v387_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v352_data, v388_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v353_data, v393_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v354_data, v394_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v355_data, v395_acc, 2, 2, 0);
          r7[4] = (v396_acc[0]);
          r7[5] = (v396_acc[1]);
          r7[6] = (v396_acc[2]);
          r7[7] = (v396_acc[3]);
          float v405_tp{};
          float v406_tp{};
          float v407_tp{};
          float v408_tp{};
          tensorforge::transpose4x4b32(v405_tp, v406_tp, v407_tp, v408_tp, v271_data, v272_data, v273_data, v274_data);
          tensorforge::VectorT<float, 4> v409_acc{};
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v336_data, v409_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v337_data, v414_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v338_data, v415_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v339_data, v416_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v344_data, v417_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v345_data, v422_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v346_data, v423_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v347_data, v424_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v352_data, v425_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v406_tp, v353_data, v430_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v407_tp, v354_data, v431_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v355_data, v432_acc, 2, 2, 0);
          r7[8] = (v433_acc[0]);
          r7[9] = (v433_acc[1]);
          r7[10] = (v433_acc[2]);
          r7[11] = (v433_acc[3]);
          // s1 = store{r>s}(localShrMem0, r7);
          if (v50_g) {
            int32_t v443_off = v32_lead + 6;
            #pragma unroll
            for (int32_t v438_i1 = 0; v438_i1 < 12; ++v438_i1) {
              float v440_data = r7[v438_i1];
              int32_t v445_a = v443_off + (v438_i1 * 12);
              s1[(v445_a ^ ((v445_a >> 4) & 15))] = v440_data;
            }
          }
          float r11[12]{};
          // r11 = load{g>r}(glb_m8);
          if (v33_g) {
            #pragma unroll
            for (int32_t v450_i1 = 0; v450_i1 < 12; ++v450_i1) {
              float v455_data = __builtin_nontemporal_load(&glb_m8[(v32_lead + (v450_i1 * 12))]);
              r11[v450_i1] = v455_data;
            }
          }
          // wait(r8 = load{g>r}(glb_m6););
          float r9[12]{};
          // r9 = +(s1 * r8) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v458_data = r8[0];
          float v459_data = r8[1];
          float v460_data = r8[2];
          float v461_data = r8[3];
          float v462_tp{};
          float v463_tp{};
          float v464_tp{};
          float v465_tp{};
          tensorforge::transpose4x4b32(v462_tp, v463_tp, v464_tp, v465_tp, v458_data, v459_data, v460_data, v461_data);
          tensorforge::VectorT<float, 4> v466_acc{};
          int32_t v471_sw = (v32_lead >> 4) & 15;
          float v473_data = s1[(v32_lead ^ v471_sw)];
          int32_t v474_a = v32_lead + 12;
          int32_t v475_sw = v474_a >> 4;
          float v478_data = s1[(v474_a ^ (v475_sw & 15))];
          int32_t v479_a = v32_lead + 24;
          int32_t v480_sw = v479_a >> 4;
          float v483_data = s1[(v479_a ^ (v480_sw & 15))];
          int32_t v484_a = v32_lead + 36;
          int32_t v485_sw = v484_a >> 4;
          float v488_data = s1[(v484_a ^ (v485_sw & 15))];
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v473_data, v466_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v478_data, v489_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v464_tp, v483_data, v490_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v465_tp, v488_data, v491_acc, 2, 0, 0);
          int32_t v493_a = v32_lead + 48;
          int32_t v494_sw = v493_a >> 4;
          float v497_data = s1[(v493_a ^ (v494_sw & 15))];
          int32_t v498_a = v32_lead + 60;
          int32_t v499_sw = v498_a >> 4;
          float v502_data = s1[(v498_a ^ (v499_sw & 15))];
          int32_t v503_a = v32_lead + 72;
          int32_t v504_sw = v503_a >> 4;
          float v507_data = s1[(v503_a ^ (v504_sw & 15))];
          int32_t v508_a = v32_lead + 84;
          int32_t v509_sw = v508_a >> 4;
          float v512_data = s1[(v508_a ^ (v509_sw & 15))];
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v497_data, v492_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v502_data, v513_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v464_tp, v507_data, v514_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v465_tp, v512_data, v515_acc, 2, 1, 0);
          int32_t v517_a = v32_lead + 96;
          int32_t v518_sw = v517_a >> 4;
          float v521_data = s1[(v517_a ^ (v518_sw & 15))];
          int32_t v522_a = v32_lead + 108;
          int32_t v523_sw = v522_a >> 4;
          float v526_data = s1[(v522_a ^ (v523_sw & 15))];
          int32_t v527_a = v32_lead + 120;
          int32_t v528_sw = v527_a >> 4;
          float v531_data = s1[(v527_a ^ (v528_sw & 15))];
          int32_t v532_a = v32_lead + 132;
          int32_t v533_sw = v532_a >> 4;
          float v536_data = s1[(v532_a ^ (v533_sw & 15))];
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v521_data, v516_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v526_data, v537_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v464_tp, v531_data, v538_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v465_tp, v536_data, v539_acc, 2, 2, 0);
          r9[0] = (v540_acc[0]);
          r9[1] = (v540_acc[1]);
          r9[2] = (v540_acc[2]);
          r9[3] = (v540_acc[3]);
          float v545_data = r8[4];
          float v546_data = r8[5];
          float v547_data = r8[6];
          float v548_data = r8[7];
          float v549_tp{};
          float v550_tp{};
          float v551_tp{};
          float v552_tp{};
          tensorforge::transpose4x4b32(v549_tp, v550_tp, v551_tp, v552_tp, v545_data, v546_data, v547_data, v548_data);
          tensorforge::VectorT<float, 4> v553_acc{};
          float v560_data = s1[(v32_lead ^ v471_sw)];
          float v565_data = s1[(v474_a ^ (v475_sw & 15))];
          float v570_data = s1[(v479_a ^ (v480_sw & 15))];
          float v575_data = s1[(v484_a ^ (v485_sw & 15))];
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v560_data, v553_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v565_data, v576_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v570_data, v577_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v575_data, v578_acc, 2, 0, 0);
          float v584_data = s1[(v493_a ^ (v494_sw & 15))];
          float v589_data = s1[(v498_a ^ (v499_sw & 15))];
          float v594_data = s1[(v503_a ^ (v504_sw & 15))];
          float v599_data = s1[(v508_a ^ (v509_sw & 15))];
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v584_data, v579_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v589_data, v600_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v594_data, v601_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v599_data, v602_acc, 2, 1, 0);
          float v608_data = s1[(v517_a ^ (v518_sw & 15))];
          float v613_data = s1[(v522_a ^ (v523_sw & 15))];
          float v618_data = s1[(v527_a ^ (v528_sw & 15))];
          float v623_data = s1[(v532_a ^ (v533_sw & 15))];
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v608_data, v603_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v613_data, v624_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v618_data, v625_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v623_data, v626_acc, 2, 2, 0);
          r9[4] = (v627_acc[0]);
          r9[5] = (v627_acc[1]);
          r9[6] = (v627_acc[2]);
          r9[7] = (v627_acc[3]);
          float v632_data = r8[8];
          float v633_data = r8[9];
          float v634_data = r8[10];
          float v635_data = r8[11];
          float v636_tp{};
          float v637_tp{};
          float v638_tp{};
          float v639_tp{};
          tensorforge::transpose4x4b32(v636_tp, v637_tp, v638_tp, v639_tp, v632_data, v633_data, v634_data, v635_data);
          tensorforge::VectorT<float, 4> v640_acc{};
          float v647_data = s1[(v32_lead ^ v471_sw)];
          float v652_data = s1[(v474_a ^ (v475_sw & 15))];
          float v657_data = s1[(v479_a ^ (v480_sw & 15))];
          float v662_data = s1[(v484_a ^ (v485_sw & 15))];
          tensorforge::VectorT<float, 4> v663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v647_data, v640_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v652_data, v663_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v657_data, v664_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v662_data, v665_acc, 2, 0, 0);
          float v671_data = s1[(v493_a ^ (v494_sw & 15))];
          float v676_data = s1[(v498_a ^ (v499_sw & 15))];
          float v681_data = s1[(v503_a ^ (v504_sw & 15))];
          float v686_data = s1[(v508_a ^ (v509_sw & 15))];
          tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v671_data, v666_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v676_data, v687_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v689_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v681_data, v688_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v686_data, v689_acc, 2, 1, 0);
          float v695_data = s1[(v517_a ^ (v518_sw & 15))];
          float v700_data = s1[(v522_a ^ (v523_sw & 15))];
          float v705_data = s1[(v527_a ^ (v528_sw & 15))];
          float v710_data = s1[(v532_a ^ (v533_sw & 15))];
          tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v636_tp, v695_data, v690_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v637_tp, v700_data, v711_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v713_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v638_tp, v705_data, v712_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v710_data, v713_acc, 2, 2, 0);
          r9[8] = (v714_acc[0]);
          r9[9] = (v714_acc[1]);
          r9[10] = (v714_acc[2]);
          r9[11] = (v714_acc[3]);
          float r10[12]{};
          // r10 = abs(glb_m7)
          bool v720_g = v32_lead < 4;
          if (v720_g) {
            #pragma unroll
            for (int32_t v721_k1 = 0; v721_k1 < 12; ++v721_k1) {
              float v726_data = glb_m7[(v32_lead + (v721_k1 * 4))];
              r10[v721_k1] = (fabsf(v726_data));
            }
          }
          // s0 = store{r>s}(localShrMem0, r10);
          if (v720_g) {
            #pragma unroll
            for (int32_t v730_i1 = 0; v730_i1 < 12; ++v730_i1) {
              float v732_data = r10[v730_i1];
              int32_t v736_a = v32_lead + (v730_i1 * 12);
              s0[(v736_a ^ ((v736_a >> 4) & 15))] = v732_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m8););
          float r12[12]{};
          // ir12 = +(r11 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir12[12]{};
          float v748_data = s0[(v32_lead ^ v471_sw)];
          float v753_data = s0[(v474_a ^ (v475_sw & 15))];
          float v758_data = s0[(v479_a ^ (v480_sw & 15))];
          float v763_data = s0[(v484_a ^ (v485_sw & 15))];
          float v764_tp{};
          float v765_tp{};
          float v766_tp{};
          float v767_tp{};
          tensorforge::transpose4x4b32(v764_tp, v765_tp, v766_tp, v767_tp, v748_data, v753_data, v758_data, v763_data);
          tensorforge::VectorT<float, 4> v768_acc{};
          float v769_data = r11[0];
          float v770_data = r11[1];
          float v771_data = r11[2];
          float v772_data = r11[3];
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v769_data, v768_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v770_data, v773_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v771_data, v774_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v772_data, v775_acc, 2, 0, 0);
          float v777_data = r11[4];
          float v778_data = r11[5];
          float v779_data = r11[6];
          float v780_data = r11[7];
          tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v777_data, v776_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v778_data, v781_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v779_data, v782_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v780_data, v783_acc, 2, 1, 0);
          float v785_data = r11[8];
          float v786_data = r11[9];
          float v787_data = r11[10];
          float v788_data = r11[11];
          tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v764_tp, v785_data, v784_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v765_tp, v786_data, v789_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v766_tp, v787_data, v790_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v767_tp, v788_data, v791_acc, 2, 2, 0);
          ir12[0] = (v792_acc[0]);
          ir12[1] = (v792_acc[1]);
          ir12[2] = (v792_acc[2]);
          ir12[3] = (v792_acc[3]);
          float v803_data = s0[(v493_a ^ (v494_sw & 15))];
          float v808_data = s0[(v498_a ^ (v499_sw & 15))];
          float v813_data = s0[(v503_a ^ (v504_sw & 15))];
          float v818_data = s0[(v508_a ^ (v509_sw & 15))];
          float v819_tp{};
          float v820_tp{};
          float v821_tp{};
          float v822_tp{};
          tensorforge::transpose4x4b32(v819_tp, v820_tp, v821_tp, v822_tp, v803_data, v808_data, v813_data, v818_data);
          tensorforge::VectorT<float, 4> v823_acc{};
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v819_tp, v769_data, v823_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v829_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v820_tp, v770_data, v828_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v821_tp, v771_data, v829_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v822_tp, v772_data, v830_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v836_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v819_tp, v777_data, v831_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v837_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v820_tp, v778_data, v836_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v838_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v821_tp, v779_data, v837_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v839_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v822_tp, v780_data, v838_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v844_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v819_tp, v785_data, v839_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v845_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v820_tp, v786_data, v844_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v846_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v821_tp, v787_data, v845_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v847_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v822_tp, v788_data, v846_acc, 2, 2, 0);
          ir12[4] = (v847_acc[0]);
          ir12[5] = (v847_acc[1]);
          ir12[6] = (v847_acc[2]);
          ir12[7] = (v847_acc[3]);
          float v858_data = s0[(v517_a ^ (v518_sw & 15))];
          float v863_data = s0[(v522_a ^ (v523_sw & 15))];
          float v868_data = s0[(v527_a ^ (v528_sw & 15))];
          float v873_data = s0[(v532_a ^ (v533_sw & 15))];
          float v874_tp{};
          float v875_tp{};
          float v876_tp{};
          float v877_tp{};
          tensorforge::transpose4x4b32(v874_tp, v875_tp, v876_tp, v877_tp, v858_data, v863_data, v868_data, v873_data);
          tensorforge::VectorT<float, 4> v878_acc{};
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v874_tp, v769_data, v878_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v875_tp, v770_data, v883_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v876_tp, v771_data, v884_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v877_tp, v772_data, v885_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v891_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v874_tp, v777_data, v886_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v892_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v875_tp, v778_data, v891_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v893_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v876_tp, v779_data, v892_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v894_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v877_tp, v780_data, v893_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v899_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v874_tp, v785_data, v894_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v900_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v875_tp, v786_data, v899_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v901_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v876_tp, v787_data, v900_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v902_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v877_tp, v788_data, v901_acc, 2, 2, 0);
          ir12[8] = (v902_acc[0]);
          ir12[9] = (v902_acc[1]);
          ir12[10] = (v902_acc[2]);
          ir12[11] = (v902_acc[3]);
          // r12 = ir12 + r9
          if (v33_g) {
            #pragma unroll
            for (int32_t v907_n1 = 0; v907_n1 < 12; ++v907_n1) {
              float v909_data = ir12[v907_n1];
              float v910_data = r9[v907_n1];
              r12[v907_n1] = (v910_data + v909_data);
            }
          }
          // glb_m5 = store{r>g}(r12);
          if (v33_g) {
            #pragma unroll
            for (int32_t v912_i1 = 0; v912_i1 < 12; ++v912_i1) {
              float v914_data = r12[v912_i1];
              glb_m5[(v32_lead + (v912_i1 * 12))] = v914_data;
            }
          }
        }
      }
    }
  }
}

