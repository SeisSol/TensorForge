// === base name ===
kernel_87bfe0a56990a2b3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_87bfe0a56990a2b3 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_87bfe0a56990a2b3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_87bfe0a56990a2b3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_87bfe0a56990a2b3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_87bfe0a56990a2b3, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_87bfe0a56990a2b3, block.x * block.y * block.z, 0));
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
void launcher_kernel_87bfe0a56990a2b3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_87bfe0a56990a2b3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_87bfe0a56990a2b3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_87bfe0a56990a2b3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_87bfe0a56990a2b3(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 144 + 0 + m3_extraOffset];
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
          // r3 = load{g>r}(glb_m3);
          if (v27_g) {
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 12; ++v44_i1) {
              float v49_data = __builtin_nontemporal_load(&glb_m3[(v26_lead + (v44_i1 * 12))]);
              r3[v44_i1] = v49_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
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
          float v92_tp{};
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          tensorforge::transpose4x4b32(v92_tp, v93_tp, v94_tp, v95_tp, v89_data, v90_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v96_acc{};
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v61_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v62_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v63_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v64_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v69_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v70_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v71_data, v110_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v72_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v77_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v78_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v79_data, v118_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v80_data, v119_acc, 2, 2, 0);
          r2[4] = (v120_acc[0]);
          r2[5] = (v120_acc[1]);
          // s0 = store{r>s, clear}(localShrMem0, r2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v123_z1 = 6; v123_z1 < 12; ++v123_z1) {
              int32_t v128_a = v26_lead + (v123_z1 * 12);
              s0[(v128_a ^ ((v128_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v27_g) {
            #pragma unroll
            for (int32_t v132_i1 = 0; v132_i1 < 6; ++v132_i1) {
              float v134_data = r2[v132_i1];
              int32_t v138_a = v26_lead + (v132_i1 * 12);
              s0[(v138_a ^ ((v138_a >> 4) & 15))] = v134_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = +(r3 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v149_data = s0[(v26_lead ^ ((v26_lead >> 4) & 15))];
          int32_t v150_a = v26_lead + 12;
          float v154_data = s0[(v150_a ^ ((v150_a >> 4) & 15))];
          int32_t v155_a = v26_lead + 24;
          float v159_data = s0[(v155_a ^ ((v155_a >> 4) & 15))];
          int32_t v160_a = v26_lead + 36;
          float v164_data = s0[(v160_a ^ ((v160_a >> 4) & 15))];
          float v165_tp{};
          float v166_tp{};
          float v167_tp{};
          float v168_tp{};
          tensorforge::transpose4x4b32(v165_tp, v166_tp, v167_tp, v168_tp, v149_data, v154_data, v159_data, v164_data);
          tensorforge::VectorT<float, 4> v169_acc{};
          float v170_data = r3[0];
          float v171_data = r3[1];
          float v172_data = r3[2];
          float v173_data = r3[3];
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v170_data, v169_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v171_data, v174_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v167_tp, v172_data, v175_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v168_tp, v173_data, v176_acc, 2, 0, 0);
          float v178_data = r3[4];
          float v179_data = r3[5];
          float v180_data = r3[6];
          float v181_data = r3[7];
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v178_data, v177_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v179_data, v182_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v167_tp, v180_data, v183_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v168_tp, v181_data, v184_acc, 2, 1, 0);
          float v186_data = r3[8];
          float v187_data = r3[9];
          float v188_data = r3[10];
          float v189_data = r3[11];
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v186_data, v185_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v166_tp, v187_data, v190_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v167_tp, v188_data, v191_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v168_tp, v189_data, v192_acc, 2, 2, 0);
          r4[0] = (v193_acc[0]);
          r4[1] = (v193_acc[1]);
          r4[2] = (v193_acc[2]);
          r4[3] = (v193_acc[3]);
          int32_t v200_a = v26_lead + 48;
          float v204_data = s0[(v200_a ^ ((v200_a >> 4) & 15))];
          int32_t v205_a = v26_lead + 60;
          float v209_data = s0[(v205_a ^ ((v205_a >> 4) & 15))];
          int32_t v210_a = v26_lead + 72;
          float v214_data = s0[(v210_a ^ ((v210_a >> 4) & 15))];
          int32_t v215_a = v26_lead + 84;
          float v219_data = s0[(v215_a ^ ((v215_a >> 4) & 15))];
          float v220_tp{};
          float v221_tp{};
          float v222_tp{};
          float v223_tp{};
          tensorforge::transpose4x4b32(v220_tp, v221_tp, v222_tp, v223_tp, v204_data, v209_data, v214_data, v219_data);
          tensorforge::VectorT<float, 4> v224_acc{};
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v170_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v171_data, v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v172_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v173_data, v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v178_data, v232_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v179_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v180_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v181_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v186_data, v240_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v187_data, v245_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v188_data, v246_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v189_data, v247_acc, 2, 2, 0);
          r4[4] = (v248_acc[0]);
          r4[5] = (v248_acc[1]);
          r4[6] = (v248_acc[2]);
          r4[7] = (v248_acc[3]);
          int32_t v255_a = v26_lead + 96;
          float v259_data = s0[(v255_a ^ ((v255_a >> 4) & 15))];
          int32_t v260_a = v26_lead + 108;
          float v264_data = s0[(v260_a ^ ((v260_a >> 4) & 15))];
          int32_t v265_a = v26_lead + 120;
          float v269_data = s0[(v265_a ^ ((v265_a >> 4) & 15))];
          int32_t v270_a = v26_lead + 132;
          float v274_data = s0[(v270_a ^ ((v270_a >> 4) & 15))];
          float v275_tp{};
          float v276_tp{};
          float v277_tp{};
          float v278_tp{};
          tensorforge::transpose4x4b32(v275_tp, v276_tp, v277_tp, v278_tp, v259_data, v264_data, v269_data, v274_data);
          tensorforge::VectorT<float, 4> v279_acc{};
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v170_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v171_data, v284_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v172_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v173_data, v286_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v178_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v179_data, v292_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v180_data, v293_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v181_data, v294_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v186_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v187_data, v300_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v188_data, v301_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v189_data, v302_acc, 2, 2, 0);
          r4[8] = (v303_acc[0]);
          r4[9] = (v303_acc[1]);
          r4[10] = (v303_acc[2]);
          r4[11] = (v303_acc[3]);
          // glb_m2 = store{r>g}(r4);
          if (v27_g) {
            #pragma unroll
            for (int32_t v308_i1 = 0; v308_i1 < 12; ++v308_i1) {
              float v310_data = r4[v308_i1];
              glb_m2[(v26_lead + (v308_i1 * 12))] = v310_data;
            }
          }
        }
      }
    }
  }
}

