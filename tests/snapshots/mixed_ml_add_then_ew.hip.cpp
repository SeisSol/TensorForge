// === base name ===
kernel_d458bf8f639fab69

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d458bf8f639fab69 = {{8, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d458bf8f639fab69(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d458bf8f639fab69(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d458bf8f639fab69(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 32, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d458bf8f639fab69, block.x * block.y * block.z, 2304 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (2304 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d458bf8f639fab69, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (2304 * sizeof(float)));
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
  config.block[0] = 8;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_d458bf8f639fab69(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d458bf8f639fab69(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d458bf8f639fab69), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_d458bf8f639fab69, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d458bf8f639fab69(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 32 per block = block 8x32x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    //   m4 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t0[i,j] += m2[i,k] × m3[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v5_batchId0 * 64 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v25_lead = v21_lead + (v22_i0 * 8);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v25_lead + (v23_i1 * 8))]);
              r0[(v22_i0 + v23_i1)] = v28_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v34_lead = v21_lead + (v31_i0 * 8);
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v34_lead + (v32_i1 * 8))]);
              r1[(v31_i0 + v32_i1)] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[8]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
            int32_t v43_lead = v21_lead + (v40_i0 * 8);
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 8; ++v41_i1) {
              float v46_data = __builtin_nontemporal_load(&glb_m2[(v43_lead + (v41_i1 * 8))]);
              r3[(v40_i0 + v41_i1)] = v46_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v49_data = r1[0];
          float v50_data = r1[1];
          float v51_data = r1[2];
          float v52_data = r1[3];
          float v53_tp{};
          float v54_tp{};
          float v55_tp{};
          float v56_tp{};
          tensorforge::transpose4x4b32(v53_tp, v54_tp, v55_tp, v56_tp, v49_data, v50_data, v51_data, v52_data);
          tensorforge::VectorT<float, 4> v57_acc{};
          float v58_data = r0[0];
          float v59_data = r0[1];
          float v60_data = r0[2];
          float v61_data = r0[3];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v57_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v62_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v63_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 1, 0, 0);
          float v66_data = r0[4];
          float v67_data = r0[5];
          float v68_data = r0[6];
          float v69_data = r0[7];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v65_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v71_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 1, 1, 0);
          r2[0] = (v73_acc[0]);
          r2[1] = (v73_acc[1]);
          r2[2] = (v73_acc[2]);
          r2[3] = (v73_acc[3]);
          float v78_data = r1[4];
          float v79_data = r1[5];
          float v80_data = r1[6];
          float v81_data = r1[7];
          float v82_tp{};
          float v83_tp{};
          float v84_tp{};
          float v85_tp{};
          tensorforge::transpose4x4b32(v82_tp, v83_tp, v84_tp, v85_tp, v78_data, v79_data, v80_data, v81_data);
          tensorforge::VectorT<float, 4> v86_acc{};
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v91_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v92_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v93_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v66_data, v94_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v99_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v100_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v101_acc, 1, 1, 0);
          r2[4] = (v102_acc[0]);
          r2[5] = (v102_acc[1]);
          r2[6] = (v102_acc[2]);
          r2[7] = (v102_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v108_i0 = 0; v108_i0 < 1; ++v108_i0) {
            int32_t v111_lead = v21_lead + (v108_i0 * 8);
            #pragma unroll
            for (int32_t v109_i1 = 0; v109_i1 < 8; ++v109_i1) {
              float v114_data = __builtin_nontemporal_load(&glb_m3[(v111_lead + (v109_i1 * 8))]);
              r4[(v108_i0 + v109_i1)] = v114_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          // wait(r4 = load{g>r}(glb_m3););
          float r5[8]{};
          // ir5 = +(r3 * r4)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir5[8]{};
          float v118_data = r4[0];
          float v119_data = r4[1];
          float v120_data = r4[2];
          float v121_data = r4[3];
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          tensorforge::transpose4x4b32(v122_tp, v123_tp, v124_tp, v125_tp, v118_data, v119_data, v120_data, v121_data);
          tensorforge::VectorT<float, 4> v126_acc{};
          float v127_data = r3[0];
          float v128_data = r3[1];
          float v129_data = r3[2];
          float v130_data = r3[3];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v127_data, v126_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v128_data, v131_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v129_data, v132_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v130_data, v133_acc, 1, 0, 0);
          float v135_data = r3[4];
          float v136_data = r3[5];
          float v137_data = r3[6];
          float v138_data = r3[7];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v135_data, v134_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v136_data, v139_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v137_data, v140_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v138_data, v141_acc, 1, 1, 0);
          ir5[0] = (v142_acc[0]);
          ir5[1] = (v142_acc[1]);
          ir5[2] = (v142_acc[2]);
          ir5[3] = (v142_acc[3]);
          float v147_data = r4[4];
          float v148_data = r4[5];
          float v149_data = r4[6];
          float v150_data = r4[7];
          float v151_tp{};
          float v152_tp{};
          float v153_tp{};
          float v154_tp{};
          tensorforge::transpose4x4b32(v151_tp, v152_tp, v153_tp, v154_tp, v147_data, v148_data, v149_data, v150_data);
          tensorforge::VectorT<float, 4> v155_acc{};
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v127_data, v155_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v128_data, v160_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v129_data, v161_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v130_data, v162_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v135_data, v163_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v136_data, v168_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v137_data, v169_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v138_data, v170_acc, 1, 1, 0);
          ir5[4] = (v171_acc[0]);
          ir5[5] = (v171_acc[1]);
          ir5[6] = (v171_acc[2]);
          ir5[7] = (v171_acc[3]);
          // r5 = ir5 + r2
          #pragma unroll
          for (int32_t v176_n0 = 0; v176_n0 < 1; ++v176_n0) {
            #pragma unroll
            for (int32_t v177_n1 = 0; v177_n1 < 8; ++v177_n1) {
              int32_t v178_a = v176_n0 + v177_n1;
              float v179_data = ir5[v178_a];
              float v180_data = r2[v178_a];
              r5[v178_a] = (v180_data + v179_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r5);
          #pragma unroll
          for (int32_t v182_i0 = 0; v182_i0 < 1; ++v182_i0) {
            int32_t v187_lead = v21_lead + (v182_i0 * 8);
            #pragma unroll
            for (int32_t v183_i1 = 0; v183_i1 < 8; ++v183_i1) {
              float v185_data = r5[(v182_i0 + v183_i1)];
              int32_t v189_a = v187_lead + (v183_i1 * 8);
              s0[(v189_a ^ ((v189_a >> 5) & 31))] = v185_data;
            }
          }
          // glb_m4 = abs(s0)
          #pragma unroll
          for (int32_t v193_k0 = 0; v193_k0 < 1; ++v193_k0) {
            int32_t v196_lead = v21_lead + (v193_k0 * 8);
            #pragma unroll
            for (int32_t v194_k1 = 0; v194_k1 < 8; ++v194_k1) {
              int32_t v198_a = v196_lead + (v194_k1 * 8);
              float v202_data = s0[(v198_a ^ ((v198_a >> 5) & 31))];
              glb_m4[v198_a] = (fabsf(v202_data));
            }
          }
        }
      }
    }
  }
}

