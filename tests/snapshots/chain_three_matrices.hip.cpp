// === base name ===
kernel_756cf29f5098135d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_756cf29f5098135d = {{32, 8, 1}, 32, 56, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_756cf29f5098135d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_756cf29f5098135d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_756cf29f5098135d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_756cf29f5098135d, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_756cf29f5098135d, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_756cf29f5098135d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_756cf29f5098135d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_756cf29f5098135d), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_756cf29f5098135d, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_756cf29f5098135d(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 56×9(56×9) {0..56}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 56×9(56×9) {0..56}×{0..9} strided
    //   m3 56×56(56×56) {0..56}×{0..56} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[56,9]],"name":"m0","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[56,9]],"name":"m2","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[56,56]],"name":"m3","ordered":false,"parts":1,"shape":[56,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[56,56]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,56]},{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 504 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 81 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 504 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 3136 + 0 + m3_extraOffset];
          float r0[18]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v20_lead = v16_lead + (v17_i0 * 32);
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 9; ++v18_i1) {
              float v23_data = __builtin_nontemporal_load(&glb_m0[(v20_lead + (v18_i1 * 56))]);
              r0[(v17_i0 + (v18_i1 * 2))] = v23_data;
            }
          }
          bool v26_g = v16_lead < 24;
          if (v26_g) {
            int32_t v29_lead = v16_lead + 32_i32;
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m0[(v29_lead + (v27_i1 * 56))]);
              r0[(1 + (v27_i1 * 2))] = v32_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m1);
          if (v16_lead < 9) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 9; ++v37_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v37_i1 * 9))]);
              r1[v37_i1] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[112]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
            int32_t v48_lead = v16_lead + (v45_i0 * 32);
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 56; ++v46_i1) {
              float v51_data = __builtin_nontemporal_load(&glb_m3[(v48_lead + (v46_i1 * 56))]);
              r3[(v45_i0 + (v46_i1 * 2))] = v51_data;
            }
          }
          if (v26_g) {
            int32_t v56_lead = v16_lead + 32_i32;
            #pragma unroll
            for (int32_t v54_i1 = 0; v54_i1 < 56; ++v54_i1) {
              float v59_data = __builtin_nontemporal_load(&glb_m3[(v56_lead + (v54_i1 * 56))]);
              r3[(1 + (v54_i1 * 2))] = v59_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[18]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 9)] [(0, 9)]
          float v63_data = r1[0];
          float v64_data = r1[1];
          float v65_data = r1[2];
          float v66_data = r1[3];
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          float v70_tp{};
          tensorforge::transpose4x4b32(v67_tp, v68_tp, v69_tp, v70_tp, v63_data, v64_data, v65_data, v66_data);
          tensorforge::VectorT<float, 4> v71_acc{};
          float v72_data = r0[0];
          float v73_data = r0[2];
          float v74_data = r0[4];
          float v75_data = r0[6];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v71_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v77_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v75_data, v78_acc, 3, 0, 0);
          float v80_data = r0[8];
          float v81_data = r0[10];
          float v82_data = r0[12];
          float v83_data = r0[14];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v79_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v85_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v83_data, v86_acc, 3, 1, 0);
          float v88_data = r0[16];
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v88_data, v87_acc, 3, 2, 0);
          r2[0] = (v90_acc[0]);
          r2[2] = (v90_acc[1]);
          r2[4] = (v90_acc[2]);
          r2[6] = (v90_acc[3]);
          tensorforge::VectorT<float, 4> v95_acc{};
          float v96_data = r0[1];
          float v97_data = r0[3];
          float v98_data = r0[5];
          float v99_data = r0[7];
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v96_data, v95_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v97_data, v100_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v98_data, v101_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v99_data, v102_acc, 3, 0, 0);
          float v104_data = r0[9];
          float v105_data = r0[11];
          float v106_data = r0[13];
          float v107_data = r0[15];
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v104_data, v103_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v105_data, v108_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v106_data, v109_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v107_data, v110_acc, 3, 1, 0);
          float v112_data = r0[17];
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v112_data, v111_acc, 3, 2, 0);
          r2[1] = (v114_acc[0]);
          r2[3] = (v114_acc[1]);
          r2[5] = (v114_acc[2]);
          r2[7] = (v114_acc[3]);
          float v119_data = r1[4];
          float v120_data = r1[5];
          float v121_data = r1[6];
          float v122_data = r1[7];
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          tensorforge::transpose4x4b32(v123_tp, v124_tp, v125_tp, v126_tp, v119_data, v120_data, v121_data, v122_data);
          tensorforge::VectorT<float, 4> v127_acc{};
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v72_data, v127_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v73_data, v132_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v74_data, v133_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v75_data, v134_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v80_data, v135_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v81_data, v140_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v82_data, v141_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v83_data, v142_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v88_data, v143_acc, 3, 2, 0);
          r2[8] = (v146_acc[0]);
          r2[10] = (v146_acc[1]);
          r2[12] = (v146_acc[2]);
          r2[14] = (v146_acc[3]);
          tensorforge::VectorT<float, 4> v151_acc{};
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v96_data, v151_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v97_data, v156_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v98_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v99_data, v158_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v104_data, v159_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v105_data, v164_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v106_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v107_data, v166_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v112_data, v167_acc, 3, 2, 0);
          r2[9] = (v170_acc[0]);
          r2[11] = (v170_acc[1]);
          r2[13] = (v170_acc[2]);
          r2[15] = (v170_acc[3]);
          float v193_acc{};
          float v194_acc{};
          float v195_data = r1[8];
          float v196_bc = tensorforge::broadcast<32, 16, 0>(v195_data);
          tensorforge::fmacdpp16<0>(v193_acc, v196_bc, v72_data);
          tensorforge::fmacdpp16<0>(v194_acc, v196_bc, v96_data);
          tensorforge::fmacdpp16<1>(v193_acc, v196_bc, v73_data);
          tensorforge::fmacdpp16<1>(v194_acc, v196_bc, v97_data);
          tensorforge::fmacdpp16<2>(v193_acc, v196_bc, v74_data);
          tensorforge::fmacdpp16<2>(v194_acc, v196_bc, v98_data);
          tensorforge::fmacdpp16<3>(v193_acc, v196_bc, v75_data);
          tensorforge::fmacdpp16<3>(v194_acc, v196_bc, v99_data);
          tensorforge::fmacdpp16<4>(v193_acc, v196_bc, v80_data);
          tensorforge::fmacdpp16<4>(v194_acc, v196_bc, v104_data);
          tensorforge::fmacdpp16<5>(v193_acc, v196_bc, v81_data);
          tensorforge::fmacdpp16<5>(v194_acc, v196_bc, v105_data);
          tensorforge::fmacdpp16<6>(v193_acc, v196_bc, v82_data);
          tensorforge::fmacdpp16<6>(v194_acc, v196_bc, v106_data);
          tensorforge::fmacdpp16<7>(v193_acc, v196_bc, v83_data);
          tensorforge::fmacdpp16<7>(v194_acc, v196_bc, v107_data);
          tensorforge::fmacdpp16<8>(v193_acc, v196_bc, v88_data);
          tensorforge::fmacdpp16<8>(v194_acc, v196_bc, v112_data);
          r2[16] = v193_acc;
          r2[17] = v194_acc;
          // wait(r3 = load{g>r}(glb_m3););
          float r4[18]{};
          // r4 = +(r3 * r2) + None
          // [(0, 56), (0, 9)] [(0, 56)]
          float v198_data = r2[0];
          float v199_data = r2[2];
          float v200_data = r2[4];
          float v201_data = r2[6];
          float v202_tp{};
          float v203_tp{};
          float v204_tp{};
          float v205_tp{};
          tensorforge::transpose4x4b32(v202_tp, v203_tp, v204_tp, v205_tp, v198_data, v199_data, v200_data, v201_data);
          float v206_data = r2[1];
          float v207_data = r2[3];
          float v208_data = r2[5];
          float v209_data = r2[7];
          float v210_tp{};
          float v211_tp{};
          float v212_tp{};
          float v213_tp{};
          tensorforge::transpose4x4b32(v210_tp, v211_tp, v212_tp, v213_tp, v206_data, v207_data, v208_data, v209_data);
          tensorforge::VectorT<float, 4> v214_acc{};
          float v215_data = r3[0];
          float v216_data = r3[2];
          float v217_data = r3[4];
          float v218_data = r3[6];
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v215_data, v214_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v216_data, v219_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v217_data, v220_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v218_data, v221_acc, 3, 0, 0);
          float v223_data = r3[8];
          float v224_data = r3[10];
          float v225_data = r3[12];
          float v226_data = r3[14];
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v223_data, v222_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v224_data, v227_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v225_data, v228_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v226_data, v229_acc, 3, 1, 0);
          float v231_data = r3[16];
          float v232_data = r3[18];
          float v233_data = r3[20];
          float v234_data = r3[22];
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v231_data, v230_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v232_data, v235_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v233_data, v236_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v234_data, v237_acc, 3, 2, 0);
          float v239_data = r3[24];
          float v240_data = r3[26];
          float v241_data = r3[28];
          float v242_data = r3[30];
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v239_data, v238_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v240_data, v243_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v241_data, v244_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v242_data, v245_acc, 3, 3, 0);
          float v247_data = r3[32];
          float v248_data = r3[34];
          float v249_data = r3[36];
          float v250_data = r3[38];
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v247_data, v246_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v248_data, v251_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v249_data, v252_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v250_data, v253_acc, 3, 4, 0);
          float v255_data = r3[40];
          float v256_data = r3[42];
          float v257_data = r3[44];
          float v258_data = r3[46];
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v255_data, v254_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v256_data, v259_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v257_data, v260_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v258_data, v261_acc, 3, 5, 0);
          float v263_data = r3[48];
          float v264_data = r3[50];
          float v265_data = r3[52];
          float v266_data = r3[54];
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v263_data, v262_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v264_data, v267_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v265_data, v268_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v266_data, v269_acc, 3, 6, 0);
          float v271_data = r3[56];
          float v272_data = r3[58];
          float v273_data = r3[60];
          float v274_data = r3[62];
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v271_data, v270_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v272_data, v275_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v273_data, v276_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v274_data, v277_acc, 3, 7, 0);
          float v279_data = r3[64];
          float v280_data = r3[66];
          float v281_data = r3[68];
          float v282_data = r3[70];
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v279_data, v278_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v280_data, v283_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v281_data, v284_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v282_data, v285_acc, 3, 0, 0);
          float v287_data = r3[72];
          float v288_data = r3[74];
          float v289_data = r3[76];
          float v290_data = r3[78];
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v287_data, v286_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v288_data, v291_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v289_data, v292_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v290_data, v293_acc, 3, 1, 0);
          float v295_data = r3[80];
          float v296_data = r3[82];
          float v297_data = r3[84];
          float v298_data = r3[86];
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v295_data, v294_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v296_data, v299_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v297_data, v300_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v298_data, v301_acc, 3, 2, 0);
          float v303_data = r3[88];
          float v304_data = r3[90];
          float v305_data = r3[92];
          float v306_data = r3[94];
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v303_data, v302_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v304_data, v307_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v305_data, v308_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v306_data, v309_acc, 3, 3, 0);
          float v311_data = r3[96];
          float v312_data = r3[98];
          float v313_data = r3[100];
          float v314_data = r3[102];
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v311_data, v310_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v312_data, v315_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v313_data, v316_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v314_data, v317_acc, 3, 4, 0);
          float v319_data = r3[104];
          float v320_data = r3[106];
          float v321_data = r3[108];
          float v322_data = r3[110];
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v319_data, v318_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v320_data, v323_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v321_data, v324_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v322_data, v325_acc, 3, 5, 0);
          r4[0] = (v326_acc[0]);
          r4[2] = (v326_acc[1]);
          r4[4] = (v326_acc[2]);
          r4[6] = (v326_acc[3]);
          tensorforge::VectorT<float, 4> v331_acc{};
          float v332_data = r3[1];
          float v333_data = r3[3];
          float v334_data = r3[5];
          float v335_data = r3[7];
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v332_data, v331_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v333_data, v336_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v334_data, v337_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v335_data, v338_acc, 3, 0, 0);
          float v340_data = r3[9];
          float v341_data = r3[11];
          float v342_data = r3[13];
          float v343_data = r3[15];
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v340_data, v339_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v341_data, v344_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v342_data, v345_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v343_data, v346_acc, 3, 1, 0);
          float v348_data = r3[17];
          float v349_data = r3[19];
          float v350_data = r3[21];
          float v351_data = r3[23];
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v348_data, v347_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v349_data, v352_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v350_data, v353_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v351_data, v354_acc, 3, 2, 0);
          float v356_data = r3[25];
          float v357_data = r3[27];
          float v358_data = r3[29];
          float v359_data = r3[31];
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v356_data, v355_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v357_data, v360_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v358_data, v361_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v359_data, v362_acc, 3, 3, 0);
          float v364_data = r3[33];
          float v365_data = r3[35];
          float v366_data = r3[37];
          float v367_data = r3[39];
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v364_data, v363_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v365_data, v368_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v366_data, v369_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v367_data, v370_acc, 3, 4, 0);
          float v372_data = r3[41];
          float v373_data = r3[43];
          float v374_data = r3[45];
          float v375_data = r3[47];
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v372_data, v371_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v373_data, v376_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v374_data, v377_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v375_data, v378_acc, 3, 5, 0);
          float v380_data = r3[49];
          float v381_data = r3[51];
          float v382_data = r3[53];
          float v383_data = r3[55];
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v380_data, v379_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v381_data, v384_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v382_data, v385_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v383_data, v386_acc, 3, 6, 0);
          float v388_data = r3[57];
          float v389_data = r3[59];
          float v390_data = r3[61];
          float v391_data = r3[63];
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v388_data, v387_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v389_data, v392_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v390_data, v393_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v391_data, v394_acc, 3, 7, 0);
          float v396_data = r3[65];
          float v397_data = r3[67];
          float v398_data = r3[69];
          float v399_data = r3[71];
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v396_data, v395_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v397_data, v400_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v398_data, v401_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v399_data, v402_acc, 3, 0, 0);
          float v404_data = r3[73];
          float v405_data = r3[75];
          float v406_data = r3[77];
          float v407_data = r3[79];
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v404_data, v403_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v405_data, v408_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v406_data, v409_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v407_data, v410_acc, 3, 1, 0);
          float v412_data = r3[81];
          float v413_data = r3[83];
          float v414_data = r3[85];
          float v415_data = r3[87];
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v412_data, v411_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v413_data, v416_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v414_data, v417_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v415_data, v418_acc, 3, 2, 0);
          float v420_data = r3[89];
          float v421_data = r3[91];
          float v422_data = r3[93];
          float v423_data = r3[95];
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v420_data, v419_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v421_data, v424_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v422_data, v425_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v423_data, v426_acc, 3, 3, 0);
          float v428_data = r3[97];
          float v429_data = r3[99];
          float v430_data = r3[101];
          float v431_data = r3[103];
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v428_data, v427_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v429_data, v432_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v430_data, v433_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v431_data, v434_acc, 3, 4, 0);
          float v436_data = r3[105];
          float v437_data = r3[107];
          float v438_data = r3[109];
          float v439_data = r3[111];
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v436_data, v435_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v437_data, v440_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v438_data, v441_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v213_tp, v439_data, v442_acc, 3, 5, 0);
          r4[1] = (v443_acc[0]);
          r4[3] = (v443_acc[1]);
          r4[5] = (v443_acc[2]);
          r4[7] = (v443_acc[3]);
          float v448_data = r2[8];
          float v449_data = r2[10];
          float v450_data = r2[12];
          float v451_data = r2[14];
          float v452_tp{};
          float v453_tp{};
          float v454_tp{};
          float v455_tp{};
          tensorforge::transpose4x4b32(v452_tp, v453_tp, v454_tp, v455_tp, v448_data, v449_data, v450_data, v451_data);
          float v456_data = r2[9];
          float v457_data = r2[11];
          float v458_data = r2[13];
          float v459_data = r2[15];
          float v460_tp{};
          float v461_tp{};
          float v462_tp{};
          float v463_tp{};
          tensorforge::transpose4x4b32(v460_tp, v461_tp, v462_tp, v463_tp, v456_data, v457_data, v458_data, v459_data);
          tensorforge::VectorT<float, 4> v464_acc{};
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v215_data, v464_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v216_data, v469_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v217_data, v470_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v218_data, v471_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v223_data, v472_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v224_data, v477_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v225_data, v478_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v226_data, v479_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v231_data, v480_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v232_data, v485_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v233_data, v486_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v234_data, v487_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v239_data, v488_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v240_data, v493_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v241_data, v494_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v242_data, v495_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v247_data, v496_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v248_data, v501_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v249_data, v502_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v250_data, v503_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v255_data, v504_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v256_data, v509_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v257_data, v510_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v258_data, v511_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v263_data, v512_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v264_data, v517_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v265_data, v518_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v266_data, v519_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v271_data, v520_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v272_data, v525_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v273_data, v526_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v274_data, v527_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v279_data, v528_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v280_data, v533_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v281_data, v534_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v282_data, v535_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v287_data, v536_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v288_data, v541_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v289_data, v542_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v290_data, v543_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v295_data, v544_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v296_data, v549_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v297_data, v550_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v298_data, v551_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v303_data, v552_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v304_data, v557_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v305_data, v558_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v306_data, v559_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v311_data, v560_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v312_data, v565_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v313_data, v566_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v314_data, v567_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v319_data, v568_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v320_data, v573_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v321_data, v574_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v322_data, v575_acc, 3, 5, 0);
          r4[8] = (v576_acc[0]);
          r4[10] = (v576_acc[1]);
          r4[12] = (v576_acc[2]);
          r4[14] = (v576_acc[3]);
          tensorforge::VectorT<float, 4> v581_acc{};
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v332_data, v581_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v333_data, v586_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v588_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v334_data, v587_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v589_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v335_data, v588_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v340_data, v589_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v341_data, v594_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v342_data, v595_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v343_data, v596_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v348_data, v597_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v349_data, v602_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v350_data, v603_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v351_data, v604_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v356_data, v605_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v357_data, v610_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v358_data, v611_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v359_data, v612_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v364_data, v613_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v365_data, v618_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v366_data, v619_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v367_data, v620_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v372_data, v621_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v373_data, v626_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v374_data, v627_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v375_data, v628_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v380_data, v629_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v381_data, v634_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v636_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v382_data, v635_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v383_data, v636_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v388_data, v637_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v643_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v389_data, v642_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v644_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v390_data, v643_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v391_data, v644_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v396_data, v645_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v397_data, v650_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v398_data, v651_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v399_data, v652_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v404_data, v653_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v405_data, v658_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v406_data, v659_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v407_data, v660_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v412_data, v661_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v413_data, v666_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v414_data, v667_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v669_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v415_data, v668_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v420_data, v669_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v421_data, v674_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v422_data, v675_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v423_data, v676_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v428_data, v677_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v429_data, v682_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v430_data, v683_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v431_data, v684_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v436_data, v685_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v437_data, v690_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v462_tp, v438_data, v691_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v463_tp, v439_data, v692_acc, 3, 5, 0);
          r4[9] = (v693_acc[0]);
          r4[11] = (v693_acc[1]);
          r4[13] = (v693_acc[2]);
          r4[15] = (v693_acc[3]);
          float v810_acc{};
          float v811_acc{};
          float v812_data = r2[16];
          float v813_data = r2[17];
          float v814_bc = tensorforge::broadcast<32, 16, 0>(v812_data);
          tensorforge::fmacdpp16<0>(v810_acc, v814_bc, v215_data);
          tensorforge::fmacdpp16<0>(v811_acc, v814_bc, v332_data);
          tensorforge::fmacdpp16<1>(v810_acc, v814_bc, v216_data);
          tensorforge::fmacdpp16<1>(v811_acc, v814_bc, v333_data);
          tensorforge::fmacdpp16<2>(v810_acc, v814_bc, v217_data);
          tensorforge::fmacdpp16<2>(v811_acc, v814_bc, v334_data);
          tensorforge::fmacdpp16<3>(v810_acc, v814_bc, v218_data);
          tensorforge::fmacdpp16<3>(v811_acc, v814_bc, v335_data);
          tensorforge::fmacdpp16<4>(v810_acc, v814_bc, v223_data);
          tensorforge::fmacdpp16<4>(v811_acc, v814_bc, v340_data);
          tensorforge::fmacdpp16<5>(v810_acc, v814_bc, v224_data);
          tensorforge::fmacdpp16<5>(v811_acc, v814_bc, v341_data);
          tensorforge::fmacdpp16<6>(v810_acc, v814_bc, v225_data);
          tensorforge::fmacdpp16<6>(v811_acc, v814_bc, v342_data);
          tensorforge::fmacdpp16<7>(v810_acc, v814_bc, v226_data);
          tensorforge::fmacdpp16<7>(v811_acc, v814_bc, v343_data);
          tensorforge::fmacdpp16<8>(v810_acc, v814_bc, v231_data);
          tensorforge::fmacdpp16<8>(v811_acc, v814_bc, v348_data);
          tensorforge::fmacdpp16<9>(v810_acc, v814_bc, v232_data);
          tensorforge::fmacdpp16<9>(v811_acc, v814_bc, v349_data);
          tensorforge::fmacdpp16<10>(v810_acc, v814_bc, v233_data);
          tensorforge::fmacdpp16<10>(v811_acc, v814_bc, v350_data);
          tensorforge::fmacdpp16<11>(v810_acc, v814_bc, v234_data);
          tensorforge::fmacdpp16<11>(v811_acc, v814_bc, v351_data);
          tensorforge::fmacdpp16<12>(v810_acc, v814_bc, v239_data);
          tensorforge::fmacdpp16<12>(v811_acc, v814_bc, v356_data);
          tensorforge::fmacdpp16<13>(v810_acc, v814_bc, v240_data);
          tensorforge::fmacdpp16<13>(v811_acc, v814_bc, v357_data);
          tensorforge::fmacdpp16<14>(v810_acc, v814_bc, v241_data);
          tensorforge::fmacdpp16<14>(v811_acc, v814_bc, v358_data);
          tensorforge::fmacdpp16<15>(v810_acc, v814_bc, v242_data);
          tensorforge::fmacdpp16<15>(v811_acc, v814_bc, v359_data);
          float v815_bc = tensorforge::broadcast<32, 16, 1>(v812_data);
          tensorforge::fmacdpp16<0>(v810_acc, v815_bc, v247_data);
          tensorforge::fmacdpp16<0>(v811_acc, v815_bc, v364_data);
          tensorforge::fmacdpp16<1>(v810_acc, v815_bc, v248_data);
          tensorforge::fmacdpp16<1>(v811_acc, v815_bc, v365_data);
          tensorforge::fmacdpp16<2>(v810_acc, v815_bc, v249_data);
          tensorforge::fmacdpp16<2>(v811_acc, v815_bc, v366_data);
          tensorforge::fmacdpp16<3>(v810_acc, v815_bc, v250_data);
          tensorforge::fmacdpp16<3>(v811_acc, v815_bc, v367_data);
          tensorforge::fmacdpp16<4>(v810_acc, v815_bc, v255_data);
          tensorforge::fmacdpp16<4>(v811_acc, v815_bc, v372_data);
          tensorforge::fmacdpp16<5>(v810_acc, v815_bc, v256_data);
          tensorforge::fmacdpp16<5>(v811_acc, v815_bc, v373_data);
          tensorforge::fmacdpp16<6>(v810_acc, v815_bc, v257_data);
          tensorforge::fmacdpp16<6>(v811_acc, v815_bc, v374_data);
          tensorforge::fmacdpp16<7>(v810_acc, v815_bc, v258_data);
          tensorforge::fmacdpp16<7>(v811_acc, v815_bc, v375_data);
          tensorforge::fmacdpp16<8>(v810_acc, v815_bc, v263_data);
          tensorforge::fmacdpp16<8>(v811_acc, v815_bc, v380_data);
          tensorforge::fmacdpp16<9>(v810_acc, v815_bc, v264_data);
          tensorforge::fmacdpp16<9>(v811_acc, v815_bc, v381_data);
          tensorforge::fmacdpp16<10>(v810_acc, v815_bc, v265_data);
          tensorforge::fmacdpp16<10>(v811_acc, v815_bc, v382_data);
          tensorforge::fmacdpp16<11>(v810_acc, v815_bc, v266_data);
          tensorforge::fmacdpp16<11>(v811_acc, v815_bc, v383_data);
          tensorforge::fmacdpp16<12>(v810_acc, v815_bc, v271_data);
          tensorforge::fmacdpp16<12>(v811_acc, v815_bc, v388_data);
          tensorforge::fmacdpp16<13>(v810_acc, v815_bc, v272_data);
          tensorforge::fmacdpp16<13>(v811_acc, v815_bc, v389_data);
          tensorforge::fmacdpp16<14>(v810_acc, v815_bc, v273_data);
          tensorforge::fmacdpp16<14>(v811_acc, v815_bc, v390_data);
          tensorforge::fmacdpp16<15>(v810_acc, v815_bc, v274_data);
          tensorforge::fmacdpp16<15>(v811_acc, v815_bc, v391_data);
          float v816_bc = tensorforge::broadcast<32, 16, 0>(v813_data);
          tensorforge::fmacdpp16<0>(v810_acc, v816_bc, v279_data);
          tensorforge::fmacdpp16<0>(v811_acc, v816_bc, v396_data);
          tensorforge::fmacdpp16<1>(v810_acc, v816_bc, v280_data);
          tensorforge::fmacdpp16<1>(v811_acc, v816_bc, v397_data);
          tensorforge::fmacdpp16<2>(v810_acc, v816_bc, v281_data);
          tensorforge::fmacdpp16<2>(v811_acc, v816_bc, v398_data);
          tensorforge::fmacdpp16<3>(v810_acc, v816_bc, v282_data);
          tensorforge::fmacdpp16<3>(v811_acc, v816_bc, v399_data);
          tensorforge::fmacdpp16<4>(v810_acc, v816_bc, v287_data);
          tensorforge::fmacdpp16<4>(v811_acc, v816_bc, v404_data);
          tensorforge::fmacdpp16<5>(v810_acc, v816_bc, v288_data);
          tensorforge::fmacdpp16<5>(v811_acc, v816_bc, v405_data);
          tensorforge::fmacdpp16<6>(v810_acc, v816_bc, v289_data);
          tensorforge::fmacdpp16<6>(v811_acc, v816_bc, v406_data);
          tensorforge::fmacdpp16<7>(v810_acc, v816_bc, v290_data);
          tensorforge::fmacdpp16<7>(v811_acc, v816_bc, v407_data);
          tensorforge::fmacdpp16<8>(v810_acc, v816_bc, v295_data);
          tensorforge::fmacdpp16<8>(v811_acc, v816_bc, v412_data);
          tensorforge::fmacdpp16<9>(v810_acc, v816_bc, v296_data);
          tensorforge::fmacdpp16<9>(v811_acc, v816_bc, v413_data);
          tensorforge::fmacdpp16<10>(v810_acc, v816_bc, v297_data);
          tensorforge::fmacdpp16<10>(v811_acc, v816_bc, v414_data);
          tensorforge::fmacdpp16<11>(v810_acc, v816_bc, v298_data);
          tensorforge::fmacdpp16<11>(v811_acc, v816_bc, v415_data);
          tensorforge::fmacdpp16<12>(v810_acc, v816_bc, v303_data);
          tensorforge::fmacdpp16<12>(v811_acc, v816_bc, v420_data);
          tensorforge::fmacdpp16<13>(v810_acc, v816_bc, v304_data);
          tensorforge::fmacdpp16<13>(v811_acc, v816_bc, v421_data);
          tensorforge::fmacdpp16<14>(v810_acc, v816_bc, v305_data);
          tensorforge::fmacdpp16<14>(v811_acc, v816_bc, v422_data);
          tensorforge::fmacdpp16<15>(v810_acc, v816_bc, v306_data);
          tensorforge::fmacdpp16<15>(v811_acc, v816_bc, v423_data);
          float v817_bc = tensorforge::broadcast<32, 16, 1>(v813_data);
          tensorforge::fmacdpp16<0>(v810_acc, v817_bc, v311_data);
          tensorforge::fmacdpp16<0>(v811_acc, v817_bc, v428_data);
          tensorforge::fmacdpp16<1>(v810_acc, v817_bc, v312_data);
          tensorforge::fmacdpp16<1>(v811_acc, v817_bc, v429_data);
          tensorforge::fmacdpp16<2>(v810_acc, v817_bc, v313_data);
          tensorforge::fmacdpp16<2>(v811_acc, v817_bc, v430_data);
          tensorforge::fmacdpp16<3>(v810_acc, v817_bc, v314_data);
          tensorforge::fmacdpp16<3>(v811_acc, v817_bc, v431_data);
          tensorforge::fmacdpp16<4>(v810_acc, v817_bc, v319_data);
          tensorforge::fmacdpp16<4>(v811_acc, v817_bc, v436_data);
          tensorforge::fmacdpp16<5>(v810_acc, v817_bc, v320_data);
          tensorforge::fmacdpp16<5>(v811_acc, v817_bc, v437_data);
          tensorforge::fmacdpp16<6>(v810_acc, v817_bc, v321_data);
          tensorforge::fmacdpp16<6>(v811_acc, v817_bc, v438_data);
          tensorforge::fmacdpp16<7>(v810_acc, v817_bc, v322_data);
          tensorforge::fmacdpp16<7>(v811_acc, v817_bc, v439_data);
          r4[16] = v810_acc;
          r4[17] = v811_acc;
          // glb_m2 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v818_i0 = 0; v818_i0 < 1; ++v818_i0) {
            int32_t v824_lead = v16_lead + (v818_i0 * 32);
            #pragma unroll
            for (int32_t v819_i1 = 0; v819_i1 < 9; ++v819_i1) {
              float v822_data = r4[(v818_i0 + (v819_i1 * 2))];
              glb_m2[(v824_lead + (v819_i1 * 56))] = v822_data;
            }
          }
          if (v26_g) {
            int32_t v832_lead = v16_lead + 32_i32;
            #pragma unroll
            for (int32_t v827_i1 = 0; v827_i1 < 9; ++v827_i1) {
              float v830_data = r4[(1 + (v827_i1 * 2))];
              glb_m2[(v832_lead + (v827_i1 * 56))] = v830_data;
            }
          }
        }
      }
    }
  }
}

