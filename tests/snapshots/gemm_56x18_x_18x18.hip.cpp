// === base name ===
kernel_b1bd1e1ea9805958

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b1bd1e1ea9805958 = {{32, 8, 1}, 32, 56, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b1bd1e1ea9805958(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b1bd1e1ea9805958(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b1bd1e1ea9805958(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b1bd1e1ea9805958, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b1bd1e1ea9805958, block.x * block.y * block.z, 0));
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
void launcher_kernel_b1bd1e1ea9805958(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b1bd1e1ea9805958(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b1bd1e1ea9805958), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b1bd1e1ea9805958, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b1bd1e1ea9805958(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 56×18(56×18) {0..56}×{0..18} strided
    //   m1 56×18(56×18) {0..56}×{0..18} strided
    //   m2 18×18(18×18) {0..18}×{0..18} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,18]],"name":"m0","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[56,18]],"name":"m1","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[18,18]],"name":"m2","ordered":false,"parts":1,"shape":[18,18],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,18]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,18]},{"addressing":"strided","bbox":[[0,0],[18,18]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[18,18]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 1008 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 1008 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 324 + 0 + m2_extraOffset];
          float r0[36]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v15_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
            int32_t v19_lead = v15_lead + (v16_i0 * 32);
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 18; ++v17_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + (v17_i1 * 56))]);
              r0[(v16_i0 + (v17_i1 * 2))] = v22_data;
            }
          }
          bool v25_g = v15_lead < 24;
          if (v25_g) {
            int32_t v28_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 18; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m1[(v28_lead + (v26_i1 * 56))]);
              r0[(1 + (v26_i1 * 2))] = v31_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          if (v15_lead < 18) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 18; ++v36_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m2[(v15_lead + (v36_i1 * 18))]);
              r1[v36_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
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
          float v54_data = r0[2];
          float v55_data = r0[4];
          float v56_data = r0[6];
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v52_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 3, 0, 0);
          float v61_data = r0[8];
          float v62_data = r0[10];
          float v63_data = r0[12];
          float v64_data = r0[14];
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v60_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 3, 1, 0);
          float v69_data = r0[16];
          float v70_data = r0[18];
          float v71_data = r0[20];
          float v72_data = r0[22];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v68_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 3, 2, 0);
          float v77_data = r0[24];
          float v78_data = r0[26];
          float v79_data = r0[28];
          float v80_data = r0[30];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v76_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v79_data, v82_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v80_data, v83_acc, 3, 3, 0);
          float v85_data = r0[32];
          float v86_data = r0[34];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v85_data, v84_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v86_data, v88_acc, 3, 4, 0);
          r2[0] = (v89_acc[0]);
          r2[2] = (v89_acc[1]);
          r2[4] = (v89_acc[2]);
          r2[6] = (v89_acc[3]);
          tensorforge::VectorT<float, 4> v94_acc{};
          float v95_data = r0[1];
          float v96_data = r0[3];
          float v97_data = r0[5];
          float v98_data = r0[7];
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v95_data, v94_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v96_data, v99_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v97_data, v100_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v98_data, v101_acc, 3, 0, 0);
          float v103_data = r0[9];
          float v104_data = r0[11];
          float v105_data = r0[13];
          float v106_data = r0[15];
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v103_data, v102_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v104_data, v107_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v105_data, v108_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v106_data, v109_acc, 3, 1, 0);
          float v111_data = r0[17];
          float v112_data = r0[19];
          float v113_data = r0[21];
          float v114_data = r0[23];
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v111_data, v110_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v112_data, v115_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v113_data, v116_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v114_data, v117_acc, 3, 2, 0);
          float v119_data = r0[25];
          float v120_data = r0[27];
          float v121_data = r0[29];
          float v122_data = r0[31];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v119_data, v118_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v120_data, v123_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v121_data, v124_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v122_data, v125_acc, 3, 3, 0);
          float v127_data = r0[33];
          float v128_data = r0[35];
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v127_data, v126_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v128_data, v130_acc, 3, 4, 0);
          r2[1] = (v131_acc[0]);
          r2[3] = (v131_acc[1]);
          r2[5] = (v131_acc[2]);
          r2[7] = (v131_acc[3]);
          float v136_data = r1[4];
          float v137_data = r1[5];
          float v138_data = r1[6];
          float v139_data = r1[7];
          float v140_tp{};
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          tensorforge::transpose4x4b32(v140_tp, v141_tp, v142_tp, v143_tp, v136_data, v137_data, v138_data, v139_data);
          tensorforge::VectorT<float, 4> v144_acc{};
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v53_data, v144_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v54_data, v149_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v55_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v56_data, v151_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v61_data, v152_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v62_data, v157_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v63_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v64_data, v159_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v69_data, v160_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v70_data, v165_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v71_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v72_data, v167_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v77_data, v168_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v78_data, v173_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v79_data, v174_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v80_data, v175_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v85_data, v176_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v86_data, v180_acc, 3, 4, 0);
          r2[8] = (v181_acc[0]);
          r2[10] = (v181_acc[1]);
          r2[12] = (v181_acc[2]);
          r2[14] = (v181_acc[3]);
          tensorforge::VectorT<float, 4> v186_acc{};
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v95_data, v186_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v96_data, v191_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v97_data, v192_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v98_data, v193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v103_data, v194_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v104_data, v199_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v105_data, v200_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v106_data, v201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v111_data, v202_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v112_data, v207_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v113_data, v208_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v114_data, v209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v119_data, v210_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v120_data, v215_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v121_data, v216_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v122_data, v217_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v127_data, v218_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v128_data, v222_acc, 3, 4, 0);
          r2[9] = (v223_acc[0]);
          r2[11] = (v223_acc[1]);
          r2[13] = (v223_acc[2]);
          r2[15] = (v223_acc[3]);
          float v228_data = r1[8];
          float v229_data = r1[9];
          float v230_data = r1[10];
          float v231_data = r1[11];
          float v232_tp{};
          float v233_tp{};
          float v234_tp{};
          float v235_tp{};
          tensorforge::transpose4x4b32(v232_tp, v233_tp, v234_tp, v235_tp, v228_data, v229_data, v230_data, v231_data);
          tensorforge::VectorT<float, 4> v236_acc{};
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v53_data, v236_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v54_data, v241_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v55_data, v242_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v56_data, v243_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v61_data, v244_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v62_data, v249_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v63_data, v250_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v64_data, v251_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v69_data, v252_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v70_data, v257_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v71_data, v258_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v72_data, v259_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v77_data, v260_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v78_data, v265_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v79_data, v266_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v80_data, v267_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v85_data, v268_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v86_data, v272_acc, 3, 4, 0);
          r2[16] = (v273_acc[0]);
          r2[18] = (v273_acc[1]);
          r2[20] = (v273_acc[2]);
          r2[22] = (v273_acc[3]);
          tensorforge::VectorT<float, 4> v278_acc{};
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v95_data, v278_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v96_data, v283_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v97_data, v284_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v98_data, v285_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v103_data, v286_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v104_data, v291_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v105_data, v292_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v106_data, v293_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v111_data, v294_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v112_data, v299_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v113_data, v300_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v114_data, v301_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v119_data, v302_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v120_data, v307_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v121_data, v308_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v122_data, v309_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v127_data, v310_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v128_data, v314_acc, 3, 4, 0);
          r2[17] = (v315_acc[0]);
          r2[19] = (v315_acc[1]);
          r2[21] = (v315_acc[2]);
          r2[23] = (v315_acc[3]);
          float v320_data = r1[12];
          float v321_data = r1[13];
          float v322_data = r1[14];
          float v323_data = r1[15];
          float v324_tp{};
          float v325_tp{};
          float v326_tp{};
          float v327_tp{};
          tensorforge::transpose4x4b32(v324_tp, v325_tp, v326_tp, v327_tp, v320_data, v321_data, v322_data, v323_data);
          tensorforge::VectorT<float, 4> v328_acc{};
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v53_data, v328_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v54_data, v333_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v55_data, v334_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v56_data, v335_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v61_data, v336_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v62_data, v341_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v63_data, v342_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v64_data, v343_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v69_data, v344_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v70_data, v349_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v71_data, v350_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v72_data, v351_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v77_data, v352_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v78_data, v357_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v79_data, v358_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v80_data, v359_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v85_data, v360_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v86_data, v364_acc, 3, 4, 0);
          r2[24] = (v365_acc[0]);
          r2[26] = (v365_acc[1]);
          r2[28] = (v365_acc[2]);
          r2[30] = (v365_acc[3]);
          tensorforge::VectorT<float, 4> v370_acc{};
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v95_data, v370_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v96_data, v375_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v97_data, v376_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v98_data, v377_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v103_data, v378_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v104_data, v383_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v105_data, v384_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v106_data, v385_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v111_data, v386_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v112_data, v391_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v113_data, v392_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v114_data, v393_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v119_data, v394_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v120_data, v399_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v121_data, v400_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v122_data, v401_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v127_data, v402_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v128_data, v406_acc, 3, 4, 0);
          r2[25] = (v407_acc[0]);
          r2[27] = (v407_acc[1]);
          r2[29] = (v407_acc[2]);
          r2[31] = (v407_acc[3]);
          float v412_data = r1[16];
          float v413_data = r1[17];
          float v415_tp{};
          float v416_tp{};
          float v417_tp{};
          float v418_tp{};
          tensorforge::transpose4x4b32(v415_tp, v416_tp, v417_tp, v418_tp, v412_data, v413_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v419_acc{};
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v53_data, v419_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v54_data, v424_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v55_data, v425_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v56_data, v426_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v61_data, v427_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v62_data, v432_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v63_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v64_data, v434_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v69_data, v435_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v70_data, v440_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v71_data, v441_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v72_data, v442_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v77_data, v443_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v78_data, v448_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v79_data, v449_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v80_data, v450_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v85_data, v451_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v86_data, v454_acc, 3, 4, 0);
          r2[32] = (v455_acc[0]);
          r2[34] = (v455_acc[1]);
          tensorforge::VectorT<float, 4> v458_acc{};
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v95_data, v458_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v96_data, v463_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v97_data, v464_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v98_data, v465_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v103_data, v466_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v104_data, v471_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v105_data, v472_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v106_data, v473_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v111_data, v474_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v112_data, v479_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v113_data, v480_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v114_data, v481_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v119_data, v482_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v120_data, v487_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v121_data, v488_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v122_data, v489_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v127_data, v490_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v128_data, v493_acc, 3, 4, 0);
          r2[33] = (v494_acc[0]);
          r2[35] = (v494_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v497_i0 = 0; v497_i0 < 1; ++v497_i0) {
            int32_t v503_lead = v15_lead + (v497_i0 * 32);
            #pragma unroll
            for (int32_t v498_i1 = 0; v498_i1 < 18; ++v498_i1) {
              float v501_data = r2[(v497_i0 + (v498_i1 * 2))];
              glb_m0[(v503_lead + (v498_i1 * 56))] = v501_data;
            }
          }
          if (v25_g) {
            int32_t v511_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v506_i1 = 0; v506_i1 < 18; ++v506_i1) {
              float v509_data = r2[(1 + (v506_i1 * 2))];
              glb_m0[(v511_lead + (v506_i1 * 56))] = v509_data;
            }
          }
        }
      }
    }
  }
}

