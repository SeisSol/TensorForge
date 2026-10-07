// === base name ===
kernel_0f7489888b658957

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0f7489888b658957 = {{32, 8, 1}, 32, 56, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0f7489888b658957(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0f7489888b658957(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0f7489888b658957(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0f7489888b658957, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0f7489888b658957, block.x * block.y * block.z, 0));
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
void launcher_kernel_0f7489888b658957(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0f7489888b658957(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0f7489888b658957), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0f7489888b658957, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0f7489888b658957(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 56×18(56×18) {0..56}×{0..18} strided
    //   m1 56×18(56×18) {0..56}×{0..18} strided
    //   m2 18×18(18×18) {0..18}×{0..18} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,18]],"name":"m0","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[56,18]],"name":"m1","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[18,18]],"name":"m2","ordered":false,"parts":1,"shape":[18,18],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,18]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,18]},{"addressing":"strided","bbox":[[0,0],[18,18]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[18,18]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 1008 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 1008 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 324 + 0 + m2_extraOffset];
          float r0[36]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v21_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v25_lead = v21_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 18; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v25_lead + (v23_i1 * 56))]);
              r0[(v22_i0 + (v23_i1 * 2))] = v28_data;
            }
          }
          bool v31_g = v21_lead < 24;
          if (v31_g) {
            int32_t v34_lead = v21_lead + 32_i32;
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 18; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v34_lead + (v32_i1 * 56))]);
              r0[(1 + (v32_i1 * 2))] = v37_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          if (v21_lead < 18) {
            #pragma unroll
            for (int32_t v42_i1 = 0; v42_i1 < 18; ++v42_i1) {
              float v47_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v42_i1 * 18))]);
              r1[v42_i1] = v47_data;
            }
          }
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          float v50_data = r1[0];
          float v51_data = r1[1];
          float v52_data = r1[2];
          float v53_data = r1[3];
          float v54_tp{};
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          tensorforge::transpose4x4b32(v54_tp, v55_tp, v56_tp, v57_tp, v50_data, v51_data, v52_data, v53_data);
          tensorforge::VectorT<float, 4> v58_acc{};
          float v59_data = r0[0];
          float v60_data = r0[2];
          float v61_data = r0[4];
          float v62_data = r0[6];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v58_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v63_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 3, 0, 0);
          float v67_data = r0[8];
          float v68_data = r0[10];
          float v69_data = r0[12];
          float v70_data = r0[14];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v66_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v71_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 3, 1, 0);
          float v75_data = r0[16];
          float v76_data = r0[18];
          float v77_data = r0[20];
          float v78_data = r0[22];
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v74_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v79_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v80_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 3, 2, 0);
          float v83_data = r0[24];
          float v84_data = r0[26];
          float v85_data = r0[28];
          float v86_data = r0[30];
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v83_data, v82_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v87_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v85_data, v88_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v86_data, v89_acc, 3, 3, 0);
          float v91_data = r0[32];
          float v92_data = r0[34];
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v91_data, v90_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v92_data, v94_acc, 3, 4, 0);
          r2[0] = (v95_acc[0]);
          r2[2] = (v95_acc[1]);
          r2[4] = (v95_acc[2]);
          r2[6] = (v95_acc[3]);
          tensorforge::VectorT<float, 4> v100_acc{};
          float v101_data = r0[1];
          float v102_data = r0[3];
          float v103_data = r0[5];
          float v104_data = r0[7];
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v101_data, v100_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v102_data, v105_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v103_data, v106_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v104_data, v107_acc, 3, 0, 0);
          float v109_data = r0[9];
          float v110_data = r0[11];
          float v111_data = r0[13];
          float v112_data = r0[15];
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v109_data, v108_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v110_data, v113_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v111_data, v114_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v112_data, v115_acc, 3, 1, 0);
          float v117_data = r0[17];
          float v118_data = r0[19];
          float v119_data = r0[21];
          float v120_data = r0[23];
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v117_data, v116_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v118_data, v121_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v119_data, v122_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v120_data, v123_acc, 3, 2, 0);
          float v125_data = r0[25];
          float v126_data = r0[27];
          float v127_data = r0[29];
          float v128_data = r0[31];
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v125_data, v124_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v126_data, v129_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v127_data, v130_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v128_data, v131_acc, 3, 3, 0);
          float v133_data = r0[33];
          float v134_data = r0[35];
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v133_data, v132_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v134_data, v136_acc, 3, 4, 0);
          r2[1] = (v137_acc[0]);
          r2[3] = (v137_acc[1]);
          r2[5] = (v137_acc[2]);
          r2[7] = (v137_acc[3]);
          float v142_data = r1[4];
          float v143_data = r1[5];
          float v144_data = r1[6];
          float v145_data = r1[7];
          float v146_tp{};
          float v147_tp{};
          float v148_tp{};
          float v149_tp{};
          tensorforge::transpose4x4b32(v146_tp, v147_tp, v148_tp, v149_tp, v142_data, v143_data, v144_data, v145_data);
          tensorforge::VectorT<float, 4> v150_acc{};
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v59_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v60_data, v155_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v61_data, v156_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v62_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v67_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v68_data, v163_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v69_data, v164_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v70_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v75_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v76_data, v171_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v77_data, v172_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v78_data, v173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v83_data, v174_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v84_data, v179_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v85_data, v180_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v86_data, v181_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v91_data, v182_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v92_data, v186_acc, 3, 4, 0);
          r2[8] = (v187_acc[0]);
          r2[10] = (v187_acc[1]);
          r2[12] = (v187_acc[2]);
          r2[14] = (v187_acc[3]);
          tensorforge::VectorT<float, 4> v192_acc{};
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v101_data, v192_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v102_data, v197_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v103_data, v198_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v104_data, v199_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v109_data, v200_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v110_data, v205_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v111_data, v206_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v112_data, v207_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v117_data, v208_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v118_data, v213_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v119_data, v214_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v120_data, v215_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v125_data, v216_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v126_data, v221_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v127_data, v222_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v128_data, v223_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v133_data, v224_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v134_data, v228_acc, 3, 4, 0);
          r2[9] = (v229_acc[0]);
          r2[11] = (v229_acc[1]);
          r2[13] = (v229_acc[2]);
          r2[15] = (v229_acc[3]);
          float v234_data = r1[8];
          float v235_data = r1[9];
          float v236_data = r1[10];
          float v237_data = r1[11];
          float v238_tp{};
          float v239_tp{};
          float v240_tp{};
          float v241_tp{};
          tensorforge::transpose4x4b32(v238_tp, v239_tp, v240_tp, v241_tp, v234_data, v235_data, v236_data, v237_data);
          tensorforge::VectorT<float, 4> v242_acc{};
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v59_data, v242_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v60_data, v247_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v61_data, v248_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v62_data, v249_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v67_data, v250_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v68_data, v255_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v69_data, v256_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v70_data, v257_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v75_data, v258_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v76_data, v263_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v77_data, v264_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v78_data, v265_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v83_data, v266_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v84_data, v271_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v85_data, v272_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v86_data, v273_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v91_data, v274_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v92_data, v278_acc, 3, 4, 0);
          r2[16] = (v279_acc[0]);
          r2[18] = (v279_acc[1]);
          r2[20] = (v279_acc[2]);
          r2[22] = (v279_acc[3]);
          tensorforge::VectorT<float, 4> v284_acc{};
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v101_data, v284_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v102_data, v289_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v103_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v104_data, v291_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v109_data, v292_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v110_data, v297_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v111_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v112_data, v299_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v117_data, v300_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v118_data, v305_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v119_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v120_data, v307_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v125_data, v308_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v126_data, v313_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v127_data, v314_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v128_data, v315_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v133_data, v316_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v134_data, v320_acc, 3, 4, 0);
          r2[17] = (v321_acc[0]);
          r2[19] = (v321_acc[1]);
          r2[21] = (v321_acc[2]);
          r2[23] = (v321_acc[3]);
          float v326_data = r1[12];
          float v327_data = r1[13];
          float v328_data = r1[14];
          float v329_data = r1[15];
          float v330_tp{};
          float v331_tp{};
          float v332_tp{};
          float v333_tp{};
          tensorforge::transpose4x4b32(v330_tp, v331_tp, v332_tp, v333_tp, v326_data, v327_data, v328_data, v329_data);
          tensorforge::VectorT<float, 4> v334_acc{};
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v59_data, v334_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v60_data, v339_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v61_data, v340_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v62_data, v341_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v67_data, v342_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v68_data, v347_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v69_data, v348_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v70_data, v349_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v75_data, v350_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v76_data, v355_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v77_data, v356_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v78_data, v357_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v83_data, v358_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v84_data, v363_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v85_data, v364_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v86_data, v365_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v91_data, v366_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v92_data, v370_acc, 3, 4, 0);
          r2[24] = (v371_acc[0]);
          r2[26] = (v371_acc[1]);
          r2[28] = (v371_acc[2]);
          r2[30] = (v371_acc[3]);
          tensorforge::VectorT<float, 4> v376_acc{};
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v101_data, v376_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v102_data, v381_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v103_data, v382_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v104_data, v383_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v109_data, v384_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v110_data, v389_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v111_data, v390_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v112_data, v391_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v117_data, v392_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v118_data, v397_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v119_data, v398_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v120_data, v399_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v125_data, v400_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v126_data, v405_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v127_data, v406_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v128_data, v407_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v133_data, v408_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v134_data, v412_acc, 3, 4, 0);
          r2[25] = (v413_acc[0]);
          r2[27] = (v413_acc[1]);
          r2[29] = (v413_acc[2]);
          r2[31] = (v413_acc[3]);
          float v418_data = r1[16];
          float v419_data = r1[17];
          float v421_tp{};
          float v422_tp{};
          float v423_tp{};
          float v424_tp{};
          tensorforge::transpose4x4b32(v421_tp, v422_tp, v423_tp, v424_tp, v418_data, v419_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v425_acc{};
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v59_data, v425_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v60_data, v430_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v61_data, v431_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v62_data, v432_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v67_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v68_data, v438_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v69_data, v439_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v70_data, v440_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v75_data, v441_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v76_data, v446_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v77_data, v447_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v78_data, v448_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v83_data, v449_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v84_data, v454_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v85_data, v455_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v86_data, v456_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v91_data, v457_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v92_data, v460_acc, 3, 4, 0);
          r2[32] = (v461_acc[0]);
          r2[34] = (v461_acc[1]);
          tensorforge::VectorT<float, 4> v464_acc{};
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v101_data, v464_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v102_data, v469_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v103_data, v470_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v104_data, v471_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v109_data, v472_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v110_data, v477_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v111_data, v478_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v112_data, v479_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v117_data, v480_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v118_data, v485_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v119_data, v486_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v120_data, v487_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v125_data, v488_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v126_data, v493_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v127_data, v494_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v128_data, v495_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v133_data, v496_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v134_data, v499_acc, 3, 4, 0);
          r2[33] = (v500_acc[0]);
          r2[35] = (v500_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v503_i0 = 0; v503_i0 < 1; ++v503_i0) {
            int32_t v509_lead = v21_lead + (v503_i0 * 32);
            #pragma unroll
            for (int32_t v504_i1 = 0; v504_i1 < 18; ++v504_i1) {
              float v507_data = r2[(v503_i0 + (v504_i1 * 2))];
              glb_m0[(v509_lead + (v504_i1 * 56))] = v507_data;
            }
          }
          if (v31_g) {
            int32_t v517_lead = v21_lead + 32_i32;
            #pragma unroll
            for (int32_t v512_i1 = 0; v512_i1 < 18; ++v512_i1) {
              float v515_data = r2[(1 + (v512_i1 * 2))];
              glb_m0[(v517_lead + (v512_i1 * 56))] = v515_data;
            }
          }
        }
      }
    }
  }
}

