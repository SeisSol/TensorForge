// === base name ===
kernel_c700b99be3b056d3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c700b99be3b056d3 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c700b99be3b056d3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c700b99be3b056d3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c700b99be3b056d3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c700b99be3b056d3, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c700b99be3b056d3, block.x * block.y * block.z, 0));
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
void launcher_kernel_c700b99be3b056d3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c700b99be3b056d3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c700b99be3b056d3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_c700b99be3b056d3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c700b99be3b056d3(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 16;
          bool v24_g = v23_lead < 12;
          if (v24_g) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v23_lead + (v25_i1 * 12))]);
              r0[v25_i1] = v30_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v24_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v33_i1 * 12))]);
              r1[v33_i1] = v38_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v24_g) {
            #pragma unroll
            for (int32_t v132_i1 = 0; v132_i1 < 12; ++v132_i1) {
              float v137_data = __builtin_nontemporal_load(&glb_m3[(v23_lead + (v132_i1 * 12))]);
              r3[v132_i1] = v137_data;
            }
          }
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v41_data = r1[0];
          float v42_data = r1[1];
          float v43_data = r1[2];
          float v44_data = r1[3];
          float v45_tp{};
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          tensorforge::transpose4x4b32(v45_tp, v46_tp, v47_tp, v48_tp, v41_data, v42_data, v43_data, v44_data);
          tensorforge::VectorT<float, 4> v49_acc{};
          float v50_data = r0[0];
          float v51_data = r0[1];
          float v52_data = r0[2];
          float v53_data = r0[3];
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v49_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          float v58_data = r0[4];
          float v59_data = r0[5];
          float v60_data = r0[6];
          float v61_data = r0[7];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v57_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          float v66_data = r0[8];
          float v67_data = r0[9];
          float v68_data = r0[10];
          float v69_data = r0[11];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v65_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v70_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
          r2[0] = (v73_acc[0]);
          r2[1] = (v73_acc[1]);
          r2[2] = (v73_acc[2]);
          r2[3] = (v73_acc[3]);
          float v78_data = r1[4];
          float v79_data = r1[5];
          float v81_tp{};
          float v82_tp{};
          float v83_tp{};
          float v84_tp{};
          tensorforge::transpose4x4b32(v81_tp, v82_tp, v83_tp, v84_tp, v78_data, v79_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v85_acc{};
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v50_data, v85_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v51_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v52_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v53_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v58_data, v93_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v59_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v60_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v61_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v66_data, v101_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v67_data, v106_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v68_data, v107_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v69_data, v108_acc, 2, 2, 0);
          r2[4] = (v109_acc[0]);
          r2[5] = (v109_acc[1]);
          // s0 = store{r>s, clear}(localShrMem0, r2);
          if (v24_g) {
            #pragma unroll
            for (int32_t v112_z1 = 6; v112_z1 < 12; ++v112_z1) {
              int32_t v117_a = v23_lead + (v112_z1 * 12);
              s0[(v117_a ^ ((v117_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v24_g) {
            #pragma unroll
            for (int32_t v121_i1 = 0; v121_i1 < 6; ++v121_i1) {
              float v123_data = r2[v121_i1];
              int32_t v127_a = v23_lead + (v121_i1 * 12);
              s0[(v127_a ^ ((v127_a >> 4) & 15))] = v123_data;
            }
          }
          float r4[12]{};
          // r4 = +(r3 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          int32_t v145_sw = v23_lead ^ ((v23_lead >> 4) & 15);
          float v146_data = s0[v145_sw];
          int32_t v147_a = v23_lead + 12;
          float v151_data = s0[(v147_a ^ ((v147_a >> 4) & 15))];
          int32_t v152_a = v23_lead + 24;
          float v156_data = s0[(v152_a ^ ((v152_a >> 4) & 15))];
          int32_t v157_a = v23_lead + 36;
          float v161_data = s0[(v157_a ^ ((v157_a >> 4) & 15))];
          float v162_tp{};
          float v163_tp{};
          float v164_tp{};
          float v165_tp{};
          tensorforge::transpose4x4b32(v162_tp, v163_tp, v164_tp, v165_tp, v146_data, v151_data, v156_data, v161_data);
          tensorforge::VectorT<float, 4> v166_acc{};
          float v167_data = r3[0];
          float v168_data = r3[1];
          float v169_data = r3[2];
          float v170_data = r3[3];
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v167_data, v166_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v168_data, v171_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v169_data, v172_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v170_data, v173_acc, 2, 0, 0);
          float v175_data = r3[4];
          float v176_data = r3[5];
          float v177_data = r3[6];
          float v178_data = r3[7];
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v175_data, v174_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v176_data, v179_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v177_data, v180_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v178_data, v181_acc, 2, 1, 0);
          float v183_data = r3[8];
          float v184_data = r3[9];
          float v185_data = r3[10];
          float v186_data = r3[11];
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v183_data, v182_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v163_tp, v184_data, v187_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v164_tp, v185_data, v188_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v165_tp, v186_data, v189_acc, 2, 2, 0);
          r4[0] = (v190_acc[0]);
          r4[1] = (v190_acc[1]);
          r4[2] = (v190_acc[2]);
          r4[3] = (v190_acc[3]);
          int32_t v197_a = v23_lead + 48;
          float v201_data = s0[(v197_a ^ ((v197_a >> 4) & 15))];
          int32_t v202_a = v23_lead + 60;
          float v206_data = s0[(v202_a ^ ((v202_a >> 4) & 15))];
          int32_t v207_a = v23_lead + 72;
          float v211_data = s0[(v207_a ^ ((v207_a >> 4) & 15))];
          int32_t v212_a = v23_lead + 84;
          float v216_data = s0[(v212_a ^ ((v212_a >> 4) & 15))];
          float v217_tp{};
          float v218_tp{};
          float v219_tp{};
          float v220_tp{};
          tensorforge::transpose4x4b32(v217_tp, v218_tp, v219_tp, v220_tp, v201_data, v206_data, v211_data, v216_data);
          tensorforge::VectorT<float, 4> v221_acc{};
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v167_data, v221_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v168_data, v226_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v169_data, v227_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v170_data, v228_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v175_data, v229_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v176_data, v234_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v177_data, v235_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v178_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v183_data, v237_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v184_data, v242_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v185_data, v243_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v186_data, v244_acc, 2, 2, 0);
          r4[4] = (v245_acc[0]);
          r4[5] = (v245_acc[1]);
          r4[6] = (v245_acc[2]);
          r4[7] = (v245_acc[3]);
          int32_t v252_a = v23_lead + 96;
          float v256_data = s0[(v252_a ^ ((v252_a >> 4) & 15))];
          int32_t v257_a = v23_lead + 108;
          float v261_data = s0[(v257_a ^ ((v257_a >> 4) & 15))];
          int32_t v262_a = v23_lead + 120;
          float v266_data = s0[(v262_a ^ ((v262_a >> 4) & 15))];
          int32_t v267_a = v23_lead + 132;
          float v271_data = s0[(v267_a ^ ((v267_a >> 4) & 15))];
          float v272_tp{};
          float v273_tp{};
          float v274_tp{};
          float v275_tp{};
          tensorforge::transpose4x4b32(v272_tp, v273_tp, v274_tp, v275_tp, v256_data, v261_data, v266_data, v271_data);
          tensorforge::VectorT<float, 4> v276_acc{};
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v167_data, v276_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v168_data, v281_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v169_data, v282_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v170_data, v283_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v175_data, v284_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v176_data, v289_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v177_data, v290_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v178_data, v291_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v183_data, v292_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v184_data, v297_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v185_data, v298_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v186_data, v299_acc, 2, 2, 0);
          r4[8] = (v300_acc[0]);
          r4[9] = (v300_acc[1]);
          r4[10] = (v300_acc[2]);
          r4[11] = (v300_acc[3]);
          // glb_m2 = store{r>g}(r4);
          if (v24_g) {
            #pragma unroll
            for (int32_t v305_i1 = 0; v305_i1 < 12; ++v305_i1) {
              float v307_data = r4[v305_i1];
              glb_m2[(v23_lead + (v305_i1 * 12))] = v307_data;
            }
          }
        }
      }
    }
  }
}

