// === base name ===
kernel_573d345a6e2c69f8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_573d345a6e2c69f8 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_573d345a6e2c69f8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_573d345a6e2c69f8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_573d345a6e2c69f8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_573d345a6e2c69f8, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_573d345a6e2c69f8, block.x * block.y * block.z, 0));
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
void launcher_kernel_573d345a6e2c69f8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_573d345a6e2c69f8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_573d345a6e2c69f8), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_573d345a6e2c69f8, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_573d345a6e2c69f8(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v5_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v5_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 16;
          bool v21_g = v20_lead < 12;
          if (v21_g) {
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 12; ++v22_i1) {
              float v27_data = __builtin_nontemporal_load(&glb_m0[(v20_lead + (v22_i1 * 12))]);
              r0[v22_i1] = v27_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v21_g) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 12; ++v30_i1) {
              float v35_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + (v30_i1 * 12))]);
              r1[v30_i1] = v35_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v21_g) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              float v43_data = __builtin_nontemporal_load(&glb_m3[(v20_lead + (v38_i1 * 12))]);
              r3[v38_i1] = v43_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v46_data = r1[0];
          float v47_data = r1[1];
          float v48_data = r1[2];
          float v49_data = r1[3];
          float v50_tp{};
          float v51_tp{};
          float v52_tp{};
          float v53_tp{};
          tensorforge::transpose4x4b32(v50_tp, v51_tp, v52_tp, v53_tp, v46_data, v47_data, v48_data, v49_data);
          tensorforge::VectorT<float, 4> v54_acc{};
          float v55_data = r0[0];
          float v56_data = r0[1];
          float v57_data = r0[2];
          float v58_data = r0[3];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v54_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v60_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v61_acc, 2, 0, 0);
          float v63_data = r0[4];
          float v64_data = r0[5];
          float v65_data = r0[6];
          float v66_data = r0[7];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v62_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v68_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 2, 1, 0);
          float v71_data = r0[8];
          float v72_data = r0[9];
          float v73_data = r0[10];
          float v74_data = r0[11];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v70_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v76_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 2, 2, 0);
          r2[0] = (v78_acc[0]);
          r2[1] = (v78_acc[1]);
          r2[2] = (v78_acc[2]);
          r2[3] = (v78_acc[3]);
          float v83_data = r1[4];
          float v84_data = r1[5];
          float v86_tp{};
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          tensorforge::transpose4x4b32(v86_tp, v87_tp, v88_tp, v89_tp, v83_data, v84_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v90_acc{};
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v55_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v56_data, v95_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v57_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v58_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v63_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v64_data, v103_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v65_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v66_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v71_data, v106_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v72_data, v111_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v73_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v74_data, v113_acc, 2, 2, 0);
          r2[4] = (v114_acc[0]);
          r2[5] = (v114_acc[1]);
          // s0 = store{r>s, clear}(localShrMem0, r2);
          if (v21_g) {
            #pragma unroll
            for (int32_t v117_z1 = 6; v117_z1 < 12; ++v117_z1) {
              int32_t v122_a = v20_lead + (v117_z1 * 12);
              s0[(v122_a ^ ((v122_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v21_g) {
            #pragma unroll
            for (int32_t v126_i1 = 0; v126_i1 < 6; ++v126_i1) {
              float v128_data = r2[v126_i1];
              int32_t v132_a = v20_lead + (v126_i1 * 12);
              s0[(v132_a ^ ((v132_a >> 4) & 15))] = v128_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = +(r3 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v143_data = s0[(v20_lead ^ ((v20_lead >> 4) & 15))];
          int32_t v144_a = v20_lead + 12;
          float v148_data = s0[(v144_a ^ ((v144_a >> 4) & 15))];
          int32_t v149_a = v20_lead + 24;
          float v153_data = s0[(v149_a ^ ((v149_a >> 4) & 15))];
          int32_t v154_a = v20_lead + 36;
          float v158_data = s0[(v154_a ^ ((v154_a >> 4) & 15))];
          float v159_tp{};
          float v160_tp{};
          float v161_tp{};
          float v162_tp{};
          tensorforge::transpose4x4b32(v159_tp, v160_tp, v161_tp, v162_tp, v143_data, v148_data, v153_data, v158_data);
          tensorforge::VectorT<float, 4> v163_acc{};
          float v164_data = r3[0];
          float v165_data = r3[1];
          float v166_data = r3[2];
          float v167_data = r3[3];
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v164_data, v163_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v165_data, v168_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v166_data, v169_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v167_data, v170_acc, 2, 0, 0);
          float v172_data = r3[4];
          float v173_data = r3[5];
          float v174_data = r3[6];
          float v175_data = r3[7];
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v172_data, v171_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v173_data, v176_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v174_data, v177_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v175_data, v178_acc, 2, 1, 0);
          float v180_data = r3[8];
          float v181_data = r3[9];
          float v182_data = r3[10];
          float v183_data = r3[11];
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v180_data, v179_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v181_data, v184_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v182_data, v185_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v183_data, v186_acc, 2, 2, 0);
          r4[0] = (v187_acc[0]);
          r4[1] = (v187_acc[1]);
          r4[2] = (v187_acc[2]);
          r4[3] = (v187_acc[3]);
          int32_t v194_a = v20_lead + 48;
          float v198_data = s0[(v194_a ^ ((v194_a >> 4) & 15))];
          int32_t v199_a = v20_lead + 60;
          float v203_data = s0[(v199_a ^ ((v199_a >> 4) & 15))];
          int32_t v204_a = v20_lead + 72;
          float v208_data = s0[(v204_a ^ ((v204_a >> 4) & 15))];
          int32_t v209_a = v20_lead + 84;
          float v213_data = s0[(v209_a ^ ((v209_a >> 4) & 15))];
          float v214_tp{};
          float v215_tp{};
          float v216_tp{};
          float v217_tp{};
          tensorforge::transpose4x4b32(v214_tp, v215_tp, v216_tp, v217_tp, v198_data, v203_data, v208_data, v213_data);
          tensorforge::VectorT<float, 4> v218_acc{};
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v164_data, v218_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v165_data, v223_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v166_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v167_data, v225_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v172_data, v226_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v173_data, v231_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v174_data, v232_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v175_data, v233_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v214_tp, v180_data, v234_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v181_data, v239_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v182_data, v240_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v183_data, v241_acc, 2, 2, 0);
          r4[4] = (v242_acc[0]);
          r4[5] = (v242_acc[1]);
          r4[6] = (v242_acc[2]);
          r4[7] = (v242_acc[3]);
          int32_t v249_a = v20_lead + 96;
          float v253_data = s0[(v249_a ^ ((v249_a >> 4) & 15))];
          int32_t v254_a = v20_lead + 108;
          float v258_data = s0[(v254_a ^ ((v254_a >> 4) & 15))];
          int32_t v259_a = v20_lead + 120;
          float v263_data = s0[(v259_a ^ ((v259_a >> 4) & 15))];
          int32_t v264_a = v20_lead + 132;
          float v268_data = s0[(v264_a ^ ((v264_a >> 4) & 15))];
          float v269_tp{};
          float v270_tp{};
          float v271_tp{};
          float v272_tp{};
          tensorforge::transpose4x4b32(v269_tp, v270_tp, v271_tp, v272_tp, v253_data, v258_data, v263_data, v268_data);
          tensorforge::VectorT<float, 4> v273_acc{};
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v164_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v165_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v166_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v167_data, v280_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v172_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v173_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v174_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v175_data, v288_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v180_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v181_data, v294_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v182_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v183_data, v296_acc, 2, 2, 0);
          r4[8] = (v297_acc[0]);
          r4[9] = (v297_acc[1]);
          r4[10] = (v297_acc[2]);
          r4[11] = (v297_acc[3]);
          // glb_m2 = store{r>g}(r4);
          if (v21_g) {
            #pragma unroll
            for (int32_t v302_i1 = 0; v302_i1 < 12; ++v302_i1) {
              float v304_data = r4[v302_i1];
              glb_m2[(v20_lead + (v302_i1 * 12))] = v304_data;
            }
          }
        }
      }
    }
  }
}

