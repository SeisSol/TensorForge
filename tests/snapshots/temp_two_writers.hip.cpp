// === base name ===
kernel_7dad415f5a461911

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7dad415f5a461911 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7dad415f5a461911(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7dad415f5a461911(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7dad415f5a461911(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7dad415f5a461911, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_7dad415f5a461911, block.x * block.y * block.z, 0));
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
void launcher_kernel_7dad415f5a461911(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7dad415f5a461911(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_7dad415f5a461911), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_7dad415f5a461911, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7dad415f5a461911(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = m4[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v11_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v27_lead = threadIdx.x % 16;
          bool v28_g = v27_lead < 6;
          if (v28_g) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v34_data = __builtin_nontemporal_load(&glb_m0[(v27_lead + (v29_i1 * 6))]);
              r0[v29_i1] = v34_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v37_g = v27_lead < 12;
          if (v37_g) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              float v43_data = __builtin_nontemporal_load(&glb_m1[(v27_lead + (v38_i1 * 12))]);
              r1[v38_i1] = v43_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v28_g) {
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 12; ++v46_i1) {
              float v51_data = __builtin_nontemporal_load(&glb_m2[(v27_lead + (v46_i1 * 6))]);
              r3[v46_i1] = v51_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v54_data = r1[0];
          float v55_data = r1[1];
          float v56_data = r1[2];
          float v57_data = r1[3];
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          tensorforge::transpose4x4b32(v58_tp, v59_tp, v60_tp, v61_tp, v54_data, v55_data, v56_data, v57_data);
          tensorforge::VectorT<float, 4> v62_acc{};
          float v63_data = r0[0];
          float v64_data = r0[1];
          float v65_data = r0[2];
          float v66_data = r0[3];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v62_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 2, 0, 0);
          float v71_data = r0[4];
          float v72_data = r0[5];
          float v73_data = r0[6];
          float v74_data = r0[7];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v70_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 2, 1, 0);
          float v79_data = r0[8];
          float v80_data = r0[9];
          float v81_data = r0[10];
          float v82_data = r0[11];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v78_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 2, 2, 0);
          r2[0] = (v86_acc[0]);
          r2[1] = (v86_acc[1]);
          r2[2] = (v86_acc[2]);
          r2[3] = (v86_acc[3]);
          float v91_data = r1[4];
          float v92_data = r1[5];
          float v93_data = r1[6];
          float v94_data = r1[7];
          float v95_tp{};
          float v96_tp{};
          float v97_tp{};
          float v98_tp{};
          tensorforge::transpose4x4b32(v95_tp, v96_tp, v97_tp, v98_tp, v91_data, v92_data, v93_data, v94_data);
          tensorforge::VectorT<float, 4> v99_acc{};
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v107_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v113_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v115_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v121_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v122_acc, 2, 2, 0);
          r2[4] = (v123_acc[0]);
          r2[5] = (v123_acc[1]);
          r2[6] = (v123_acc[2]);
          r2[7] = (v123_acc[3]);
          float v128_data = r1[8];
          float v129_data = r1[9];
          float v130_data = r1[10];
          float v131_data = r1[11];
          float v132_tp{};
          float v133_tp{};
          float v134_tp{};
          float v135_tp{};
          tensorforge::transpose4x4b32(v132_tp, v133_tp, v134_tp, v135_tp, v128_data, v129_data, v130_data, v131_data);
          tensorforge::VectorT<float, 4> v136_acc{};
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v136_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v142_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v143_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v150_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v151_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v79_data, v152_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v80_data, v157_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v81_data, v158_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v82_data, v159_acc, 2, 2, 0);
          r2[8] = (v160_acc[0]);
          r2[9] = (v160_acc[1]);
          r2[10] = (v160_acc[2]);
          r2[11] = (v160_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v28_g) {
            #pragma unroll
            for (int32_t v165_i1 = 0; v165_i1 < 12; ++v165_i1) {
              float v167_data = r2[v165_i1];
              int32_t v171_a = v27_lead + (v165_i1 * 12);
              s0[(v171_a ^ ((v171_a >> 4) & 15))] = v167_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v37_g) {
            #pragma unroll
            for (int32_t v176_i1 = 0; v176_i1 < 12; ++v176_i1) {
              float v181_data = __builtin_nontemporal_load(&glb_m4[(v27_lead + (v176_i1 * 12))]);
              r5[v176_i1] = v181_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          tensorforge::transpose4x4b32(v188_tp, v189_tp, v190_tp, v191_tp, v54_data, v55_data, v56_data, v57_data);
          tensorforge::VectorT<float, 4> v192_acc{};
          float v193_data = r3[0];
          float v194_data = r3[1];
          float v195_data = r3[2];
          float v196_data = r3[3];
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v193_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v194_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v195_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v196_data, v199_acc, 2, 0, 0);
          float v201_data = r3[4];
          float v202_data = r3[5];
          float v203_data = r3[6];
          float v204_data = r3[7];
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v201_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v202_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v203_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v204_data, v207_acc, 2, 1, 0);
          float v209_data = r3[8];
          float v210_data = r3[9];
          float v211_data = r3[10];
          float v212_data = r3[11];
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v209_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v210_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v211_data, v214_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v212_data, v215_acc, 2, 2, 0);
          r4[0] = (v216_acc[0]);
          r4[1] = (v216_acc[1]);
          r4[2] = (v216_acc[2]);
          r4[3] = (v216_acc[3]);
          float v225_tp{};
          float v226_tp{};
          float v227_tp{};
          float v228_tp{};
          tensorforge::transpose4x4b32(v225_tp, v226_tp, v227_tp, v228_tp, v91_data, v92_data, v93_data, v94_data);
          tensorforge::VectorT<float, 4> v229_acc{};
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v193_data, v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v194_data, v234_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v195_data, v235_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v196_data, v236_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v201_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v202_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v203_data, v243_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v204_data, v244_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v209_data, v245_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v210_data, v250_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v211_data, v251_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v212_data, v252_acc, 2, 2, 0);
          r4[4] = (v253_acc[0]);
          r4[5] = (v253_acc[1]);
          r4[6] = (v253_acc[2]);
          r4[7] = (v253_acc[3]);
          float v262_tp{};
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          tensorforge::transpose4x4b32(v262_tp, v263_tp, v264_tp, v265_tp, v128_data, v129_data, v130_data, v131_data);
          tensorforge::VectorT<float, 4> v266_acc{};
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v193_data, v266_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v194_data, v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v195_data, v272_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v196_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v201_data, v274_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v202_data, v279_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v203_data, v280_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v204_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v209_data, v282_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v210_data, v287_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v211_data, v288_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v212_data, v289_acc, 2, 2, 0);
          r4[8] = (v290_acc[0]);
          r4[9] = (v290_acc[1]);
          r4[10] = (v290_acc[2]);
          r4[11] = (v290_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v28_g) {
            int32_t v300_off = v27_lead + 6;
            #pragma unroll
            for (int32_t v295_i1 = 0; v295_i1 < 12; ++v295_i1) {
              float v297_data = r4[v295_i1];
              int32_t v302_a = v300_off + (v295_i1 * 12);
              s0[(v302_a ^ ((v302_a >> 4) & 15))] = v297_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v313_data = s0[(v27_lead ^ ((v27_lead >> 4) & 15))];
          int32_t v314_a = v27_lead + 12;
          float v318_data = s0[(v314_a ^ ((v314_a >> 4) & 15))];
          int32_t v319_a = v27_lead + 24;
          float v323_data = s0[(v319_a ^ ((v319_a >> 4) & 15))];
          int32_t v324_a = v27_lead + 36;
          float v328_data = s0[(v324_a ^ ((v324_a >> 4) & 15))];
          float v329_tp{};
          float v330_tp{};
          float v331_tp{};
          float v332_tp{};
          tensorforge::transpose4x4b32(v329_tp, v330_tp, v331_tp, v332_tp, v313_data, v318_data, v323_data, v328_data);
          tensorforge::VectorT<float, 4> v333_acc{};
          float v334_data = r5[0];
          float v335_data = r5[1];
          float v336_data = r5[2];
          float v337_data = r5[3];
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v334_data, v333_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v335_data, v338_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v336_data, v339_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v337_data, v340_acc, 2, 0, 0);
          float v342_data = r5[4];
          float v343_data = r5[5];
          float v344_data = r5[6];
          float v345_data = r5[7];
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v342_data, v341_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v343_data, v346_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v344_data, v347_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v345_data, v348_acc, 2, 1, 0);
          float v350_data = r5[8];
          float v351_data = r5[9];
          float v352_data = r5[10];
          float v353_data = r5[11];
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v350_data, v349_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v330_tp, v351_data, v354_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v352_data, v355_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v353_data, v356_acc, 2, 2, 0);
          r6[0] = (v357_acc[0]);
          r6[1] = (v357_acc[1]);
          r6[2] = (v357_acc[2]);
          r6[3] = (v357_acc[3]);
          int32_t v364_a = v27_lead + 48;
          float v368_data = s0[(v364_a ^ ((v364_a >> 4) & 15))];
          int32_t v369_a = v27_lead + 60;
          float v373_data = s0[(v369_a ^ ((v369_a >> 4) & 15))];
          int32_t v374_a = v27_lead + 72;
          float v378_data = s0[(v374_a ^ ((v374_a >> 4) & 15))];
          int32_t v379_a = v27_lead + 84;
          float v383_data = s0[(v379_a ^ ((v379_a >> 4) & 15))];
          float v384_tp{};
          float v385_tp{};
          float v386_tp{};
          float v387_tp{};
          tensorforge::transpose4x4b32(v384_tp, v385_tp, v386_tp, v387_tp, v368_data, v373_data, v378_data, v383_data);
          tensorforge::VectorT<float, 4> v388_acc{};
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v334_data, v388_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v335_data, v393_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v386_tp, v336_data, v394_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v387_tp, v337_data, v395_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v342_data, v396_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v343_data, v401_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v386_tp, v344_data, v402_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v387_tp, v345_data, v403_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v350_data, v404_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v385_tp, v351_data, v409_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v386_tp, v352_data, v410_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v387_tp, v353_data, v411_acc, 2, 2, 0);
          r6[4] = (v412_acc[0]);
          r6[5] = (v412_acc[1]);
          r6[6] = (v412_acc[2]);
          r6[7] = (v412_acc[3]);
          int32_t v419_a = v27_lead + 96;
          float v423_data = s0[(v419_a ^ ((v419_a >> 4) & 15))];
          int32_t v424_a = v27_lead + 108;
          float v428_data = s0[(v424_a ^ ((v424_a >> 4) & 15))];
          int32_t v429_a = v27_lead + 120;
          float v433_data = s0[(v429_a ^ ((v429_a >> 4) & 15))];
          int32_t v434_a = v27_lead + 132;
          float v438_data = s0[(v434_a ^ ((v434_a >> 4) & 15))];
          float v439_tp{};
          float v440_tp{};
          float v441_tp{};
          float v442_tp{};
          tensorforge::transpose4x4b32(v439_tp, v440_tp, v441_tp, v442_tp, v423_data, v428_data, v433_data, v438_data);
          tensorforge::VectorT<float, 4> v443_acc{};
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v334_data, v443_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v335_data, v448_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v336_data, v449_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v337_data, v450_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v342_data, v451_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v343_data, v456_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v344_data, v457_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v345_data, v458_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v350_data, v459_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v351_data, v464_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v352_data, v465_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v353_data, v466_acc, 2, 2, 0);
          r6[8] = (v467_acc[0]);
          r6[9] = (v467_acc[1]);
          r6[10] = (v467_acc[2]);
          r6[11] = (v467_acc[3]);
          // glb_m3 = store{r>g}(r6);
          if (v37_g) {
            #pragma unroll
            for (int32_t v472_i1 = 0; v472_i1 < 12; ++v472_i1) {
              float v474_data = r6[v472_i1];
              glb_m3[(v27_lead + (v472_i1 * 12))] = v474_data;
            }
          }
        }
      }
    }
  }
}

