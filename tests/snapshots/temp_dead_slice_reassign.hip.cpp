// === base name ===
kernel_3d8977b662371ba4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3d8977b662371ba4 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3d8977b662371ba4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3d8977b662371ba4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3d8977b662371ba4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3d8977b662371ba4, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_3d8977b662371ba4, block.x * block.y * block.z, 0));
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
void launcher_kernel_3d8977b662371ba4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3d8977b662371ba4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3d8977b662371ba4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_3d8977b662371ba4, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3d8977b662371ba4(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 13312 B shared, occupancy grid
    // operands:
    //   m0 6×12(6×12) {0..6}×{0..12} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 6×12(6×12) {0..6}×{0..12} strided
    //   m3 6×12(6×12) {0..6}×{0..12} strided
    //   m4 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j] = m2[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
    //   m4[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 72 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v11_batchId0 * 144 + 0 + m4_extraOffset];
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
            int32_t v170_off = v27_lead + 6;
            #pragma unroll
            for (int32_t v165_i1 = 0; v165_i1 < 12; ++v165_i1) {
              float v167_data = r2[v165_i1];
              int32_t v172_a = v170_off + (v165_i1 * 12);
              s0[(v172_a ^ ((v172_a >> 4) & 15))] = v167_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m3);
          if (v28_g) {
            #pragma unroll
            for (int32_t v177_i1 = 0; v177_i1 < 12; ++v177_i1) {
              float v182_data = __builtin_nontemporal_load(&glb_m3[(v27_lead + (v177_i1 * 6))]);
              r5[v177_i1] = v182_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          tensorforge::transpose4x4b32(v189_tp, v190_tp, v191_tp, v192_tp, v54_data, v55_data, v56_data, v57_data);
          tensorforge::VectorT<float, 4> v193_acc{};
          float v194_data = r3[0];
          float v195_data = r3[1];
          float v196_data = r3[2];
          float v197_data = r3[3];
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v194_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v195_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v196_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v197_data, v200_acc, 2, 0, 0);
          float v202_data = r3[4];
          float v203_data = r3[5];
          float v204_data = r3[6];
          float v205_data = r3[7];
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v202_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v203_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v204_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v205_data, v208_acc, 2, 1, 0);
          float v210_data = r3[8];
          float v211_data = r3[9];
          float v212_data = r3[10];
          float v213_data = r3[11];
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v210_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v211_data, v214_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v212_data, v215_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v213_data, v216_acc, 2, 2, 0);
          r4[0] = (v217_acc[0]);
          r4[1] = (v217_acc[1]);
          r4[2] = (v217_acc[2]);
          r4[3] = (v217_acc[3]);
          float v226_tp{};
          float v227_tp{};
          float v228_tp{};
          float v229_tp{};
          tensorforge::transpose4x4b32(v226_tp, v227_tp, v228_tp, v229_tp, v91_data, v92_data, v93_data, v94_data);
          tensorforge::VectorT<float, 4> v230_acc{};
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v194_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v195_data, v235_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v196_data, v236_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v197_data, v237_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v202_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v203_data, v243_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v204_data, v244_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v205_data, v245_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v210_data, v246_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v211_data, v251_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v212_data, v252_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v213_data, v253_acc, 2, 2, 0);
          r4[4] = (v254_acc[0]);
          r4[5] = (v254_acc[1]);
          r4[6] = (v254_acc[2]);
          r4[7] = (v254_acc[3]);
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          tensorforge::transpose4x4b32(v263_tp, v264_tp, v265_tp, v266_tp, v128_data, v129_data, v130_data, v131_data);
          tensorforge::VectorT<float, 4> v267_acc{};
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v194_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v195_data, v272_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v196_data, v273_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v197_data, v274_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v202_data, v275_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v203_data, v280_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v204_data, v281_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v205_data, v282_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v210_data, v283_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v211_data, v288_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v212_data, v289_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v213_data, v290_acc, 2, 2, 0);
          r4[8] = (v291_acc[0]);
          r4[9] = (v291_acc[1]);
          r4[10] = (v291_acc[2]);
          r4[11] = (v291_acc[3]);
          // s0 = store{r>s, clear}(localShrMem0, r4);
          if ((v27_lead >= 6) && v37_g) {
            #pragma unroll
            for (int32_t v298_z1 = 0; v298_z1 < 12; ++v298_z1) {
              int32_t v303_a = v27_lead + (v298_z1 * 12);
              s0[(v303_a ^ ((v303_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v28_g) {
            #pragma unroll
            for (int32_t v307_i1 = 0; v307_i1 < 12; ++v307_i1) {
              float v309_data = r4[v307_i1];
              int32_t v313_a = v27_lead + (v307_i1 * 12);
              s0[(v313_a ^ ((v313_a >> 4) & 15))] = v309_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = +(r5) + None
          // [(0, 6), (0, 12)] []
          float v318_data = r5[0];
          float v319_data = r6[0];
          r6[0] = (v319_data + v318_data);
          float v321_data = r5[1];
          float v322_data = r6[1];
          r6[1] = (v322_data + v321_data);
          float v324_data = r5[2];
          float v325_data = r6[2];
          r6[2] = (v325_data + v324_data);
          float v327_data = r5[3];
          float v328_data = r6[3];
          r6[3] = (v328_data + v327_data);
          float v330_data = r5[4];
          float v331_data = r6[4];
          r6[4] = (v331_data + v330_data);
          float v333_data = r5[5];
          float v334_data = r6[5];
          r6[5] = (v334_data + v333_data);
          float v336_data = r5[6];
          float v337_data = r6[6];
          r6[6] = (v337_data + v336_data);
          float v339_data = r5[7];
          float v340_data = r6[7];
          r6[7] = (v340_data + v339_data);
          float v342_data = r5[8];
          float v343_data = r6[8];
          r6[8] = (v343_data + v342_data);
          float v345_data = r5[9];
          float v346_data = r6[9];
          r6[9] = (v346_data + v345_data);
          float v348_data = r5[10];
          float v349_data = r6[10];
          r6[10] = (v349_data + v348_data);
          float v351_data = r5[11];
          float v352_data = r6[11];
          r6[11] = (v352_data + v351_data);
          // s0 = store{r>s}(localShrMem0, r6);
          if (v28_g) {
            int32_t v359_off = v27_lead + 6;
            #pragma unroll
            for (int32_t v354_i1 = 0; v354_i1 < 12; ++v354_i1) {
              float v356_data = r6[v354_i1];
              int32_t v361_a = v359_off + (v354_i1 * 12);
              s0[(v361_a ^ ((v361_a >> 4) & 15))] = v356_data;
            }
          }
          float r7[12]{};
          // r7 = +(s0) + None
          // [(0, 12), (0, 12)] []
          float v372_data = v37_g ? (s0[(v27_lead ^ ((v27_lead >> 4) & 15))]) : (0.0f);
          float v373_data = r7[0];
          r7[0] = (v373_data + v372_data);
          int32_t v375_a = v27_lead + 12;
          float v379_data = v37_g ? (s0[(v375_a ^ ((v375_a >> 4) & 15))]) : (0.0f);
          float v380_data = r7[1];
          r7[1] = (v380_data + v379_data);
          int32_t v382_a = v27_lead + 24;
          float v386_data = v37_g ? (s0[(v382_a ^ ((v382_a >> 4) & 15))]) : (0.0f);
          float v387_data = r7[2];
          r7[2] = (v387_data + v386_data);
          int32_t v389_a = v27_lead + 36;
          float v393_data = v37_g ? (s0[(v389_a ^ ((v389_a >> 4) & 15))]) : (0.0f);
          float v394_data = r7[3];
          r7[3] = (v394_data + v393_data);
          int32_t v396_a = v27_lead + 48;
          float v400_data = v37_g ? (s0[(v396_a ^ ((v396_a >> 4) & 15))]) : (0.0f);
          float v401_data = r7[4];
          r7[4] = (v401_data + v400_data);
          int32_t v403_a = v27_lead + 60;
          float v407_data = v37_g ? (s0[(v403_a ^ ((v403_a >> 4) & 15))]) : (0.0f);
          float v408_data = r7[5];
          r7[5] = (v408_data + v407_data);
          int32_t v410_a = v27_lead + 72;
          float v414_data = v37_g ? (s0[(v410_a ^ ((v410_a >> 4) & 15))]) : (0.0f);
          float v415_data = r7[6];
          r7[6] = (v415_data + v414_data);
          int32_t v417_a = v27_lead + 84;
          float v421_data = v37_g ? (s0[(v417_a ^ ((v417_a >> 4) & 15))]) : (0.0f);
          float v422_data = r7[7];
          r7[7] = (v422_data + v421_data);
          int32_t v424_a = v27_lead + 96;
          float v428_data = v37_g ? (s0[(v424_a ^ ((v424_a >> 4) & 15))]) : (0.0f);
          float v429_data = r7[8];
          r7[8] = (v429_data + v428_data);
          int32_t v431_a = v27_lead + 108;
          float v435_data = v37_g ? (s0[(v431_a ^ ((v431_a >> 4) & 15))]) : (0.0f);
          float v436_data = r7[9];
          r7[9] = (v436_data + v435_data);
          int32_t v438_a = v27_lead + 120;
          float v442_data = v37_g ? (s0[(v438_a ^ ((v438_a >> 4) & 15))]) : (0.0f);
          float v443_data = r7[10];
          r7[10] = (v443_data + v442_data);
          int32_t v445_a = v27_lead + 132;
          float v449_data = v37_g ? (s0[(v445_a ^ ((v445_a >> 4) & 15))]) : (0.0f);
          float v450_data = r7[11];
          r7[11] = (v450_data + v449_data);
          // glb_m4 = store{r>g}(r7);
          if (v37_g) {
            #pragma unroll
            for (int32_t v452_i1 = 0; v452_i1 < 12; ++v452_i1) {
              float v454_data = r7[v452_i1];
              glb_m4[(v27_lead + (v452_i1 * 12))] = v454_data;
            }
          }
        }
      }
    }
  }
}

