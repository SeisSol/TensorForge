// === base name ===
kernel_ab0f9ae8694fc902

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ab0f9ae8694fc902 = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ab0f9ae8694fc902(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ab0f9ae8694fc902(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ab0f9ae8694fc902(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ab0f9ae8694fc902, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_ab0f9ae8694fc902, block.x * block.y * block.z, 0));
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
void launcher_kernel_ab0f9ae8694fc902(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ab0f9ae8694fc902(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ab0f9ae8694fc902), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_ab0f9ae8694fc902, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ab0f9ae8694fc902(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
    //   m3 32×32(4×12) {4..8}×{0..12} strided
    //   m4 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = t0[i,k] × m2[k,j]
    //   t0 12×12(12×12) {0..12}×{0..12} pointer_based({4..8}×{0..12}) = abs(N)
    //   m4[i,j] = t1[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[4,0],[8,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 48 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v11_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v27_lead = threadIdx.x % 16;
          bool v28_g = v27_lead < 12;
          if (v28_g) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v34_data = __builtin_nontemporal_load(&glb_m0[(v27_lead + (v29_i1 * 12))]);
              r0[v29_i1] = v34_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v28_g) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v27_lead + (v37_i1 * 12))]);
              r1[v37_i1] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v28_g) {
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              float v50_data = __builtin_nontemporal_load(&glb_m2[(v27_lead + (v45_i1 * 12))]);
              r3[v45_i1] = v50_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v53_data = r1[0];
          float v54_data = r1[1];
          float v55_data = r1[2];
          float v56_data = r1[3];
          float v57_tp{};
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          tensorforge::transpose4x4b32(v57_tp, v58_tp, v59_tp, v60_tp, v53_data, v54_data, v55_data, v56_data);
          tensorforge::VectorT<float, 4> v61_acc{};
          float v62_data = r0[0];
          float v63_data = r0[1];
          float v64_data = r0[2];
          float v65_data = r0[3];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 2, 0, 0);
          float v70_data = r0[4];
          float v71_data = r0[5];
          float v72_data = r0[6];
          float v73_data = r0[7];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 2, 1, 0);
          float v78_data = r0[8];
          float v79_data = r0[9];
          float v80_data = r0[10];
          float v81_data = r0[11];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v77_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 2, 2, 0);
          r2[0] = (v85_acc[0]);
          r2[1] = (v85_acc[1]);
          r2[2] = (v85_acc[2]);
          r2[3] = (v85_acc[3]);
          float v90_data = r1[4];
          float v91_data = r1[5];
          float v92_data = r1[6];
          float v93_data = r1[7];
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          float v97_tp{};
          tensorforge::transpose4x4b32(v94_tp, v95_tp, v96_tp, v97_tp, v90_data, v91_data, v92_data, v93_data);
          tensorforge::VectorT<float, 4> v98_acc{};
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v112_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v113_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v119_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v121_acc, 2, 2, 0);
          r2[4] = (v122_acc[0]);
          r2[5] = (v122_acc[1]);
          r2[6] = (v122_acc[2]);
          r2[7] = (v122_acc[3]);
          float v127_data = r1[8];
          float v128_data = r1[9];
          float v129_data = r1[10];
          float v130_data = r1[11];
          float v131_tp{};
          float v132_tp{};
          float v133_tp{};
          float v134_tp{};
          tensorforge::transpose4x4b32(v131_tp, v132_tp, v133_tp, v134_tp, v127_data, v128_data, v129_data, v130_data);
          tensorforge::VectorT<float, 4> v135_acc{};
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v140_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v64_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v142_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v148_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v72_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v150_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v78_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v79_data, v156_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v80_data, v157_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v81_data, v158_acc, 2, 2, 0);
          r2[8] = (v159_acc[0]);
          r2[9] = (v159_acc[1]);
          r2[10] = (v159_acc[2]);
          r2[11] = (v159_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v28_g) {
            #pragma unroll
            for (int32_t v164_i1 = 0; v164_i1 < 12; ++v164_i1) {
              float v166_data = r2[v164_i1];
              int32_t v170_a = v27_lead + (v164_i1 * 12);
              s0[(v170_a ^ ((v170_a >> 4) & 15))] = v166_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v175_data = r3[0];
          float v176_data = r3[1];
          float v177_data = r3[2];
          float v178_data = r3[3];
          float v179_tp{};
          float v180_tp{};
          float v181_tp{};
          float v182_tp{};
          tensorforge::transpose4x4b32(v179_tp, v180_tp, v181_tp, v182_tp, v175_data, v176_data, v177_data, v178_data);
          tensorforge::VectorT<float, 4> v183_acc{};
          int32_t v188_sw = (v27_lead >> 4) & 15;
          float v190_data = s0[(v27_lead ^ v188_sw)];
          int32_t v191_a = v27_lead + 12;
          int32_t v192_sw = v191_a >> 4;
          float v195_data = s0[(v191_a ^ (v192_sw & 15))];
          int32_t v196_a = v27_lead + 24;
          int32_t v197_sw = v196_a >> 4;
          float v200_data = s0[(v196_a ^ (v197_sw & 15))];
          int32_t v201_a = v27_lead + 36;
          int32_t v202_sw = v201_a >> 4;
          float v205_data = s0[(v201_a ^ (v202_sw & 15))];
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v190_data, v183_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v195_data, v206_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v200_data, v207_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v205_data, v208_acc, 2, 0, 0);
          int32_t v210_a = v27_lead + 48;
          int32_t v211_sw = v210_a >> 4;
          float v214_data = s0[(v210_a ^ (v211_sw & 15))];
          int32_t v215_a = v27_lead + 60;
          int32_t v216_sw = v215_a >> 4;
          float v219_data = s0[(v215_a ^ (v216_sw & 15))];
          int32_t v220_a = v27_lead + 72;
          int32_t v221_sw = v220_a >> 4;
          float v224_data = s0[(v220_a ^ (v221_sw & 15))];
          int32_t v225_a = v27_lead + 84;
          int32_t v226_sw = v225_a >> 4;
          float v229_data = s0[(v225_a ^ (v226_sw & 15))];
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v214_data, v209_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v219_data, v230_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v224_data, v231_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v229_data, v232_acc, 2, 1, 0);
          int32_t v234_a = v27_lead + 96;
          int32_t v235_sw = v234_a >> 4;
          float v238_data = s0[(v234_a ^ (v235_sw & 15))];
          int32_t v239_a = v27_lead + 108;
          int32_t v240_sw = v239_a >> 4;
          float v243_data = s0[(v239_a ^ (v240_sw & 15))];
          int32_t v244_a = v27_lead + 120;
          int32_t v245_sw = v244_a >> 4;
          float v248_data = s0[(v244_a ^ (v245_sw & 15))];
          int32_t v249_a = v27_lead + 132;
          int32_t v250_sw = v249_a >> 4;
          float v253_data = s0[(v249_a ^ (v250_sw & 15))];
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v238_data, v233_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v243_data, v254_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v248_data, v255_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v253_data, v256_acc, 2, 2, 0);
          r4[0] = (v257_acc[0]);
          r4[1] = (v257_acc[1]);
          r4[2] = (v257_acc[2]);
          r4[3] = (v257_acc[3]);
          float v262_data = r3[4];
          float v263_data = r3[5];
          float v264_data = r3[6];
          float v265_data = r3[7];
          float v266_tp{};
          float v267_tp{};
          float v268_tp{};
          float v269_tp{};
          tensorforge::transpose4x4b32(v266_tp, v267_tp, v268_tp, v269_tp, v262_data, v263_data, v264_data, v265_data);
          tensorforge::VectorT<float, 4> v270_acc{};
          float v277_data = s0[(v27_lead ^ v188_sw)];
          float v282_data = s0[(v191_a ^ (v192_sw & 15))];
          float v287_data = s0[(v196_a ^ (v197_sw & 15))];
          float v292_data = s0[(v201_a ^ (v202_sw & 15))];
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v277_data, v270_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v282_data, v293_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v287_data, v294_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v292_data, v295_acc, 2, 0, 0);
          float v301_data = s0[(v210_a ^ (v211_sw & 15))];
          float v306_data = s0[(v215_a ^ (v216_sw & 15))];
          float v311_data = s0[(v220_a ^ (v221_sw & 15))];
          float v316_data = s0[(v225_a ^ (v226_sw & 15))];
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v301_data, v296_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v306_data, v317_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v311_data, v318_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v316_data, v319_acc, 2, 1, 0);
          float v325_data = s0[(v234_a ^ (v235_sw & 15))];
          float v330_data = s0[(v239_a ^ (v240_sw & 15))];
          float v335_data = s0[(v244_a ^ (v245_sw & 15))];
          float v340_data = s0[(v249_a ^ (v250_sw & 15))];
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v325_data, v320_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v330_data, v341_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v335_data, v342_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v340_data, v343_acc, 2, 2, 0);
          r4[4] = (v344_acc[0]);
          r4[5] = (v344_acc[1]);
          r4[6] = (v344_acc[2]);
          r4[7] = (v344_acc[3]);
          float v349_data = r3[8];
          float v350_data = r3[9];
          float v351_data = r3[10];
          float v352_data = r3[11];
          float v353_tp{};
          float v354_tp{};
          float v355_tp{};
          float v356_tp{};
          tensorforge::transpose4x4b32(v353_tp, v354_tp, v355_tp, v356_tp, v349_data, v350_data, v351_data, v352_data);
          tensorforge::VectorT<float, 4> v357_acc{};
          float v364_data = s0[(v27_lead ^ v188_sw)];
          float v369_data = s0[(v191_a ^ (v192_sw & 15))];
          float v374_data = s0[(v196_a ^ (v197_sw & 15))];
          float v379_data = s0[(v201_a ^ (v202_sw & 15))];
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v364_data, v357_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v369_data, v380_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v374_data, v381_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v379_data, v382_acc, 2, 0, 0);
          float v388_data = s0[(v210_a ^ (v211_sw & 15))];
          float v393_data = s0[(v215_a ^ (v216_sw & 15))];
          float v398_data = s0[(v220_a ^ (v221_sw & 15))];
          float v403_data = s0[(v225_a ^ (v226_sw & 15))];
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v388_data, v383_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v393_data, v404_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v398_data, v405_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v403_data, v406_acc, 2, 1, 0);
          float v412_data = s0[(v234_a ^ (v235_sw & 15))];
          float v417_data = s0[(v239_a ^ (v240_sw & 15))];
          float v422_data = s0[(v244_a ^ (v245_sw & 15))];
          float v427_data = s0[(v249_a ^ (v250_sw & 15))];
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v412_data, v407_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v417_data, v428_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v422_data, v429_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v427_data, v430_acc, 2, 2, 0);
          r4[8] = (v431_acc[0]);
          r4[9] = (v431_acc[1]);
          r4[10] = (v431_acc[2]);
          r4[11] = (v431_acc[3]);
          float r5[12]{};
          // r5 = abs(glb_m3)
          bool v437_g = v27_lead < 4;
          if (v437_g) {
            int32_t v442_a = (v27_lead + 4) - 4;
            #pragma unroll
            for (int32_t v438_k1 = 0; v438_k1 < 12; ++v438_k1) {
              float v445_data = glb_m3[(v442_a + (v438_k1 * 4))];
              r5[v438_k1] = (fabsf(v445_data));
            }
          }
          // s0 = store{r>s, clear}(localShrMem0, r5);
          if (v437_g) {
            #pragma unroll
            for (int32_t v449_z1 = 0; v449_z1 < 12; ++v449_z1) {
              int32_t v454_a = v27_lead + (v449_z1 * 12);
              s0[(v454_a ^ ((v454_a >> 4) & 15))] = 0.0f;
            }
          }
          if ((v27_lead >= 8) && v28_g) {
            #pragma unroll
            for (int32_t v460_z1 = 0; v460_z1 < 12; ++v460_z1) {
              int32_t v465_a = v27_lead + (v460_z1 * 12);
              s0[(v465_a ^ ((v465_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v437_g) {
            int32_t v474_off = v27_lead + 4;
            #pragma unroll
            for (int32_t v469_i1 = 0; v469_i1 < 12; ++v469_i1) {
              float v471_data = r5[v469_i1];
              int32_t v476_a = v474_off + (v469_i1 * 12);
              s0[(v476_a ^ ((v476_a >> 4) & 15))] = v471_data;
            }
          }
          float r6[12]{};
          // r6 = +(r4 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v487_data = s0[(v27_lead ^ v188_sw)];
          float v492_data = s0[(v191_a ^ (v192_sw & 15))];
          float v497_data = s0[(v196_a ^ (v197_sw & 15))];
          float v502_data = s0[(v201_a ^ (v202_sw & 15))];
          float v503_tp{};
          float v504_tp{};
          float v505_tp{};
          float v506_tp{};
          tensorforge::transpose4x4b32(v503_tp, v504_tp, v505_tp, v506_tp, v487_data, v492_data, v497_data, v502_data);
          tensorforge::VectorT<float, 4> v507_acc{};
          float v508_data = r4[0];
          float v509_data = r4[1];
          float v510_data = r4[2];
          float v511_data = r4[3];
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v508_data, v507_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v509_data, v512_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v510_data, v513_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v511_data, v514_acc, 2, 0, 0);
          float v516_data = r4[4];
          float v517_data = r4[5];
          float v518_data = r4[6];
          float v519_data = r4[7];
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v516_data, v515_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v517_data, v520_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v518_data, v521_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v519_data, v522_acc, 2, 1, 0);
          float v524_data = r4[8];
          float v525_data = r4[9];
          float v526_data = r4[10];
          float v527_data = r4[11];
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v524_data, v523_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v525_data, v528_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v526_data, v529_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v527_data, v530_acc, 2, 2, 0);
          r6[0] = (v531_acc[0]);
          r6[1] = (v531_acc[1]);
          r6[2] = (v531_acc[2]);
          r6[3] = (v531_acc[3]);
          float v542_data = s0[(v210_a ^ (v211_sw & 15))];
          float v547_data = s0[(v215_a ^ (v216_sw & 15))];
          float v552_data = s0[(v220_a ^ (v221_sw & 15))];
          float v557_data = s0[(v225_a ^ (v226_sw & 15))];
          float v558_tp{};
          float v559_tp{};
          float v560_tp{};
          float v561_tp{};
          tensorforge::transpose4x4b32(v558_tp, v559_tp, v560_tp, v561_tp, v542_data, v547_data, v552_data, v557_data);
          tensorforge::VectorT<float, 4> v562_acc{};
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v508_data, v562_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v509_data, v567_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v560_tp, v510_data, v568_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v561_tp, v511_data, v569_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v516_data, v570_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v517_data, v575_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v560_tp, v518_data, v576_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v561_tp, v519_data, v577_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v524_data, v578_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v525_data, v583_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v560_tp, v526_data, v584_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v561_tp, v527_data, v585_acc, 2, 2, 0);
          r6[4] = (v586_acc[0]);
          r6[5] = (v586_acc[1]);
          r6[6] = (v586_acc[2]);
          r6[7] = (v586_acc[3]);
          float v597_data = s0[(v234_a ^ (v235_sw & 15))];
          float v602_data = s0[(v239_a ^ (v240_sw & 15))];
          float v607_data = s0[(v244_a ^ (v245_sw & 15))];
          float v612_data = s0[(v249_a ^ (v250_sw & 15))];
          float v613_tp{};
          float v614_tp{};
          float v615_tp{};
          float v616_tp{};
          tensorforge::transpose4x4b32(v613_tp, v614_tp, v615_tp, v616_tp, v597_data, v602_data, v607_data, v612_data);
          tensorforge::VectorT<float, 4> v617_acc{};
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v613_tp, v508_data, v617_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v614_tp, v509_data, v622_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v615_tp, v510_data, v623_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v616_tp, v511_data, v624_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v613_tp, v516_data, v625_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v631_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v614_tp, v517_data, v630_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v615_tp, v518_data, v631_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v616_tp, v519_data, v632_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v613_tp, v524_data, v633_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v614_tp, v525_data, v638_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v615_tp, v526_data, v639_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v641_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v616_tp, v527_data, v640_acc, 2, 2, 0);
          r6[8] = (v641_acc[0]);
          r6[9] = (v641_acc[1]);
          r6[10] = (v641_acc[2]);
          r6[11] = (v641_acc[3]);
          // glb_m4 = store{r>g}(r6);
          if (v28_g) {
            #pragma unroll
            for (int32_t v646_i1 = 0; v646_i1 < 12; ++v646_i1) {
              float v648_data = r6[v646_i1];
              glb_m4[(v27_lead + (v646_i1 * 12))] = v648_data;
            }
          }
        }
      }
    }
  }
}

