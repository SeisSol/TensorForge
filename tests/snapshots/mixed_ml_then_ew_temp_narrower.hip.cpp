// === base name ===
kernel_682aff8bb3c3f7ba

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_682aff8bb3c3f7ba = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_682aff8bb3c3f7ba(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_682aff8bb3c3f7ba(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_682aff8bb3c3f7ba(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_682aff8bb3c3f7ba, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_682aff8bb3c3f7ba, block.x * block.y * block.z, 0));
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
void launcher_kernel_682aff8bb3c3f7ba(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_682aff8bb3c3f7ba(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_682aff8bb3c3f7ba), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_682aff8bb3c3f7ba, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_682aff8bb3c3f7ba(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 48 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v8_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 16;
          bool v25_g = v24_lead < 12;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m0[(v24_lead + (v26_i1 * 12))]);
              r0[v26_i1] = v31_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          if (v25_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v34_i1 * 12))]);
              r1[v34_i1] = v39_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v164_i1 = 0; v164_i1 < 12; ++v164_i1) {
              float v169_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + (v164_i1 * 12))]);
              r3[v164_i1] = v169_data;
            }
          }
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v42_data = r1[0];
          float v43_data = r1[1];
          float v44_data = r1[2];
          float v45_data = r1[3];
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
          tensorforge::VectorT<float, 4> v50_acc{};
          float v51_data = r0[0];
          float v52_data = r0[1];
          float v53_data = r0[2];
          float v54_data = r0[3];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v50_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          float v59_data = r0[4];
          float v60_data = r0[5];
          float v61_data = r0[6];
          float v62_data = r0[7];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v58_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          float v67_data = r0[8];
          float v68_data = r0[9];
          float v69_data = r0[10];
          float v70_data = r0[11];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v66_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          r2[0] = (v74_acc[0]);
          r2[1] = (v74_acc[1]);
          r2[2] = (v74_acc[2]);
          r2[3] = (v74_acc[3]);
          float v79_data = r1[4];
          float v80_data = r1[5];
          float v81_data = r1[6];
          float v82_data = r1[7];
          float v83_tp{};
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          tensorforge::transpose4x4b32(v83_tp, v84_tp, v85_tp, v86_tp, v79_data, v80_data, v81_data, v82_data);
          tensorforge::VectorT<float, 4> v87_acc{};
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v51_data, v87_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v94_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v95_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v102_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v103_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v108_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v109_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v110_acc, 2, 2, 0);
          r2[4] = (v111_acc[0]);
          r2[5] = (v111_acc[1]);
          r2[6] = (v111_acc[2]);
          r2[7] = (v111_acc[3]);
          float v116_data = r1[8];
          float v117_data = r1[9];
          float v118_data = r1[10];
          float v119_data = r1[11];
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          tensorforge::transpose4x4b32(v120_tp, v121_tp, v122_tp, v123_tp, v116_data, v117_data, v118_data, v119_data);
          tensorforge::VectorT<float, 4> v124_acc{};
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v51_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v52_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v53_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v54_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v59_data, v132_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v60_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v61_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v62_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v67_data, v140_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v68_data, v145_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v69_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v70_data, v147_acc, 2, 2, 0);
          r2[8] = (v148_acc[0]);
          r2[9] = (v148_acc[1]);
          r2[10] = (v148_acc[2]);
          r2[11] = (v148_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v153_i1 = 0; v153_i1 < 12; ++v153_i1) {
              float v155_data = r2[v153_i1];
              int32_t v159_a = v24_lead + (v153_i1 * 12);
              s0[(v159_a ^ ((v159_a >> 4) & 15))] = v155_data;
            }
          }
          float r4[12]{};
          // r4 = +(s0 * r3) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v172_data = r3[0];
          float v173_data = r3[1];
          float v174_data = r3[2];
          float v175_data = r3[3];
          float v176_tp{};
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          tensorforge::transpose4x4b32(v176_tp, v177_tp, v178_tp, v179_tp, v172_data, v173_data, v174_data, v175_data);
          tensorforge::VectorT<float, 4> v180_acc{};
          int32_t v185_sw = (v24_lead >> 4) & 15;
          int32_t v186_sw = v24_lead ^ v185_sw;
          float v187_data = s0[v186_sw];
          int32_t v188_a = v24_lead + 12;
          int32_t v189_sw = v188_a >> 4;
          float v192_data = s0[(v188_a ^ (v189_sw & 15))];
          int32_t v193_a = v24_lead + 24;
          int32_t v194_sw = v193_a >> 4;
          float v197_data = s0[(v193_a ^ (v194_sw & 15))];
          int32_t v198_a = v24_lead + 36;
          int32_t v199_sw = v198_a >> 4;
          float v202_data = s0[(v198_a ^ (v199_sw & 15))];
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v187_data, v180_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v192_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v197_data, v204_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v202_data, v205_acc, 2, 0, 0);
          int32_t v207_a = v24_lead + 48;
          int32_t v208_sw = v207_a >> 4;
          float v211_data = s0[(v207_a ^ (v208_sw & 15))];
          int32_t v212_a = v24_lead + 60;
          int32_t v213_sw = v212_a >> 4;
          float v216_data = s0[(v212_a ^ (v213_sw & 15))];
          int32_t v217_a = v24_lead + 72;
          int32_t v218_sw = v217_a >> 4;
          float v221_data = s0[(v217_a ^ (v218_sw & 15))];
          int32_t v222_a = v24_lead + 84;
          int32_t v223_sw = v222_a >> 4;
          float v226_data = s0[(v222_a ^ (v223_sw & 15))];
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v211_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v216_data, v227_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v221_data, v228_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v226_data, v229_acc, 2, 1, 0);
          int32_t v231_a = v24_lead + 96;
          int32_t v232_sw = v231_a >> 4;
          float v235_data = s0[(v231_a ^ (v232_sw & 15))];
          int32_t v236_a = v24_lead + 108;
          int32_t v237_sw = v236_a >> 4;
          float v240_data = s0[(v236_a ^ (v237_sw & 15))];
          int32_t v241_a = v24_lead + 120;
          int32_t v242_sw = v241_a >> 4;
          float v245_data = s0[(v241_a ^ (v242_sw & 15))];
          int32_t v246_a = v24_lead + 132;
          int32_t v247_sw = v246_a >> 4;
          float v250_data = s0[(v246_a ^ (v247_sw & 15))];
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v235_data, v230_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v240_data, v251_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v245_data, v252_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v250_data, v253_acc, 2, 2, 0);
          r4[0] = (v254_acc[0]);
          r4[1] = (v254_acc[1]);
          r4[2] = (v254_acc[2]);
          r4[3] = (v254_acc[3]);
          float v259_data = r3[4];
          float v260_data = r3[5];
          float v261_data = r3[6];
          float v262_data = r3[7];
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          tensorforge::transpose4x4b32(v263_tp, v264_tp, v265_tp, v266_tp, v259_data, v260_data, v261_data, v262_data);
          tensorforge::VectorT<float, 4> v267_acc{};
          float v274_data = s0[(v24_lead ^ v185_sw)];
          float v279_data = s0[(v188_a ^ (v189_sw & 15))];
          float v284_data = s0[(v193_a ^ (v194_sw & 15))];
          float v289_data = s0[(v198_a ^ (v199_sw & 15))];
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v274_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v279_data, v290_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v284_data, v291_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v289_data, v292_acc, 2, 0, 0);
          float v298_data = s0[(v207_a ^ (v208_sw & 15))];
          float v303_data = s0[(v212_a ^ (v213_sw & 15))];
          float v308_data = s0[(v217_a ^ (v218_sw & 15))];
          float v313_data = s0[(v222_a ^ (v223_sw & 15))];
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v298_data, v293_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v303_data, v314_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v308_data, v315_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v313_data, v316_acc, 2, 1, 0);
          float v322_data = s0[(v231_a ^ (v232_sw & 15))];
          float v327_data = s0[(v236_a ^ (v237_sw & 15))];
          float v332_data = s0[(v241_a ^ (v242_sw & 15))];
          float v337_data = s0[(v246_a ^ (v247_sw & 15))];
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v322_data, v317_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v327_data, v338_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v332_data, v339_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v337_data, v340_acc, 2, 2, 0);
          r4[4] = (v341_acc[0]);
          r4[5] = (v341_acc[1]);
          r4[6] = (v341_acc[2]);
          r4[7] = (v341_acc[3]);
          float v346_data = r3[8];
          float v347_data = r3[9];
          float v348_data = r3[10];
          float v349_data = r3[11];
          float v350_tp{};
          float v351_tp{};
          float v352_tp{};
          float v353_tp{};
          tensorforge::transpose4x4b32(v350_tp, v351_tp, v352_tp, v353_tp, v346_data, v347_data, v348_data, v349_data);
          tensorforge::VectorT<float, 4> v354_acc{};
          float v361_data = s0[(v24_lead ^ v185_sw)];
          float v366_data = s0[(v188_a ^ (v189_sw & 15))];
          float v371_data = s0[(v193_a ^ (v194_sw & 15))];
          float v376_data = s0[(v198_a ^ (v199_sw & 15))];
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v361_data, v354_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v366_data, v377_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v371_data, v378_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v376_data, v379_acc, 2, 0, 0);
          float v385_data = s0[(v207_a ^ (v208_sw & 15))];
          float v390_data = s0[(v212_a ^ (v213_sw & 15))];
          float v395_data = s0[(v217_a ^ (v218_sw & 15))];
          float v400_data = s0[(v222_a ^ (v223_sw & 15))];
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v385_data, v380_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v390_data, v401_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v395_data, v402_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v400_data, v403_acc, 2, 1, 0);
          float v409_data = s0[(v231_a ^ (v232_sw & 15))];
          float v414_data = s0[(v236_a ^ (v237_sw & 15))];
          float v419_data = s0[(v241_a ^ (v242_sw & 15))];
          float v424_data = s0[(v246_a ^ (v247_sw & 15))];
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v409_data, v404_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v414_data, v425_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v419_data, v426_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v424_data, v427_acc, 2, 2, 0);
          r4[8] = (v428_acc[0]);
          r4[9] = (v428_acc[1]);
          r4[10] = (v428_acc[2]);
          r4[11] = (v428_acc[3]);
          float r5[12]{};
          // r5 = abs(glb_m3)
          bool v434_g = v24_lead < 4;
          if (v434_g) {
            int32_t v439_a = (v24_lead + 4) - 4;
            #pragma unroll
            for (int32_t v435_k1 = 0; v435_k1 < 12; ++v435_k1) {
              float v442_data = glb_m3[(v439_a + (v435_k1 * 4))];
              r5[v435_k1] = (fabsf(v442_data));
            }
          }
          // s0 = store{r>s, clear}(localShrMem0, r5);
          if (v434_g) {
            #pragma unroll
            for (int32_t v446_z1 = 0; v446_z1 < 12; ++v446_z1) {
              int32_t v451_a = v24_lead + (v446_z1 * 12);
              s0[(v451_a ^ ((v451_a >> 4) & 15))] = 0.0f;
            }
          }
          if ((v24_lead >= 8) && v25_g) {
            #pragma unroll
            for (int32_t v457_z1 = 0; v457_z1 < 12; ++v457_z1) {
              int32_t v462_a = v24_lead + (v457_z1 * 12);
              s0[(v462_a ^ ((v462_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v434_g) {
            int32_t v471_off = v24_lead + 4;
            #pragma unroll
            for (int32_t v466_i1 = 0; v466_i1 < 12; ++v466_i1) {
              float v468_data = r5[v466_i1];
              int32_t v473_a = v471_off + (v466_i1 * 12);
              s0[(v473_a ^ ((v473_a >> 4) & 15))] = v468_data;
            }
          }
          float r6[12]{};
          // r6 = +(r4 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          int32_t v483_sw = v24_lead ^ v185_sw;
          float v484_data = s0[v483_sw];
          float v489_data = s0[(v188_a ^ (v189_sw & 15))];
          float v494_data = s0[(v193_a ^ (v194_sw & 15))];
          float v499_data = s0[(v198_a ^ (v199_sw & 15))];
          float v500_tp{};
          float v501_tp{};
          float v502_tp{};
          float v503_tp{};
          tensorforge::transpose4x4b32(v500_tp, v501_tp, v502_tp, v503_tp, v484_data, v489_data, v494_data, v499_data);
          tensorforge::VectorT<float, 4> v504_acc{};
          float v505_data = r4[0];
          float v506_data = r4[1];
          float v507_data = r4[2];
          float v508_data = r4[3];
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v505_data, v504_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v506_data, v509_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v507_data, v510_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v508_data, v511_acc, 2, 0, 0);
          float v513_data = r4[4];
          float v514_data = r4[5];
          float v515_data = r4[6];
          float v516_data = r4[7];
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v513_data, v512_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v514_data, v517_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v515_data, v518_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v516_data, v519_acc, 2, 1, 0);
          float v521_data = r4[8];
          float v522_data = r4[9];
          float v523_data = r4[10];
          float v524_data = r4[11];
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v521_data, v520_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v522_data, v525_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v523_data, v526_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v524_data, v527_acc, 2, 2, 0);
          r6[0] = (v528_acc[0]);
          r6[1] = (v528_acc[1]);
          r6[2] = (v528_acc[2]);
          r6[3] = (v528_acc[3]);
          float v539_data = s0[(v207_a ^ (v208_sw & 15))];
          float v544_data = s0[(v212_a ^ (v213_sw & 15))];
          float v549_data = s0[(v217_a ^ (v218_sw & 15))];
          float v554_data = s0[(v222_a ^ (v223_sw & 15))];
          float v555_tp{};
          float v556_tp{};
          float v557_tp{};
          float v558_tp{};
          tensorforge::transpose4x4b32(v555_tp, v556_tp, v557_tp, v558_tp, v539_data, v544_data, v549_data, v554_data);
          tensorforge::VectorT<float, 4> v559_acc{};
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v505_data, v559_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v506_data, v564_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v507_data, v565_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v508_data, v566_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v513_data, v567_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v514_data, v572_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v515_data, v573_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v516_data, v574_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v555_tp, v521_data, v575_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v522_data, v580_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v523_data, v581_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v524_data, v582_acc, 2, 2, 0);
          r6[4] = (v583_acc[0]);
          r6[5] = (v583_acc[1]);
          r6[6] = (v583_acc[2]);
          r6[7] = (v583_acc[3]);
          float v594_data = s0[(v231_a ^ (v232_sw & 15))];
          float v599_data = s0[(v236_a ^ (v237_sw & 15))];
          float v604_data = s0[(v241_a ^ (v242_sw & 15))];
          float v609_data = s0[(v246_a ^ (v247_sw & 15))];
          float v610_tp{};
          float v611_tp{};
          float v612_tp{};
          float v613_tp{};
          tensorforge::transpose4x4b32(v610_tp, v611_tp, v612_tp, v613_tp, v594_data, v599_data, v604_data, v609_data);
          tensorforge::VectorT<float, 4> v614_acc{};
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v610_tp, v505_data, v614_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v611_tp, v506_data, v619_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v612_tp, v507_data, v620_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v613_tp, v508_data, v621_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v610_tp, v513_data, v622_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v611_tp, v514_data, v627_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v612_tp, v515_data, v628_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v613_tp, v516_data, v629_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v610_tp, v521_data, v630_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v636_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v611_tp, v522_data, v635_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v612_tp, v523_data, v636_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v613_tp, v524_data, v637_acc, 2, 2, 0);
          r6[8] = (v638_acc[0]);
          r6[9] = (v638_acc[1]);
          r6[10] = (v638_acc[2]);
          r6[11] = (v638_acc[3]);
          // glb_m4 = store{r>g}(r6);
          if (v25_g) {
            #pragma unroll
            for (int32_t v643_i1 = 0; v643_i1 < 12; ++v643_i1) {
              float v645_data = r6[v643_i1];
              glb_m4[(v24_lead + (v643_i1 * 12))] = v645_data;
            }
          }
        }
      }
    }
  }
}

