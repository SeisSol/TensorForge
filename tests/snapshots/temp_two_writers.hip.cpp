// === base name ===
kernel_10f7d6da3bcc316c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_10f7d6da3bcc316c = {{16, 16, 1}, 16, 12, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_10f7d6da3bcc316c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_10f7d6da3bcc316c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_10f7d6da3bcc316c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_10f7d6da3bcc316c, block.x * block.y * block.z, 3328 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (3328 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_10f7d6da3bcc316c, block.x * block.y * block.z, 0));
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
void launcher_kernel_10f7d6da3bcc316c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_10f7d6da3bcc316c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_10f7d6da3bcc316c), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_10f7d6da3bcc316c, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_10f7d6da3bcc316c(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 72 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 72 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v8_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 16;
          bool v25_g = v24_lead < 6;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m0[(v24_lead + (v26_i1 * 6))]);
              r0[v26_i1] = v31_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m1);
          bool v34_g = v24_lead < 12;
          if (v34_g) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v35_i1 * 12))]);
              r1[v35_i1] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + (v43_i1 * 6))]);
              r3[v43_i1] = v48_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v51_data = r1[0];
          float v52_data = r1[1];
          float v53_data = r1[2];
          float v54_data = r1[3];
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
          tensorforge::VectorT<float, 4> v59_acc{};
          float v60_data = r0[0];
          float v61_data = r0[1];
          float v62_data = r0[2];
          float v63_data = r0[3];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v59_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 2, 0, 0);
          float v68_data = r0[4];
          float v69_data = r0[5];
          float v70_data = r0[6];
          float v71_data = r0[7];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 1, 0);
          float v76_data = r0[8];
          float v77_data = r0[9];
          float v78_data = r0[10];
          float v79_data = r0[11];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v75_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v80_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 2, 0);
          r2[0] = (v83_acc[0]);
          r2[1] = (v83_acc[1]);
          r2[2] = (v83_acc[2]);
          r2[3] = (v83_acc[3]);
          float v88_data = r1[4];
          float v89_data = r1[5];
          float v90_data = r1[6];
          float v91_data = r1[7];
          float v92_tp{};
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          tensorforge::transpose4x4b32(v92_tp, v93_tp, v94_tp, v95_tp, v88_data, v89_data, v90_data, v91_data);
          tensorforge::VectorT<float, 4> v96_acc{};
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v110_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v118_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v119_acc, 2, 2, 0);
          r2[4] = (v120_acc[0]);
          r2[5] = (v120_acc[1]);
          r2[6] = (v120_acc[2]);
          r2[7] = (v120_acc[3]);
          float v125_data = r1[8];
          float v126_data = r1[9];
          float v127_data = r1[10];
          float v128_data = r1[11];
          float v129_tp{};
          float v130_tp{};
          float v131_tp{};
          float v132_tp{};
          tensorforge::transpose4x4b32(v129_tp, v130_tp, v131_tp, v132_tp, v125_data, v126_data, v127_data, v128_data);
          tensorforge::VectorT<float, 4> v133_acc{};
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v60_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v61_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v62_data, v139_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v63_data, v140_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v68_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v69_data, v146_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v70_data, v147_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v71_data, v148_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v76_data, v149_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v77_data, v154_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v78_data, v155_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v79_data, v156_acc, 2, 2, 0);
          r2[8] = (v157_acc[0]);
          r2[9] = (v157_acc[1]);
          r2[10] = (v157_acc[2]);
          r2[11] = (v157_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v162_i1 = 0; v162_i1 < 12; ++v162_i1) {
              float v164_data = r2[v162_i1];
              int32_t v168_a = v24_lead + (v162_i1 * 12);
              s0[(v168_a ^ ((v168_a >> 4) & 15))] = v164_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v34_g) {
            #pragma unroll
            for (int32_t v173_i1 = 0; v173_i1 < 12; ++v173_i1) {
              float v178_data = __builtin_nontemporal_load(&glb_m4[(v24_lead + (v173_i1 * 12))]);
              r5[v173_i1] = v178_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          float v188_tp{};
          tensorforge::transpose4x4b32(v185_tp, v186_tp, v187_tp, v188_tp, v51_data, v52_data, v53_data, v54_data);
          tensorforge::VectorT<float, 4> v189_acc{};
          float v190_data = r3[0];
          float v191_data = r3[1];
          float v192_data = r3[2];
          float v193_data = r3[3];
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v190_data, v189_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v191_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v192_data, v195_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v193_data, v196_acc, 2, 0, 0);
          float v198_data = r3[4];
          float v199_data = r3[5];
          float v200_data = r3[6];
          float v201_data = r3[7];
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v198_data, v197_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v199_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v200_data, v203_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v201_data, v204_acc, 2, 1, 0);
          float v206_data = r3[8];
          float v207_data = r3[9];
          float v208_data = r3[10];
          float v209_data = r3[11];
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v206_data, v205_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v207_data, v210_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v208_data, v211_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v209_data, v212_acc, 2, 2, 0);
          r4[0] = (v213_acc[0]);
          r4[1] = (v213_acc[1]);
          r4[2] = (v213_acc[2]);
          r4[3] = (v213_acc[3]);
          float v222_tp{};
          float v223_tp{};
          float v224_tp{};
          float v225_tp{};
          tensorforge::transpose4x4b32(v222_tp, v223_tp, v224_tp, v225_tp, v88_data, v89_data, v90_data, v91_data);
          tensorforge::VectorT<float, 4> v226_acc{};
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v190_data, v226_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v191_data, v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v192_data, v232_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v193_data, v233_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v198_data, v234_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v199_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v200_data, v240_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v201_data, v241_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v206_data, v242_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v207_data, v247_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v208_data, v248_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v209_data, v249_acc, 2, 2, 0);
          r4[4] = (v250_acc[0]);
          r4[5] = (v250_acc[1]);
          r4[6] = (v250_acc[2]);
          r4[7] = (v250_acc[3]);
          float v259_tp{};
          float v260_tp{};
          float v261_tp{};
          float v262_tp{};
          tensorforge::transpose4x4b32(v259_tp, v260_tp, v261_tp, v262_tp, v125_data, v126_data, v127_data, v128_data);
          tensorforge::VectorT<float, 4> v263_acc{};
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v190_data, v263_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v191_data, v268_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v192_data, v269_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v193_data, v270_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v198_data, v271_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v199_data, v276_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v200_data, v277_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v201_data, v278_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v206_data, v279_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v207_data, v284_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v208_data, v285_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v209_data, v286_acc, 2, 2, 0);
          r4[8] = (v287_acc[0]);
          r4[9] = (v287_acc[1]);
          r4[10] = (v287_acc[2]);
          r4[11] = (v287_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v25_g) {
            int32_t v297_off = v24_lead + 6;
            #pragma unroll
            for (int32_t v292_i1 = 0; v292_i1 < 12; ++v292_i1) {
              float v294_data = r4[v292_i1];
              int32_t v299_a = v297_off + (v292_i1 * 12);
              s0[(v299_a ^ ((v299_a >> 4) & 15))] = v294_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          int32_t v309_sw = v24_lead ^ ((v24_lead >> 4) & 15);
          float v310_data = s0[v309_sw];
          int32_t v311_a = v24_lead + 12;
          float v315_data = s0[(v311_a ^ ((v311_a >> 4) & 15))];
          int32_t v316_a = v24_lead + 24;
          float v320_data = s0[(v316_a ^ ((v316_a >> 4) & 15))];
          int32_t v321_a = v24_lead + 36;
          float v325_data = s0[(v321_a ^ ((v321_a >> 4) & 15))];
          float v326_tp{};
          float v327_tp{};
          float v328_tp{};
          float v329_tp{};
          tensorforge::transpose4x4b32(v326_tp, v327_tp, v328_tp, v329_tp, v310_data, v315_data, v320_data, v325_data);
          tensorforge::VectorT<float, 4> v330_acc{};
          float v331_data = r5[0];
          float v332_data = r5[1];
          float v333_data = r5[2];
          float v334_data = r5[3];
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v331_data, v330_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v332_data, v335_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v333_data, v336_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v334_data, v337_acc, 2, 0, 0);
          float v339_data = r5[4];
          float v340_data = r5[5];
          float v341_data = r5[6];
          float v342_data = r5[7];
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v339_data, v338_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v340_data, v343_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v341_data, v344_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v342_data, v345_acc, 2, 1, 0);
          float v347_data = r5[8];
          float v348_data = r5[9];
          float v349_data = r5[10];
          float v350_data = r5[11];
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v347_data, v346_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v348_data, v351_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v349_data, v352_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v329_tp, v350_data, v353_acc, 2, 2, 0);
          r6[0] = (v354_acc[0]);
          r6[1] = (v354_acc[1]);
          r6[2] = (v354_acc[2]);
          r6[3] = (v354_acc[3]);
          int32_t v361_a = v24_lead + 48;
          float v365_data = s0[(v361_a ^ ((v361_a >> 4) & 15))];
          int32_t v366_a = v24_lead + 60;
          float v370_data = s0[(v366_a ^ ((v366_a >> 4) & 15))];
          int32_t v371_a = v24_lead + 72;
          float v375_data = s0[(v371_a ^ ((v371_a >> 4) & 15))];
          int32_t v376_a = v24_lead + 84;
          float v380_data = s0[(v376_a ^ ((v376_a >> 4) & 15))];
          float v381_tp{};
          float v382_tp{};
          float v383_tp{};
          float v384_tp{};
          tensorforge::transpose4x4b32(v381_tp, v382_tp, v383_tp, v384_tp, v365_data, v370_data, v375_data, v380_data);
          tensorforge::VectorT<float, 4> v385_acc{};
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v331_data, v385_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v332_data, v390_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v333_data, v391_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v334_data, v392_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v339_data, v393_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v340_data, v398_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v341_data, v399_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v342_data, v400_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v381_tp, v347_data, v401_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v382_tp, v348_data, v406_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v383_tp, v349_data, v407_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v384_tp, v350_data, v408_acc, 2, 2, 0);
          r6[4] = (v409_acc[0]);
          r6[5] = (v409_acc[1]);
          r6[6] = (v409_acc[2]);
          r6[7] = (v409_acc[3]);
          int32_t v416_a = v24_lead + 96;
          float v420_data = s0[(v416_a ^ ((v416_a >> 4) & 15))];
          int32_t v421_a = v24_lead + 108;
          float v425_data = s0[(v421_a ^ ((v421_a >> 4) & 15))];
          int32_t v426_a = v24_lead + 120;
          float v430_data = s0[(v426_a ^ ((v426_a >> 4) & 15))];
          int32_t v431_a = v24_lead + 132;
          float v435_data = s0[(v431_a ^ ((v431_a >> 4) & 15))];
          float v436_tp{};
          float v437_tp{};
          float v438_tp{};
          float v439_tp{};
          tensorforge::transpose4x4b32(v436_tp, v437_tp, v438_tp, v439_tp, v420_data, v425_data, v430_data, v435_data);
          tensorforge::VectorT<float, 4> v440_acc{};
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v331_data, v440_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v332_data, v445_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v333_data, v446_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v334_data, v447_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v339_data, v448_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v340_data, v453_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v341_data, v454_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v342_data, v455_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v436_tp, v347_data, v456_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v437_tp, v348_data, v461_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v438_tp, v349_data, v462_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v350_data, v463_acc, 2, 2, 0);
          r6[8] = (v464_acc[0]);
          r6[9] = (v464_acc[1]);
          r6[10] = (v464_acc[2]);
          r6[11] = (v464_acc[3]);
          // glb_m3 = store{r>g}(r6);
          if (v34_g) {
            #pragma unroll
            for (int32_t v469_i1 = 0; v469_i1 < 12; ++v469_i1) {
              float v471_data = r6[v469_i1];
              glb_m3[(v24_lead + (v469_i1 * 12))] = v471_data;
            }
          }
        }
      }
    }
  }
}

