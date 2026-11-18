// === base name ===
kernel_17f89ff2c8e72ee2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_17f89ff2c8e72ee2 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_17f89ff2c8e72ee2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_17f89ff2c8e72ee2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_17f89ff2c8e72ee2(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_17f89ff2c8e72ee2, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_17f89ff2c8e72ee2, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_17f89ff2c8e72ee2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_17f89ff2c8e72ee2(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_17f89ff2c8e72ee2), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_17f89ff2c8e72ee2, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m1_stride, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_17f89ff2c8e72ee2(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, size_t m1_stride, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×7(12×7) {0..12}×{0..7} strided
    //   m1 12×20(12×20) {0..12}×{0..20} indexed(stride=runtime)
    //   m2 20×7(20×7) {0..20}×{0..7} indexed
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,7]],"name":"m0","ordered":false,"parts":1,"shape":[12,7],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"indexed","alias":"B","bbox":[[0,0],[20,7]],"name":"m2","ordered":false,"parts":1,"shape":[20,7],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,7]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,7]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"indexed","bbox":[[0,0],[20,7]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,7]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      float r0[20]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v184_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v184_flagSecondAhead < numElements0) ? v184_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[static_cast<size_t>(elementIndices[v3_batchId1]) * m1_stride + 0 + m1_extraOffset];
      int32_t v194_lead = threadIdx.x % 16;
      bool v195_g = v194_lead < 12;
      bool v212_g = allowed_peel && v195_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v204_i1 = 0; v204_i1 < 20; ++v204_i1) {
        float v207_data = v212_g ? (__builtin_nontemporal_load(&peel_glb_m1[(v194_lead + (v204_i1 * 12))])) : (0.0f);
        r0[v204_i1] = v207_data;
      }
      uint32_t v188_flagWord = flagWordFirst;
      uint32_t v189_flagWordNext = flagWordSecond;
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[static_cast<size_t>(elementIndices[v10_batchId1]) * m1_stride + 0 + m1_extraOffset];
        size_t v217_flagAheadAhead = v10_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v217_flagAheadAhead < numElements0) ? v217_flagAheadAhead : v10_batchId1)];
        const bool allowed = static_cast<bool>(v188_flagWord);
        const bool allowed_next = static_cast<bool>(v189_flagWordNext);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 84 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[v7_batchId0]) * 140 + 0 + m2_extraOffset];
          float r1[14]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v34_lead = v194_lead + (v31_i0 * 16);
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 7; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m2[(v34_lead + (v32_i1 * 20))]);
              r1[(v31_i0 + (v32_i1 * 2))] = v37_data;
            }
          }
          int32_t v42_lead = v194_lead + 16_i32;
          bool v45_g = v194_lead < 4;
          #pragma unroll
          for (int32_t v40_i1 = 0; v40_i1 < 7; ++v40_i1) {
            float v46_data = v45_g ? (__builtin_nontemporal_load(&glb_m2[(v42_lead + (v40_i1 * 20))])) : (0.0f);
            r1[(1 + (v40_i1 * 2))] = v46_data;
          }
          float r2[7]{};
          // ir2 = +(r0 * r1)
          // [(0, 12), (0, 7)] [(0, 20)]
          float ir2[7]{};
          float v51_data = r1[0];
          float v52_data = r1[2];
          float v53_data = r1[4];
          float v54_data = r1[6];
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32Fused(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
          float v59_data = r1[1];
          float v60_data = r1[3];
          float v61_data = r1[5];
          float v62_data = r1[7];
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          tensorforge::transpose4x4b32Fused(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v67_acc{};
          float v68_data = r0[0];
          float v69_data = r0[1];
          float v70_data = r0[2];
          float v71_data = r0[3];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 0, 0);
          float v76_data = r0[4];
          float v77_data = r0[5];
          float v78_data = r0[6];
          float v79_data = r0[7];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v80_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 1, 0);
          float v84_data = r0[8];
          float v85_data = r0[9];
          float v86_data = r0[10];
          float v87_data = r0[11];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v85_data, v88_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v86_data, v89_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v87_data, v90_acc, 2, 2, 0);
          float v92_data = r0[12];
          float v93_data = r0[13];
          float v94_data = r0[14];
          float v95_data = r0[15];
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v92_data, v91_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v93_data, v96_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v94_data, v97_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v98_acc, 2, 3, 0);
          float v100_data = r0[16];
          float v101_data = r0[17];
          float v102_data = r0[18];
          float v103_data = r0[19];
          bool v213_g = allowed_next && v195_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v208_i1 = 0; v208_i1 < 20; ++v208_i1) {
            float v211_data = v213_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v194_lead + (v208_i1 * 12))])) : (0.0f);
            r0[v208_i1] = v211_data;
          }
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v100_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v101_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v102_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v103_data, v106_acc, 2, 0, 0);
          ir2[0] = (v107_acc[0]);
          ir2[1] = (v107_acc[1]);
          ir2[2] = (v107_acc[2]);
          ir2[3] = (v107_acc[3]);
          float v112_data = r1[8];
          float v113_data = r1[10];
          float v114_data = r1[12];
          float v116_tp{};
          float v117_tp{};
          float v118_tp{};
          float v119_tp{};
          tensorforge::transpose4x4b32Fused(v116_tp, v117_tp, v118_tp, v119_tp, v112_data, v113_data, v114_data, 0.0f);
          float v120_data = r1[9];
          float v121_data = r1[11];
          float v122_data = r1[13];
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          tensorforge::transpose4x4b32Fused(v123_tp, v124_tp, v125_tp, v126_tp, v120_data, v121_data, v122_data, 0.0f);
          tensorforge::VectorT<float, 4> v127_acc{};
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v68_data, v127_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v69_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v70_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v71_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v76_data, v135_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v77_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v78_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v79_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v84_data, v143_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v85_data, v148_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v86_data, v149_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v87_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v92_data, v151_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v93_data, v156_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v94_data, v157_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v95_data, v158_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v100_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v101_data, v164_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v102_data, v165_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v103_data, v166_acc, 2, 0, 0);
          ir2[4] = (v167_acc[0]);
          ir2[5] = (v167_acc[1]);
          ir2[6] = (v167_acc[2]);
          // r2 = ir2
          // glb_m0 = store{r>g}(r2);
          if (v195_g) {
            #pragma unroll
            for (int32_t v176_i1 = 0; v176_i1 < 7; ++v176_i1) {
              float v178_data = ir2[v176_i1];
              glb_m0[(v194_lead + (v176_i1 * 12))] = v178_data;
            }
          }
        }
        else {
          bool v214_g = allowed_next && v195_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v200_i1 = 0; v200_i1 < 20; ++v200_i1) {
            float v203_data = v214_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v194_lead + (v200_i1 * 12))])) : (0.0f);
            r0[v200_i1] = v203_data;
          }
        }
        v188_flagWord = v189_flagWordNext;
        v189_flagWordNext = flagWordAhead;
      }
    }
  }
}

