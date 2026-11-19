// === base name ===
kernel_b6ab67eff79e6ff2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b6ab67eff79e6ff2 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b6ab67eff79e6ff2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b6ab67eff79e6ff2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b6ab67eff79e6ff2(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b6ab67eff79e6ff2, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b6ab67eff79e6ff2, block.x * block.y * block.z, 0));
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
void launcher_kernel_b6ab67eff79e6ff2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b6ab67eff79e6ff2(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b6ab67eff79e6ff2), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b6ab67eff79e6ff2, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m1_stride, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b6ab67eff79e6ff2(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, size_t m1_stride, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
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
      size_t v185_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v185_flagSecondAhead < numElements0) ? v185_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
      int32_t v196_lead = threadIdx.x % 16;
      bool v197_g = v196_lead < 12;
      bool v215_g = allowed_peel && v197_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v207_i1 = 0; v207_i1 < 20; ++v207_i1) {
        float v210_data = v215_g ? (__builtin_nontemporal_load(&peel_glb_m1[(v196_lead + (v207_i1 * 12))])) : (0.0f);
        r0[v207_i1] = v210_data;
      }
      uint32_t v189_flagWord = flagWordFirst;
      uint32_t v190_flagWordNext = flagWordSecond;
      size_t v225_elementIndex = peel_elementIndex;
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v10_batchId1]);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
        size_t v220_flagAheadAhead = v10_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v220_flagAheadAhead < numElements0) ? v220_flagAheadAhead : v10_batchId1)];
        const bool allowed = static_cast<bool>(v189_flagWord);
        const bool allowed_next = static_cast<bool>(v190_flagWordNext);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 84 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v225_elementIndex * 140 + 0 + m2_extraOffset];
          float r1[14]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
            int32_t v35_lead = v196_lead + (v32_i0 * 16);
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 7; ++v33_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v35_lead + (v33_i1 * 20))]);
              r1[(v32_i0 + (v33_i1 * 2))] = v38_data;
            }
          }
          int32_t v43_lead = v196_lead + 16_i32;
          bool v46_g = v196_lead < 4;
          #pragma unroll
          for (int32_t v41_i1 = 0; v41_i1 < 7; ++v41_i1) {
            float v47_data = v46_g ? (__builtin_nontemporal_load(&glb_m2[(v43_lead + (v41_i1 * 20))])) : (0.0f);
            r1[(1 + (v41_i1 * 2))] = v47_data;
          }
          float r2[7]{};
          // ir2 = +(r0 * r1)
          // [(0, 12), (0, 7)] [(0, 20)]
          float ir2[7]{};
          float v52_data = r1[0];
          float v53_data = r1[2];
          float v54_data = r1[4];
          float v55_data = r1[6];
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          float v59_tp{};
          tensorforge::transpose4x4b32Fused(v56_tp, v57_tp, v58_tp, v59_tp, v52_data, v53_data, v54_data, v55_data);
          float v60_data = r1[1];
          float v61_data = r1[3];
          float v62_data = r1[5];
          float v63_data = r1[7];
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          float v67_tp{};
          tensorforge::transpose4x4b32Fused(v64_tp, v65_tp, v66_tp, v67_tp, v60_data, v61_data, v62_data, v63_data);
          tensorforge::VectorT<float, 4> v68_acc{};
          float v69_data = r0[0];
          float v70_data = r0[1];
          float v71_data = r0[2];
          float v72_data = r0[3];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v68_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 0, 0);
          float v77_data = r0[4];
          float v78_data = r0[5];
          float v79_data = r0[6];
          float v80_data = r0[7];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v76_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 1, 0);
          float v85_data = r0[8];
          float v86_data = r0[9];
          float v87_data = r0[10];
          float v88_data = r0[11];
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v85_data, v84_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v86_data, v89_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v87_data, v90_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v88_data, v91_acc, 2, 2, 0);
          float v93_data = r0[12];
          float v94_data = r0[13];
          float v95_data = r0[14];
          float v96_data = r0[15];
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v93_data, v92_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v94_data, v97_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v98_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v96_data, v99_acc, 2, 3, 0);
          float v101_data = r0[16];
          float v102_data = r0[17];
          float v103_data = r0[18];
          float v104_data = r0[19];
          bool v216_g = allowed_next && v197_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v211_i1 = 0; v211_i1 < 20; ++v211_i1) {
            float v214_data = v216_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v196_lead + (v211_i1 * 12))])) : (0.0f);
            r0[v211_i1] = v214_data;
          }
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v101_data, v100_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v102_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v103_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v104_data, v107_acc, 2, 0, 0);
          ir2[0] = (v108_acc[0]);
          ir2[1] = (v108_acc[1]);
          ir2[2] = (v108_acc[2]);
          ir2[3] = (v108_acc[3]);
          float v113_data = r1[8];
          float v114_data = r1[10];
          float v115_data = r1[12];
          float v117_tp{};
          float v118_tp{};
          float v119_tp{};
          float v120_tp{};
          tensorforge::transpose4x4b32Fused(v117_tp, v118_tp, v119_tp, v120_tp, v113_data, v114_data, v115_data, 0.0f);
          float v121_data = r1[9];
          float v122_data = r1[11];
          float v123_data = r1[13];
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          tensorforge::transpose4x4b32Fused(v124_tp, v125_tp, v126_tp, v127_tp, v121_data, v122_data, v123_data, 0.0f);
          tensorforge::VectorT<float, 4> v128_acc{};
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v69_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v70_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v71_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v72_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v77_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v78_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v79_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v80_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v85_data, v144_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v86_data, v149_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v87_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v88_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v93_data, v152_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v94_data, v157_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v95_data, v158_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v96_data, v159_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v101_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v102_data, v165_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v103_data, v166_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v104_data, v167_acc, 2, 0, 0);
          ir2[4] = (v168_acc[0]);
          ir2[5] = (v168_acc[1]);
          ir2[6] = (v168_acc[2]);
          // r2 = ir2
          // glb_m0 = store{r>g}(r2);
          if (v197_g) {
            #pragma unroll
            for (int32_t v177_i1 = 0; v177_i1 < 7; ++v177_i1) {
              float v179_data = ir2[v177_i1];
              glb_m0[(v196_lead + (v177_i1 * 12))] = v179_data;
            }
          }
        }
        else {
          bool v217_g = allowed_next && v197_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v203_i1 = 0; v203_i1 < 20; ++v203_i1) {
            float v206_data = v217_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v196_lead + (v203_i1 * 12))])) : (0.0f);
            r0[v203_i1] = v206_data;
          }
        }
        v189_flagWord = v190_flagWordNext;
        v190_flagWordNext = flagWordAhead;
        v225_elementIndex = wrap_elementIndex;
      }
    }
  }
}

