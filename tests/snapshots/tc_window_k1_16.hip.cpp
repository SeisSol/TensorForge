// === base name ===
kernel_ea221985747a1592

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ea221985747a1592 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ea221985747a1592(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ea221985747a1592(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ea221985747a1592(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ea221985747a1592, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_ea221985747a1592, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(float)));
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
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ea221985747a1592(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ea221985747a1592(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ea221985747a1592), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_ea221985747a1592, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ea221985747a1592(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×16) {0..16}×{1..17} none ordered
    //   m2 20×9(16×9) {1..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,17]],"name":"m1","ordered":true,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 4;
      size_t v14_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v30_lead = threadIdx.x % 16;
      bool v31_g = v30_lead >= 1;
      bool v41_g = v30_lead < 1;
      int32_t v75_a = ((v30_lead + ((threadIdx.y % 4) * 16)) * 4) + -16;
      int32_t v177_a = (v30_lead * 4) + -16;
      int32_t v187_lead = v30_lead + 16_i32;
      int32_t v189_a = (v187_lead * 4) + -16;
      int32_t v201_a = ((v30_lead + 32_i32) * 4) + -16;
      int32_t v213_a = ((v30_lead + 48_i32) * 4) + -16;
      float r0[18]{};
      size_t v236_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdActive0 = v236_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v236_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v236_row : v14_batchIdGroup0First) * 144 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      if (v31_g) {
        int32_t v244_a = v30_lead - 1;
        #pragma unroll
        for (int32_t v245_i1 = 0; v245_i1 < 9; ++v245_i1) {
          float v248_data = __builtin_nontemporal_load(&peel_glb_m2[(v244_a + (v245_i1 * 16))]);
          r0[(v245_i1 * 2)] = v248_data;
        }
      }
      if (v41_g) {
        int32_t v250_a = v187_lead - 1;
        #pragma unroll
        for (int32_t v251_i1 = 0; v251_i1 < 9; ++v251_i1) {
          float v254_data = __builtin_nontemporal_load(&peel_glb_m2[(v250_a + (v251_i1 * 16))]);
          r0[(1 + (v251_i1 * 2))] = v254_data;
        }
      }
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v233_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v235_groupNext = (v233_groupNextAhead < numElements0) ? v233_groupNextAhead : v15_batchIdGroup0;
        size_t v240_row = v235_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdActive0 = v240_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v240_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v240_row : v235_groupNext) * 144 + 0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 144 + 0 + m0_extraOffset];
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 9)] [(1, 17)]
        float v52_data = r0[0];
        float v53_data = r0[2];
        float v54_data = r0[4];
        float v55_data = r0[6];
        float v56_tp{};
        float v57_tp{};
        float v58_tp{};
        float v59_tp{};
        tensorforge::transpose4x4b32(v56_tp, v57_tp, v58_tp, v59_tp, v52_data, v53_data, v54_data, v55_data);
        float v60_data = r0[1];
        float v61_data = r0[3];
        float v62_data = r0[5];
        float v63_data = r0[7];
        float v64_tp{};
        float v65_tp{};
        float v66_tp{};
        float v67_tp{};
        tensorforge::transpose4x4b32(v64_tp, v65_tp, v66_tp, v67_tp, v60_data, v61_data, v62_data, v63_data);
        tensorforge::VectorT<float, 4> v68_acc{};
        tensorforge::VectorT<float, 4> v76_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v75_a];
        float v77_el = v76_data[0];
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v77_el, v68_acc, 2, 0, 4);
        float v79_el = v76_data[1];
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_el, v78_acc, 2, 0, 4);
        float v81_el = v76_data[2];
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v81_el, v80_acc, 2, 0, 4);
        float v83_el = v76_data[3];
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v83_el, v82_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v77_el, v84_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_el, v86_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v81_el, v88_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v83_el, v90_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v77_el, v92_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_el, v94_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v81_el, v96_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v83_el, v98_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v77_el, v100_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_el, v102_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v81_el, v104_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v83_el, v106_acc, 2, 0, 7);
        r1[0] = (v108_acc[0]);
        r1[1] = (v108_acc[1]);
        r1[2] = (v108_acc[2]);
        r1[3] = (v108_acc[3]);
        float v113_data = r0[8];
        float v114_data = r0[10];
        float v115_data = r0[12];
        float v116_data = r0[14];
        float v117_tp{};
        float v118_tp{};
        float v119_tp{};
        float v120_tp{};
        tensorforge::transpose4x4b32(v117_tp, v118_tp, v119_tp, v120_tp, v113_data, v114_data, v115_data, v116_data);
        float v121_data = r0[9];
        float v122_data = r0[11];
        float v123_data = r0[13];
        float v124_data = r0[15];
        float v125_tp{};
        float v126_tp{};
        float v127_tp{};
        float v128_tp{};
        tensorforge::transpose4x4b32(v125_tp, v126_tp, v127_tp, v128_tp, v121_data, v122_data, v123_data, v124_data);
        tensorforge::VectorT<float, 4> v129_acc{};
        tensorforge::VectorT<float, 4> v137_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v75_a];
        float v138_el = v137_data[0];
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v138_el, v129_acc, 2, 0, 4);
        float v140_el = v137_data[1];
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v140_el, v139_acc, 2, 0, 4);
        float v142_el = v137_data[2];
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v142_el, v141_acc, 2, 0, 4);
        float v144_el = v137_data[3];
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v144_el, v143_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v138_el, v145_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v140_el, v147_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v142_el, v149_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v144_el, v151_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v138_el, v153_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v140_el, v155_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v142_el, v157_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v144_el, v159_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v138_el, v161_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v140_el, v163_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v142_el, v165_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v144_el, v167_acc, 2, 0, 7);
        r1[4] = (v169_acc[0]);
        r1[5] = (v169_acc[1]);
        r1[6] = (v169_acc[2]);
        r1[7] = (v169_acc[3]);
        tensorforge::VectorT<float, 4> v178_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v177_a];
        float v179_el = v178_data[0];
        float v181_el = v178_data[1];
        float v183_el = v178_data[2];
        float v185_el = v178_data[3];
        tensorforge::VectorT<float, 4> v190_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v189_a];
        float v191_el = v190_data[0];
        float v193_el = v190_data[1];
        float v195_el = v190_data[2];
        float v197_el = v190_data[3];
        tensorforge::VectorT<float, 4> v202_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v201_a];
        float v203_el = v202_data[0];
        float v205_el = v202_data[1];
        float v207_el = v202_data[2];
        float v209_el = v202_data[3];
        tensorforge::VectorT<float, 4> v214_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v213_a];
        float v215_el = v214_data[0];
        float v217_el = v214_data[1];
        float v219_el = v214_data[2];
        float v221_el = v214_data[3];
        float v222_acc{};
        float v223_data = r0[16];
        float v224_data = r0[17];
        // r0 = load{g>r}(wrap_glb_m2);
        if (v31_g) {
          int32_t v257_a = v30_lead - 1;
          #pragma unroll
          for (int32_t v258_i1 = 0; v258_i1 < 9; ++v258_i1) {
            float v261_data = __builtin_nontemporal_load(&wrap_glb_m2[(v257_a + (v258_i1 * 16))]);
            r0[(v258_i1 * 2)] = v261_data;
          }
        }
        if (v41_g) {
          int32_t v263_a = v187_lead - 1;
          #pragma unroll
          for (int32_t v264_i1 = 0; v264_i1 < 9; ++v264_i1) {
            float v267_data = __builtin_nontemporal_load(&wrap_glb_m2[(v263_a + (v264_i1 * 16))]);
            r0[(1 + (v264_i1 * 2))] = v267_data;
          }
        }
        tensorforge::fmacdpp16<1>(v222_acc, v223_data, v179_el);
        tensorforge::fmacdpp16<2>(v222_acc, v223_data, v181_el);
        tensorforge::fmacdpp16<3>(v222_acc, v223_data, v183_el);
        tensorforge::fmacdpp16<4>(v222_acc, v223_data, v185_el);
        tensorforge::fmacdpp16<5>(v222_acc, v223_data, v191_el);
        tensorforge::fmacdpp16<6>(v222_acc, v223_data, v193_el);
        tensorforge::fmacdpp16<7>(v222_acc, v223_data, v195_el);
        tensorforge::fmacdpp16<8>(v222_acc, v223_data, v197_el);
        tensorforge::fmacdpp16<9>(v222_acc, v223_data, v203_el);
        tensorforge::fmacdpp16<10>(v222_acc, v223_data, v205_el);
        tensorforge::fmacdpp16<11>(v222_acc, v223_data, v207_el);
        tensorforge::fmacdpp16<12>(v222_acc, v223_data, v209_el);
        tensorforge::fmacdpp16<13>(v222_acc, v223_data, v215_el);
        tensorforge::fmacdpp16<14>(v222_acc, v223_data, v217_el);
        tensorforge::fmacdpp16<15>(v222_acc, v223_data, v219_el);
        tensorforge::fmacdpp16<0>(v222_acc, v224_data, v221_el);
        r1[8] = v222_acc;
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v225_i0 = 0; v225_i0 < 1; ++v225_i0) {
          #pragma unroll
          for (int32_t v226_i1 = 0; v226_i1 < 9; ++v226_i1) {
            float v228_data = r1[(v225_i0 + v226_i1)];
            if (batchIdActive0) {
              glb_m0[((v30_lead + (v225_i0 * 16)) + (v226_i1 * 16))] = v228_data;
            }
          }
        }
      }
    }
  }
}

