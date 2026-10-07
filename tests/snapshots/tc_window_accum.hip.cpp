// === base name ===
kernel_bd96f57ddd6ac130

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bd96f57ddd6ac130 = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bd96f57ddd6ac130(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bd96f57ddd6ac130(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bd96f57ddd6ac130(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_bd96f57ddd6ac130, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_bd96f57ddd6ac130, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (640 * sizeof(float)));
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
  config.sharedMemBytes = 640 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_bd96f57ddd6ac130(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bd96f57ddd6ac130(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_bd96f57ddd6ac130), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_bd96f57ddd6ac130, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_bd96f57ddd6ac130(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 2560 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 16×20(10×17) {0..10}×{1..18} none
    //   m2 20×9(17×9) {1..18}×{0..9} strided
    //   m3 16×20(10×18) {0..10}×{1..19} none
    //   m4 20×9(18×9) {1..19}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 384];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[192];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 180) {
        float v12_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      }
      __syncthreads();
      size_t v14_batchIdLane0 = threadIdx.y % 4;
      int32_t v31_lead = threadIdx.x % 16;
      bool v32_g = v31_lead >= 1;
      bool v42_g = v31_lead < 2;
      bool v208_g = v31_lead < 3;
      int32_t v74_a = v31_lead + ((threadIdx.y % 4) * 10);
      int32_t v81_a = v74_a + 40;
      int32_t v87_a = v74_a + 80;
      int32_t v93_a = v74_a + 120;
      int32_t v99_a = v31_lead + 160;
      int32_t v163_a = v31_lead + 10;
      int32_t v165_a = v31_lead + 20;
      int32_t v167_a = v31_lead + 30;
      int32_t v169_a = v31_lead + 40;
      int32_t v171_a = v31_lead + 50;
      int32_t v173_a = v31_lead + 60;
      int32_t v175_a = v31_lead + 70;
      int32_t v177_a = v31_lead + 80;
      int32_t v179_a = v31_lead + 90;
      int32_t v181_a = v31_lead + 100;
      int32_t v183_a = v31_lead + 110;
      int32_t v185_a = v31_lead + 120;
      int32_t v187_a = v31_lead + 130;
      int32_t v189_a = v31_lead + 140;
      int32_t v191_a = v31_lead + 150;
      int32_t v269_a = v31_lead + 170;
      bool v373_g = v31_lead < 10;
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v16_row = v15_batchIdGroup0 + v14_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v18_batchId0 * 153 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v18_batchId0 * 162 + 0 + m4_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        if (v32_g) {
          int32_t v36_a = v31_lead - 1;
          #pragma unroll
          for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
            float v39_data = __builtin_nontemporal_load(&glb_m2[(v36_a + (v33_i1 * 17))]);
            r0[(v33_i1 * 2)] = v39_data;
          }
        }
        if (v42_g) {
          int32_t v46_a = (v31_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v43_i1 = 0; v43_i1 < 9; ++v43_i1) {
            float v49_data = __builtin_nontemporal_load(&glb_m2[(v46_a + (v43_i1 * 17))]);
            r0[(1 + (v43_i1 * 2))] = v49_data;
          }
        }
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        if (v32_g) {
          int32_t v202_a = v31_lead - 1;
          #pragma unroll
          for (int32_t v199_i1 = 0; v199_i1 < 9; ++v199_i1) {
            float v205_data = __builtin_nontemporal_load(&glb_m4[(v202_a + (v199_i1 * 18))]);
            r2[(v199_i1 * 2)] = v205_data;
          }
        }
        if (v208_g) {
          int32_t v212_a = (v31_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v209_i1 = 0; v209_i1 < 9; ++v209_i1) {
            float v215_data = __builtin_nontemporal_load(&glb_m4[(v212_a + (v209_i1 * 18))]);
            r2[(1 + (v209_i1 * 2))] = v215_data;
          }
        }
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(1, 18)]
        float v53_data = r0[0];
        float v54_data = r0[2];
        float v55_data = r0[4];
        float v56_data = r0[6];
        float v57_tp{};
        float v58_tp{};
        float v59_tp{};
        float v60_tp{};
        tensorforge::transpose4x4b32(v57_tp, v58_tp, v59_tp, v60_tp, v53_data, v54_data, v55_data, v56_data);
        float v61_data = r0[1];
        float v62_data = r0[3];
        float v63_data = r0[5];
        float v64_data = r0[7];
        float v65_tp{};
        float v66_tp{};
        float v67_tp{};
        float v68_tp{};
        tensorforge::transpose4x4b32(v65_tp, v66_tp, v67_tp, v68_tp, v61_data, v62_data, v63_data, v64_data);
        tensorforge::VectorT<float, 4> v69_acc{};
        float v76_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v76_data, v69_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v76_data, v77_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v76_data, v78_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v76_data, v79_acc, 2, 1, 7);
        float v82_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v82_data, v80_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v82_data, v83_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v82_data, v84_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v82_data, v85_acc, 2, 2, 7);
        float v88_data = glb_m1[v87_a];
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v88_data, v86_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v88_data, v89_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v88_data, v90_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v88_data, v91_acc, 2, 3, 7);
        float v94_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v94_data, v92_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v94_data, v95_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v94_data, v96_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v94_data, v97_acc, 2, 0, 7);
        float v100_data = glb_m1[v99_a];
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v100_data, v98_acc, 2, 0, 0);
        r1[0] = (v101_acc[0]);
        r1[1] = (v101_acc[1]);
        r1[2] = (v101_acc[2]);
        r1[3] = (v101_acc[3]);
        float v106_data = r0[8];
        float v107_data = r0[10];
        float v108_data = r0[12];
        float v109_data = r0[14];
        float v110_tp{};
        float v111_tp{};
        float v112_tp{};
        float v113_tp{};
        tensorforge::transpose4x4b32(v110_tp, v111_tp, v112_tp, v113_tp, v106_data, v107_data, v108_data, v109_data);
        float v114_data = r0[9];
        float v115_data = r0[11];
        float v116_data = r0[13];
        float v117_data = r0[15];
        float v118_tp{};
        float v119_tp{};
        float v120_tp{};
        float v121_tp{};
        tensorforge::transpose4x4b32(v118_tp, v119_tp, v120_tp, v121_tp, v114_data, v115_data, v116_data, v117_data);
        tensorforge::VectorT<float, 4> v122_acc{};
        float v129_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v129_data, v122_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v129_data, v130_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v129_data, v131_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v129_data, v132_acc, 2, 1, 7);
        float v135_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v135_data, v133_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v135_data, v136_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v135_data, v137_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v135_data, v138_acc, 2, 2, 7);
        float v141_data = glb_m1[v87_a];
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v141_data, v139_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v141_data, v142_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v141_data, v143_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v141_data, v144_acc, 2, 3, 7);
        float v147_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v147_data, v145_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v147_data, v148_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v147_data, v149_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v147_data, v150_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v100_data, v151_acc, 2, 0, 0);
        r1[4] = (v154_acc[0]);
        r1[5] = (v154_acc[1]);
        r1[6] = (v154_acc[2]);
        r1[7] = (v154_acc[3]);
        float v162_data = glb_m1[v31_lead];
        float v164_data = glb_m1[v163_a];
        float v166_data = glb_m1[v165_a];
        float v168_data = glb_m1[v167_a];
        float v170_data = glb_m1[v169_a];
        float v172_data = glb_m1[v171_a];
        float v174_data = glb_m1[v173_a];
        float v176_data = glb_m1[v175_a];
        float v178_data = glb_m1[v177_a];
        float v180_data = glb_m1[v179_a];
        float v182_data = glb_m1[v181_a];
        float v184_data = glb_m1[v183_a];
        float v186_data = glb_m1[v185_a];
        float v188_data = glb_m1[v187_a];
        float v190_data = glb_m1[v189_a];
        float v192_data = glb_m1[v191_a];
        float v195_acc{};
        float v196_data = r0[16];
        float v197_data = r0[17];
        tensorforge::fmacdpp16<1>(v195_acc, v196_data, v162_data);
        tensorforge::fmacdpp16<2>(v195_acc, v196_data, v164_data);
        tensorforge::fmacdpp16<3>(v195_acc, v196_data, v166_data);
        tensorforge::fmacdpp16<4>(v195_acc, v196_data, v168_data);
        tensorforge::fmacdpp16<5>(v195_acc, v196_data, v170_data);
        tensorforge::fmacdpp16<6>(v195_acc, v196_data, v172_data);
        tensorforge::fmacdpp16<7>(v195_acc, v196_data, v174_data);
        tensorforge::fmacdpp16<8>(v195_acc, v196_data, v176_data);
        tensorforge::fmacdpp16<9>(v195_acc, v196_data, v178_data);
        tensorforge::fmacdpp16<10>(v195_acc, v196_data, v180_data);
        tensorforge::fmacdpp16<11>(v195_acc, v196_data, v182_data);
        tensorforge::fmacdpp16<12>(v195_acc, v196_data, v184_data);
        tensorforge::fmacdpp16<13>(v195_acc, v196_data, v186_data);
        tensorforge::fmacdpp16<14>(v195_acc, v196_data, v188_data);
        tensorforge::fmacdpp16<15>(v195_acc, v196_data, v190_data);
        tensorforge::fmacdpp16<0>(v195_acc, v197_data, v192_data);
        tensorforge::fmacdpp16<1>(v195_acc, v197_data, v100_data);
        r1[8] = v195_acc;
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(1, 19)]
        float ir3[9]{};
        float v220_data = r2[0];
        float v221_data = r2[2];
        float v222_data = r2[4];
        float v223_data = r2[6];
        float v224_tp{};
        float v225_tp{};
        float v226_tp{};
        float v227_tp{};
        tensorforge::transpose4x4b32(v224_tp, v225_tp, v226_tp, v227_tp, v220_data, v221_data, v222_data, v223_data);
        float v228_data = r2[1];
        float v229_data = r2[3];
        float v230_data = r2[5];
        float v231_data = r2[7];
        float v232_tp{};
        float v233_tp{};
        float v234_tp{};
        float v235_tp{};
        tensorforge::transpose4x4b32(v232_tp, v233_tp, v234_tp, v235_tp, v228_data, v229_data, v230_data, v231_data);
        tensorforge::VectorT<float, 4> v236_acc{};
        float v243_data = glb_m3[v74_a];
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v243_data, v236_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v243_data, v244_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v243_data, v245_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v243_data, v246_acc, 2, 1, 7);
        float v249_data = glb_m3[v81_a];
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v249_data, v247_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v249_data, v250_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v249_data, v251_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v249_data, v252_acc, 2, 2, 7);
        float v255_data = glb_m3[v87_a];
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v255_data, v253_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v255_data, v256_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v255_data, v257_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v255_data, v258_acc, 2, 3, 7);
        float v261_data = glb_m3[v93_a];
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v261_data, v259_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v261_data, v262_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v261_data, v263_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v261_data, v264_acc, 2, 0, 7);
        float v267_data = glb_m3[v99_a];
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v267_data, v265_acc, 2, 0, 0);
        float v270_data = glb_m3[v269_a];
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v234_tp, v270_data, v268_acc, 2, 0, 0);
        ir3[0] = (v271_acc[0]);
        ir3[1] = (v271_acc[1]);
        ir3[2] = (v271_acc[2]);
        ir3[3] = (v271_acc[3]);
        float v276_data = r2[8];
        float v277_data = r2[10];
        float v278_data = r2[12];
        float v279_data = r2[14];
        float v280_tp{};
        float v281_tp{};
        float v282_tp{};
        float v283_tp{};
        tensorforge::transpose4x4b32(v280_tp, v281_tp, v282_tp, v283_tp, v276_data, v277_data, v278_data, v279_data);
        float v284_data = r2[9];
        float v285_data = r2[11];
        float v286_data = r2[13];
        float v287_data = r2[15];
        float v288_tp{};
        float v289_tp{};
        float v290_tp{};
        float v291_tp{};
        tensorforge::transpose4x4b32(v288_tp, v289_tp, v290_tp, v291_tp, v284_data, v285_data, v286_data, v287_data);
        tensorforge::VectorT<float, 4> v292_acc{};
        float v299_data = glb_m3[v74_a];
        tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v299_data, v292_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v299_data, v300_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v299_data, v301_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v299_data, v302_acc, 2, 1, 7);
        float v305_data = glb_m3[v81_a];
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v305_data, v303_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v305_data, v306_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v305_data, v307_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v305_data, v308_acc, 2, 2, 7);
        float v311_data = glb_m3[v87_a];
        tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v311_data, v309_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v311_data, v312_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v311_data, v313_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v311_data, v314_acc, 2, 3, 7);
        float v317_data = glb_m3[v93_a];
        tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v317_data, v315_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v317_data, v318_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v317_data, v319_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v317_data, v320_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v267_data, v321_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v290_tp, v270_data, v324_acc, 2, 0, 0);
        ir3[4] = (v327_acc[0]);
        ir3[5] = (v327_acc[1]);
        ir3[6] = (v327_acc[2]);
        ir3[7] = (v327_acc[3]);
        float v335_data = glb_m3[v31_lead];
        float v337_data = glb_m3[v163_a];
        float v339_data = glb_m3[v165_a];
        float v341_data = glb_m3[v167_a];
        float v343_data = glb_m3[v169_a];
        float v345_data = glb_m3[v171_a];
        float v347_data = glb_m3[v173_a];
        float v349_data = glb_m3[v175_a];
        float v351_data = glb_m3[v177_a];
        float v353_data = glb_m3[v179_a];
        float v355_data = glb_m3[v181_a];
        float v357_data = glb_m3[v183_a];
        float v359_data = glb_m3[v185_a];
        float v361_data = glb_m3[v187_a];
        float v363_data = glb_m3[v189_a];
        float v365_data = glb_m3[v191_a];
        float v370_acc{};
        float v371_data = r2[16];
        float v372_data = r2[17];
        tensorforge::fmacdpp16<1>(v370_acc, v371_data, v335_data);
        tensorforge::fmacdpp16<2>(v370_acc, v371_data, v337_data);
        tensorforge::fmacdpp16<3>(v370_acc, v371_data, v339_data);
        tensorforge::fmacdpp16<4>(v370_acc, v371_data, v341_data);
        tensorforge::fmacdpp16<5>(v370_acc, v371_data, v343_data);
        tensorforge::fmacdpp16<6>(v370_acc, v371_data, v345_data);
        tensorforge::fmacdpp16<7>(v370_acc, v371_data, v347_data);
        tensorforge::fmacdpp16<8>(v370_acc, v371_data, v349_data);
        tensorforge::fmacdpp16<9>(v370_acc, v371_data, v351_data);
        tensorforge::fmacdpp16<10>(v370_acc, v371_data, v353_data);
        tensorforge::fmacdpp16<11>(v370_acc, v371_data, v355_data);
        tensorforge::fmacdpp16<12>(v370_acc, v371_data, v357_data);
        tensorforge::fmacdpp16<13>(v370_acc, v371_data, v359_data);
        tensorforge::fmacdpp16<14>(v370_acc, v371_data, v361_data);
        tensorforge::fmacdpp16<15>(v370_acc, v371_data, v363_data);
        tensorforge::fmacdpp16<0>(v370_acc, v372_data, v365_data);
        tensorforge::fmacdpp16<1>(v370_acc, v372_data, v267_data);
        tensorforge::fmacdpp16<2>(v370_acc, v372_data, v270_data);
        ir3[8] = v370_acc;
        // r3 = ir3 + r1
        if (v373_g) {
          #pragma unroll
          for (int32_t v374_n1 = 0; v374_n1 < 9; ++v374_n1) {
            float v376_data = ir3[v374_n1];
            float v377_data = r1[v374_n1];
            r3[v374_n1] = (v377_data + v376_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v373_g) {
          #pragma unroll
          for (int32_t v380_i1 = 0; v380_i1 < 9; ++v380_i1) {
            float v382_data = r3[v380_i1];
            if (batchIdActive0) {
              glb_m0[(v31_lead + (v380_i1 * 10))] = v382_data;
            }
          }
        }
      }
    }
  }
}

