// === base name ===
kernel_cc2bb18f29337d43

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cc2bb18f29337d43 = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cc2bb18f29337d43(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cc2bb18f29337d43(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cc2bb18f29337d43(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_cc2bb18f29337d43, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_cc2bb18f29337d43, block.x * block.y * block.z, 0));
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
void launcher_kernel_cc2bb18f29337d43(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cc2bb18f29337d43(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_cc2bb18f29337d43), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_cc2bb18f29337d43, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_cc2bb18f29337d43(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 2560 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 10×17(10×17) {0..10}×{0..17} none
    //   m2 17×9(17×9) {0..17}×{0..9} strided
    //   m3 10×17(10×17) {0..10}×{0..17} none
    //   m4 17×9(17×9) {0..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v12_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v14_batchIdLane0 = threadIdx.y % 4;
      int32_t v31_lead = threadIdx.x % 16;
      bool v41_g = v31_lead < 1;
      int32_t v90_a = v31_lead + ((threadIdx.y % 4) * 10);
      int32_t v97_a = v90_a + 40;
      int32_t v103_a = v90_a + 80;
      int32_t v109_a = v90_a + 120;
      int32_t v115_a = v31_lead + 160;
      int32_t v179_a = v31_lead + 10;
      int32_t v181_a = v31_lead + 20;
      int32_t v183_a = v31_lead + 30;
      int32_t v185_a = v31_lead + 40;
      int32_t v187_a = v31_lead + 50;
      int32_t v189_a = v31_lead + 60;
      int32_t v191_a = v31_lead + 70;
      int32_t v193_a = v31_lead + 80;
      int32_t v195_a = v31_lead + 90;
      int32_t v197_a = v31_lead + 100;
      int32_t v199_a = v31_lead + 110;
      int32_t v201_a = v31_lead + 120;
      int32_t v203_a = v31_lead + 130;
      int32_t v205_a = v31_lead + 140;
      int32_t v207_a = v31_lead + 150;
      bool v361_g = v31_lead < 10;
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v16_row = v15_batchIdGroup0 + v14_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v18_batchId0 * 153 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v18_batchId0 * 153 + 0 + m4_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
          int32_t v35_lead = v31_lead + (v32_i0 * 16);
          #pragma unroll
          for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
            float v38_data = __builtin_nontemporal_load(&glb_m2[(v35_lead + (v33_i1 * 17))]);
            r0[(v32_i0 + (v33_i1 * 2))] = v38_data;
          }
        }
        if (v41_g) {
          int32_t v44_lead = v31_lead + 16_i32;
          #pragma unroll
          for (int32_t v42_i1 = 0; v42_i1 < 9; ++v42_i1) {
            float v47_data = __builtin_nontemporal_load(&glb_m2[(v44_lead + (v42_i1 * 17))]);
            r0[(1 + (v42_i1 * 2))] = v47_data;
          }
        }
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
          int32_t v54_lead = v31_lead + (v51_i0 * 16);
          #pragma unroll
          for (int32_t v52_i1 = 0; v52_i1 < 9; ++v52_i1) {
            float v57_data = __builtin_nontemporal_load(&glb_m4[(v54_lead + (v52_i1 * 17))]);
            r2[(v51_i0 + (v52_i1 * 2))] = v57_data;
          }
        }
        if (v41_g) {
          int32_t v62_lead = v31_lead + 16_i32;
          #pragma unroll
          for (int32_t v60_i1 = 0; v60_i1 < 9; ++v60_i1) {
            float v65_data = __builtin_nontemporal_load(&glb_m4[(v62_lead + (v60_i1 * 17))]);
            r2[(1 + (v60_i1 * 2))] = v65_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(0, 17)]
        float v69_data = r0[0];
        float v70_data = r0[2];
        float v71_data = r0[4];
        float v72_data = r0[6];
        float v73_tp{};
        float v74_tp{};
        float v75_tp{};
        float v76_tp{};
        tensorforge::transpose4x4b32(v73_tp, v74_tp, v75_tp, v76_tp, v69_data, v70_data, v71_data, v72_data);
        float v77_data = r0[1];
        float v78_data = r0[3];
        float v79_data = r0[5];
        float v80_data = r0[7];
        float v81_tp{};
        float v82_tp{};
        float v83_tp{};
        float v84_tp{};
        tensorforge::transpose4x4b32(v81_tp, v82_tp, v83_tp, v84_tp, v77_data, v78_data, v79_data, v80_data);
        tensorforge::VectorT<float, 4> v85_acc{};
        float v92_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v92_data, v85_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v92_data, v93_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v92_data, v94_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v92_data, v95_acc, 2, 0, 7);
        float v98_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v98_data, v96_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v98_data, v99_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v98_data, v100_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v98_data, v101_acc, 2, 1, 7);
        float v104_data = glb_m1[v103_a];
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v104_data, v102_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v104_data, v105_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v104_data, v106_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v104_data, v107_acc, 2, 2, 7);
        float v110_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v110_data, v108_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v110_data, v111_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v110_data, v112_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v110_data, v113_acc, 2, 3, 7);
        float v116_data = glb_m1[v115_a];
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v116_data, v114_acc, 2, 0, 0);
        r1[0] = (v117_acc[0]);
        r1[1] = (v117_acc[1]);
        r1[2] = (v117_acc[2]);
        r1[3] = (v117_acc[3]);
        float v122_data = r0[8];
        float v123_data = r0[10];
        float v124_data = r0[12];
        float v125_data = r0[14];
        float v126_tp{};
        float v127_tp{};
        float v128_tp{};
        float v129_tp{};
        tensorforge::transpose4x4b32(v126_tp, v127_tp, v128_tp, v129_tp, v122_data, v123_data, v124_data, v125_data);
        float v130_data = r0[9];
        float v131_data = r0[11];
        float v132_data = r0[13];
        float v133_data = r0[15];
        float v134_tp{};
        float v135_tp{};
        float v136_tp{};
        float v137_tp{};
        tensorforge::transpose4x4b32(v134_tp, v135_tp, v136_tp, v137_tp, v130_data, v131_data, v132_data, v133_data);
        tensorforge::VectorT<float, 4> v138_acc{};
        float v145_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v145_data, v138_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v145_data, v146_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v145_data, v147_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v145_data, v148_acc, 2, 0, 7);
        float v151_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v151_data, v149_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v151_data, v152_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v151_data, v153_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v151_data, v154_acc, 2, 1, 7);
        float v157_data = glb_m1[v103_a];
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v157_data, v155_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v157_data, v158_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v157_data, v159_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v157_data, v160_acc, 2, 2, 7);
        float v163_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v163_data, v161_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v163_data, v164_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v163_data, v165_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v163_data, v166_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v116_data, v167_acc, 2, 0, 0);
        r1[4] = (v170_acc[0]);
        r1[5] = (v170_acc[1]);
        r1[6] = (v170_acc[2]);
        r1[7] = (v170_acc[3]);
        float v178_data = glb_m1[v31_lead];
        float v180_data = glb_m1[v179_a];
        float v182_data = glb_m1[v181_a];
        float v184_data = glb_m1[v183_a];
        float v186_data = glb_m1[v185_a];
        float v188_data = glb_m1[v187_a];
        float v190_data = glb_m1[v189_a];
        float v192_data = glb_m1[v191_a];
        float v194_data = glb_m1[v193_a];
        float v196_data = glb_m1[v195_a];
        float v198_data = glb_m1[v197_a];
        float v200_data = glb_m1[v199_a];
        float v202_data = glb_m1[v201_a];
        float v204_data = glb_m1[v203_a];
        float v206_data = glb_m1[v205_a];
        float v208_data = glb_m1[v207_a];
        float v211_acc{};
        float v212_data = r0[16];
        float v213_data = r0[17];
        tensorforge::fmacdpp16<0>(v211_acc, v212_data, v178_data);
        tensorforge::fmacdpp16<1>(v211_acc, v212_data, v180_data);
        tensorforge::fmacdpp16<2>(v211_acc, v212_data, v182_data);
        tensorforge::fmacdpp16<3>(v211_acc, v212_data, v184_data);
        tensorforge::fmacdpp16<4>(v211_acc, v212_data, v186_data);
        tensorforge::fmacdpp16<5>(v211_acc, v212_data, v188_data);
        tensorforge::fmacdpp16<6>(v211_acc, v212_data, v190_data);
        tensorforge::fmacdpp16<7>(v211_acc, v212_data, v192_data);
        tensorforge::fmacdpp16<8>(v211_acc, v212_data, v194_data);
        tensorforge::fmacdpp16<9>(v211_acc, v212_data, v196_data);
        tensorforge::fmacdpp16<10>(v211_acc, v212_data, v198_data);
        tensorforge::fmacdpp16<11>(v211_acc, v212_data, v200_data);
        tensorforge::fmacdpp16<12>(v211_acc, v212_data, v202_data);
        tensorforge::fmacdpp16<13>(v211_acc, v212_data, v204_data);
        tensorforge::fmacdpp16<14>(v211_acc, v212_data, v206_data);
        tensorforge::fmacdpp16<15>(v211_acc, v212_data, v208_data);
        tensorforge::fmacdpp16<0>(v211_acc, v213_data, v116_data);
        r1[8] = v211_acc;
        // wait(r2 = load{g>r}(glb_m4););
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(0, 17)]
        float ir3[9]{};
        float v216_data = r2[0];
        float v217_data = r2[2];
        float v218_data = r2[4];
        float v219_data = r2[6];
        float v220_tp{};
        float v221_tp{};
        float v222_tp{};
        float v223_tp{};
        tensorforge::transpose4x4b32(v220_tp, v221_tp, v222_tp, v223_tp, v216_data, v217_data, v218_data, v219_data);
        float v224_data = r2[1];
        float v225_data = r2[3];
        float v226_data = r2[5];
        float v227_data = r2[7];
        float v228_tp{};
        float v229_tp{};
        float v230_tp{};
        float v231_tp{};
        tensorforge::transpose4x4b32(v228_tp, v229_tp, v230_tp, v231_tp, v224_data, v225_data, v226_data, v227_data);
        tensorforge::VectorT<float, 4> v232_acc{};
        float v239_data = glb_m3[v90_a];
        tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v239_data, v232_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v239_data, v240_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v239_data, v241_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v239_data, v242_acc, 2, 0, 7);
        float v245_data = glb_m3[v97_a];
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v245_data, v243_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v245_data, v246_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v245_data, v247_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v245_data, v248_acc, 2, 1, 7);
        float v251_data = glb_m3[v103_a];
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v251_data, v249_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v251_data, v252_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v251_data, v253_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v251_data, v254_acc, 2, 2, 7);
        float v257_data = glb_m3[v109_a];
        tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v257_data, v255_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v257_data, v258_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v257_data, v259_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v257_data, v260_acc, 2, 3, 7);
        float v263_data = glb_m3[v115_a];
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v263_data, v261_acc, 2, 0, 0);
        ir3[0] = (v264_acc[0]);
        ir3[1] = (v264_acc[1]);
        ir3[2] = (v264_acc[2]);
        ir3[3] = (v264_acc[3]);
        float v269_data = r2[8];
        float v270_data = r2[10];
        float v271_data = r2[12];
        float v272_data = r2[14];
        float v273_tp{};
        float v274_tp{};
        float v275_tp{};
        float v276_tp{};
        tensorforge::transpose4x4b32(v273_tp, v274_tp, v275_tp, v276_tp, v269_data, v270_data, v271_data, v272_data);
        float v277_data = r2[9];
        float v278_data = r2[11];
        float v279_data = r2[13];
        float v280_data = r2[15];
        float v281_tp{};
        float v282_tp{};
        float v283_tp{};
        float v284_tp{};
        tensorforge::transpose4x4b32(v281_tp, v282_tp, v283_tp, v284_tp, v277_data, v278_data, v279_data, v280_data);
        tensorforge::VectorT<float, 4> v285_acc{};
        float v292_data = glb_m3[v90_a];
        tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v292_data, v285_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v292_data, v293_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v292_data, v294_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v292_data, v295_acc, 2, 0, 7);
        float v298_data = glb_m3[v97_a];
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v298_data, v296_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v298_data, v299_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v298_data, v300_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v298_data, v301_acc, 2, 1, 7);
        float v304_data = glb_m3[v103_a];
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v304_data, v302_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v304_data, v305_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v304_data, v306_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v304_data, v307_acc, 2, 2, 7);
        float v310_data = glb_m3[v109_a];
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v310_data, v308_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v310_data, v311_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v310_data, v312_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v310_data, v313_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v263_data, v314_acc, 2, 0, 0);
        ir3[4] = (v317_acc[0]);
        ir3[5] = (v317_acc[1]);
        ir3[6] = (v317_acc[2]);
        ir3[7] = (v317_acc[3]);
        float v325_data = glb_m3[v31_lead];
        float v327_data = glb_m3[v179_a];
        float v329_data = glb_m3[v181_a];
        float v331_data = glb_m3[v183_a];
        float v333_data = glb_m3[v185_a];
        float v335_data = glb_m3[v187_a];
        float v337_data = glb_m3[v189_a];
        float v339_data = glb_m3[v191_a];
        float v341_data = glb_m3[v193_a];
        float v343_data = glb_m3[v195_a];
        float v345_data = glb_m3[v197_a];
        float v347_data = glb_m3[v199_a];
        float v349_data = glb_m3[v201_a];
        float v351_data = glb_m3[v203_a];
        float v353_data = glb_m3[v205_a];
        float v355_data = glb_m3[v207_a];
        float v358_acc{};
        float v359_data = r2[16];
        float v360_data = r2[17];
        tensorforge::fmacdpp16<0>(v358_acc, v359_data, v325_data);
        tensorforge::fmacdpp16<1>(v358_acc, v359_data, v327_data);
        tensorforge::fmacdpp16<2>(v358_acc, v359_data, v329_data);
        tensorforge::fmacdpp16<3>(v358_acc, v359_data, v331_data);
        tensorforge::fmacdpp16<4>(v358_acc, v359_data, v333_data);
        tensorforge::fmacdpp16<5>(v358_acc, v359_data, v335_data);
        tensorforge::fmacdpp16<6>(v358_acc, v359_data, v337_data);
        tensorforge::fmacdpp16<7>(v358_acc, v359_data, v339_data);
        tensorforge::fmacdpp16<8>(v358_acc, v359_data, v341_data);
        tensorforge::fmacdpp16<9>(v358_acc, v359_data, v343_data);
        tensorforge::fmacdpp16<10>(v358_acc, v359_data, v345_data);
        tensorforge::fmacdpp16<11>(v358_acc, v359_data, v347_data);
        tensorforge::fmacdpp16<12>(v358_acc, v359_data, v349_data);
        tensorforge::fmacdpp16<13>(v358_acc, v359_data, v351_data);
        tensorforge::fmacdpp16<14>(v358_acc, v359_data, v353_data);
        tensorforge::fmacdpp16<15>(v358_acc, v359_data, v355_data);
        tensorforge::fmacdpp16<0>(v358_acc, v360_data, v263_data);
        ir3[8] = v358_acc;
        // r3 = ir3 + r1
        if (v361_g) {
          #pragma unroll
          for (int32_t v362_n1 = 0; v362_n1 < 9; ++v362_n1) {
            float v364_data = ir3[v362_n1];
            float v365_data = r1[v362_n1];
            r3[v362_n1] = (v365_data + v364_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v361_g) {
          #pragma unroll
          for (int32_t v368_i1 = 0; v368_i1 < 9; ++v368_i1) {
            float v370_data = r3[v368_i1];
            if (batchIdActive0) {
              glb_m0[(v31_lead + (v368_i1 * 10))] = v370_data;
            }
          }
        }
      }
    }
  }
}

