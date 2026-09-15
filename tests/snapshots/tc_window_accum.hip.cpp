// === base name ===
kernel_6f840c33df1cdd0a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6f840c33df1cdd0a = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6f840c33df1cdd0a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6f840c33df1cdd0a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6f840c33df1cdd0a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6f840c33df1cdd0a, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6f840c33df1cdd0a, block.x * block.y * block.z, 0));
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
void launcher_kernel_6f840c33df1cdd0a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6f840c33df1cdd0a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6f840c33df1cdd0a), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_6f840c33df1cdd0a, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6f840c33df1cdd0a(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 384];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[192];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 180) {
        float v8_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v8_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v10_batchIdLane0 = threadIdx.y % 4;
      int32_t v28_lead = threadIdx.x % 16;
      bool v29_g = v28_lead >= 1;
      bool v39_g = v28_lead < 2;
      bool v59_g = v28_lead < 3;
      int32_t v91_a = v28_lead + ((threadIdx.y % 4) * 10);
      int32_t v98_a = v91_a + 40;
      int32_t v104_a = v91_a + 80;
      int32_t v110_a = v91_a + 120;
      int32_t v116_a = v28_lead + 160;
      int32_t v180_a = v28_lead + 10;
      int32_t v182_a = v28_lead + 20;
      int32_t v184_a = v28_lead + 30;
      int32_t v186_a = v28_lead + 40;
      int32_t v188_a = v28_lead + 50;
      int32_t v190_a = v28_lead + 60;
      int32_t v192_a = v28_lead + 70;
      int32_t v194_a = v28_lead + 80;
      int32_t v196_a = v28_lead + 90;
      int32_t v198_a = v28_lead + 100;
      int32_t v200_a = v28_lead + 110;
      int32_t v202_a = v28_lead + 120;
      int32_t v204_a = v28_lead + 130;
      int32_t v206_a = v28_lead + 140;
      int32_t v208_a = v28_lead + 150;
      int32_t v266_a = v28_lead + 170;
      bool v370_g = v28_lead < 10;
      for (size_t v11_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v11_batchIdGroup0 < numElements0; v11_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v12_row = v11_batchIdGroup0 + v10_batchIdLane0;
        const bool batchIdActive0 = v12_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v12_row]));
        size_t v14_batchId0 = batchIdActive0 ? v12_row : v11_batchIdGroup0;
        size_t v15_ahead1 = v14_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v15_ahead1 < numElements0) ? v15_ahead1 : v14_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v14_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v14_batchId0 * 153 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v14_batchId0 * 162 + 0 + m4_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        if (v29_g) {
          int32_t v33_a = v28_lead - 1;
          #pragma unroll
          for (int32_t v30_i1 = 0; v30_i1 < 9; ++v30_i1) {
            float v36_data = __builtin_nontemporal_load(&glb_m2[(v33_a + (v30_i1 * 17))]);
            r0[(v30_i1 * 2)] = v36_data;
          }
        }
        if (v39_g) {
          int32_t v43_a = (v28_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v40_i1 = 0; v40_i1 < 9; ++v40_i1) {
            float v46_data = __builtin_nontemporal_load(&glb_m2[(v43_a + (v40_i1 * 17))]);
            r0[(1 + (v40_i1 * 2))] = v46_data;
          }
        }
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        if (v29_g) {
          int32_t v53_a = v28_lead - 1;
          #pragma unroll
          for (int32_t v50_i1 = 0; v50_i1 < 9; ++v50_i1) {
            float v56_data = __builtin_nontemporal_load(&glb_m4[(v53_a + (v50_i1 * 18))]);
            r2[(v50_i1 * 2)] = v56_data;
          }
        }
        if (v59_g) {
          int32_t v63_a = (v28_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v60_i1 = 0; v60_i1 < 9; ++v60_i1) {
            float v66_data = __builtin_nontemporal_load(&glb_m4[(v63_a + (v60_i1 * 18))]);
            r2[(1 + (v60_i1 * 2))] = v66_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(1, 18)]
        float v70_data = r0[0];
        float v71_data = r0[2];
        float v72_data = r0[4];
        float v73_data = r0[6];
        float v74_tp{};
        float v75_tp{};
        float v76_tp{};
        float v77_tp{};
        tensorforge::transpose4x4b32(v74_tp, v75_tp, v76_tp, v77_tp, v70_data, v71_data, v72_data, v73_data);
        float v78_data = r0[1];
        float v79_data = r0[3];
        float v80_data = r0[5];
        float v81_data = r0[7];
        float v82_tp{};
        float v83_tp{};
        float v84_tp{};
        float v85_tp{};
        tensorforge::transpose4x4b32(v82_tp, v83_tp, v84_tp, v85_tp, v78_data, v79_data, v80_data, v81_data);
        tensorforge::VectorT<float, 4> v86_acc{};
        float v93_data = glb_m1[v91_a];
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v93_data, v86_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v93_data, v94_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v93_data, v95_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v93_data, v96_acc, 2, 1, 7);
        float v99_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v99_data, v97_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v99_data, v100_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v99_data, v101_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v99_data, v102_acc, 2, 2, 7);
        float v105_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v105_data, v103_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v105_data, v106_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v105_data, v107_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v105_data, v108_acc, 2, 3, 7);
        float v111_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v111_data, v109_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v111_data, v112_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v111_data, v113_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v111_data, v114_acc, 2, 0, 7);
        float v117_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v117_data, v115_acc, 2, 0, 0);
        r1[0] = (v118_acc[0]);
        r1[1] = (v118_acc[1]);
        r1[2] = (v118_acc[2]);
        r1[3] = (v118_acc[3]);
        float v123_data = r0[8];
        float v124_data = r0[10];
        float v125_data = r0[12];
        float v126_data = r0[14];
        float v127_tp{};
        float v128_tp{};
        float v129_tp{};
        float v130_tp{};
        tensorforge::transpose4x4b32(v127_tp, v128_tp, v129_tp, v130_tp, v123_data, v124_data, v125_data, v126_data);
        float v131_data = r0[9];
        float v132_data = r0[11];
        float v133_data = r0[13];
        float v134_data = r0[15];
        float v135_tp{};
        float v136_tp{};
        float v137_tp{};
        float v138_tp{};
        tensorforge::transpose4x4b32(v135_tp, v136_tp, v137_tp, v138_tp, v131_data, v132_data, v133_data, v134_data);
        tensorforge::VectorT<float, 4> v139_acc{};
        float v146_data = glb_m1[v91_a];
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v146_data, v139_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v146_data, v147_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v146_data, v148_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v146_data, v149_acc, 2, 1, 7);
        float v152_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v152_data, v150_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v152_data, v153_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v152_data, v154_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v152_data, v155_acc, 2, 2, 7);
        float v158_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v158_data, v156_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v158_data, v159_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v158_data, v160_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v158_data, v161_acc, 2, 3, 7);
        float v164_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v164_data, v162_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v164_data, v165_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v164_data, v166_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v164_data, v167_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v117_data, v168_acc, 2, 0, 0);
        r1[4] = (v171_acc[0]);
        r1[5] = (v171_acc[1]);
        r1[6] = (v171_acc[2]);
        r1[7] = (v171_acc[3]);
        float v179_data = glb_m1[v28_lead];
        float v181_data = glb_m1[v180_a];
        float v183_data = glb_m1[v182_a];
        float v185_data = glb_m1[v184_a];
        float v187_data = glb_m1[v186_a];
        float v189_data = glb_m1[v188_a];
        float v191_data = glb_m1[v190_a];
        float v193_data = glb_m1[v192_a];
        float v195_data = glb_m1[v194_a];
        float v197_data = glb_m1[v196_a];
        float v199_data = glb_m1[v198_a];
        float v201_data = glb_m1[v200_a];
        float v203_data = glb_m1[v202_a];
        float v205_data = glb_m1[v204_a];
        float v207_data = glb_m1[v206_a];
        float v209_data = glb_m1[v208_a];
        float v212_acc{};
        float v213_data = r0[16];
        float v214_data = r0[17];
        tensorforge::fmacdpp16<1>(v212_acc, v213_data, v179_data);
        tensorforge::fmacdpp16<2>(v212_acc, v213_data, v181_data);
        tensorforge::fmacdpp16<3>(v212_acc, v213_data, v183_data);
        tensorforge::fmacdpp16<4>(v212_acc, v213_data, v185_data);
        tensorforge::fmacdpp16<5>(v212_acc, v213_data, v187_data);
        tensorforge::fmacdpp16<6>(v212_acc, v213_data, v189_data);
        tensorforge::fmacdpp16<7>(v212_acc, v213_data, v191_data);
        tensorforge::fmacdpp16<8>(v212_acc, v213_data, v193_data);
        tensorforge::fmacdpp16<9>(v212_acc, v213_data, v195_data);
        tensorforge::fmacdpp16<10>(v212_acc, v213_data, v197_data);
        tensorforge::fmacdpp16<11>(v212_acc, v213_data, v199_data);
        tensorforge::fmacdpp16<12>(v212_acc, v213_data, v201_data);
        tensorforge::fmacdpp16<13>(v212_acc, v213_data, v203_data);
        tensorforge::fmacdpp16<14>(v212_acc, v213_data, v205_data);
        tensorforge::fmacdpp16<15>(v212_acc, v213_data, v207_data);
        tensorforge::fmacdpp16<0>(v212_acc, v214_data, v209_data);
        tensorforge::fmacdpp16<1>(v212_acc, v214_data, v117_data);
        r1[8] = v212_acc;
        // wait(r2 = load{g>r}(glb_m4););
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(1, 19)]
        float ir3[9]{};
        float v217_data = r2[0];
        float v218_data = r2[2];
        float v219_data = r2[4];
        float v220_data = r2[6];
        float v221_tp{};
        float v222_tp{};
        float v223_tp{};
        float v224_tp{};
        tensorforge::transpose4x4b32(v221_tp, v222_tp, v223_tp, v224_tp, v217_data, v218_data, v219_data, v220_data);
        float v225_data = r2[1];
        float v226_data = r2[3];
        float v227_data = r2[5];
        float v228_data = r2[7];
        float v229_tp{};
        float v230_tp{};
        float v231_tp{};
        float v232_tp{};
        tensorforge::transpose4x4b32(v229_tp, v230_tp, v231_tp, v232_tp, v225_data, v226_data, v227_data, v228_data);
        tensorforge::VectorT<float, 4> v233_acc{};
        float v240_data = glb_m3[v91_a];
        tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v240_data, v233_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v240_data, v241_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v240_data, v242_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v240_data, v243_acc, 2, 1, 7);
        float v246_data = glb_m3[v98_a];
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v246_data, v244_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v246_data, v247_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v246_data, v248_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v246_data, v249_acc, 2, 2, 7);
        float v252_data = glb_m3[v104_a];
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v252_data, v250_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v252_data, v253_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v252_data, v254_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v221_tp, v252_data, v255_acc, 2, 3, 7);
        float v258_data = glb_m3[v110_a];
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v222_tp, v258_data, v256_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v258_data, v259_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v258_data, v260_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v258_data, v261_acc, 2, 0, 7);
        float v264_data = glb_m3[v116_a];
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v264_data, v262_acc, 2, 0, 0);
        float v267_data = glb_m3[v266_a];
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v267_data, v265_acc, 2, 0, 0);
        ir3[0] = (v268_acc[0]);
        ir3[1] = (v268_acc[1]);
        ir3[2] = (v268_acc[2]);
        ir3[3] = (v268_acc[3]);
        float v273_data = r2[8];
        float v274_data = r2[10];
        float v275_data = r2[12];
        float v276_data = r2[14];
        float v277_tp{};
        float v278_tp{};
        float v279_tp{};
        float v280_tp{};
        tensorforge::transpose4x4b32(v277_tp, v278_tp, v279_tp, v280_tp, v273_data, v274_data, v275_data, v276_data);
        float v281_data = r2[9];
        float v282_data = r2[11];
        float v283_data = r2[13];
        float v284_data = r2[15];
        float v285_tp{};
        float v286_tp{};
        float v287_tp{};
        float v288_tp{};
        tensorforge::transpose4x4b32(v285_tp, v286_tp, v287_tp, v288_tp, v281_data, v282_data, v283_data, v284_data);
        tensorforge::VectorT<float, 4> v289_acc{};
        float v296_data = glb_m3[v91_a];
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v296_data, v289_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v296_data, v297_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v296_data, v298_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v296_data, v299_acc, 2, 1, 7);
        float v302_data = glb_m3[v98_a];
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v302_data, v300_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v302_data, v303_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v302_data, v304_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v302_data, v305_acc, 2, 2, 7);
        float v308_data = glb_m3[v104_a];
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v308_data, v306_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v308_data, v309_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v308_data, v310_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v308_data, v311_acc, 2, 3, 7);
        float v314_data = glb_m3[v110_a];
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v314_data, v312_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v314_data, v315_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v314_data, v316_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v314_data, v317_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v264_data, v318_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v267_data, v321_acc, 2, 0, 0);
        ir3[4] = (v324_acc[0]);
        ir3[5] = (v324_acc[1]);
        ir3[6] = (v324_acc[2]);
        ir3[7] = (v324_acc[3]);
        float v332_data = glb_m3[v28_lead];
        float v334_data = glb_m3[v180_a];
        float v336_data = glb_m3[v182_a];
        float v338_data = glb_m3[v184_a];
        float v340_data = glb_m3[v186_a];
        float v342_data = glb_m3[v188_a];
        float v344_data = glb_m3[v190_a];
        float v346_data = glb_m3[v192_a];
        float v348_data = glb_m3[v194_a];
        float v350_data = glb_m3[v196_a];
        float v352_data = glb_m3[v198_a];
        float v354_data = glb_m3[v200_a];
        float v356_data = glb_m3[v202_a];
        float v358_data = glb_m3[v204_a];
        float v360_data = glb_m3[v206_a];
        float v362_data = glb_m3[v208_a];
        float v367_acc{};
        float v368_data = r2[16];
        float v369_data = r2[17];
        tensorforge::fmacdpp16<1>(v367_acc, v368_data, v332_data);
        tensorforge::fmacdpp16<2>(v367_acc, v368_data, v334_data);
        tensorforge::fmacdpp16<3>(v367_acc, v368_data, v336_data);
        tensorforge::fmacdpp16<4>(v367_acc, v368_data, v338_data);
        tensorforge::fmacdpp16<5>(v367_acc, v368_data, v340_data);
        tensorforge::fmacdpp16<6>(v367_acc, v368_data, v342_data);
        tensorforge::fmacdpp16<7>(v367_acc, v368_data, v344_data);
        tensorforge::fmacdpp16<8>(v367_acc, v368_data, v346_data);
        tensorforge::fmacdpp16<9>(v367_acc, v368_data, v348_data);
        tensorforge::fmacdpp16<10>(v367_acc, v368_data, v350_data);
        tensorforge::fmacdpp16<11>(v367_acc, v368_data, v352_data);
        tensorforge::fmacdpp16<12>(v367_acc, v368_data, v354_data);
        tensorforge::fmacdpp16<13>(v367_acc, v368_data, v356_data);
        tensorforge::fmacdpp16<14>(v367_acc, v368_data, v358_data);
        tensorforge::fmacdpp16<15>(v367_acc, v368_data, v360_data);
        tensorforge::fmacdpp16<0>(v367_acc, v369_data, v362_data);
        tensorforge::fmacdpp16<1>(v367_acc, v369_data, v264_data);
        tensorforge::fmacdpp16<2>(v367_acc, v369_data, v267_data);
        ir3[8] = v367_acc;
        // r3 = ir3 + r1
        if (v370_g) {
          #pragma unroll
          for (int32_t v371_n1 = 0; v371_n1 < 9; ++v371_n1) {
            float v373_data = ir3[v371_n1];
            float v374_data = r1[v371_n1];
            r3[v371_n1] = (v374_data + v373_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v370_g) {
          #pragma unroll
          for (int32_t v377_i1 = 0; v377_i1 < 9; ++v377_i1) {
            float v379_data = r3[v377_i1];
            if (batchIdActive0) {
              glb_m0[(v28_lead + (v377_i1 * 10))] = v379_data;
            }
          }
        }
      }
    }
  }
}

