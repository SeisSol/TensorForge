// === base name ===
kernel_63a0ddf83076ade0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_63a0ddf83076ade0 = {{32, 8, 1}, 32, 56, 1, 8, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_63a0ddf83076ade0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_63a0ddf83076ade0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_63a0ddf83076ade0(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_63a0ddf83076ade0, block.x * block.y * block.z, 1792 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1792 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_63a0ddf83076ade0, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (1792 * sizeof(float)));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_63a0ddf83076ade0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_63a0ddf83076ade0(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_63a0ddf83076ade0), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_63a0ddf83076ade0, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_63a0ddf83076ade0(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 7168 B shared, occupancy grid
    // operands:
    //   m0 56×18(56×18) {0..56}×{0..18} strided
    //   m1 56×32(56×32) {0..56}×{0..32} none
    //   m2 32×18(32×18) {0..32}×{0..18} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,18]],"name":"m0","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,32]],"name":"m1","ordered":false,"parts":1,"shape":[56,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[32,18]],"name":"m2","ordered":false,"parts":1,"shape":[32,18],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,18]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,32]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,32]},{"addressing":"strided","bbox":[[0,0],[32,18]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,18]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 1792];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 4> v5_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      tensorforge::VectorT<float, 2> v6_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024]);
      *(tensorforge::VectorT<float, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v6_ld;
      float v7_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536] = v7_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v9_batchIdLane0 = threadIdx.y % 2;
      int32_t v26_lead = threadIdx.x % 32;
      int32_t v46_r = (threadIdx.y % 2) * 56;
      int32_t v49_a = v26_lead + v46_r;
      int32_t v54_a = v49_a + 112;
      int32_t v58_a = v49_a + 224;
      int32_t v62_a = v49_a + 336;
      int32_t v66_a = v49_a + 448;
      int32_t v70_a = v49_a + 560;
      int32_t v74_a = v49_a + 672;
      int32_t v78_a = v49_a + 784;
      int32_t v82_a = v49_a + 896;
      int32_t v86_a = v49_a + 1008;
      int32_t v90_a = v49_a + 1120;
      int32_t v94_a = v49_a + 1232;
      int32_t v98_a = v49_a + 1344;
      int32_t v102_a = v49_a + 1456;
      int32_t v106_a = v49_a + 1568;
      int32_t v110_a = v49_a + 1680;
      int32_t v122_lead = v26_lead + 32_i32;
      int32_t v123_a = v122_lead + v46_r;
      int32_t v128_a = v123_a + 112;
      int32_t v132_a = v123_a + 224;
      int32_t v136_a = v123_a + 336;
      int32_t v140_a = v123_a + 448;
      int32_t v144_a = v123_a + 560;
      int32_t v148_a = v123_a + 672;
      int32_t v152_a = v123_a + 784;
      int32_t v156_a = v123_a + 896;
      int32_t v160_a = v123_a + 1008;
      int32_t v164_a = v123_a + 1120;
      int32_t v168_a = v123_a + 1232;
      int32_t v172_a = v123_a + 1344;
      int32_t v176_a = v123_a + 1456;
      int32_t v180_a = v123_a + 1568;
      int32_t v184_a = v123_a + 1680;
      bool v820_g = v26_lead < 24;
      for (size_t v10_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v10_batchIdGroup0 < numElements0; v10_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v11_row = v10_batchIdGroup0 + v9_batchIdLane0;
        const bool batchIdActive0 = v11_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v11_row]));
        size_t v13_batchId0 = batchIdActive0 ? v11_row : v10_batchIdGroup0;
        size_t v14_ahead1 = v13_batchId0 + (gridDim.x * blockDim.y);
        size_t v17_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v13_batchId0 * 1008 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v13_batchId0 * 576 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
          int32_t v30_lead = v26_lead + (v27_i0 * 32);
          #pragma unroll
          for (int32_t v28_i1 = 0; v28_i1 < 18; ++v28_i1) {
            float v33_data = __builtin_nontemporal_load(&glb_m2[(v30_lead + (v28_i1 * 32))]);
            r0[(v27_i0 + v28_i1)] = v33_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[36]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 56), (0, 18)] [(0, 32)]
        float v36_data = r0[0];
        float v37_data = r0[1];
        float v38_data = r0[2];
        float v39_data = r0[3];
        float v40_tp{};
        float v41_tp{};
        float v42_tp{};
        float v43_tp{};
        tensorforge::transpose4x4b32(v40_tp, v41_tp, v42_tp, v43_tp, v36_data, v37_data, v38_data, v39_data);
        tensorforge::VectorT<float, 4> v44_acc{};
        float v51_data = glb_m1[v49_a];
        tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v51_data, v44_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v51_data, v52_acc, 3, 0, 2);
        float v55_data = glb_m1[v54_a];
        tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v53_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v55_data, v56_acc, 3, 0, 2);
        float v59_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v59_data, v57_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v59_data, v60_acc, 3, 1, 2);
        float v63_data = glb_m1[v62_a];
        tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v63_data, v61_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v63_data, v64_acc, 3, 1, 2);
        float v67_data = glb_m1[v66_a];
        tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v67_data, v65_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v67_data, v68_acc, 3, 2, 2);
        float v71_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v71_data, v69_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v71_data, v72_acc, 3, 2, 2);
        float v75_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v75_data, v73_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v75_data, v76_acc, 3, 3, 2);
        float v79_data = glb_m1[v78_a];
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v79_data, v77_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v79_data, v80_acc, 3, 3, 2);
        float v83_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v83_data, v81_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v83_data, v84_acc, 3, 4, 2);
        float v87_data = glb_m1[v86_a];
        tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v87_data, v85_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v87_data, v88_acc, 3, 4, 2);
        float v91_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v91_data, v89_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v91_data, v92_acc, 3, 5, 2);
        float v95_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v95_data, v93_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v95_data, v96_acc, 3, 5, 2);
        float v99_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v99_data, v97_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v99_data, v100_acc, 3, 6, 2);
        float v103_data = glb_m1[v102_a];
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v103_data, v101_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v103_data, v104_acc, 3, 6, 2);
        float v107_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v107_data, v105_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v107_data, v108_acc, 3, 7, 2);
        float v111_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v111_data, v109_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v111_data, v112_acc, 3, 7, 2);
        r1[0] = (v113_acc[0]);
        r1[2] = (v113_acc[1]);
        r1[4] = (v113_acc[2]);
        r1[6] = (v113_acc[3]);
        tensorforge::VectorT<float, 4> v118_acc{};
        float v125_data = glb_m1[v123_a];
        tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v125_data, v118_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v125_data, v126_acc, 3, 0, 2);
        float v129_data = glb_m1[v128_a];
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v129_data, v127_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v129_data, v130_acc, 3, 0, 2);
        float v133_data = glb_m1[v132_a];
        tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v133_data, v131_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v133_data, v134_acc, 3, 1, 2);
        float v137_data = glb_m1[v136_a];
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v137_data, v135_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v137_data, v138_acc, 3, 1, 2);
        float v141_data = glb_m1[v140_a];
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v141_data, v139_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v141_data, v142_acc, 3, 2, 2);
        float v145_data = glb_m1[v144_a];
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v145_data, v143_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v145_data, v146_acc, 3, 2, 2);
        float v149_data = glb_m1[v148_a];
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v149_data, v147_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v149_data, v150_acc, 3, 3, 2);
        float v153_data = glb_m1[v152_a];
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v153_data, v151_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v153_data, v154_acc, 3, 3, 2);
        float v157_data = glb_m1[v156_a];
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v157_data, v155_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v157_data, v158_acc, 3, 4, 2);
        float v161_data = glb_m1[v160_a];
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v161_data, v159_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v161_data, v162_acc, 3, 4, 2);
        float v165_data = glb_m1[v164_a];
        tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v165_data, v163_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v165_data, v166_acc, 3, 5, 2);
        float v169_data = glb_m1[v168_a];
        tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v169_data, v167_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v169_data, v170_acc, 3, 5, 2);
        float v173_data = glb_m1[v172_a];
        tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v173_data, v171_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v173_data, v174_acc, 3, 6, 2);
        float v177_data = glb_m1[v176_a];
        tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v177_data, v175_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v177_data, v178_acc, 3, 6, 2);
        float v181_data = glb_m1[v180_a];
        tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v181_data, v179_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v181_data, v182_acc, 3, 7, 2);
        float v185_data = glb_m1[v184_a];
        tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v185_data, v183_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v185_data, v186_acc, 3, 7, 2);
        r1[1] = (v187_acc[0]);
        r1[3] = (v187_acc[1]);
        r1[5] = (v187_acc[2]);
        r1[7] = (v187_acc[3]);
        float v192_data = r0[4];
        float v193_data = r0[5];
        float v194_data = r0[6];
        float v195_data = r0[7];
        float v196_tp{};
        float v197_tp{};
        float v198_tp{};
        float v199_tp{};
        tensorforge::transpose4x4b32(v196_tp, v197_tp, v198_tp, v199_tp, v192_data, v193_data, v194_data, v195_data);
        tensorforge::VectorT<float, 4> v200_acc{};
        float v207_data = glb_m1[v49_a];
        tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v207_data, v200_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v207_data, v208_acc, 3, 0, 2);
        float v211_data = glb_m1[v54_a];
        tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v211_data, v209_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v211_data, v212_acc, 3, 0, 2);
        float v215_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v215_data, v213_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v215_data, v216_acc, 3, 1, 2);
        float v219_data = glb_m1[v62_a];
        tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v219_data, v217_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v219_data, v220_acc, 3, 1, 2);
        float v223_data = glb_m1[v66_a];
        tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v223_data, v221_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v223_data, v224_acc, 3, 2, 2);
        float v227_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v227_data, v225_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v227_data, v228_acc, 3, 2, 2);
        float v231_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v231_data, v229_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v231_data, v232_acc, 3, 3, 2);
        float v235_data = glb_m1[v78_a];
        tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v235_data, v233_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v235_data, v236_acc, 3, 3, 2);
        float v239_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v239_data, v237_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v239_data, v240_acc, 3, 4, 2);
        float v243_data = glb_m1[v86_a];
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v243_data, v241_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v243_data, v244_acc, 3, 4, 2);
        float v247_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v247_data, v245_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v247_data, v248_acc, 3, 5, 2);
        float v251_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v251_data, v249_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v251_data, v252_acc, 3, 5, 2);
        float v255_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v255_data, v253_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v255_data, v256_acc, 3, 6, 2);
        float v259_data = glb_m1[v102_a];
        tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v259_data, v257_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v259_data, v260_acc, 3, 6, 2);
        float v263_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v263_data, v261_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v263_data, v264_acc, 3, 7, 2);
        float v267_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v267_data, v265_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v267_data, v268_acc, 3, 7, 2);
        r1[8] = (v269_acc[0]);
        r1[10] = (v269_acc[1]);
        r1[12] = (v269_acc[2]);
        r1[14] = (v269_acc[3]);
        tensorforge::VectorT<float, 4> v274_acc{};
        float v281_data = glb_m1[v123_a];
        tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v281_data, v274_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v281_data, v282_acc, 3, 0, 2);
        float v285_data = glb_m1[v128_a];
        tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v285_data, v283_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v285_data, v286_acc, 3, 0, 2);
        float v289_data = glb_m1[v132_a];
        tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v289_data, v287_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v289_data, v290_acc, 3, 1, 2);
        float v293_data = glb_m1[v136_a];
        tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v293_data, v291_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v293_data, v294_acc, 3, 1, 2);
        float v297_data = glb_m1[v140_a];
        tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v297_data, v295_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v297_data, v298_acc, 3, 2, 2);
        float v301_data = glb_m1[v144_a];
        tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v301_data, v299_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v301_data, v302_acc, 3, 2, 2);
        float v305_data = glb_m1[v148_a];
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v305_data, v303_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v305_data, v306_acc, 3, 3, 2);
        float v309_data = glb_m1[v152_a];
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v309_data, v307_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v309_data, v310_acc, 3, 3, 2);
        float v313_data = glb_m1[v156_a];
        tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v313_data, v311_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v313_data, v314_acc, 3, 4, 2);
        float v317_data = glb_m1[v160_a];
        tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v317_data, v315_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v317_data, v318_acc, 3, 4, 2);
        float v321_data = glb_m1[v164_a];
        tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v321_data, v319_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v321_data, v322_acc, 3, 5, 2);
        float v325_data = glb_m1[v168_a];
        tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v325_data, v323_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v325_data, v326_acc, 3, 5, 2);
        float v329_data = glb_m1[v172_a];
        tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v329_data, v327_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v329_data, v330_acc, 3, 6, 2);
        float v333_data = glb_m1[v176_a];
        tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v333_data, v331_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v333_data, v334_acc, 3, 6, 2);
        float v337_data = glb_m1[v180_a];
        tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v337_data, v335_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v197_tp, v337_data, v338_acc, 3, 7, 2);
        float v341_data = glb_m1[v184_a];
        tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v198_tp, v341_data, v339_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v341_data, v342_acc, 3, 7, 2);
        r1[9] = (v343_acc[0]);
        r1[11] = (v343_acc[1]);
        r1[13] = (v343_acc[2]);
        r1[15] = (v343_acc[3]);
        float v348_data = r0[8];
        float v349_data = r0[9];
        float v350_data = r0[10];
        float v351_data = r0[11];
        float v352_tp{};
        float v353_tp{};
        float v354_tp{};
        float v355_tp{};
        tensorforge::transpose4x4b32(v352_tp, v353_tp, v354_tp, v355_tp, v348_data, v349_data, v350_data, v351_data);
        tensorforge::VectorT<float, 4> v356_acc{};
        float v363_data = glb_m1[v49_a];
        tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v363_data, v356_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v363_data, v364_acc, 3, 0, 2);
        float v367_data = glb_m1[v54_a];
        tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v367_data, v365_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v367_data, v368_acc, 3, 0, 2);
        float v371_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v371_data, v369_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v371_data, v372_acc, 3, 1, 2);
        float v375_data = glb_m1[v62_a];
        tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v375_data, v373_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v375_data, v376_acc, 3, 1, 2);
        float v379_data = glb_m1[v66_a];
        tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v379_data, v377_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v379_data, v380_acc, 3, 2, 2);
        float v383_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v383_data, v381_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v383_data, v384_acc, 3, 2, 2);
        float v387_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v387_data, v385_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v387_data, v388_acc, 3, 3, 2);
        float v391_data = glb_m1[v78_a];
        tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v391_data, v389_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v391_data, v392_acc, 3, 3, 2);
        float v395_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v395_data, v393_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v395_data, v396_acc, 3, 4, 2);
        float v399_data = glb_m1[v86_a];
        tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v399_data, v397_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v399_data, v400_acc, 3, 4, 2);
        float v403_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v403_data, v401_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v403_data, v404_acc, 3, 5, 2);
        float v407_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v407_data, v405_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v407_data, v408_acc, 3, 5, 2);
        float v411_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v411_data, v409_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v411_data, v412_acc, 3, 6, 2);
        float v415_data = glb_m1[v102_a];
        tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v415_data, v413_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v415_data, v416_acc, 3, 6, 2);
        float v419_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v419_data, v417_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v419_data, v420_acc, 3, 7, 2);
        float v423_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v423_data, v421_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v423_data, v424_acc, 3, 7, 2);
        r1[16] = (v425_acc[0]);
        r1[18] = (v425_acc[1]);
        r1[20] = (v425_acc[2]);
        r1[22] = (v425_acc[3]);
        tensorforge::VectorT<float, 4> v430_acc{};
        float v437_data = glb_m1[v123_a];
        tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v437_data, v430_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v437_data, v438_acc, 3, 0, 2);
        float v441_data = glb_m1[v128_a];
        tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v441_data, v439_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v441_data, v442_acc, 3, 0, 2);
        float v445_data = glb_m1[v132_a];
        tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v445_data, v443_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v445_data, v446_acc, 3, 1, 2);
        float v449_data = glb_m1[v136_a];
        tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v449_data, v447_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v449_data, v450_acc, 3, 1, 2);
        float v453_data = glb_m1[v140_a];
        tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v453_data, v451_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v453_data, v454_acc, 3, 2, 2);
        float v457_data = glb_m1[v144_a];
        tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v457_data, v455_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v457_data, v458_acc, 3, 2, 2);
        float v461_data = glb_m1[v148_a];
        tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v461_data, v459_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v461_data, v462_acc, 3, 3, 2);
        float v465_data = glb_m1[v152_a];
        tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v465_data, v463_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v465_data, v466_acc, 3, 3, 2);
        float v469_data = glb_m1[v156_a];
        tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v469_data, v467_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v469_data, v470_acc, 3, 4, 2);
        float v473_data = glb_m1[v160_a];
        tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v473_data, v471_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v473_data, v474_acc, 3, 4, 2);
        float v477_data = glb_m1[v164_a];
        tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v477_data, v475_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v477_data, v478_acc, 3, 5, 2);
        float v481_data = glb_m1[v168_a];
        tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v481_data, v479_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v481_data, v482_acc, 3, 5, 2);
        float v485_data = glb_m1[v172_a];
        tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v485_data, v483_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v485_data, v486_acc, 3, 6, 2);
        float v489_data = glb_m1[v176_a];
        tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v489_data, v487_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v489_data, v490_acc, 3, 6, 2);
        float v493_data = glb_m1[v180_a];
        tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v352_tp, v493_data, v491_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v493_data, v494_acc, 3, 7, 2);
        float v497_data = glb_m1[v184_a];
        tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v497_data, v495_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v497_data, v498_acc, 3, 7, 2);
        r1[17] = (v499_acc[0]);
        r1[19] = (v499_acc[1]);
        r1[21] = (v499_acc[2]);
        r1[23] = (v499_acc[3]);
        float v504_data = r0[12];
        float v505_data = r0[13];
        float v506_data = r0[14];
        float v507_data = r0[15];
        float v508_tp{};
        float v509_tp{};
        float v510_tp{};
        float v511_tp{};
        tensorforge::transpose4x4b32(v508_tp, v509_tp, v510_tp, v511_tp, v504_data, v505_data, v506_data, v507_data);
        tensorforge::VectorT<float, 4> v512_acc{};
        float v519_data = glb_m1[v49_a];
        tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v519_data, v512_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v519_data, v520_acc, 3, 0, 2);
        float v523_data = glb_m1[v54_a];
        tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v523_data, v521_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v523_data, v524_acc, 3, 0, 2);
        float v527_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v527_data, v525_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v527_data, v528_acc, 3, 1, 2);
        float v531_data = glb_m1[v62_a];
        tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v531_data, v529_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v531_data, v532_acc, 3, 1, 2);
        float v535_data = glb_m1[v66_a];
        tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v535_data, v533_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v535_data, v536_acc, 3, 2, 2);
        float v539_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v539_data, v537_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v539_data, v540_acc, 3, 2, 2);
        float v543_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v543_data, v541_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v543_data, v544_acc, 3, 3, 2);
        float v547_data = glb_m1[v78_a];
        tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v547_data, v545_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v547_data, v548_acc, 3, 3, 2);
        float v551_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v551_data, v549_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v551_data, v552_acc, 3, 4, 2);
        float v555_data = glb_m1[v86_a];
        tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v555_data, v553_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v555_data, v556_acc, 3, 4, 2);
        float v559_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v559_data, v557_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v559_data, v560_acc, 3, 5, 2);
        float v563_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v563_data, v561_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v563_data, v564_acc, 3, 5, 2);
        float v567_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v567_data, v565_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v567_data, v568_acc, 3, 6, 2);
        float v571_data = glb_m1[v102_a];
        tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v571_data, v569_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v571_data, v572_acc, 3, 6, 2);
        float v575_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v575_data, v573_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v575_data, v576_acc, 3, 7, 2);
        float v579_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v579_data, v577_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v579_data, v580_acc, 3, 7, 2);
        r1[24] = (v581_acc[0]);
        r1[26] = (v581_acc[1]);
        r1[28] = (v581_acc[2]);
        r1[30] = (v581_acc[3]);
        tensorforge::VectorT<float, 4> v586_acc{};
        float v593_data = glb_m1[v123_a];
        tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v593_data, v586_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v593_data, v594_acc, 3, 0, 2);
        float v597_data = glb_m1[v128_a];
        tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v597_data, v595_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v599_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v597_data, v598_acc, 3, 0, 2);
        float v601_data = glb_m1[v132_a];
        tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v601_data, v599_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v601_data, v602_acc, 3, 1, 2);
        float v605_data = glb_m1[v136_a];
        tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v605_data, v603_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v607_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v605_data, v606_acc, 3, 1, 2);
        float v609_data = glb_m1[v140_a];
        tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v609_data, v607_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v609_data, v610_acc, 3, 2, 2);
        float v613_data = glb_m1[v144_a];
        tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v613_data, v611_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v613_data, v614_acc, 3, 2, 2);
        float v617_data = glb_m1[v148_a];
        tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v617_data, v615_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v617_data, v618_acc, 3, 3, 2);
        float v621_data = glb_m1[v152_a];
        tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v621_data, v619_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v623_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v621_data, v622_acc, 3, 3, 2);
        float v625_data = glb_m1[v156_a];
        tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v625_data, v623_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v625_data, v626_acc, 3, 4, 2);
        float v629_data = glb_m1[v160_a];
        tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v629_data, v627_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v631_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v629_data, v630_acc, 3, 4, 2);
        float v633_data = glb_m1[v164_a];
        tensorforge::VectorT<float, 4> v634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v633_data, v631_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v633_data, v634_acc, 3, 5, 2);
        float v637_data = glb_m1[v168_a];
        tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v637_data, v635_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v639_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v637_data, v638_acc, 3, 5, 2);
        float v641_data = glb_m1[v172_a];
        tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v641_data, v639_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v643_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v641_data, v642_acc, 3, 6, 2);
        float v645_data = glb_m1[v176_a];
        tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v645_data, v643_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v645_data, v646_acc, 3, 6, 2);
        float v649_data = glb_m1[v180_a];
        tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v649_data, v647_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v509_tp, v649_data, v650_acc, 3, 7, 2);
        float v653_data = glb_m1[v184_a];
        tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v510_tp, v653_data, v651_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v653_data, v654_acc, 3, 7, 2);
        r1[25] = (v655_acc[0]);
        r1[27] = (v655_acc[1]);
        r1[29] = (v655_acc[2]);
        r1[31] = (v655_acc[3]);
        float v660_data = r0[16];
        float v661_data = r0[17];
        float v663_tp{};
        float v664_tp{};
        float v665_tp{};
        float v666_tp{};
        tensorforge::transpose4x4b32(v663_tp, v664_tp, v665_tp, v666_tp, v660_data, v661_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v667_acc{};
        float v674_data = glb_m1[v49_a];
        tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v674_data, v667_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v674_data, v675_acc, 3, 0, 2);
        float v678_data = glb_m1[v54_a];
        tensorforge::VectorT<float, 4> v679_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v678_data, v676_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v680_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v678_data, v679_acc, 3, 0, 2);
        float v682_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v682_data, v680_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v682_data, v683_acc, 3, 1, 2);
        float v686_data = glb_m1[v62_a];
        tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v686_data, v684_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v686_data, v687_acc, 3, 1, 2);
        float v690_data = glb_m1[v66_a];
        tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v690_data, v688_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v690_data, v691_acc, 3, 2, 2);
        float v694_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v695_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v694_data, v692_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v694_data, v695_acc, 3, 2, 2);
        float v698_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v698_data, v696_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v698_data, v699_acc, 3, 3, 2);
        float v702_data = glb_m1[v78_a];
        tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v702_data, v700_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v702_data, v703_acc, 3, 3, 2);
        float v706_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v706_data, v704_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v706_data, v707_acc, 3, 4, 2);
        float v710_data = glb_m1[v86_a];
        tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v710_data, v708_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v710_data, v711_acc, 3, 4, 2);
        float v714_data = glb_m1[v90_a];
        tensorforge::VectorT<float, 4> v715_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v714_data, v712_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v716_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v714_data, v715_acc, 3, 5, 2);
        float v718_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v719_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v718_data, v716_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v720_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v718_data, v719_acc, 3, 5, 2);
        float v722_data = glb_m1[v98_a];
        tensorforge::VectorT<float, 4> v723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v722_data, v720_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v724_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v722_data, v723_acc, 3, 6, 2);
        float v726_data = glb_m1[v102_a];
        tensorforge::VectorT<float, 4> v727_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v726_data, v724_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v728_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v726_data, v727_acc, 3, 6, 2);
        float v730_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v731_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v730_data, v728_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v732_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v730_data, v731_acc, 3, 7, 2);
        float v734_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v735_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v734_data, v732_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v736_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v734_data, v735_acc, 3, 7, 2);
        r1[32] = (v736_acc[0]);
        r1[34] = (v736_acc[1]);
        tensorforge::VectorT<float, 4> v739_acc{};
        float v746_data = glb_m1[v123_a];
        tensorforge::VectorT<float, 4> v747_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v746_data, v739_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v748_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v746_data, v747_acc, 3, 0, 2);
        float v750_data = glb_m1[v128_a];
        tensorforge::VectorT<float, 4> v751_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v750_data, v748_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v750_data, v751_acc, 3, 0, 2);
        float v754_data = glb_m1[v132_a];
        tensorforge::VectorT<float, 4> v755_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v754_data, v752_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v756_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v754_data, v755_acc, 3, 1, 2);
        float v758_data = glb_m1[v136_a];
        tensorforge::VectorT<float, 4> v759_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v758_data, v756_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v758_data, v759_acc, 3, 1, 2);
        float v762_data = glb_m1[v140_a];
        tensorforge::VectorT<float, 4> v763_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v762_data, v760_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v764_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v762_data, v763_acc, 3, 2, 2);
        float v766_data = glb_m1[v144_a];
        tensorforge::VectorT<float, 4> v767_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v766_data, v764_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v766_data, v767_acc, 3, 2, 2);
        float v770_data = glb_m1[v148_a];
        tensorforge::VectorT<float, 4> v771_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v770_data, v768_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v770_data, v771_acc, 3, 3, 2);
        float v774_data = glb_m1[v152_a];
        tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v774_data, v772_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v774_data, v775_acc, 3, 3, 2);
        float v778_data = glb_m1[v156_a];
        tensorforge::VectorT<float, 4> v779_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v778_data, v776_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v778_data, v779_acc, 3, 4, 2);
        float v782_data = glb_m1[v160_a];
        tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v782_data, v780_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v782_data, v783_acc, 3, 4, 2);
        float v786_data = glb_m1[v164_a];
        tensorforge::VectorT<float, 4> v787_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v786_data, v784_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v788_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v786_data, v787_acc, 3, 5, 2);
        float v790_data = glb_m1[v168_a];
        tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v790_data, v788_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v790_data, v791_acc, 3, 5, 2);
        float v794_data = glb_m1[v172_a];
        tensorforge::VectorT<float, 4> v795_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v794_data, v792_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v794_data, v795_acc, 3, 6, 2);
        float v798_data = glb_m1[v176_a];
        tensorforge::VectorT<float, 4> v799_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v798_data, v796_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v800_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v798_data, v799_acc, 3, 6, 2);
        float v802_data = glb_m1[v180_a];
        tensorforge::VectorT<float, 4> v803_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v663_tp, v802_data, v800_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v804_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v664_tp, v802_data, v803_acc, 3, 7, 2);
        float v806_data = glb_m1[v184_a];
        tensorforge::VectorT<float, 4> v807_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v665_tp, v806_data, v804_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v808_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v806_data, v807_acc, 3, 7, 2);
        r1[33] = (v808_acc[0]);
        r1[35] = (v808_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v811_i0 = 0; v811_i0 < 1; ++v811_i0) {
          #pragma unroll
          for (int32_t v812_i1 = 0; v812_i1 < 18; ++v812_i1) {
            float v815_data = r1[(v811_i0 + (v812_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v26_lead + (v811_i0 * 32)) + (v812_i1 * 56))] = v815_data;
            }
          }
        }
        if (v820_g) {
          #pragma unroll
          for (int32_t v821_i1 = 0; v821_i1 < 18; ++v821_i1) {
            float v824_data = r1[(1 + (v821_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[(v122_lead + (v821_i1 * 56))] = v824_data;
            }
          }
        }
      }
    }
  }
}

