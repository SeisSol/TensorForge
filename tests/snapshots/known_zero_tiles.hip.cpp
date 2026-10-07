// === base name ===
kernel_4dd06735b523f67e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4dd06735b523f67e = {{32, 8, 1}, 32, 56, 1, 8, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4dd06735b523f67e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4dd06735b523f67e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4dd06735b523f67e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4dd06735b523f67e, block.x * block.y * block.z, 1792 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1792 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4dd06735b523f67e, block.x * block.y * block.z, 0));
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
void launcher_kernel_4dd06735b523f67e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4dd06735b523f67e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4dd06735b523f67e), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_4dd06735b523f67e, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4dd06735b523f67e(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 7168 B shared, occupancy grid
    // operands:
    //   m0 56×18(56×18) {0..56}×{0..18} strided
    //   m1 56×32(56×32) {0..56}×{0..32} none
    //   m2 32×18(32×18) {0..32}×{0..18} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,18]],"name":"m0","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,32]],"name":"m1","ordered":false,"parts":1,"shape":[56,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[32,18]],"name":"m2","ordered":false,"parts":1,"shape":[32,18],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,18]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,32]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,32]},{"addressing":"strided","bbox":[[0,0],[32,18]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,18]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 1792];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 4> v9_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      tensorforge::VectorT<float, 2> v10_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024]);
      *(tensorforge::VectorT<float, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v10_ld;
      float v11_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536] = v11_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v13_batchIdLane0 = threadIdx.y % 2;
      int32_t v29_lead = threadIdx.x % 32;
      int32_t v49_r = (threadIdx.y % 2) * 56;
      int32_t v52_a = v29_lead + v49_r;
      int32_t v57_a = v52_a + 112;
      int32_t v61_a = v52_a + 224;
      int32_t v65_a = v52_a + 336;
      int32_t v69_a = v52_a + 448;
      int32_t v73_a = v52_a + 560;
      int32_t v77_a = v52_a + 672;
      int32_t v81_a = v52_a + 784;
      int32_t v85_a = v52_a + 896;
      int32_t v89_a = v52_a + 1008;
      int32_t v93_a = v52_a + 1120;
      int32_t v97_a = v52_a + 1232;
      int32_t v101_a = v52_a + 1344;
      int32_t v105_a = v52_a + 1456;
      int32_t v109_a = v52_a + 1568;
      int32_t v113_a = v52_a + 1680;
      int32_t v125_lead = v29_lead + 32_i32;
      int32_t v126_a = v125_lead + v49_r;
      int32_t v131_a = v126_a + 112;
      int32_t v135_a = v126_a + 224;
      int32_t v139_a = v126_a + 336;
      int32_t v143_a = v126_a + 448;
      int32_t v147_a = v126_a + 560;
      int32_t v151_a = v126_a + 672;
      int32_t v155_a = v126_a + 784;
      int32_t v159_a = v126_a + 896;
      int32_t v163_a = v126_a + 1008;
      int32_t v167_a = v126_a + 1120;
      int32_t v171_a = v126_a + 1232;
      int32_t v175_a = v126_a + 1344;
      int32_t v179_a = v126_a + 1456;
      int32_t v183_a = v126_a + 1568;
      int32_t v187_a = v126_a + 1680;
      bool v823_g = v29_lead < 24;
      for (size_t v14_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v14_batchIdGroup0 < numElements0; v14_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v15_row = v14_batchIdGroup0 + v13_batchIdLane0;
        const bool batchIdActive0 = v15_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v15_row]));
        size_t v17_batchId0 = batchIdActive0 ? v15_row : v14_batchIdGroup0;
        size_t v18_ahead1 = v17_batchId0 + (gridDim.x * blockDim.y);
        size_t v20_batchId1 = (v18_ahead1 < numElements0) ? v18_ahead1 : v17_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v17_batchId0 * 1008 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v17_batchId0 * 576 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
          int32_t v33_lead = v29_lead + (v30_i0 * 32);
          #pragma unroll
          for (int32_t v31_i1 = 0; v31_i1 < 18; ++v31_i1) {
            float v36_data = __builtin_nontemporal_load(&glb_m2[(v33_lead + (v31_i1 * 32))]);
            r0[(v30_i0 + v31_i1)] = v36_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[36]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 56), (0, 18)] [(0, 32)]
        float v39_data = r0[0];
        float v40_data = r0[1];
        float v41_data = r0[2];
        float v42_data = r0[3];
        float v43_tp{};
        float v44_tp{};
        float v45_tp{};
        float v46_tp{};
        tensorforge::transpose4x4b32(v43_tp, v44_tp, v45_tp, v46_tp, v39_data, v40_data, v41_data, v42_data);
        tensorforge::VectorT<float, 4> v47_acc{};
        float v54_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v54_data, v47_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v54_data, v55_acc, 3, 0, 2);
        float v58_data = glb_m1[v57_a];
        tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v56_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v58_data, v59_acc, 3, 0, 2);
        float v62_data = glb_m1[v61_a];
        tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v62_data, v60_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v62_data, v63_acc, 3, 1, 2);
        float v66_data = glb_m1[v65_a];
        tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v64_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v66_data, v67_acc, 3, 1, 2);
        float v70_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v70_data, v68_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v70_data, v71_acc, 3, 2, 2);
        float v74_data = glb_m1[v73_a];
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v72_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v74_data, v75_acc, 3, 2, 2);
        float v78_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v78_data, v76_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v78_data, v79_acc, 3, 3, 2);
        float v82_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v82_data, v80_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v82_data, v83_acc, 3, 3, 2);
        float v86_data = glb_m1[v85_a];
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v86_data, v84_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v86_data, v87_acc, 3, 4, 2);
        float v90_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v90_data, v88_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v90_data, v91_acc, 3, 4, 2);
        float v94_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v94_data, v92_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v94_data, v95_acc, 3, 5, 2);
        float v98_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v98_data, v96_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v98_data, v99_acc, 3, 5, 2);
        float v102_data = glb_m1[v101_a];
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v102_data, v100_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v102_data, v103_acc, 3, 6, 2);
        float v106_data = glb_m1[v105_a];
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v106_data, v104_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v106_data, v107_acc, 3, 6, 2);
        float v110_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v110_data, v108_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v110_data, v111_acc, 3, 7, 2);
        float v114_data = glb_m1[v113_a];
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v114_data, v112_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v114_data, v115_acc, 3, 7, 2);
        r1[0] = (v116_acc[0]);
        r1[2] = (v116_acc[1]);
        r1[4] = (v116_acc[2]);
        r1[6] = (v116_acc[3]);
        tensorforge::VectorT<float, 4> v121_acc{};
        float v128_data = glb_m1[v126_a];
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v128_data, v121_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v128_data, v129_acc, 3, 0, 2);
        float v132_data = glb_m1[v131_a];
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v132_data, v130_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v132_data, v133_acc, 3, 0, 2);
        float v136_data = glb_m1[v135_a];
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v136_data, v134_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v136_data, v137_acc, 3, 1, 2);
        float v140_data = glb_m1[v139_a];
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v140_data, v138_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v140_data, v141_acc, 3, 1, 2);
        float v144_data = glb_m1[v143_a];
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v144_data, v142_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v144_data, v145_acc, 3, 2, 2);
        float v148_data = glb_m1[v147_a];
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v148_data, v146_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v148_data, v149_acc, 3, 2, 2);
        float v152_data = glb_m1[v151_a];
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v152_data, v150_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v152_data, v153_acc, 3, 3, 2);
        float v156_data = glb_m1[v155_a];
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v156_data, v154_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v156_data, v157_acc, 3, 3, 2);
        float v160_data = glb_m1[v159_a];
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v160_data, v158_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v160_data, v161_acc, 3, 4, 2);
        float v164_data = glb_m1[v163_a];
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v164_data, v162_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v164_data, v165_acc, 3, 4, 2);
        float v168_data = glb_m1[v167_a];
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v168_data, v166_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v168_data, v169_acc, 3, 5, 2);
        float v172_data = glb_m1[v171_a];
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v172_data, v170_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v172_data, v173_acc, 3, 5, 2);
        float v176_data = glb_m1[v175_a];
        tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v176_data, v174_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v176_data, v177_acc, 3, 6, 2);
        float v180_data = glb_m1[v179_a];
        tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v180_data, v178_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v180_data, v181_acc, 3, 6, 2);
        float v184_data = glb_m1[v183_a];
        tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v184_data, v182_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v184_data, v185_acc, 3, 7, 2);
        float v188_data = glb_m1[v187_a];
        tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v188_data, v186_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v188_data, v189_acc, 3, 7, 2);
        r1[1] = (v190_acc[0]);
        r1[3] = (v190_acc[1]);
        r1[5] = (v190_acc[2]);
        r1[7] = (v190_acc[3]);
        float v195_data = r0[4];
        float v196_data = r0[5];
        float v197_data = r0[6];
        float v198_data = r0[7];
        float v199_tp{};
        float v200_tp{};
        float v201_tp{};
        float v202_tp{};
        tensorforge::transpose4x4b32(v199_tp, v200_tp, v201_tp, v202_tp, v195_data, v196_data, v197_data, v198_data);
        tensorforge::VectorT<float, 4> v203_acc{};
        float v210_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v210_data, v203_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v210_data, v211_acc, 3, 0, 2);
        float v214_data = glb_m1[v57_a];
        tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v214_data, v212_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v214_data, v215_acc, 3, 0, 2);
        float v218_data = glb_m1[v61_a];
        tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v218_data, v216_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v218_data, v219_acc, 3, 1, 2);
        float v222_data = glb_m1[v65_a];
        tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v222_data, v220_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v222_data, v223_acc, 3, 1, 2);
        float v226_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v226_data, v224_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v226_data, v227_acc, 3, 2, 2);
        float v230_data = glb_m1[v73_a];
        tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v230_data, v228_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v230_data, v231_acc, 3, 2, 2);
        float v234_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v234_data, v232_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v234_data, v235_acc, 3, 3, 2);
        float v238_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v238_data, v236_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v238_data, v239_acc, 3, 3, 2);
        float v242_data = glb_m1[v85_a];
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v242_data, v240_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v242_data, v243_acc, 3, 4, 2);
        float v246_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v246_data, v244_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v246_data, v247_acc, 3, 4, 2);
        float v250_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v250_data, v248_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v250_data, v251_acc, 3, 5, 2);
        float v254_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v254_data, v252_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v254_data, v255_acc, 3, 5, 2);
        float v258_data = glb_m1[v101_a];
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v258_data, v256_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v258_data, v259_acc, 3, 6, 2);
        float v262_data = glb_m1[v105_a];
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v262_data, v260_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v262_data, v263_acc, 3, 6, 2);
        float v266_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v266_data, v264_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v266_data, v267_acc, 3, 7, 2);
        float v270_data = glb_m1[v113_a];
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v270_data, v268_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v270_data, v271_acc, 3, 7, 2);
        r1[8] = (v272_acc[0]);
        r1[10] = (v272_acc[1]);
        r1[12] = (v272_acc[2]);
        r1[14] = (v272_acc[3]);
        tensorforge::VectorT<float, 4> v277_acc{};
        float v284_data = glb_m1[v126_a];
        tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v284_data, v277_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v284_data, v285_acc, 3, 0, 2);
        float v288_data = glb_m1[v131_a];
        tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v288_data, v286_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v288_data, v289_acc, 3, 0, 2);
        float v292_data = glb_m1[v135_a];
        tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v292_data, v290_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v292_data, v293_acc, 3, 1, 2);
        float v296_data = glb_m1[v139_a];
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v296_data, v294_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v296_data, v297_acc, 3, 1, 2);
        float v300_data = glb_m1[v143_a];
        tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v300_data, v298_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v300_data, v301_acc, 3, 2, 2);
        float v304_data = glb_m1[v147_a];
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v304_data, v302_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v304_data, v305_acc, 3, 2, 2);
        float v308_data = glb_m1[v151_a];
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v308_data, v306_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v308_data, v309_acc, 3, 3, 2);
        float v312_data = glb_m1[v155_a];
        tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v312_data, v310_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v312_data, v313_acc, 3, 3, 2);
        float v316_data = glb_m1[v159_a];
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v316_data, v314_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v316_data, v317_acc, 3, 4, 2);
        float v320_data = glb_m1[v163_a];
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v320_data, v318_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v320_data, v321_acc, 3, 4, 2);
        float v324_data = glb_m1[v167_a];
        tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v324_data, v322_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v324_data, v325_acc, 3, 5, 2);
        float v328_data = glb_m1[v171_a];
        tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v328_data, v326_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v328_data, v329_acc, 3, 5, 2);
        float v332_data = glb_m1[v175_a];
        tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v332_data, v330_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v332_data, v333_acc, 3, 6, 2);
        float v336_data = glb_m1[v179_a];
        tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v336_data, v334_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v336_data, v337_acc, 3, 6, 2);
        float v340_data = glb_m1[v183_a];
        tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v340_data, v338_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v340_data, v341_acc, 3, 7, 2);
        float v344_data = glb_m1[v187_a];
        tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v344_data, v342_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v344_data, v345_acc, 3, 7, 2);
        r1[9] = (v346_acc[0]);
        r1[11] = (v346_acc[1]);
        r1[13] = (v346_acc[2]);
        r1[15] = (v346_acc[3]);
        float v351_data = r0[8];
        float v352_data = r0[9];
        float v353_data = r0[10];
        float v354_data = r0[11];
        float v355_tp{};
        float v356_tp{};
        float v357_tp{};
        float v358_tp{};
        tensorforge::transpose4x4b32(v355_tp, v356_tp, v357_tp, v358_tp, v351_data, v352_data, v353_data, v354_data);
        tensorforge::VectorT<float, 4> v359_acc{};
        float v366_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v366_data, v359_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v366_data, v367_acc, 3, 0, 2);
        float v370_data = glb_m1[v57_a];
        tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v370_data, v368_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v370_data, v371_acc, 3, 0, 2);
        float v374_data = glb_m1[v61_a];
        tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v374_data, v372_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v374_data, v375_acc, 3, 1, 2);
        float v378_data = glb_m1[v65_a];
        tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v378_data, v376_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v378_data, v379_acc, 3, 1, 2);
        float v382_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v382_data, v380_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v382_data, v383_acc, 3, 2, 2);
        float v386_data = glb_m1[v73_a];
        tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v386_data, v384_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v386_data, v387_acc, 3, 2, 2);
        float v390_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v390_data, v388_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v390_data, v391_acc, 3, 3, 2);
        float v394_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v394_data, v392_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v394_data, v395_acc, 3, 3, 2);
        float v398_data = glb_m1[v85_a];
        tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v398_data, v396_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v398_data, v399_acc, 3, 4, 2);
        float v402_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v402_data, v400_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v402_data, v403_acc, 3, 4, 2);
        float v406_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v406_data, v404_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v406_data, v407_acc, 3, 5, 2);
        float v410_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v410_data, v408_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v410_data, v411_acc, 3, 5, 2);
        float v414_data = glb_m1[v101_a];
        tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v414_data, v412_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v414_data, v415_acc, 3, 6, 2);
        float v418_data = glb_m1[v105_a];
        tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v418_data, v416_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v418_data, v419_acc, 3, 6, 2);
        float v422_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v422_data, v420_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v422_data, v423_acc, 3, 7, 2);
        float v426_data = glb_m1[v113_a];
        tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v426_data, v424_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v426_data, v427_acc, 3, 7, 2);
        r1[16] = (v428_acc[0]);
        r1[18] = (v428_acc[1]);
        r1[20] = (v428_acc[2]);
        r1[22] = (v428_acc[3]);
        tensorforge::VectorT<float, 4> v433_acc{};
        float v440_data = glb_m1[v126_a];
        tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v440_data, v433_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v440_data, v441_acc, 3, 0, 2);
        float v444_data = glb_m1[v131_a];
        tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v444_data, v442_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v444_data, v445_acc, 3, 0, 2);
        float v448_data = glb_m1[v135_a];
        tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v448_data, v446_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v448_data, v449_acc, 3, 1, 2);
        float v452_data = glb_m1[v139_a];
        tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v452_data, v450_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v452_data, v453_acc, 3, 1, 2);
        float v456_data = glb_m1[v143_a];
        tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v456_data, v454_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v456_data, v457_acc, 3, 2, 2);
        float v460_data = glb_m1[v147_a];
        tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v460_data, v458_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v460_data, v461_acc, 3, 2, 2);
        float v464_data = glb_m1[v151_a];
        tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v464_data, v462_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v464_data, v465_acc, 3, 3, 2);
        float v468_data = glb_m1[v155_a];
        tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v468_data, v466_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v468_data, v469_acc, 3, 3, 2);
        float v472_data = glb_m1[v159_a];
        tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v472_data, v470_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v472_data, v473_acc, 3, 4, 2);
        float v476_data = glb_m1[v163_a];
        tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v476_data, v474_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v476_data, v477_acc, 3, 4, 2);
        float v480_data = glb_m1[v167_a];
        tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v480_data, v478_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v480_data, v481_acc, 3, 5, 2);
        float v484_data = glb_m1[v171_a];
        tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v484_data, v482_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v484_data, v485_acc, 3, 5, 2);
        float v488_data = glb_m1[v175_a];
        tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v488_data, v486_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v488_data, v489_acc, 3, 6, 2);
        float v492_data = glb_m1[v179_a];
        tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v492_data, v490_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v492_data, v493_acc, 3, 6, 2);
        float v496_data = glb_m1[v183_a];
        tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v496_data, v494_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v496_data, v497_acc, 3, 7, 2);
        float v500_data = glb_m1[v187_a];
        tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v500_data, v498_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v500_data, v501_acc, 3, 7, 2);
        r1[17] = (v502_acc[0]);
        r1[19] = (v502_acc[1]);
        r1[21] = (v502_acc[2]);
        r1[23] = (v502_acc[3]);
        float v507_data = r0[12];
        float v508_data = r0[13];
        float v509_data = r0[14];
        float v510_data = r0[15];
        float v511_tp{};
        float v512_tp{};
        float v513_tp{};
        float v514_tp{};
        tensorforge::transpose4x4b32(v511_tp, v512_tp, v513_tp, v514_tp, v507_data, v508_data, v509_data, v510_data);
        tensorforge::VectorT<float, 4> v515_acc{};
        float v522_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v522_data, v515_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v522_data, v523_acc, 3, 0, 2);
        float v526_data = glb_m1[v57_a];
        tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v526_data, v524_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v526_data, v527_acc, 3, 0, 2);
        float v530_data = glb_m1[v61_a];
        tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v530_data, v528_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v530_data, v531_acc, 3, 1, 2);
        float v534_data = glb_m1[v65_a];
        tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v534_data, v532_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v534_data, v535_acc, 3, 1, 2);
        float v538_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v538_data, v536_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v538_data, v539_acc, 3, 2, 2);
        float v542_data = glb_m1[v73_a];
        tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v542_data, v540_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v542_data, v543_acc, 3, 2, 2);
        float v546_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v546_data, v544_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v546_data, v547_acc, 3, 3, 2);
        float v550_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v550_data, v548_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v550_data, v551_acc, 3, 3, 2);
        float v554_data = glb_m1[v85_a];
        tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v554_data, v552_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v554_data, v555_acc, 3, 4, 2);
        float v558_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v558_data, v556_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v558_data, v559_acc, 3, 4, 2);
        float v562_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v562_data, v560_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v562_data, v563_acc, 3, 5, 2);
        float v566_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v566_data, v564_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v566_data, v567_acc, 3, 5, 2);
        float v570_data = glb_m1[v101_a];
        tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v570_data, v568_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v570_data, v571_acc, 3, 6, 2);
        float v574_data = glb_m1[v105_a];
        tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v574_data, v572_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v574_data, v575_acc, 3, 6, 2);
        float v578_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v578_data, v576_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v578_data, v579_acc, 3, 7, 2);
        float v582_data = glb_m1[v113_a];
        tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v582_data, v580_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v582_data, v583_acc, 3, 7, 2);
        r1[24] = (v584_acc[0]);
        r1[26] = (v584_acc[1]);
        r1[28] = (v584_acc[2]);
        r1[30] = (v584_acc[3]);
        tensorforge::VectorT<float, 4> v589_acc{};
        float v596_data = glb_m1[v126_a];
        tensorforge::VectorT<float, 4> v597_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v596_data, v589_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v596_data, v597_acc, 3, 0, 2);
        float v600_data = glb_m1[v131_a];
        tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v600_data, v598_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v600_data, v601_acc, 3, 0, 2);
        float v604_data = glb_m1[v135_a];
        tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v604_data, v602_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v604_data, v605_acc, 3, 1, 2);
        float v608_data = glb_m1[v139_a];
        tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v608_data, v606_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v608_data, v609_acc, 3, 1, 2);
        float v612_data = glb_m1[v143_a];
        tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v612_data, v610_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v612_data, v613_acc, 3, 2, 2);
        float v616_data = glb_m1[v147_a];
        tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v616_data, v614_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v616_data, v617_acc, 3, 2, 2);
        float v620_data = glb_m1[v151_a];
        tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v620_data, v618_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v620_data, v621_acc, 3, 3, 2);
        float v624_data = glb_m1[v155_a];
        tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v624_data, v622_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v624_data, v625_acc, 3, 3, 2);
        float v628_data = glb_m1[v159_a];
        tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v628_data, v626_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v628_data, v629_acc, 3, 4, 2);
        float v632_data = glb_m1[v163_a];
        tensorforge::VectorT<float, 4> v633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v632_data, v630_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v632_data, v633_acc, 3, 4, 2);
        float v636_data = glb_m1[v167_a];
        tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v636_data, v634_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v636_data, v637_acc, 3, 5, 2);
        float v640_data = glb_m1[v171_a];
        tensorforge::VectorT<float, 4> v641_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v640_data, v638_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v640_data, v641_acc, 3, 5, 2);
        float v644_data = glb_m1[v175_a];
        tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v644_data, v642_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v644_data, v645_acc, 3, 6, 2);
        float v648_data = glb_m1[v179_a];
        tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v648_data, v646_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v648_data, v649_acc, 3, 6, 2);
        float v652_data = glb_m1[v183_a];
        tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v511_tp, v652_data, v650_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v652_data, v653_acc, 3, 7, 2);
        float v656_data = glb_m1[v187_a];
        tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v513_tp, v656_data, v654_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v656_data, v657_acc, 3, 7, 2);
        r1[25] = (v658_acc[0]);
        r1[27] = (v658_acc[1]);
        r1[29] = (v658_acc[2]);
        r1[31] = (v658_acc[3]);
        float v663_data = r0[16];
        float v664_data = r0[17];
        float v666_tp{};
        float v667_tp{};
        float v668_tp{};
        float v669_tp{};
        tensorforge::transpose4x4b32(v666_tp, v667_tp, v668_tp, v669_tp, v663_data, v664_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v670_acc{};
        float v677_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v678_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v677_data, v670_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v679_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v677_data, v678_acc, 3, 0, 2);
        float v681_data = glb_m1[v57_a];
        tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v681_data, v679_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v681_data, v682_acc, 3, 0, 2);
        float v685_data = glb_m1[v61_a];
        tensorforge::VectorT<float, 4> v686_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v685_data, v683_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v685_data, v686_acc, 3, 1, 2);
        float v689_data = glb_m1[v65_a];
        tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v689_data, v687_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v689_data, v690_acc, 3, 1, 2);
        float v693_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v694_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v693_data, v691_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v695_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v693_data, v694_acc, 3, 2, 2);
        float v697_data = glb_m1[v73_a];
        tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v697_data, v695_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v697_data, v698_acc, 3, 2, 2);
        float v701_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v701_data, v699_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v701_data, v702_acc, 3, 3, 2);
        float v705_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v705_data, v703_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v705_data, v706_acc, 3, 3, 2);
        float v709_data = glb_m1[v85_a];
        tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v709_data, v707_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v709_data, v710_acc, 3, 4, 2);
        float v713_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v713_data, v711_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v715_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v713_data, v714_acc, 3, 4, 2);
        float v717_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v718_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v717_data, v715_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v719_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v717_data, v718_acc, 3, 5, 2);
        float v721_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v721_data, v719_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v721_data, v722_acc, 3, 5, 2);
        float v725_data = glb_m1[v101_a];
        tensorforge::VectorT<float, 4> v726_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v725_data, v723_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v727_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v725_data, v726_acc, 3, 6, 2);
        float v729_data = glb_m1[v105_a];
        tensorforge::VectorT<float, 4> v730_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v729_data, v727_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v731_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v729_data, v730_acc, 3, 6, 2);
        float v733_data = glb_m1[v109_a];
        tensorforge::VectorT<float, 4> v734_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v733_data, v731_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v735_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v733_data, v734_acc, 3, 7, 2);
        float v737_data = glb_m1[v113_a];
        tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v737_data, v735_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v739_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v737_data, v738_acc, 3, 7, 2);
        r1[32] = (v739_acc[0]);
        r1[34] = (v739_acc[1]);
        tensorforge::VectorT<float, 4> v742_acc{};
        float v749_data = glb_m1[v126_a];
        tensorforge::VectorT<float, 4> v750_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v749_data, v742_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v751_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v749_data, v750_acc, 3, 0, 2);
        float v753_data = glb_m1[v131_a];
        tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v753_data, v751_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v755_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v753_data, v754_acc, 3, 0, 2);
        float v757_data = glb_m1[v135_a];
        tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v757_data, v755_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v759_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v757_data, v758_acc, 3, 1, 2);
        float v761_data = glb_m1[v139_a];
        tensorforge::VectorT<float, 4> v762_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v761_data, v759_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v763_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v761_data, v762_acc, 3, 1, 2);
        float v765_data = glb_m1[v143_a];
        tensorforge::VectorT<float, 4> v766_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v765_data, v763_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v767_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v765_data, v766_acc, 3, 2, 2);
        float v769_data = glb_m1[v147_a];
        tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v769_data, v767_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v771_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v769_data, v770_acc, 3, 2, 2);
        float v773_data = glb_m1[v151_a];
        tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v773_data, v771_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v773_data, v774_acc, 3, 3, 2);
        float v777_data = glb_m1[v155_a];
        tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v777_data, v775_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v779_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v777_data, v778_acc, 3, 3, 2);
        float v781_data = glb_m1[v159_a];
        tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v781_data, v779_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v783_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v781_data, v782_acc, 3, 4, 2);
        float v785_data = glb_m1[v163_a];
        tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v785_data, v783_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v787_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v785_data, v786_acc, 3, 4, 2);
        float v789_data = glb_m1[v167_a];
        tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v789_data, v787_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v789_data, v790_acc, 3, 5, 2);
        float v793_data = glb_m1[v171_a];
        tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v793_data, v791_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v795_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v793_data, v794_acc, 3, 5, 2);
        float v797_data = glb_m1[v175_a];
        tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v797_data, v795_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v799_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v797_data, v798_acc, 3, 6, 2);
        float v801_data = glb_m1[v179_a];
        tensorforge::VectorT<float, 4> v802_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v801_data, v799_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v803_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v801_data, v802_acc, 3, 6, 2);
        float v805_data = glb_m1[v183_a];
        tensorforge::VectorT<float, 4> v806_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v666_tp, v805_data, v803_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v807_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v667_tp, v805_data, v806_acc, 3, 7, 2);
        float v809_data = glb_m1[v187_a];
        tensorforge::VectorT<float, 4> v810_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v668_tp, v809_data, v807_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v811_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v809_data, v810_acc, 3, 7, 2);
        r1[33] = (v811_acc[0]);
        r1[35] = (v811_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v814_i0 = 0; v814_i0 < 1; ++v814_i0) {
          #pragma unroll
          for (int32_t v815_i1 = 0; v815_i1 < 18; ++v815_i1) {
            float v818_data = r1[(v814_i0 + (v815_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v29_lead + (v814_i0 * 32)) + (v815_i1 * 56))] = v818_data;
            }
          }
        }
        if (v823_g) {
          #pragma unroll
          for (int32_t v824_i1 = 0; v824_i1 < 18; ++v824_i1) {
            float v827_data = r1[(1 + (v824_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[(v125_lead + (v824_i1 * 56))] = v827_data;
            }
          }
        }
      }
    }
  }
}

