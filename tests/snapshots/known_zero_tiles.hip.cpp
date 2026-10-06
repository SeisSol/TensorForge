// === base name ===
kernel_6e4e05d09469158b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6e4e05d09469158b = {{32, 8, 1}, 32, 56, 1, 8, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6e4e05d09469158b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6e4e05d09469158b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6e4e05d09469158b(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6e4e05d09469158b, block.x * block.y * block.z, 1792 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1792 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6e4e05d09469158b, block.x * block.y * block.z, 0));
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
void launcher_kernel_6e4e05d09469158b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6e4e05d09469158b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6e4e05d09469158b), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_6e4e05d09469158b, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6e4e05d09469158b(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 4> v12_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      tensorforge::VectorT<float, 2> v13_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024]);
      *(tensorforge::VectorT<float, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v13_ld;
      float v14_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 1536] = v14_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v16_batchIdLane0 = threadIdx.y % 2;
      int32_t v32_lead = threadIdx.x % 32;
      int32_t v52_r = (threadIdx.y % 2) * 56;
      int32_t v55_a = v32_lead + v52_r;
      int32_t v60_a = v55_a + 112;
      int32_t v64_a = v55_a + 224;
      int32_t v68_a = v55_a + 336;
      int32_t v72_a = v55_a + 448;
      int32_t v76_a = v55_a + 560;
      int32_t v80_a = v55_a + 672;
      int32_t v84_a = v55_a + 784;
      int32_t v88_a = v55_a + 896;
      int32_t v92_a = v55_a + 1008;
      int32_t v96_a = v55_a + 1120;
      int32_t v100_a = v55_a + 1232;
      int32_t v104_a = v55_a + 1344;
      int32_t v108_a = v55_a + 1456;
      int32_t v112_a = v55_a + 1568;
      int32_t v116_a = v55_a + 1680;
      int32_t v128_lead = v32_lead + 32_i32;
      int32_t v129_a = v128_lead + v52_r;
      int32_t v134_a = v129_a + 112;
      int32_t v138_a = v129_a + 224;
      int32_t v142_a = v129_a + 336;
      int32_t v146_a = v129_a + 448;
      int32_t v150_a = v129_a + 560;
      int32_t v154_a = v129_a + 672;
      int32_t v158_a = v129_a + 784;
      int32_t v162_a = v129_a + 896;
      int32_t v166_a = v129_a + 1008;
      int32_t v170_a = v129_a + 1120;
      int32_t v174_a = v129_a + 1232;
      int32_t v178_a = v129_a + 1344;
      int32_t v182_a = v129_a + 1456;
      int32_t v186_a = v129_a + 1568;
      int32_t v190_a = v129_a + 1680;
      bool v826_g = v32_lead < 24;
      for (size_t v17_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v17_batchIdGroup0 < numElements0; v17_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v18_row = v17_batchIdGroup0 + v16_batchIdLane0;
        const bool batchIdActive0 = v18_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v18_row]));
        size_t v20_batchId0 = batchIdActive0 ? v18_row : v17_batchIdGroup0;
        size_t v21_ahead1 = v20_batchId0 + (gridDim.x * blockDim.y);
        size_t v23_batchId1 = (v21_ahead1 < numElements0) ? v21_ahead1 : v20_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v20_batchId0 * 1008 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v20_batchId0 * 576 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
          int32_t v36_lead = v32_lead + (v33_i0 * 32);
          #pragma unroll
          for (int32_t v34_i1 = 0; v34_i1 < 18; ++v34_i1) {
            float v39_data = __builtin_nontemporal_load(&glb_m2[(v36_lead + (v34_i1 * 32))]);
            r0[(v33_i0 + v34_i1)] = v39_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[36]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 56), (0, 18)] [(0, 32)]
        float v42_data = r0[0];
        float v43_data = r0[1];
        float v44_data = r0[2];
        float v45_data = r0[3];
        float v46_tp{};
        float v47_tp{};
        float v48_tp{};
        float v49_tp{};
        tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
        tensorforge::VectorT<float, 4> v50_acc{};
        float v57_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v57_data, v50_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v57_data, v58_acc, 3, 0, 2);
        float v61_data = glb_m1[v60_a];
        tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v59_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v61_data, v62_acc, 3, 0, 2);
        float v65_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v65_data, v63_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v65_data, v66_acc, 3, 1, 2);
        float v69_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v67_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v69_data, v70_acc, 3, 1, 2);
        float v73_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v73_data, v71_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v73_data, v74_acc, 3, 2, 2);
        float v77_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v75_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v77_data, v78_acc, 3, 2, 2);
        float v81_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v81_data, v79_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v81_data, v82_acc, 3, 3, 2);
        float v85_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v85_data, v83_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v85_data, v86_acc, 3, 3, 2);
        float v89_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v89_data, v87_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v89_data, v90_acc, 3, 4, 2);
        float v93_data = glb_m1[v92_a];
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v93_data, v91_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v93_data, v94_acc, 3, 4, 2);
        float v97_data = glb_m1[v96_a];
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v97_data, v95_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v97_data, v98_acc, 3, 5, 2);
        float v101_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v101_data, v99_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v101_data, v102_acc, 3, 5, 2);
        float v105_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v105_data, v103_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v105_data, v106_acc, 3, 6, 2);
        float v109_data = glb_m1[v108_a];
        tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v109_data, v107_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v109_data, v110_acc, 3, 6, 2);
        float v113_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v113_data, v111_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v113_data, v114_acc, 3, 7, 2);
        float v117_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v117_data, v115_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v117_data, v118_acc, 3, 7, 2);
        r1[0] = (v119_acc[0]);
        r1[2] = (v119_acc[1]);
        r1[4] = (v119_acc[2]);
        r1[6] = (v119_acc[3]);
        tensorforge::VectorT<float, 4> v124_acc{};
        float v131_data = glb_m1[v129_a];
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v131_data, v124_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v131_data, v132_acc, 3, 0, 2);
        float v135_data = glb_m1[v134_a];
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v135_data, v133_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v135_data, v136_acc, 3, 0, 2);
        float v139_data = glb_m1[v138_a];
        tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v139_data, v137_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v139_data, v140_acc, 3, 1, 2);
        float v143_data = glb_m1[v142_a];
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v143_data, v141_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v143_data, v144_acc, 3, 1, 2);
        float v147_data = glb_m1[v146_a];
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v147_data, v145_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v147_data, v148_acc, 3, 2, 2);
        float v151_data = glb_m1[v150_a];
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v151_data, v149_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v151_data, v152_acc, 3, 2, 2);
        float v155_data = glb_m1[v154_a];
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v155_data, v153_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v155_data, v156_acc, 3, 3, 2);
        float v159_data = glb_m1[v158_a];
        tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v159_data, v157_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v159_data, v160_acc, 3, 3, 2);
        float v163_data = glb_m1[v162_a];
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v163_data, v161_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v163_data, v164_acc, 3, 4, 2);
        float v167_data = glb_m1[v166_a];
        tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v167_data, v165_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v167_data, v168_acc, 3, 4, 2);
        float v171_data = glb_m1[v170_a];
        tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v171_data, v169_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v171_data, v172_acc, 3, 5, 2);
        float v175_data = glb_m1[v174_a];
        tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v175_data, v173_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v175_data, v176_acc, 3, 5, 2);
        float v179_data = glb_m1[v178_a];
        tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v179_data, v177_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v179_data, v180_acc, 3, 6, 2);
        float v183_data = glb_m1[v182_a];
        tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v183_data, v181_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v183_data, v184_acc, 3, 6, 2);
        float v187_data = glb_m1[v186_a];
        tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v187_data, v185_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v187_data, v188_acc, 3, 7, 2);
        float v191_data = glb_m1[v190_a];
        tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v191_data, v189_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v191_data, v192_acc, 3, 7, 2);
        r1[1] = (v193_acc[0]);
        r1[3] = (v193_acc[1]);
        r1[5] = (v193_acc[2]);
        r1[7] = (v193_acc[3]);
        float v198_data = r0[4];
        float v199_data = r0[5];
        float v200_data = r0[6];
        float v201_data = r0[7];
        float v202_tp{};
        float v203_tp{};
        float v204_tp{};
        float v205_tp{};
        tensorforge::transpose4x4b32(v202_tp, v203_tp, v204_tp, v205_tp, v198_data, v199_data, v200_data, v201_data);
        tensorforge::VectorT<float, 4> v206_acc{};
        float v213_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v213_data, v206_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v213_data, v214_acc, 3, 0, 2);
        float v217_data = glb_m1[v60_a];
        tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v217_data, v215_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v217_data, v218_acc, 3, 0, 2);
        float v221_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v221_data, v219_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v221_data, v222_acc, 3, 1, 2);
        float v225_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v225_data, v223_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v225_data, v226_acc, 3, 1, 2);
        float v229_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v229_data, v227_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v229_data, v230_acc, 3, 2, 2);
        float v233_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v233_data, v231_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v233_data, v234_acc, 3, 2, 2);
        float v237_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v237_data, v235_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v237_data, v238_acc, 3, 3, 2);
        float v241_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v241_data, v239_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v241_data, v242_acc, 3, 3, 2);
        float v245_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v245_data, v243_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v245_data, v246_acc, 3, 4, 2);
        float v249_data = glb_m1[v92_a];
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v249_data, v247_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v249_data, v250_acc, 3, 4, 2);
        float v253_data = glb_m1[v96_a];
        tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v253_data, v251_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v253_data, v254_acc, 3, 5, 2);
        float v257_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v257_data, v255_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v257_data, v258_acc, 3, 5, 2);
        float v261_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v261_data, v259_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v261_data, v262_acc, 3, 6, 2);
        float v265_data = glb_m1[v108_a];
        tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v265_data, v263_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v265_data, v266_acc, 3, 6, 2);
        float v269_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v269_data, v267_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v269_data, v270_acc, 3, 7, 2);
        float v273_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v273_data, v271_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v273_data, v274_acc, 3, 7, 2);
        r1[8] = (v275_acc[0]);
        r1[10] = (v275_acc[1]);
        r1[12] = (v275_acc[2]);
        r1[14] = (v275_acc[3]);
        tensorforge::VectorT<float, 4> v280_acc{};
        float v287_data = glb_m1[v129_a];
        tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v287_data, v280_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v287_data, v288_acc, 3, 0, 2);
        float v291_data = glb_m1[v134_a];
        tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v291_data, v289_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v291_data, v292_acc, 3, 0, 2);
        float v295_data = glb_m1[v138_a];
        tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v295_data, v293_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v295_data, v296_acc, 3, 1, 2);
        float v299_data = glb_m1[v142_a];
        tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v299_data, v297_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v299_data, v300_acc, 3, 1, 2);
        float v303_data = glb_m1[v146_a];
        tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v303_data, v301_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v303_data, v304_acc, 3, 2, 2);
        float v307_data = glb_m1[v150_a];
        tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v307_data, v305_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v307_data, v308_acc, 3, 2, 2);
        float v311_data = glb_m1[v154_a];
        tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v311_data, v309_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v311_data, v312_acc, 3, 3, 2);
        float v315_data = glb_m1[v158_a];
        tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v315_data, v313_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v315_data, v316_acc, 3, 3, 2);
        float v319_data = glb_m1[v162_a];
        tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v319_data, v317_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v319_data, v320_acc, 3, 4, 2);
        float v323_data = glb_m1[v166_a];
        tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v323_data, v321_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v323_data, v324_acc, 3, 4, 2);
        float v327_data = glb_m1[v170_a];
        tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v327_data, v325_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v327_data, v328_acc, 3, 5, 2);
        float v331_data = glb_m1[v174_a];
        tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v331_data, v329_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v331_data, v332_acc, 3, 5, 2);
        float v335_data = glb_m1[v178_a];
        tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v335_data, v333_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v335_data, v336_acc, 3, 6, 2);
        float v339_data = glb_m1[v182_a];
        tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v339_data, v337_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v339_data, v340_acc, 3, 6, 2);
        float v343_data = glb_m1[v186_a];
        tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v343_data, v341_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v343_data, v344_acc, 3, 7, 2);
        float v347_data = glb_m1[v190_a];
        tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v347_data, v345_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v347_data, v348_acc, 3, 7, 2);
        r1[9] = (v349_acc[0]);
        r1[11] = (v349_acc[1]);
        r1[13] = (v349_acc[2]);
        r1[15] = (v349_acc[3]);
        float v354_data = r0[8];
        float v355_data = r0[9];
        float v356_data = r0[10];
        float v357_data = r0[11];
        float v358_tp{};
        float v359_tp{};
        float v360_tp{};
        float v361_tp{};
        tensorforge::transpose4x4b32(v358_tp, v359_tp, v360_tp, v361_tp, v354_data, v355_data, v356_data, v357_data);
        tensorforge::VectorT<float, 4> v362_acc{};
        float v369_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v369_data, v362_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v369_data, v370_acc, 3, 0, 2);
        float v373_data = glb_m1[v60_a];
        tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v373_data, v371_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v373_data, v374_acc, 3, 0, 2);
        float v377_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v377_data, v375_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v377_data, v378_acc, 3, 1, 2);
        float v381_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v381_data, v379_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v381_data, v382_acc, 3, 1, 2);
        float v385_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v385_data, v383_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v385_data, v386_acc, 3, 2, 2);
        float v389_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v389_data, v387_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v389_data, v390_acc, 3, 2, 2);
        float v393_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v393_data, v391_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v393_data, v394_acc, 3, 3, 2);
        float v397_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v397_data, v395_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v397_data, v398_acc, 3, 3, 2);
        float v401_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v401_data, v399_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v401_data, v402_acc, 3, 4, 2);
        float v405_data = glb_m1[v92_a];
        tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v405_data, v403_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v405_data, v406_acc, 3, 4, 2);
        float v409_data = glb_m1[v96_a];
        tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v409_data, v407_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v409_data, v410_acc, 3, 5, 2);
        float v413_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v413_data, v411_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v413_data, v414_acc, 3, 5, 2);
        float v417_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v417_data, v415_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v417_data, v418_acc, 3, 6, 2);
        float v421_data = glb_m1[v108_a];
        tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v421_data, v419_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v421_data, v422_acc, 3, 6, 2);
        float v425_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v425_data, v423_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v425_data, v426_acc, 3, 7, 2);
        float v429_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v429_data, v427_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v429_data, v430_acc, 3, 7, 2);
        r1[16] = (v431_acc[0]);
        r1[18] = (v431_acc[1]);
        r1[20] = (v431_acc[2]);
        r1[22] = (v431_acc[3]);
        tensorforge::VectorT<float, 4> v436_acc{};
        float v443_data = glb_m1[v129_a];
        tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v443_data, v436_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v443_data, v444_acc, 3, 0, 2);
        float v447_data = glb_m1[v134_a];
        tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v447_data, v445_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v447_data, v448_acc, 3, 0, 2);
        float v451_data = glb_m1[v138_a];
        tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v451_data, v449_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v451_data, v452_acc, 3, 1, 2);
        float v455_data = glb_m1[v142_a];
        tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v455_data, v453_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v455_data, v456_acc, 3, 1, 2);
        float v459_data = glb_m1[v146_a];
        tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v459_data, v457_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v459_data, v460_acc, 3, 2, 2);
        float v463_data = glb_m1[v150_a];
        tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v463_data, v461_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v463_data, v464_acc, 3, 2, 2);
        float v467_data = glb_m1[v154_a];
        tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v467_data, v465_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v467_data, v468_acc, 3, 3, 2);
        float v471_data = glb_m1[v158_a];
        tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v471_data, v469_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v471_data, v472_acc, 3, 3, 2);
        float v475_data = glb_m1[v162_a];
        tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v475_data, v473_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v475_data, v476_acc, 3, 4, 2);
        float v479_data = glb_m1[v166_a];
        tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v479_data, v477_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v479_data, v480_acc, 3, 4, 2);
        float v483_data = glb_m1[v170_a];
        tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v483_data, v481_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v483_data, v484_acc, 3, 5, 2);
        float v487_data = glb_m1[v174_a];
        tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v487_data, v485_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v487_data, v488_acc, 3, 5, 2);
        float v491_data = glb_m1[v178_a];
        tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v491_data, v489_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v491_data, v492_acc, 3, 6, 2);
        float v495_data = glb_m1[v182_a];
        tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v495_data, v493_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v495_data, v496_acc, 3, 6, 2);
        float v499_data = glb_m1[v186_a];
        tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v499_data, v497_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v499_data, v500_acc, 3, 7, 2);
        float v503_data = glb_m1[v190_a];
        tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v503_data, v501_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v503_data, v504_acc, 3, 7, 2);
        r1[17] = (v505_acc[0]);
        r1[19] = (v505_acc[1]);
        r1[21] = (v505_acc[2]);
        r1[23] = (v505_acc[3]);
        float v510_data = r0[12];
        float v511_data = r0[13];
        float v512_data = r0[14];
        float v513_data = r0[15];
        float v514_tp{};
        float v515_tp{};
        float v516_tp{};
        float v517_tp{};
        tensorforge::transpose4x4b32(v514_tp, v515_tp, v516_tp, v517_tp, v510_data, v511_data, v512_data, v513_data);
        tensorforge::VectorT<float, 4> v518_acc{};
        float v525_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v525_data, v518_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v525_data, v526_acc, 3, 0, 2);
        float v529_data = glb_m1[v60_a];
        tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v529_data, v527_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v529_data, v530_acc, 3, 0, 2);
        float v533_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v533_data, v531_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v533_data, v534_acc, 3, 1, 2);
        float v537_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v537_data, v535_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v537_data, v538_acc, 3, 1, 2);
        float v541_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v541_data, v539_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v541_data, v542_acc, 3, 2, 2);
        float v545_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v545_data, v543_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v545_data, v546_acc, 3, 2, 2);
        float v549_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v549_data, v547_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v549_data, v550_acc, 3, 3, 2);
        float v553_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v553_data, v551_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v553_data, v554_acc, 3, 3, 2);
        float v557_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v557_data, v555_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v557_data, v558_acc, 3, 4, 2);
        float v561_data = glb_m1[v92_a];
        tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v561_data, v559_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v561_data, v562_acc, 3, 4, 2);
        float v565_data = glb_m1[v96_a];
        tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v565_data, v563_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v565_data, v566_acc, 3, 5, 2);
        float v569_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v569_data, v567_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v569_data, v570_acc, 3, 5, 2);
        float v573_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v573_data, v571_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v573_data, v574_acc, 3, 6, 2);
        float v577_data = glb_m1[v108_a];
        tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v577_data, v575_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v577_data, v578_acc, 3, 6, 2);
        float v581_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v581_data, v579_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v581_data, v582_acc, 3, 7, 2);
        float v585_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v585_data, v583_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v585_data, v586_acc, 3, 7, 2);
        r1[24] = (v587_acc[0]);
        r1[26] = (v587_acc[1]);
        r1[28] = (v587_acc[2]);
        r1[30] = (v587_acc[3]);
        tensorforge::VectorT<float, 4> v592_acc{};
        float v599_data = glb_m1[v129_a];
        tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v599_data, v592_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v599_data, v600_acc, 3, 0, 2);
        float v603_data = glb_m1[v134_a];
        tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v603_data, v601_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v605_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v603_data, v604_acc, 3, 0, 2);
        float v607_data = glb_m1[v138_a];
        tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v607_data, v605_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v607_data, v608_acc, 3, 1, 2);
        float v611_data = glb_m1[v142_a];
        tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v611_data, v609_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v611_data, v612_acc, 3, 1, 2);
        float v615_data = glb_m1[v146_a];
        tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v615_data, v613_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v615_data, v616_acc, 3, 2, 2);
        float v619_data = glb_m1[v150_a];
        tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v619_data, v617_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v619_data, v620_acc, 3, 2, 2);
        float v623_data = glb_m1[v154_a];
        tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v623_data, v621_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v623_data, v624_acc, 3, 3, 2);
        float v627_data = glb_m1[v158_a];
        tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v627_data, v625_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v627_data, v628_acc, 3, 3, 2);
        float v631_data = glb_m1[v162_a];
        tensorforge::VectorT<float, 4> v632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v631_data, v629_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v631_data, v632_acc, 3, 4, 2);
        float v635_data = glb_m1[v166_a];
        tensorforge::VectorT<float, 4> v636_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v635_data, v633_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v635_data, v636_acc, 3, 4, 2);
        float v639_data = glb_m1[v170_a];
        tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v639_data, v637_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v641_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v639_data, v640_acc, 3, 5, 2);
        float v643_data = glb_m1[v174_a];
        tensorforge::VectorT<float, 4> v644_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v643_data, v641_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v643_data, v644_acc, 3, 5, 2);
        float v647_data = glb_m1[v178_a];
        tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v647_data, v645_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v647_data, v648_acc, 3, 6, 2);
        float v651_data = glb_m1[v182_a];
        tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v651_data, v649_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v651_data, v652_acc, 3, 6, 2);
        float v655_data = glb_m1[v186_a];
        tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v514_tp, v655_data, v653_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v515_tp, v655_data, v656_acc, 3, 7, 2);
        float v659_data = glb_m1[v190_a];
        tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v516_tp, v659_data, v657_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v517_tp, v659_data, v660_acc, 3, 7, 2);
        r1[25] = (v661_acc[0]);
        r1[27] = (v661_acc[1]);
        r1[29] = (v661_acc[2]);
        r1[31] = (v661_acc[3]);
        float v666_data = r0[16];
        float v667_data = r0[17];
        float v669_tp{};
        float v670_tp{};
        float v671_tp{};
        float v672_tp{};
        tensorforge::transpose4x4b32(v669_tp, v670_tp, v671_tp, v672_tp, v666_data, v667_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v673_acc{};
        float v680_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v680_data, v673_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v680_data, v681_acc, 3, 0, 2);
        float v684_data = glb_m1[v60_a];
        tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v684_data, v682_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v686_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v684_data, v685_acc, 3, 0, 2);
        float v688_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v689_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v688_data, v686_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v688_data, v689_acc, 3, 1, 2);
        float v692_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v692_data, v690_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v694_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v692_data, v693_acc, 3, 1, 2);
        float v696_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v697_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v696_data, v694_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v696_data, v697_acc, 3, 2, 2);
        float v700_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v700_data, v698_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v700_data, v701_acc, 3, 2, 2);
        float v704_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v704_data, v702_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v704_data, v705_acc, 3, 3, 2);
        float v708_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v708_data, v706_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v708_data, v709_acc, 3, 3, 2);
        float v712_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v713_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v712_data, v710_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v712_data, v713_acc, 3, 4, 2);
        float v716_data = glb_m1[v92_a];
        tensorforge::VectorT<float, 4> v717_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v716_data, v714_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v718_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v716_data, v717_acc, 3, 4, 2);
        float v720_data = glb_m1[v96_a];
        tensorforge::VectorT<float, 4> v721_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v720_data, v718_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v720_data, v721_acc, 3, 5, 2);
        float v724_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v725_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v724_data, v722_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v726_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v724_data, v725_acc, 3, 5, 2);
        float v728_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v729_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v728_data, v726_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v730_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v728_data, v729_acc, 3, 6, 2);
        float v732_data = glb_m1[v108_a];
        tensorforge::VectorT<float, 4> v733_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v732_data, v730_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v734_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v732_data, v733_acc, 3, 6, 2);
        float v736_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v737_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v736_data, v734_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v736_data, v737_acc, 3, 7, 2);
        float v740_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v741_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v740_data, v738_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v742_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v740_data, v741_acc, 3, 7, 2);
        r1[32] = (v742_acc[0]);
        r1[34] = (v742_acc[1]);
        tensorforge::VectorT<float, 4> v745_acc{};
        float v752_data = glb_m1[v129_a];
        tensorforge::VectorT<float, 4> v753_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v752_data, v745_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v752_data, v753_acc, 3, 0, 2);
        float v756_data = glb_m1[v134_a];
        tensorforge::VectorT<float, 4> v757_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v756_data, v754_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v756_data, v757_acc, 3, 0, 2);
        float v760_data = glb_m1[v138_a];
        tensorforge::VectorT<float, 4> v761_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v760_data, v758_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v762_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v760_data, v761_acc, 3, 1, 2);
        float v764_data = glb_m1[v142_a];
        tensorforge::VectorT<float, 4> v765_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v764_data, v762_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v766_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v764_data, v765_acc, 3, 1, 2);
        float v768_data = glb_m1[v146_a];
        tensorforge::VectorT<float, 4> v769_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v768_data, v766_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v768_data, v769_acc, 3, 2, 2);
        float v772_data = glb_m1[v150_a];
        tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v772_data, v770_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v772_data, v773_acc, 3, 2, 2);
        float v776_data = glb_m1[v154_a];
        tensorforge::VectorT<float, 4> v777_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v776_data, v774_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v776_data, v777_acc, 3, 3, 2);
        float v780_data = glb_m1[v158_a];
        tensorforge::VectorT<float, 4> v781_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v780_data, v778_acc, 3, 3, 1);
        tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v780_data, v781_acc, 3, 3, 2);
        float v784_data = glb_m1[v162_a];
        tensorforge::VectorT<float, 4> v785_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v784_data, v782_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v784_data, v785_acc, 3, 4, 2);
        float v788_data = glb_m1[v166_a];
        tensorforge::VectorT<float, 4> v789_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v788_data, v786_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v788_data, v789_acc, 3, 4, 2);
        float v792_data = glb_m1[v170_a];
        tensorforge::VectorT<float, 4> v793_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v792_data, v790_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v792_data, v793_acc, 3, 5, 2);
        float v796_data = glb_m1[v174_a];
        tensorforge::VectorT<float, 4> v797_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v796_data, v794_acc, 3, 5, 1);
        tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v796_data, v797_acc, 3, 5, 2);
        float v800_data = glb_m1[v178_a];
        tensorforge::VectorT<float, 4> v801_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v800_data, v798_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v802_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v800_data, v801_acc, 3, 6, 2);
        float v804_data = glb_m1[v182_a];
        tensorforge::VectorT<float, 4> v805_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v804_data, v802_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v806_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v804_data, v805_acc, 3, 6, 2);
        float v808_data = glb_m1[v186_a];
        tensorforge::VectorT<float, 4> v809_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v669_tp, v808_data, v806_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v810_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v670_tp, v808_data, v809_acc, 3, 7, 2);
        float v812_data = glb_m1[v190_a];
        tensorforge::VectorT<float, 4> v813_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v671_tp, v812_data, v810_acc, 3, 7, 1);
        tensorforge::VectorT<float, 4> v814_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v672_tp, v812_data, v813_acc, 3, 7, 2);
        r1[33] = (v814_acc[0]);
        r1[35] = (v814_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v817_i0 = 0; v817_i0 < 1; ++v817_i0) {
          #pragma unroll
          for (int32_t v818_i1 = 0; v818_i1 < 18; ++v818_i1) {
            float v821_data = r1[(v817_i0 + (v818_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v32_lead + (v817_i0 * 32)) + (v818_i1 * 56))] = v821_data;
            }
          }
        }
        if (v826_g) {
          #pragma unroll
          for (int32_t v827_i1 = 0; v827_i1 < 18; ++v827_i1) {
            float v830_data = r1[(1 + (v827_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[(v128_lead + (v827_i1 * 56))] = v830_data;
            }
          }
        }
      }
    }
  }
}

