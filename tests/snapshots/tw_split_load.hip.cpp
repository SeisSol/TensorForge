// === base name ===
kernel_399d5ca0b03101d8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_399d5ca0b03101d8 = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_399d5ca0b03101d8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_399d5ca0b03101d8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_399d5ca0b03101d8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_399d5ca0b03101d8, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_399d5ca0b03101d8, block.x * block.y * block.z, 0));
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
void launcher_kernel_399d5ca0b03101d8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_399d5ca0b03101d8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_399d5ca0b03101d8), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_399d5ca0b03101d8, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_399d5ca0b03101d8(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
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
      __syncthreads();
      size_t v14_batchIdLane0 = threadIdx.y % 4;
      size_t v17_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v34_lead = threadIdx.x % 16;
      int32_t v46_lead = v34_lead + 16_i32;
      bool v49_g = v34_lead < 1;
      int32_t v75_a = v34_lead + ((threadIdx.y % 4) * 10);
      int32_t v82_a = v75_a + 40;
      int32_t v88_a = v75_a + 80;
      int32_t v94_a = v75_a + 120;
      int32_t v100_a = v34_lead + 160;
      int32_t v164_a = v34_lead + 10;
      int32_t v166_a = v34_lead + 20;
      int32_t v168_a = v34_lead + 30;
      int32_t v170_a = v34_lead + 40;
      int32_t v172_a = v34_lead + 50;
      int32_t v174_a = v34_lead + 60;
      int32_t v176_a = v34_lead + 70;
      int32_t v178_a = v34_lead + 80;
      int32_t v180_a = v34_lead + 90;
      int32_t v182_a = v34_lead + 100;
      int32_t v184_a = v34_lead + 110;
      int32_t v186_a = v34_lead + 120;
      int32_t v188_a = v34_lead + 130;
      int32_t v190_a = v34_lead + 140;
      int32_t v192_a = v34_lead + 150;
      bool v365_g = v34_lead < 10;
      float r0[18]{};
      size_t v382_row = v17_batchIdGroup0First + v14_batchIdLane0;
      const bool peel_batchIdActive0 = v382_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v382_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v382_row : v17_batchIdGroup0First) * 153 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v390_i0 = 0; v390_i0 < 1; ++v390_i0) {
        int32_t v392_lead = v34_lead + (v390_i0 * 16);
        #pragma unroll
        for (int32_t v393_i1 = 0; v393_i1 < 9; ++v393_i1) {
          float v396_data = __builtin_nontemporal_load(&peel_glb_m2[(v392_lead + (v393_i1 * 17))]);
          r0[(v390_i0 + (v393_i1 * 2))] = v396_data;
        }
      }
      #pragma unroll
      for (int32_t v399_i1 = 0; v399_i1 < 9; ++v399_i1) {
        float v402_data = v49_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v46_lead + (v399_i1 * 17))])) : (0.0f);
        r0[(1 + (v399_i1 * 2))] = v402_data;
      }
      for (size_t v18_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v18_batchIdGroup0 < numElements0; v18_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v379_groupNextAhead = v18_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v381_groupNext = (v379_groupNextAhead < numElements0) ? v379_groupNextAhead : v18_batchIdGroup0;
        size_t v386_row = v381_groupNext + v14_batchIdLane0;
        const bool wrap_batchIdActive0 = v386_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v386_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v386_row : v381_groupNext) * 153 + 0 + m2_extraOffset];
        size_t v19_row = v18_batchIdGroup0 + v14_batchIdLane0;
        const bool batchIdActive0 = v19_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v19_row]));
        size_t v21_batchId0 = batchIdActive0 ? v19_row : v18_batchIdGroup0;
        size_t v22_ahead1 = v21_batchId0 + (gridDim.x * blockDim.y);
        size_t v24_batchId1 = (v22_ahead1 < numElements0) ? v22_ahead1 : v21_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v21_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v21_batchId0 * 153 + 0 + m4_extraOffset];
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v200_i0 = 0; v200_i0 < 1; ++v200_i0) {
          int32_t v203_lead = v34_lead + (v200_i0 * 16);
          #pragma unroll
          for (int32_t v201_i1 = 0; v201_i1 < 9; ++v201_i1) {
            float v206_data = __builtin_nontemporal_load(&glb_m4[(v203_lead + (v201_i1 * 17))]);
            r2[(v200_i0 + (v201_i1 * 2))] = v206_data;
          }
        }
        #pragma unroll
        for (int32_t v209_i1 = 0; v209_i1 < 9; ++v209_i1) {
          float v215_data = v49_g ? (__builtin_nontemporal_load(&glb_m4[(v46_lead + (v209_i1 * 17))])) : (0.0f);
          r2[(1 + (v209_i1 * 2))] = v215_data;
        }
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(0, 17)]
        float v54_data = r0[0];
        float v55_data = r0[2];
        float v56_data = r0[4];
        float v57_data = r0[6];
        float v58_tp{};
        float v59_tp{};
        float v60_tp{};
        float v61_tp{};
        tensorforge::transpose4x4b32(v58_tp, v59_tp, v60_tp, v61_tp, v54_data, v55_data, v56_data, v57_data);
        float v62_data = r0[1];
        float v63_data = r0[3];
        float v64_data = r0[5];
        float v65_data = r0[7];
        float v66_tp{};
        float v67_tp{};
        float v68_tp{};
        float v69_tp{};
        tensorforge::transpose4x4b32(v66_tp, v67_tp, v68_tp, v69_tp, v62_data, v63_data, v64_data, v65_data);
        tensorforge::VectorT<float, 4> v70_acc{};
        float v77_data = glb_m1[v75_a];
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v77_data, v70_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v77_data, v78_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v77_data, v79_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v77_data, v80_acc, 2, 0, 7);
        float v83_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v83_data, v81_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v83_data, v84_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v83_data, v85_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v83_data, v86_acc, 2, 1, 7);
        float v89_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v89_data, v87_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v89_data, v90_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v89_data, v91_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v89_data, v92_acc, 2, 2, 7);
        float v95_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v93_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v95_data, v96_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v95_data, v97_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v95_data, v98_acc, 2, 3, 7);
        float v101_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v101_data, v99_acc, 2, 0, 0);
        r1[0] = (v102_acc[0]);
        r1[1] = (v102_acc[1]);
        r1[2] = (v102_acc[2]);
        r1[3] = (v102_acc[3]);
        float v107_data = r0[8];
        float v108_data = r0[10];
        float v109_data = r0[12];
        float v110_data = r0[14];
        float v111_tp{};
        float v112_tp{};
        float v113_tp{};
        float v114_tp{};
        tensorforge::transpose4x4b32(v111_tp, v112_tp, v113_tp, v114_tp, v107_data, v108_data, v109_data, v110_data);
        float v115_data = r0[9];
        float v116_data = r0[11];
        float v117_data = r0[13];
        float v118_data = r0[15];
        float v119_tp{};
        float v120_tp{};
        float v121_tp{};
        float v122_tp{};
        tensorforge::transpose4x4b32(v119_tp, v120_tp, v121_tp, v122_tp, v115_data, v116_data, v117_data, v118_data);
        tensorforge::VectorT<float, 4> v123_acc{};
        float v130_data = glb_m1[v75_a];
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v130_data, v123_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v130_data, v131_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v130_data, v132_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v130_data, v133_acc, 2, 0, 7);
        float v136_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v136_data, v134_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v136_data, v137_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v136_data, v138_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v136_data, v139_acc, 2, 1, 7);
        float v142_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v142_data, v140_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v142_data, v143_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v142_data, v144_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v142_data, v145_acc, 2, 2, 7);
        float v148_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v148_data, v146_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v148_data, v149_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v148_data, v150_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v148_data, v151_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v101_data, v152_acc, 2, 0, 0);
        r1[4] = (v155_acc[0]);
        r1[5] = (v155_acc[1]);
        r1[6] = (v155_acc[2]);
        r1[7] = (v155_acc[3]);
        float v163_data = glb_m1[v34_lead];
        float v165_data = glb_m1[v164_a];
        float v167_data = glb_m1[v166_a];
        float v169_data = glb_m1[v168_a];
        float v171_data = glb_m1[v170_a];
        float v173_data = glb_m1[v172_a];
        float v175_data = glb_m1[v174_a];
        float v177_data = glb_m1[v176_a];
        float v179_data = glb_m1[v178_a];
        float v181_data = glb_m1[v180_a];
        float v183_data = glb_m1[v182_a];
        float v185_data = glb_m1[v184_a];
        float v187_data = glb_m1[v186_a];
        float v189_data = glb_m1[v188_a];
        float v191_data = glb_m1[v190_a];
        float v193_data = glb_m1[v192_a];
        float v196_acc{};
        float v197_data = r0[16];
        float v198_data = r0[17];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v405_i0 = 0; v405_i0 < 1; ++v405_i0) {
          int32_t v407_lead = v34_lead + (v405_i0 * 16);
          #pragma unroll
          for (int32_t v408_i1 = 0; v408_i1 < 9; ++v408_i1) {
            float v411_data = __builtin_nontemporal_load(&wrap_glb_m2[(v407_lead + (v408_i1 * 17))]);
            r0[(v405_i0 + (v408_i1 * 2))] = v411_data;
          }
        }
        #pragma unroll
        for (int32_t v414_i1 = 0; v414_i1 < 9; ++v414_i1) {
          float v417_data = v49_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v46_lead + (v414_i1 * 17))])) : (0.0f);
          r0[(1 + (v414_i1 * 2))] = v417_data;
        }
        tensorforge::fmacdpp16<0>(v196_acc, v197_data, v163_data);
        tensorforge::fmacdpp16<1>(v196_acc, v197_data, v165_data);
        tensorforge::fmacdpp16<2>(v196_acc, v197_data, v167_data);
        tensorforge::fmacdpp16<3>(v196_acc, v197_data, v169_data);
        tensorforge::fmacdpp16<4>(v196_acc, v197_data, v171_data);
        tensorforge::fmacdpp16<5>(v196_acc, v197_data, v173_data);
        tensorforge::fmacdpp16<6>(v196_acc, v197_data, v175_data);
        tensorforge::fmacdpp16<7>(v196_acc, v197_data, v177_data);
        tensorforge::fmacdpp16<8>(v196_acc, v197_data, v179_data);
        tensorforge::fmacdpp16<9>(v196_acc, v197_data, v181_data);
        tensorforge::fmacdpp16<10>(v196_acc, v197_data, v183_data);
        tensorforge::fmacdpp16<11>(v196_acc, v197_data, v185_data);
        tensorforge::fmacdpp16<12>(v196_acc, v197_data, v187_data);
        tensorforge::fmacdpp16<13>(v196_acc, v197_data, v189_data);
        tensorforge::fmacdpp16<14>(v196_acc, v197_data, v191_data);
        tensorforge::fmacdpp16<15>(v196_acc, v197_data, v193_data);
        tensorforge::fmacdpp16<0>(v196_acc, v198_data, v101_data);
        r1[8] = v196_acc;
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(0, 17)]
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
        float v243_data = glb_m3[v75_a];
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v243_data, v236_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v243_data, v244_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v243_data, v245_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v243_data, v246_acc, 2, 0, 7);
        float v249_data = glb_m3[v82_a];
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v249_data, v247_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v249_data, v250_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v249_data, v251_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v249_data, v252_acc, 2, 1, 7);
        float v255_data = glb_m3[v88_a];
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v255_data, v253_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v255_data, v256_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v255_data, v257_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v255_data, v258_acc, 2, 2, 7);
        float v261_data = glb_m3[v94_a];
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v261_data, v259_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v261_data, v262_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v261_data, v263_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v261_data, v264_acc, 2, 3, 7);
        float v267_data = glb_m3[v100_a];
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v267_data, v265_acc, 2, 0, 0);
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
        float v296_data = glb_m3[v75_a];
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v296_data, v289_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v296_data, v297_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v296_data, v298_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v296_data, v299_acc, 2, 0, 7);
        float v302_data = glb_m3[v82_a];
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v302_data, v300_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v302_data, v303_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v302_data, v304_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v302_data, v305_acc, 2, 1, 7);
        float v308_data = glb_m3[v88_a];
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v308_data, v306_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v308_data, v309_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v308_data, v310_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v308_data, v311_acc, 2, 2, 7);
        float v314_data = glb_m3[v94_a];
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v314_data, v312_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v314_data, v315_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v314_data, v316_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v314_data, v317_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v267_data, v318_acc, 2, 0, 0);
        ir3[4] = (v321_acc[0]);
        ir3[5] = (v321_acc[1]);
        ir3[6] = (v321_acc[2]);
        ir3[7] = (v321_acc[3]);
        float v329_data = glb_m3[v34_lead];
        float v331_data = glb_m3[v164_a];
        float v333_data = glb_m3[v166_a];
        float v335_data = glb_m3[v168_a];
        float v337_data = glb_m3[v170_a];
        float v339_data = glb_m3[v172_a];
        float v341_data = glb_m3[v174_a];
        float v343_data = glb_m3[v176_a];
        float v345_data = glb_m3[v178_a];
        float v347_data = glb_m3[v180_a];
        float v349_data = glb_m3[v182_a];
        float v351_data = glb_m3[v184_a];
        float v353_data = glb_m3[v186_a];
        float v355_data = glb_m3[v188_a];
        float v357_data = glb_m3[v190_a];
        float v359_data = glb_m3[v192_a];
        float v362_acc{};
        float v363_data = r2[16];
        float v364_data = r2[17];
        tensorforge::fmacdpp16<0>(v362_acc, v363_data, v329_data);
        tensorforge::fmacdpp16<1>(v362_acc, v363_data, v331_data);
        tensorforge::fmacdpp16<2>(v362_acc, v363_data, v333_data);
        tensorforge::fmacdpp16<3>(v362_acc, v363_data, v335_data);
        tensorforge::fmacdpp16<4>(v362_acc, v363_data, v337_data);
        tensorforge::fmacdpp16<5>(v362_acc, v363_data, v339_data);
        tensorforge::fmacdpp16<6>(v362_acc, v363_data, v341_data);
        tensorforge::fmacdpp16<7>(v362_acc, v363_data, v343_data);
        tensorforge::fmacdpp16<8>(v362_acc, v363_data, v345_data);
        tensorforge::fmacdpp16<9>(v362_acc, v363_data, v347_data);
        tensorforge::fmacdpp16<10>(v362_acc, v363_data, v349_data);
        tensorforge::fmacdpp16<11>(v362_acc, v363_data, v351_data);
        tensorforge::fmacdpp16<12>(v362_acc, v363_data, v353_data);
        tensorforge::fmacdpp16<13>(v362_acc, v363_data, v355_data);
        tensorforge::fmacdpp16<14>(v362_acc, v363_data, v357_data);
        tensorforge::fmacdpp16<15>(v362_acc, v363_data, v359_data);
        tensorforge::fmacdpp16<0>(v362_acc, v364_data, v267_data);
        ir3[8] = v362_acc;
        // r3 = ir3 + r1
        if (v365_g) {
          #pragma unroll
          for (int32_t v366_n1 = 0; v366_n1 < 9; ++v366_n1) {
            float v368_data = ir3[v366_n1];
            float v369_data = r1[v366_n1];
            r3[v366_n1] = (v369_data + v368_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v365_g) {
          #pragma unroll
          for (int32_t v372_i1 = 0; v372_i1 < 9; ++v372_i1) {
            float v374_data = r3[v372_i1];
            if (batchIdActive0) {
              glb_m0[(v34_lead + (v372_i1 * 10))] = v374_data;
            }
          }
        }
      }
    }
  }
}

