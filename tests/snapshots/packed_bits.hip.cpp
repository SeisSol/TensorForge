// === base name ===
kernel_aab6a515c31a39ad

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_aab6a515c31a39ad = {{32, 8, 1}, 32, 24, 1, 8, 768, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_aab6a515c31a39ad(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_aab6a515c31a39ad(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_aab6a515c31a39ad(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_aab6a515c31a39ad, block.x * block.y * block.z, 192 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (192 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_aab6a515c31a39ad, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (192 * sizeof(float)));
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
  config.sharedMemBytes = 192 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_aab6a515c31a39ad(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_aab6a515c31a39ad(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_aab6a515c31a39ad), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_aab6a515c31a39ad, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_aab6a515c31a39ad(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: pack_operands=bits
    // launch: 32 lanes (24 active) x 8 per block = block 32x8x1, 768 B shared, occupancy grid
    // operands:
    //   m0 24×6(24×6) {0..24}×{0..6} strided
    //   m1 24×7(24×7) {0..24}×{0..7} none
    //   m2 7×6(7×6) {0..7}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":192}],"shared_bytes":768,"shared_elements":192,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[24,7]],"name":"m1","ordered":false,"parts":1,"shape":[24,7],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[7,6]],"name":"m2","ordered":false,"parts":1,"shape":[7,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[24,7]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,7]},{"addressing":"strided","bbox":[[0,0],[7,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[7,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 192];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + (threadIdx.y * 32)) < 168) {
        float v14_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v14_ld;
      }
      __syncthreads();
      size_t v16_batchIdLane0 = threadIdx.y % 2;
      size_t v19_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v35_lead = threadIdx.x % 32;
      bool v41_g = v35_lead < 7;
      int32_t v58_a = v35_lead + ((threadIdx.y % 2) * 24);
      int32_t v63_a = v58_a + 48;
      int32_t v67_a = v58_a + 96;
      int32_t v71_a = v35_lead + 144;
      bool v108_g = v35_lead < 24;
      float r0[6]{};
      size_t v119_row = v19_batchIdGroup0First + v16_batchIdLane0;
      const bool peel_batchIdActive0 = v119_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v119_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v119_row : v19_batchIdGroup0First) * 42 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v127_i1 = 0; v127_i1 < 6; ++v127_i1) {
        float v130_data = v41_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v35_lead + (v127_i1 * 7))])) : (0.0f);
        r0[v127_i1] = v130_data;
      }
      for (size_t v20_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v20_batchIdGroup0 < numElements0; v20_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v116_groupNextAhead = v20_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v118_groupNext = (v116_groupNextAhead < numElements0) ? v116_groupNextAhead : v20_batchIdGroup0;
        size_t v123_row = v118_groupNext + v16_batchIdLane0;
        const bool wrap_batchIdActive0 = v123_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v123_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v123_row : v118_groupNext) * 42 + 0 + m2_extraOffset];
        size_t v21_row = v20_batchIdGroup0 + v16_batchIdLane0;
        const bool batchIdActive0 = v21_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v21_row]));
        size_t v23_batchId0 = batchIdActive0 ? v21_row : v20_batchIdGroup0;
        size_t v24_ahead1 = v23_batchId0 + (gridDim.x * blockDim.y);
        size_t v26_batchId1 = (v24_ahead1 < numElements0) ? v24_ahead1 : v23_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v23_batchId0 * 144 + 0 + m0_extraOffset];
        float r1[6]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 24), (0, 6)] [(0, 7)]
        float v45_data = r0[0];
        float v46_data = r0[1];
        float v47_data = r0[2];
        float v48_data = r0[3];
        float v49_tp{};
        float v50_tp{};
        float v51_tp{};
        float v52_tp{};
        tensorforge::transpose4x4b32(v49_tp, v50_tp, v51_tp, v52_tp, v45_data, v46_data, v47_data, v48_data);
        tensorforge::VectorT<float, 4> v53_acc{};
        float v60_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v60_data, v53_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v60_data, v61_acc, 3, 0, 2);
        float v64_data = glb_m1[v63_a];
        tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v62_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v64_data, v65_acc, 3, 0, 2);
        float v68_data = glb_m1[v67_a];
        tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v68_data, v66_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v68_data, v69_acc, 3, 1, 2);
        float v72_data = glb_m1[v71_a];
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v70_acc, 3, 1, 0);
        r1[0] = (v73_acc[0]);
        r1[1] = (v73_acc[1]);
        r1[2] = (v73_acc[2]);
        r1[3] = (v73_acc[3]);
        float v78_data = r0[4];
        float v79_data = r0[5];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v131_i1 = 0; v131_i1 < 6; ++v131_i1) {
          float v134_data = v41_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v35_lead + (v131_i1 * 7))])) : (0.0f);
          r0[v131_i1] = v134_data;
        }
        float v81_tp{};
        float v82_tp{};
        float v83_tp{};
        float v84_tp{};
        tensorforge::transpose4x4b32(v81_tp, v82_tp, v83_tp, v84_tp, v78_data, v79_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v85_acc{};
        float v92_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v92_data, v85_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v92_data, v93_acc, 3, 0, 2);
        float v96_data = glb_m1[v63_a];
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v96_data, v94_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v96_data, v97_acc, 3, 0, 2);
        float v100_data = glb_m1[v67_a];
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v100_data, v98_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v100_data, v101_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v72_data, v102_acc, 3, 1, 0);
        r1[4] = (v105_acc[0]);
        r1[5] = (v105_acc[1]);
        // glb_m0 = store{r>g}(r1);
        if (v108_g) {
          #pragma unroll
          for (int32_t v109_i1 = 0; v109_i1 < 6; ++v109_i1) {
            float v111_data = r1[v109_i1];
            if (batchIdActive0) {
              glb_m0[(v35_lead + (v109_i1 * 24))] = v111_data;
            }
          }
        }
      }
    }
  }
}

