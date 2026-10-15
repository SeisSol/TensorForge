// === base name ===
kernel_5d4acaa94ee73971

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5d4acaa94ee73971 = {{32, 8, 1}, 32, 24, 1, 8, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5d4acaa94ee73971(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5d4acaa94ee73971(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5d4acaa94ee73971(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5d4acaa94ee73971, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5d4acaa94ee73971, block.x * block.y * block.z, 0));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5d4acaa94ee73971(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5d4acaa94ee73971(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5d4acaa94ee73971), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5d4acaa94ee73971, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5d4acaa94ee73971(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: pack_operands=bits
    // launch: 32 lanes (24 active) x 8 per block = block 32x8x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 24×6(24×6) {0..24}×{0..6} strided
    //   m1 24×7(24×7) {0..24}×{0..7} none ordered
    //   m2 7×6(7×6) {0..7}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[24,7]],"name":"m1","ordered":true,"parts":1,"shape":[24,7],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[7,6]],"name":"m2","ordered":false,"parts":1,"shape":[7,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[24,7]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,7]},{"addressing":"strided","bbox":[[0,0],[7,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[7,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 2;
      size_t v14_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v30_lead = threadIdx.x % 32;
      bool v36_g = v30_lead < 7;
      int32_t v51_vlead = v30_lead * 4;
      int32_t v64_vlead = (v30_lead + 32_i32) * 4;
      bool v111_g = v30_lead < 24;
      float r0[6]{};
      size_t v122_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdActive0 = v122_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v122_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v122_row : v14_batchIdGroup0First) * 42 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v130_i1 = 0; v130_i1 < 6; ++v130_i1) {
        float v133_data = v36_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v30_lead + (v130_i1 * 7))])) : (0.0f);
        r0[v130_i1] = v133_data;
      }
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v119_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v121_groupNext = (v119_groupNextAhead < numElements0) ? v119_groupNextAhead : v15_batchIdGroup0;
        size_t v126_row = v121_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdActive0 = v126_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v126_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v126_row : v121_groupNext) * 42 + 0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 144 + 0 + m0_extraOffset];
        float r1[6]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 24), (0, 6)] [(0, 7)]
        float v40_data = r0[0];
        float v41_data = r0[1];
        float v42_data = r0[2];
        float v43_data = r0[3];
        float v44_tp{};
        float v45_tp{};
        float v46_tp{};
        float v47_tp{};
        tensorforge::transpose4x4b32(v44_tp, v45_tp, v46_tp, v47_tp, v40_data, v41_data, v42_data, v43_data);
        tensorforge::VectorT<float, 4> v48_acc{};
        tensorforge::VectorT<float, 4> v53_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v51_vlead];
        float v54_el = v53_data[0];
        tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v54_el, v48_acc, 3, 0, 0);
        float v56_el = v53_data[1];
        tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v56_el, v55_acc, 3, 0, 0);
        float v58_el = v53_data[2];
        tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v58_el, v57_acc, 3, 0, 0);
        float v60_el = v53_data[3];
        tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_el, v59_acc, 3, 0, 0);
        tensorforge::VectorT<float, 4> v66_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v64_vlead];
        tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, (v66_data[0]), v61_acc, 3, 1, 0);
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, (v66_data[1]), v68_acc, 3, 1, 0);
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, (v66_data[2]), v70_acc, 3, 1, 0);
        r1[0] = (v72_acc[0]);
        r1[1] = (v72_acc[1]);
        r1[2] = (v72_acc[2]);
        r1[3] = (v72_acc[3]);
        float v77_data = r0[4];
        float v78_data = r0[5];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v134_i1 = 0; v134_i1 < 6; ++v134_i1) {
          float v137_data = v36_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v30_lead + (v134_i1 * 7))])) : (0.0f);
          r0[v134_i1] = v137_data;
        }
        float v80_tp{};
        float v81_tp{};
        float v82_tp{};
        float v83_tp{};
        tensorforge::transpose4x4b32(v80_tp, v81_tp, v82_tp, v83_tp, v77_data, v78_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v84_acc{};
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v54_el, v84_acc, 3, 0, 0);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v56_el, v91_acc, 3, 0, 0);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_el, v93_acc, 3, 0, 0);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v60_el, v95_acc, 3, 0, 0);
        tensorforge::VectorT<float, 4> v102_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v64_vlead];
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, (v102_data[0]), v97_acc, 3, 1, 0);
        tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, (v102_data[1]), v104_acc, 3, 1, 0);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, (v102_data[2]), v106_acc, 3, 1, 0);
        r1[4] = (v108_acc[0]);
        r1[5] = (v108_acc[1]);
        // glb_m0 = store{r>g}(r1);
        if (v111_g) {
          #pragma unroll
          for (int32_t v112_i1 = 0; v112_i1 < 6; ++v112_i1) {
            float v114_data = r1[v112_i1];
            if (batchIdActive0) {
              glb_m0[(v30_lead + (v112_i1 * 24))] = v114_data;
            }
          }
        }
      }
    }
  }
}

