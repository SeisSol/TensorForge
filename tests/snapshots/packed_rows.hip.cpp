// === base name ===
kernel_256f9d17eecec653

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_256f9d17eecec653 = {{32, 8, 1}, 32, 40, 1, 8, 1280, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_256f9d17eecec653(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_256f9d17eecec653(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_256f9d17eecec653(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_256f9d17eecec653, block.x * block.y * block.z, 320 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (320 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_256f9d17eecec653, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (320 * sizeof(float)));
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
  config.sharedMemBytes = 320 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_256f9d17eecec653(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_256f9d17eecec653(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_256f9d17eecec653), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_256f9d17eecec653, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_256f9d17eecec653(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: pack_operands=rows
    // launch: 32 lanes (40 active) x 8 per block = block 32x8x1, 1280 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":320}],"shared_bytes":1280,"shared_elements":320,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 320];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      if ((threadIdx.x + (threadIdx.y * 32)) < 64) {
        float v15_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v15_ld;
      }
      __syncthreads();
      size_t v17_batchIdLane0 = threadIdx.y % 2;
      size_t v20_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v36_lead = threadIdx.x % 32;
      bool v42_g = v36_lead < 8;
      int32_t v59_a = v36_lead + ((threadIdx.y % 2) * 40);
      int32_t v64_a = v59_a + 80;
      int32_t v68_a = v59_a + 160;
      int32_t v72_a = v59_a + 240;
      float r0[6]{};
      size_t v140_row = v20_batchIdGroup0First + v17_batchIdLane0;
      const bool peel_batchIdActive0 = v140_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v140_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v140_row : v20_batchIdGroup0First) * 48 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v148_i1 = 0; v148_i1 < 6; ++v148_i1) {
        float v151_data = v42_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v36_lead + (v148_i1 * 8))])) : (0.0f);
        r0[v148_i1] = v151_data;
      }
      for (size_t v21_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v21_batchIdGroup0 < numElements0; v21_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v137_groupNextAhead = v21_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v139_groupNext = (v137_groupNextAhead < numElements0) ? v137_groupNextAhead : v21_batchIdGroup0;
        size_t v144_row = v139_groupNext + v17_batchIdLane0;
        const bool wrap_batchIdActive0 = v144_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v144_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v144_row : v139_groupNext) * 48 + 0 + m2_extraOffset];
        size_t v22_row = v21_batchIdGroup0 + v17_batchIdLane0;
        const bool batchIdActive0 = v22_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v22_row]));
        size_t v24_batchId0 = batchIdActive0 ? v22_row : v21_batchIdGroup0;
        size_t v25_ahead1 = v24_batchId0 + (gridDim.x * blockDim.y);
        size_t v27_batchId1 = (v25_ahead1 < numElements0) ? v25_ahead1 : v24_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v24_batchId0 * 240 + 0 + m0_extraOffset];
        float r1[12]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 40), (0, 6)] [(0, 8)]
        float v46_data = r0[0];
        float v47_data = r0[1];
        float v48_data = r0[2];
        float v49_data = r0[3];
        float v50_tp{};
        float v51_tp{};
        float v52_tp{};
        float v53_tp{};
        tensorforge::transpose4x4b32(v50_tp, v51_tp, v52_tp, v53_tp, v46_data, v47_data, v48_data, v49_data);
        tensorforge::VectorT<float, 4> v54_acc{};
        float v61_data = glb_m1[v59_a];
        tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v61_data, v54_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v61_data, v62_acc, 3, 0, 2);
        float v65_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v63_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v65_data, v66_acc, 3, 0, 2);
        float v69_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v69_data, v67_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v69_data, v70_acc, 3, 1, 2);
        float v73_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v71_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v73_data, v74_acc, 3, 1, 2);
        r1[0] = (v75_acc[0]);
        r1[2] = (v75_acc[1]);
        r1[4] = (v75_acc[2]);
        r1[6] = (v75_acc[3]);
        tensorforge::VectorT<float, 4> v80_acc{};
        r1[1] = (v80_acc[0]);
        r1[3] = (v80_acc[1]);
        r1[5] = (v80_acc[2]);
        r1[7] = (v80_acc[3]);
        float v85_data = r0[4];
        float v86_data = r0[5];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v152_i1 = 0; v152_i1 < 6; ++v152_i1) {
          float v155_data = v42_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v36_lead + (v152_i1 * 8))])) : (0.0f);
          r0[v152_i1] = v155_data;
        }
        float v88_tp{};
        float v89_tp{};
        float v90_tp{};
        float v91_tp{};
        tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v85_data, v86_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v92_acc{};
        float v99_data = glb_m1[v59_a];
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v99_data, v92_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v99_data, v100_acc, 3, 0, 2);
        float v103_data = glb_m1[v64_a];
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v103_data, v101_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v103_data, v104_acc, 3, 0, 2);
        float v107_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v107_data, v105_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v107_data, v108_acc, 3, 1, 2);
        float v111_data = glb_m1[v72_a];
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v111_data, v109_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v111_data, v112_acc, 3, 1, 2);
        r1[8] = (v113_acc[0]);
        r1[10] = (v113_acc[1]);
        tensorforge::VectorT<float, 4> v116_acc{};
        r1[9] = (v116_acc[0]);
        r1[11] = (v116_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v119_i0 = 0; v119_i0 < 1; ++v119_i0) {
          #pragma unroll
          for (int32_t v120_i1 = 0; v120_i1 < 6; ++v120_i1) {
            float v123_data = r1[(v119_i0 + (v120_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v36_lead + (v119_i0 * 32)) + (v120_i1 * 40))] = v123_data;
            }
          }
        }
        if (v42_g) {
          #pragma unroll
          for (int32_t v129_i1 = 0; v129_i1 < 6; ++v129_i1) {
            float v132_data = r1[(1 + (v129_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v36_lead + 32_i32) + (v129_i1 * 40))] = v132_data;
            }
          }
        }
      }
    }
  }
}

