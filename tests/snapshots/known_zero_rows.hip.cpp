// === base name ===
kernel_1af781d84d542919

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1af781d84d542919 = {{32, 8, 1}, 32, 40, 1, 8, 1280, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1af781d84d542919(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1af781d84d542919(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1af781d84d542919(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1af781d84d542919, block.x * block.y * block.z, 320 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (320 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_1af781d84d542919, block.x * block.y * block.z, 0));
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
void launcher_kernel_1af781d84d542919(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1af781d84d542919(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_1af781d84d542919), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_1af781d84d542919, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_1af781d84d542919(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (40 active) x 8 per block = block 32x8x1, 1280 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":320}],"shared_bytes":1280,"shared_elements":320,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 320];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 64) {
        float v6_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v6_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v8_batchIdLane0 = threadIdx.y % 2;
      int32_t v25_lead = threadIdx.x % 32;
      bool v26_g = v25_lead < 8;
      int32_t v45_r = (threadIdx.y % 2) * 40;
      int32_t v48_a = v25_lead + v45_r;
      int32_t v52_a = v48_a + 80;
      int32_t v55_a = v48_a + 160;
      int32_t v58_a = v48_a + 240;
      int32_t v70_lead = v25_lead + 32_i32;
      int32_t v71_a = v70_lead + v45_r;
      int32_t v76_a = v71_a + 80;
      int32_t v80_a = v71_a + 160;
      int32_t v84_a = v71_a + 240;
      for (size_t v9_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v9_batchIdGroup0 < numElements0; v9_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v10_row = v9_batchIdGroup0 + v8_batchIdLane0;
        const bool batchIdActive0 = v10_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v10_row]));
        size_t v12_batchId0 = batchIdActive0 ? v10_row : v9_batchIdGroup0;
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v16_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v12_batchId0 * 240 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v12_batchId0 * 48 + 0 + m2_extraOffset];
        float r0[6]{};
        // r0 = load{g>r}(glb_m2);
        if (v26_g) {
          #pragma unroll
          for (int32_t v27_i1 = 0; v27_i1 < 6; ++v27_i1) {
            float v32_data = __builtin_nontemporal_load(&glb_m2[(v25_lead + (v27_i1 * 8))]);
            r0[v27_i1] = v32_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[12]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 40), (0, 6)] [(0, 8)]
        float v35_data = r0[0];
        float v36_data = r0[1];
        float v37_data = r0[2];
        float v38_data = r0[3];
        float v39_tp{};
        float v40_tp{};
        float v41_tp{};
        float v42_tp{};
        tensorforge::transpose4x4b32(v39_tp, v40_tp, v41_tp, v42_tp, v35_data, v36_data, v37_data, v38_data);
        tensorforge::VectorT<float, 4> v43_acc{};
        float v50_data = glb_m1[v48_a];
        tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v50_data, v43_acc, 3, 0, 2);
        float v53_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v53_data, v51_acc, 3, 0, 2);
        float v56_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v56_data, v54_acc, 3, 1, 1);
        float v59_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v59_data, v57_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v59_data, v60_acc, 3, 1, 2);
        r1[0] = (v61_acc[0]);
        r1[2] = (v61_acc[1]);
        r1[4] = (v61_acc[2]);
        r1[6] = (v61_acc[3]);
        tensorforge::VectorT<float, 4> v66_acc{};
        float v73_data = glb_m1[v71_a];
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v73_data, v66_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v73_data, v74_acc, 3, 0, 2);
        float v77_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v77_data, v75_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v77_data, v78_acc, 3, 0, 2);
        float v81_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v81_data, v79_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v81_data, v82_acc, 3, 1, 2);
        float v85_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v41_tp, v85_data, v83_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v85_data, v86_acc, 3, 1, 2);
        r1[1] = (v87_acc[0]);
        r1[3] = (v87_acc[1]);
        r1[5] = (v87_acc[2]);
        r1[7] = (v87_acc[3]);
        float v92_data = r0[4];
        float v93_data = r0[5];
        float v95_tp{};
        float v96_tp{};
        float v97_tp{};
        float v98_tp{};
        tensorforge::transpose4x4b32(v95_tp, v96_tp, v97_tp, v98_tp, v92_data, v93_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v99_acc{};
        float v106_data = glb_m1[v48_a];
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v106_data, v99_acc, 3, 0, 2);
        float v109_data = glb_m1[v52_a];
        tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v109_data, v107_acc, 3, 0, 2);
        float v112_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v112_data, v110_acc, 3, 1, 1);
        float v115_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v115_data, v113_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v115_data, v116_acc, 3, 1, 2);
        r1[8] = (v117_acc[0]);
        r1[10] = (v117_acc[1]);
        tensorforge::VectorT<float, 4> v120_acc{};
        float v127_data = glb_m1[v71_a];
        tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v127_data, v120_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v127_data, v128_acc, 3, 0, 2);
        float v131_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v131_data, v129_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v131_data, v132_acc, 3, 0, 2);
        float v135_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v135_data, v133_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v135_data, v136_acc, 3, 1, 2);
        float v139_data = glb_m1[v84_a];
        tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v139_data, v137_acc, 3, 1, 1);
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v139_data, v140_acc, 3, 1, 2);
        r1[9] = (v141_acc[0]);
        r1[11] = (v141_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v144_i0 = 0; v144_i0 < 1; ++v144_i0) {
          #pragma unroll
          for (int32_t v145_i1 = 0; v145_i1 < 6; ++v145_i1) {
            float v148_data = r1[(v144_i0 + (v145_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v25_lead + (v144_i0 * 32)) + (v145_i1 * 40))] = v148_data;
            }
          }
        }
        if (v26_g) {
          #pragma unroll
          for (int32_t v153_i1 = 0; v153_i1 < 6; ++v153_i1) {
            float v156_data = r1[(1 + (v153_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[(v70_lead + (v153_i1 * 40))] = v156_data;
            }
          }
        }
      }
    }
  }
}

