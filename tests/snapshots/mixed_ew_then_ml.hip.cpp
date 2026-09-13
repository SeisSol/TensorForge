// === base name ===
kernel_6c6d7cbd67b79045

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6c6d7cbd67b79045 = {{8, 32, 1}, 8, 8, 1, 32, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6c6d7cbd67b79045(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6c6d7cbd67b79045(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6c6d7cbd67b79045(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 32, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6c6d7cbd67b79045, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6c6d7cbd67b79045, block.x * block.y * block.z, 0));
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
  config.block[0] = 8;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_6c6d7cbd67b79045(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6c6d7cbd67b79045(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6c6d7cbd67b79045), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_6c6d7cbd67b79045, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6c6d7cbd67b79045(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 32 per block = block 8x32x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   TMP = abs(A)
    //   m1[i,j] = t0[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[8 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 64 + 0 + m2_extraOffset];
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v18_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v22_lead = v18_lead + (v19_i0 * 8);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m2[(v22_lead + (v20_i1 * 8))]);
              r1[(v19_i0 + v20_i1)] = v25_data;
            }
          }
          float r0[8]{};
          // r0 = abs(glb_m0)
          #pragma unroll
          for (int32_t v28_k0 = 0; v28_k0 < 1; ++v28_k0) {
            int32_t v31_lead = v18_lead + (v28_k0 * 8);
            #pragma unroll
            for (int32_t v29_k1 = 0; v29_k1 < 8; ++v29_k1) {
              float v34_data = glb_m0[(v31_lead + (v29_k1 * 8))];
              r0[(v28_k0 + v29_k1)] = (fabsf(v34_data));
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v38_data = r1[0];
          float v39_data = r1[1];
          float v40_data = r1[2];
          float v41_data = r1[3];
          float v42_tp{};
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          tensorforge::transpose4x4b32(v42_tp, v43_tp, v44_tp, v45_tp, v38_data, v39_data, v40_data, v41_data);
          tensorforge::VectorT<float, 4> v46_acc{};
          float v47_data = r0[0];
          float v48_data = r0[1];
          float v49_data = r0[2];
          float v50_data = r0[3];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v47_data, v46_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v51_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 1, 0, 0);
          float v55_data = r0[4];
          float v56_data = r0[5];
          float v57_data = r0[6];
          float v58_data = r0[7];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v54_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v59_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 1, 1, 0);
          r2[0] = (v62_acc[0]);
          r2[1] = (v62_acc[1]);
          r2[2] = (v62_acc[2]);
          r2[3] = (v62_acc[3]);
          float v67_data = r1[4];
          float v68_data = r1[5];
          float v69_data = r1[6];
          float v70_data = r1[7];
          float v71_tp{};
          float v72_tp{};
          float v73_tp{};
          float v74_tp{};
          tensorforge::transpose4x4b32(v71_tp, v72_tp, v73_tp, v74_tp, v67_data, v68_data, v69_data, v70_data);
          tensorforge::VectorT<float, 4> v75_acc{};
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v47_data, v75_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v48_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v49_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v50_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v55_data, v83_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v56_data, v88_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v57_data, v89_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v58_data, v90_acc, 1, 1, 0);
          r2[4] = (v91_acc[0]);
          r2[5] = (v91_acc[1]);
          r2[6] = (v91_acc[2]);
          r2[7] = (v91_acc[3]);
          // glb_m1 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v96_i0 = 0; v96_i0 < 1; ++v96_i0) {
            int32_t v101_lead = v18_lead + (v96_i0 * 8);
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 8; ++v97_i1) {
              float v99_data = r2[(v96_i0 + v97_i1)];
              glb_m1[(v101_lead + (v97_i1 * 8))] = v99_data;
            }
          }
        }
      }
    }
  }
}

