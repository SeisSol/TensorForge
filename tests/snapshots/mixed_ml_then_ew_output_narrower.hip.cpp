// === base name ===
kernel_e34fd4d36bf1425f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e34fd4d36bf1425f = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e34fd4d36bf1425f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e34fd4d36bf1425f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e34fd4d36bf1425f(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e34fd4d36bf1425f, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e34fd4d36bf1425f, block.x * block.y * block.z, 0));
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
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e34fd4d36bf1425f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e34fd4d36bf1425f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e34fd4d36bf1425f), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_e34fd4d36bf1425f, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e34fd4d36bf1425f(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(12×12) {0..12}×{0..12} strided
    //   m3 32×32(4×12) {4..8}×{0..12} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   D = abs(N)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 144 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 144 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v10_batchId0 * 48 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 16;
          bool v26_g = v25_lead < 12;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m1[(v25_lead + (v27_i1 * 12))]);
              r0[v27_i1] = v32_data;
            }
          }
          float r1[12]{};
          // r1 = load{g>r}(glb_m2);
          if (v26_g) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m2[(v25_lead + (v35_i1 * 12))]);
              r1[v35_i1] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float v43_data = r1[0];
          float v44_data = r1[1];
          float v45_data = r1[2];
          float v46_data = r1[3];
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          tensorforge::transpose4x4b32(v47_tp, v48_tp, v49_tp, v50_tp, v43_data, v44_data, v45_data, v46_data);
          tensorforge::VectorT<float, 4> v51_acc{};
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v51_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 2, 0, 0);
          float v60_data = r0[4];
          float v61_data = r0[5];
          float v62_data = r0[6];
          float v63_data = r0[7];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v59_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 1, 0);
          float v68_data = r0[8];
          float v69_data = r0[9];
          float v70_data = r0[10];
          float v71_data = r0[11];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v67_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 2, 0);
          r2[0] = (v75_acc[0]);
          r2[1] = (v75_acc[1]);
          r2[2] = (v75_acc[2]);
          r2[3] = (v75_acc[3]);
          float v80_data = r1[4];
          float v81_data = r1[5];
          float v82_data = r1[6];
          float v83_data = r1[7];
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          float v87_tp{};
          tensorforge::transpose4x4b32(v84_tp, v85_tp, v86_tp, v87_tp, v80_data, v81_data, v82_data, v83_data);
          tensorforge::VectorT<float, 4> v88_acc{};
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v88_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v94_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v95_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v102_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v103_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v104_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v109_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v110_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v111_acc, 2, 2, 0);
          r2[4] = (v112_acc[0]);
          r2[5] = (v112_acc[1]);
          r2[6] = (v112_acc[2]);
          r2[7] = (v112_acc[3]);
          float v117_data = r1[8];
          float v118_data = r1[9];
          float v119_data = r1[10];
          float v120_data = r1[11];
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          tensorforge::transpose4x4b32(v121_tp, v122_tp, v123_tp, v124_tp, v117_data, v118_data, v119_data, v120_data);
          tensorforge::VectorT<float, 4> v125_acc{};
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v52_data, v125_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v53_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v54_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v55_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v60_data, v133_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v61_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v62_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v63_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v68_data, v141_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v69_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v70_data, v147_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v71_data, v148_acc, 2, 2, 0);
          r2[8] = (v149_acc[0]);
          r2[9] = (v149_acc[1]);
          r2[10] = (v149_acc[2]);
          r2[11] = (v149_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v26_g) {
            #pragma unroll
            for (int32_t v154_i1 = 0; v154_i1 < 12; ++v154_i1) {
              float v156_data = r2[v154_i1];
              glb_m0[(v25_lead + (v154_i1 * 12))] = v156_data;
            }
          }
          float r3[12]{};
          // r3 = abs(glb_m3)
          bool v162_g = v25_lead < 4;
          if (v162_g) {
            int32_t v167_a = (v25_lead + 4) - 4;
            #pragma unroll
            for (int32_t v163_k1 = 0; v163_k1 < 12; ++v163_k1) {
              float v170_data = glb_m3[(v167_a + (v163_k1 * 4))];
              r3[v163_k1] = (fabsf(v170_data));
            }
          }
          // glb_m0 = store{r>g}(r3);
          if (v162_g) {
            int32_t v179_off = v25_lead + 4;
            #pragma unroll
            for (int32_t v174_i1 = 0; v174_i1 < 12; ++v174_i1) {
              float v176_data = r3[v174_i1];
              glb_m0[(v179_off + (v174_i1 * 12))] = v176_data;
            }
          }
          if (v25_lead >= 12) {
            int32_t v187_off = (v25_lead + -16_i32) + 4;
            #pragma unroll
            for (int32_t v183_z1 = 0; v183_z1 < 12; ++v183_z1) {
              glb_m0[(v187_off + (v183_z1 * 12))] = 0.0f;
            }
          }
          if ((v25_lead >= 4) && (v25_lead < 8)) {
            int32_t v197_off = v25_lead + 4;
            #pragma unroll
            for (int32_t v193_z1 = 0; v193_z1 < 12; ++v193_z1) {
              glb_m0[(v197_off + (v193_z1 * 12))] = 0.0f;
            }
          }
        }
      }
    }
  }
}

