// === base name ===
kernel_b47325a8724d42ab

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b47325a8724d42ab = {{16, 16, 1}, 16, 9, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b47325a8724d42ab(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b47325a8724d42ab(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b47325a8724d42ab(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b47325a8724d42ab, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b47325a8724d42ab, block.x * block.y * block.z, 0));
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
void launcher_kernel_b47325a8724d42ab(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b47325a8724d42ab(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b47325a8724d42ab), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b47325a8724d42ab, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b47325a8724d42ab(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (9 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 9×9(9×9) {0..9}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 9×9(9×9) {0..9}×{0..9} strided
    //   m3 ()  scalar
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j] × m3[]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 81 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v10_batchId0 * 81 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v24_lead = threadIdx.x % 16;
          bool v25_g = v24_lead < 9;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 9; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v26_i1 * 9))]);
              r0[v26_i1] = v31_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + (v34_i1 * 9))]);
              r1[v34_i1] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // ir2 = +(r0 * r1)
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir2[9]{};
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
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v67_acc, 2, 2, 0);
          ir2[0] = (v70_acc[0]);
          ir2[1] = (v70_acc[1]);
          ir2[2] = (v70_acc[2]);
          ir2[3] = (v70_acc[3]);
          float v75_data = r1[4];
          float v76_data = r1[5];
          float v77_data = r1[6];
          float v78_data = r1[7];
          float v79_tp{};
          float v80_tp{};
          float v81_tp{};
          float v82_tp{};
          tensorforge::transpose4x4b32(v79_tp, v80_tp, v81_tp, v82_tp, v75_data, v76_data, v77_data, v78_data);
          tensorforge::VectorT<float, 4> v83_acc{};
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v52_data, v83_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v53_data, v88_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v54_data, v89_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v55_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v60_data, v91_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v61_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v62_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v63_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v68_data, v99_acc, 2, 2, 0);
          ir2[4] = (v102_acc[0]);
          ir2[5] = (v102_acc[1]);
          ir2[6] = (v102_acc[2]);
          ir2[7] = (v102_acc[3]);
          float v116_acc{};
          float v117_data = r1[8];
          tensorforge::fmacdpp16<0>(v116_acc, v117_data, v52_data);
          tensorforge::fmacdpp16<1>(v116_acc, v117_data, v53_data);
          tensorforge::fmacdpp16<2>(v116_acc, v117_data, v54_data);
          tensorforge::fmacdpp16<3>(v116_acc, v117_data, v55_data);
          tensorforge::fmacdpp16<4>(v116_acc, v117_data, v60_data);
          tensorforge::fmacdpp16<5>(v116_acc, v117_data, v61_data);
          tensorforge::fmacdpp16<6>(v116_acc, v117_data, v62_data);
          tensorforge::fmacdpp16<7>(v116_acc, v117_data, v63_data);
          tensorforge::fmacdpp16<8>(v116_acc, v117_data, v68_data);
          ir2[8] = v116_acc;
          // r2 = ir2 * glb_m3
          if (v25_g) {
            #pragma unroll
            for (int32_t v119_n1 = 0; v119_n1 < 9; ++v119_n1) {
              float v121_data = ir2[v119_n1];
              r2[v119_n1] = (v121_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v123_i1 = 0; v123_i1 < 9; ++v123_i1) {
              float v125_data = r2[v123_i1];
              glb_m0[(v24_lead + (v123_i1 * 9))] = v125_data;
            }
          }
        }
      }
    }
  }
}

