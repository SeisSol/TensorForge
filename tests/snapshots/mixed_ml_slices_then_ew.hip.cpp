// === base name ===
kernel_ca213a58b6189366

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ca213a58b6189366 = {{8, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ca213a58b6189366(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ca213a58b6189366(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ca213a58b6189366(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 32, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ca213a58b6189366, block.x * block.y * block.z, 2304 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (2304 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_ca213a58b6189366, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (2304 * sizeof(float)));
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ca213a58b6189366(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ca213a58b6189366(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ca213a58b6189366), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_ca213a58b6189366, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ca213a58b6189366(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 32 per block = block 8x32x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×4(8×4) {0..8}×{0..4} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 64 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v11_batchId0 * 32 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 32 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
            int32_t v30_lead = v26_lead + (v27_i0 * 8);
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m0[(v30_lead + (v28_i1 * 8))]);
              r0[(v27_i0 + v28_i1)] = v33_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
            int32_t v39_lead = v26_lead + (v36_i0 * 8);
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 4; ++v37_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v39_lead + (v37_i1 * 8))]);
              r1[(v36_i0 + v37_i1)] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[4]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
            int32_t v48_lead = v26_lead + (v45_i0 * 8);
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 4; ++v46_i1) {
              float v51_data = __builtin_nontemporal_load(&glb_m2[(v48_lead + (v46_i1 * 8))]);
              r3[(v45_i0 + v46_i1)] = v51_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[4]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v54_data = r1[0];
          float v55_data = r1[1];
          float v56_data = r1[2];
          float v57_data = r1[3];
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          tensorforge::transpose4x4b32(v58_tp, v59_tp, v60_tp, v61_tp, v54_data, v55_data, v56_data, v57_data);
          tensorforge::VectorT<float, 4> v62_acc{};
          float v63_data = r0[0];
          float v64_data = r0[1];
          float v65_data = r0[2];
          float v66_data = r0[3];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v62_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v67_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 1, 0, 0);
          float v71_data = r0[4];
          float v72_data = r0[5];
          float v73_data = r0[6];
          float v74_data = r0[7];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v70_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 1, 1, 0);
          r2[0] = (v78_acc[0]);
          r2[1] = (v78_acc[1]);
          r2[2] = (v78_acc[2]);
          r2[3] = (v78_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v83_i0 = 0; v83_i0 < 1; ++v83_i0) {
            int32_t v88_lead = v26_lead + (v83_i0 * 8);
            #pragma unroll
            for (int32_t v84_i1 = 0; v84_i1 < 4; ++v84_i1) {
              float v86_data = r2[(v83_i0 + v84_i1)];
              int32_t v90_a = v88_lead + (v84_i1 * 8);
              s0[(v90_a ^ ((v90_a >> 5) & 31))] = v86_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[4]{};
          // r4 = +(r0 * r3) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v95_data = r3[0];
          float v96_data = r3[1];
          float v97_data = r3[2];
          float v98_data = r3[3];
          float v99_tp{};
          float v100_tp{};
          float v101_tp{};
          float v102_tp{};
          tensorforge::transpose4x4b32(v99_tp, v100_tp, v101_tp, v102_tp, v95_data, v96_data, v97_data, v98_data);
          tensorforge::VectorT<float, 4> v103_acc{};
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v63_data, v103_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v64_data, v108_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v65_data, v109_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v66_data, v110_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v71_data, v111_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v72_data, v116_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v73_data, v117_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v74_data, v118_acc, 1, 1, 0);
          r4[0] = (v119_acc[0]);
          r4[1] = (v119_acc[1]);
          r4[2] = (v119_acc[2]);
          r4[3] = (v119_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          #pragma unroll
          for (int32_t v124_i0 = 0; v124_i0 < 1; ++v124_i0) {
            int32_t v129_lead = v26_lead + (v124_i0 * 8);
            #pragma unroll
            for (int32_t v125_i1 = 0; v125_i1 < 4; ++v125_i1) {
              float v127_data = r4[(v124_i0 + v125_i1)];
              int32_t v132_a = v129_lead + ((v125_i1 + 4) * 8);
              s0[(v132_a ^ ((v132_a >> 5) & 31))] = v127_data;
            }
          }
          // glb_m3 = abs(s0)
          #pragma unroll
          for (int32_t v136_k0 = 0; v136_k0 < 1; ++v136_k0) {
            int32_t v139_lead = v26_lead + (v136_k0 * 8);
            #pragma unroll
            for (int32_t v137_k1 = 0; v137_k1 < 8; ++v137_k1) {
              int32_t v141_a = v139_lead + (v137_k1 * 8);
              float v145_data = s0[(v141_a ^ ((v141_a >> 5) & 31))];
              glb_m3[v141_a] = (fabsf(v145_data));
            }
          }
        }
      }
    }
  }
}

