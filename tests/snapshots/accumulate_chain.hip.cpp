// === base name ===
kernel_4a33efad4d1eacb2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4a33efad4d1eacb2 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4a33efad4d1eacb2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4a33efad4d1eacb2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4a33efad4d1eacb2(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4a33efad4d1eacb2, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4a33efad4d1eacb2, block.x * block.y * block.z, 0));
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
void launcher_kernel_4a33efad4d1eacb2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4a33efad4d1eacb2(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4a33efad4d1eacb2), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m7;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m8;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_4a33efad4d1eacb2, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, m7Arg, m7_extraOffset, m8Arg, m8_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4a33efad4d1eacb2(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m7, size_t m7_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m8, size_t m8_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 12×8(12×8) {0..12}×{0..8} strided
    //   m3 12×12(12×12) {0..12}×{0..12} strided
    //   m4 12×8(12×8) {0..12}×{0..8} strided
    //   m5 12×12(12×12) {0..12}×{0..12} strided
    //   m6 12×8(12×8) {0..12}×{0..8} strided
    //   m7 12×12(12×12) {0..12}×{0..12} strided
    //   m8 12×8(12×8) {0..12}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    //   m0[i,j] += m5[i,k] × m6[k,j]
    //   m0[i,j] += m7[i,k] × m8[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 96 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 96 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v10_batchId0 * 144 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v10_batchId0 * 96 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v10_batchId0 * 144 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v10_batchId0 * 96 + 0 + m6_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m7 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m7[v10_batchId0 * 144 + 0 + m7_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m8 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m8[v10_batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v30_lead = threadIdx.x % 16;
          bool v31_g = v30_lead < 12;
          if (v31_g) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v30_lead + (v32_i1 * 12))]);
              r0[v32_i1] = v37_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          if (v31_g) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 8; ++v40_i1) {
              float v45_data = __builtin_nontemporal_load(&glb_m2[(v30_lead + (v40_i1 * 12))]);
              r1[v40_i1] = v45_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v31_g) {
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 12; ++v48_i1) {
              float v53_data = __builtin_nontemporal_load(&glb_m3[(v30_lead + (v48_i1 * 12))]);
              r3[v48_i1] = v53_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v56_data = r1[0];
          float v57_data = r1[1];
          float v58_data = r1[2];
          float v59_data = r1[3];
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v56_data, v57_data, v58_data, v59_data);
          tensorforge::VectorT<float, 4> v64_acc{};
          float v65_data = r0[0];
          float v66_data = r0[1];
          float v67_data = r0[2];
          float v68_data = r0[3];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 2, 0, 0);
          float v73_data = r0[4];
          float v74_data = r0[5];
          float v75_data = r0[6];
          float v76_data = r0[7];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v72_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v79_acc, 2, 1, 0);
          float v81_data = r0[8];
          float v82_data = r0[9];
          float v83_data = r0[10];
          float v84_data = r0[11];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v80_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v87_acc, 2, 2, 0);
          r2[0] = (v88_acc[0]);
          r2[1] = (v88_acc[1]);
          r2[2] = (v88_acc[2]);
          r2[3] = (v88_acc[3]);
          float v93_data = r1[4];
          float v94_data = r1[5];
          float v95_data = r1[6];
          float v96_data = r1[7];
          float v97_tp{};
          float v98_tp{};
          float v99_tp{};
          float v100_tp{};
          tensorforge::transpose4x4b32(v97_tp, v98_tp, v99_tp, v100_tp, v93_data, v94_data, v95_data, v96_data);
          tensorforge::VectorT<float, 4> v101_acc{};
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v65_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v73_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v114_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v115_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v116_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v81_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v123_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v124_acc, 2, 2, 0);
          r2[4] = (v125_acc[0]);
          r2[5] = (v125_acc[1]);
          r2[6] = (v125_acc[2]);
          r2[7] = (v125_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v31_g) {
            #pragma unroll
            for (int32_t v131_i1 = 0; v131_i1 < 8; ++v131_i1) {
              float v136_data = __builtin_nontemporal_load(&glb_m4[(v30_lead + (v131_i1 * 12))]);
              r4[v131_i1] = v136_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v31_g) {
            #pragma unroll
            for (int32_t v139_i1 = 0; v139_i1 < 12; ++v139_i1) {
              float v144_data = __builtin_nontemporal_load(&glb_m5[(v30_lead + (v139_i1 * 12))]);
              r6[v139_i1] = v144_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // ir5 = +(r3 * r4)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v148_data = r4[0];
          float v149_data = r4[1];
          float v150_data = r4[2];
          float v151_data = r4[3];
          float v152_tp{};
          float v153_tp{};
          float v154_tp{};
          float v155_tp{};
          tensorforge::transpose4x4b32(v152_tp, v153_tp, v154_tp, v155_tp, v148_data, v149_data, v150_data, v151_data);
          tensorforge::VectorT<float, 4> v156_acc{};
          float v157_data = r3[0];
          float v158_data = r3[1];
          float v159_data = r3[2];
          float v160_data = r3[3];
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v157_data, v156_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v158_data, v161_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v159_data, v162_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v160_data, v163_acc, 2, 0, 0);
          float v165_data = r3[4];
          float v166_data = r3[5];
          float v167_data = r3[6];
          float v168_data = r3[7];
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v165_data, v164_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v166_data, v169_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v167_data, v170_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v168_data, v171_acc, 2, 1, 0);
          float v173_data = r3[8];
          float v174_data = r3[9];
          float v175_data = r3[10];
          float v176_data = r3[11];
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v173_data, v172_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v174_data, v177_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v175_data, v178_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v176_data, v179_acc, 2, 2, 0);
          ir5[0] = (v180_acc[0]);
          ir5[1] = (v180_acc[1]);
          ir5[2] = (v180_acc[2]);
          ir5[3] = (v180_acc[3]);
          float v185_data = r4[4];
          float v186_data = r4[5];
          float v187_data = r4[6];
          float v188_data = r4[7];
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          tensorforge::transpose4x4b32(v189_tp, v190_tp, v191_tp, v192_tp, v185_data, v186_data, v187_data, v188_data);
          tensorforge::VectorT<float, 4> v193_acc{};
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v157_data, v193_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v158_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v159_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v160_data, v200_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v165_data, v201_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v166_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v167_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v168_data, v208_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v173_data, v209_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v174_data, v214_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v175_data, v215_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v176_data, v216_acc, 2, 2, 0);
          ir5[4] = (v217_acc[0]);
          ir5[5] = (v217_acc[1]);
          ir5[6] = (v217_acc[2]);
          ir5[7] = (v217_acc[3]);
          // r5 = ir5 + r2
          if (v31_g) {
            #pragma unroll
            for (int32_t v222_n1 = 0; v222_n1 < 8; ++v222_n1) {
              float v224_data = ir5[v222_n1];
              float v225_data = r2[v222_n1];
              r5[v222_n1] = (v225_data + v224_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v31_g) {
            #pragma unroll
            for (int32_t v228_i1 = 0; v228_i1 < 8; ++v228_i1) {
              float v233_data = __builtin_nontemporal_load(&glb_m6[(v30_lead + (v228_i1 * 12))]);
              r7[v228_i1] = v233_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v31_g) {
            #pragma unroll
            for (int32_t v236_i1 = 0; v236_i1 < 12; ++v236_i1) {
              float v241_data = __builtin_nontemporal_load(&glb_m7[(v30_lead + (v236_i1 * 12))]);
              r9[v236_i1] = v241_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // ir8 = +(r6 * r7)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v245_data = r7[0];
          float v246_data = r7[1];
          float v247_data = r7[2];
          float v248_data = r7[3];
          float v249_tp{};
          float v250_tp{};
          float v251_tp{};
          float v252_tp{};
          tensorforge::transpose4x4b32(v249_tp, v250_tp, v251_tp, v252_tp, v245_data, v246_data, v247_data, v248_data);
          tensorforge::VectorT<float, 4> v253_acc{};
          float v254_data = r6[0];
          float v255_data = r6[1];
          float v256_data = r6[2];
          float v257_data = r6[3];
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v254_data, v253_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v255_data, v258_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v256_data, v259_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v257_data, v260_acc, 2, 0, 0);
          float v262_data = r6[4];
          float v263_data = r6[5];
          float v264_data = r6[6];
          float v265_data = r6[7];
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v262_data, v261_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v263_data, v266_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v264_data, v267_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v265_data, v268_acc, 2, 1, 0);
          float v270_data = r6[8];
          float v271_data = r6[9];
          float v272_data = r6[10];
          float v273_data = r6[11];
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v270_data, v269_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v271_data, v274_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v272_data, v275_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v273_data, v276_acc, 2, 2, 0);
          ir8[0] = (v277_acc[0]);
          ir8[1] = (v277_acc[1]);
          ir8[2] = (v277_acc[2]);
          ir8[3] = (v277_acc[3]);
          float v282_data = r7[4];
          float v283_data = r7[5];
          float v284_data = r7[6];
          float v285_data = r7[7];
          float v286_tp{};
          float v287_tp{};
          float v288_tp{};
          float v289_tp{};
          tensorforge::transpose4x4b32(v286_tp, v287_tp, v288_tp, v289_tp, v282_data, v283_data, v284_data, v285_data);
          tensorforge::VectorT<float, 4> v290_acc{};
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v254_data, v290_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v255_data, v295_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v256_data, v296_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v257_data, v297_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v262_data, v298_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v263_data, v303_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v264_data, v304_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v265_data, v305_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v270_data, v306_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v271_data, v311_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v272_data, v312_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v273_data, v313_acc, 2, 2, 0);
          ir8[4] = (v314_acc[0]);
          ir8[5] = (v314_acc[1]);
          ir8[6] = (v314_acc[2]);
          ir8[7] = (v314_acc[3]);
          // r8 = ir8 + r5
          if (v31_g) {
            #pragma unroll
            for (int32_t v319_n1 = 0; v319_n1 < 8; ++v319_n1) {
              float v321_data = ir8[v319_n1];
              float v322_data = r5[v319_n1];
              r8[v319_n1] = (v322_data + v321_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          if (v31_g) {
            #pragma unroll
            for (int32_t v325_i1 = 0; v325_i1 < 8; ++v325_i1) {
              float v330_data = __builtin_nontemporal_load(&glb_m8[(v30_lead + (v325_i1 * 12))]);
              r10[v325_i1] = v330_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // ir11 = +(r9 * r10)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v334_data = r10[0];
          float v335_data = r10[1];
          float v336_data = r10[2];
          float v337_data = r10[3];
          float v338_tp{};
          float v339_tp{};
          float v340_tp{};
          float v341_tp{};
          tensorforge::transpose4x4b32(v338_tp, v339_tp, v340_tp, v341_tp, v334_data, v335_data, v336_data, v337_data);
          tensorforge::VectorT<float, 4> v342_acc{};
          float v343_data = r9[0];
          float v344_data = r9[1];
          float v345_data = r9[2];
          float v346_data = r9[3];
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v343_data, v342_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v344_data, v347_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v345_data, v348_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v346_data, v349_acc, 2, 0, 0);
          float v351_data = r9[4];
          float v352_data = r9[5];
          float v353_data = r9[6];
          float v354_data = r9[7];
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v351_data, v350_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v352_data, v355_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v353_data, v356_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v354_data, v357_acc, 2, 1, 0);
          float v359_data = r9[8];
          float v360_data = r9[9];
          float v361_data = r9[10];
          float v362_data = r9[11];
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v359_data, v358_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v360_data, v363_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v361_data, v364_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v341_tp, v362_data, v365_acc, 2, 2, 0);
          ir11[0] = (v366_acc[0]);
          ir11[1] = (v366_acc[1]);
          ir11[2] = (v366_acc[2]);
          ir11[3] = (v366_acc[3]);
          float v371_data = r10[4];
          float v372_data = r10[5];
          float v373_data = r10[6];
          float v374_data = r10[7];
          float v375_tp{};
          float v376_tp{};
          float v377_tp{};
          float v378_tp{};
          tensorforge::transpose4x4b32(v375_tp, v376_tp, v377_tp, v378_tp, v371_data, v372_data, v373_data, v374_data);
          tensorforge::VectorT<float, 4> v379_acc{};
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v343_data, v379_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v344_data, v384_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v345_data, v385_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v346_data, v386_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v351_data, v387_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v352_data, v392_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v353_data, v393_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v354_data, v394_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v375_tp, v359_data, v395_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v360_data, v400_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v361_data, v401_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v362_data, v402_acc, 2, 2, 0);
          ir11[4] = (v403_acc[0]);
          ir11[5] = (v403_acc[1]);
          ir11[6] = (v403_acc[2]);
          ir11[7] = (v403_acc[3]);
          // r11 = ir11 + r8
          if (v31_g) {
            #pragma unroll
            for (int32_t v408_n1 = 0; v408_n1 < 8; ++v408_n1) {
              float v410_data = ir11[v408_n1];
              float v411_data = r8[v408_n1];
              r11[v408_n1] = (v411_data + v410_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v31_g) {
            #pragma unroll
            for (int32_t v413_i1 = 0; v413_i1 < 8; ++v413_i1) {
              float v415_data = r11[v413_i1];
              glb_m0[(v30_lead + (v413_i1 * 12))] = v415_data;
            }
          }
        }
      }
    }
  }
}

