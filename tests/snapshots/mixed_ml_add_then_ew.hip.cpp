// === base name ===
kernel_555f38d1bdf4ea38

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_555f38d1bdf4ea38 = {{8, 32, 1}, 8, 8, 1, 32, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_555f38d1bdf4ea38(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_555f38d1bdf4ea38(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_555f38d1bdf4ea38(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 32, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_555f38d1bdf4ea38, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_555f38d1bdf4ea38, block.x * block.y * block.z, 0));
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
void launcher_kernel_555f38d1bdf4ea38(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_555f38d1bdf4ea38(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_555f38d1bdf4ea38), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_555f38d1bdf4ea38, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_555f38d1bdf4ea38(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 32 per block = block 8x32x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    //   m4 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t0[i,j] += m2[i,k] × m3[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[8 * threadIdx.y + 0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 8);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v27_lead + (v25_i1 * 8))]);
              r0[(v24_i0 + v25_i1)] = v30_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
            int32_t v36_lead = v23_lead + (v33_i0 * 8);
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m1[(v36_lead + (v34_i1 * 8))]);
              r1[(v33_i0 + v34_i1)] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[8]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v42_i0 = 0; v42_i0 < 1; ++v42_i0) {
            int32_t v45_lead = v23_lead + (v42_i0 * 8);
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 8; ++v43_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m2[(v45_lead + (v43_i1 * 8))]);
              r3[(v42_i0 + v43_i1)] = v48_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v51_data = r1[0];
          float v52_data = r1[1];
          float v53_data = r1[2];
          float v54_data = r1[3];
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
          tensorforge::VectorT<float, 4> v59_acc{};
          float v60_data = r0[0];
          float v61_data = r0[1];
          float v62_data = r0[2];
          float v63_data = r0[3];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v59_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 1, 0, 0);
          float v68_data = r0[4];
          float v69_data = r0[5];
          float v70_data = r0[6];
          float v71_data = r0[7];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 1, 1, 0);
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
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v93_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v94_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v95_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v96_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v101_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v102_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v103_acc, 1, 1, 0);
          r2[4] = (v104_acc[0]);
          r2[5] = (v104_acc[1]);
          r2[6] = (v104_acc[2]);
          r2[7] = (v104_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v110_i0 = 0; v110_i0 < 1; ++v110_i0) {
            int32_t v113_lead = v23_lead + (v110_i0 * 8);
            #pragma unroll
            for (int32_t v111_i1 = 0; v111_i1 < 8; ++v111_i1) {
              float v116_data = __builtin_nontemporal_load(&glb_m3[(v113_lead + (v111_i1 * 8))]);
              r4[(v110_i0 + v111_i1)] = v116_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          // wait(r4 = load{g>r}(glb_m3););
          float r5[8]{};
          // ir5 = +(r3 * r4)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir5[8]{};
          float v120_data = r4[0];
          float v121_data = r4[1];
          float v122_data = r4[2];
          float v123_data = r4[3];
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          tensorforge::transpose4x4b32(v124_tp, v125_tp, v126_tp, v127_tp, v120_data, v121_data, v122_data, v123_data);
          tensorforge::VectorT<float, 4> v128_acc{};
          float v129_data = r3[0];
          float v130_data = r3[1];
          float v131_data = r3[2];
          float v132_data = r3[3];
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v129_data, v128_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v130_data, v133_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v131_data, v134_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v132_data, v135_acc, 1, 0, 0);
          float v137_data = r3[4];
          float v138_data = r3[5];
          float v139_data = r3[6];
          float v140_data = r3[7];
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v137_data, v136_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v138_data, v141_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v139_data, v142_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v140_data, v143_acc, 1, 1, 0);
          ir5[0] = (v144_acc[0]);
          ir5[1] = (v144_acc[1]);
          ir5[2] = (v144_acc[2]);
          ir5[3] = (v144_acc[3]);
          float v149_data = r4[4];
          float v150_data = r4[5];
          float v151_data = r4[6];
          float v152_data = r4[7];
          float v153_tp{};
          float v154_tp{};
          float v155_tp{};
          float v156_tp{};
          tensorforge::transpose4x4b32(v153_tp, v154_tp, v155_tp, v156_tp, v149_data, v150_data, v151_data, v152_data);
          tensorforge::VectorT<float, 4> v157_acc{};
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v129_data, v157_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v130_data, v162_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v131_data, v163_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v132_data, v164_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v137_data, v165_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v138_data, v170_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v139_data, v171_acc, 1, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v140_data, v172_acc, 1, 1, 0);
          ir5[4] = (v173_acc[0]);
          ir5[5] = (v173_acc[1]);
          ir5[6] = (v173_acc[2]);
          ir5[7] = (v173_acc[3]);
          // r5 = ir5 + r2
          #pragma unroll
          for (int32_t v178_n0 = 0; v178_n0 < 1; ++v178_n0) {
            #pragma unroll
            for (int32_t v179_n1 = 0; v179_n1 < 8; ++v179_n1) {
              int32_t v180_a = v178_n0 + v179_n1;
              float v181_data = ir5[v180_a];
              float v182_data = r2[v180_a];
              r5[v180_a] = (v182_data + v181_data);
            }
          }
          // glb_m4 = abs(r5)
          #pragma unroll
          for (int32_t v184_k0 = 0; v184_k0 < 1; ++v184_k0) {
            int32_t v190_lead = v23_lead + (v184_k0 * 8);
            #pragma unroll
            for (int32_t v185_k1 = 0; v185_k1 < 8; ++v185_k1) {
              float v187_data = r5[(v184_k0 + v185_k1)];
              glb_m4[(v190_lead + (v185_k1 * 8))] = (fabsf(v187_data));
            }
          }
        }
      }
    }
  }
}

