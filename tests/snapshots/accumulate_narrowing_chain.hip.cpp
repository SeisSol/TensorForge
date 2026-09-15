// === base name ===
kernel_0bef5e3ddb00d499

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0bef5e3ddb00d499 = {{32, 8, 1}, 32, 20, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0bef5e3ddb00d499(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0bef5e3ddb00d499(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0bef5e3ddb00d499(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0bef5e3ddb00d499, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0bef5e3ddb00d499, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0bef5e3ddb00d499(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0bef5e3ddb00d499(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0bef5e3ddb00d499), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0bef5e3ddb00d499, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0bef5e3ddb00d499(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 20×9(20×9) {0..20}×{0..9} strided
    //   m1 20×9(20×9) {0..20}×{0..9} strided
    //   m2 10×9(10×9) {0..10}×{0..9} strided
    //   m3 4×9(4×9) {0..4}×{0..9} strided
    //   m4 1×9(1×9) {0..1}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,j]
    //   m0[i,j] += m2[i,j]
    //   m0[i,j] += m3[i,j]
    //   m0[i,j] += m4[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"F0","bbox":[[0,0],[10,9]],"name":"m2","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[4,9]],"name":"m3","ordered":false,"parts":1,"shape":[4,9],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[1,9]],"name":"m4","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[4,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[1,9]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 180 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 180 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 90 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 36 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 9 + 0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 32;
          bool v18_g = v17_lead < 20;
          if (v18_g) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v19_i1 * 20))]);
              r0[v19_i1] = v24_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m2);
          if (v17_lead < 10) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v28_i1 * 10))]);
              r2[v28_i1] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 20), (0, 9)] []
          float v36_data = r0[0];
          float v37_data = r1[0];
          r1[0] = (v37_data + v36_data);
          float v39_data = r0[1];
          float v40_data = r1[1];
          r1[1] = (v40_data + v39_data);
          float v42_data = r0[2];
          float v43_data = r1[2];
          r1[2] = (v43_data + v42_data);
          float v45_data = r0[3];
          float v46_data = r1[3];
          r1[3] = (v46_data + v45_data);
          float v48_data = r0[4];
          float v49_data = r1[4];
          r1[4] = (v49_data + v48_data);
          float v51_data = r0[5];
          float v52_data = r1[5];
          r1[5] = (v52_data + v51_data);
          float v54_data = r0[6];
          float v55_data = r1[6];
          r1[6] = (v55_data + v54_data);
          float v57_data = r0[7];
          float v58_data = r1[7];
          r1[7] = (v58_data + v57_data);
          float v60_data = r0[8];
          float v61_data = r1[8];
          r1[8] = (v61_data + v60_data);
          float r4[9]{};
          // r4 = load{g>r}(glb_m3);
          if (v17_lead < 4) {
            #pragma unroll
            for (int32_t v65_i1 = 0; v65_i1 < 9; ++v65_i1) {
              float v70_data = __builtin_nontemporal_load(&glb_m3[(v17_lead + (v65_i1 * 4))]);
              r4[v65_i1] = v70_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 10), (0, 9)] []
          float ir3[9]{};
          float v74_data = r2[0];
          float v75_data = ir3[0];
          ir3[0] = (v75_data + v74_data);
          float v77_data = r2[1];
          float v78_data = ir3[1];
          ir3[1] = (v78_data + v77_data);
          float v80_data = r2[2];
          float v81_data = ir3[2];
          ir3[2] = (v81_data + v80_data);
          float v83_data = r2[3];
          float v84_data = ir3[3];
          ir3[3] = (v84_data + v83_data);
          float v86_data = r2[4];
          float v87_data = ir3[4];
          ir3[4] = (v87_data + v86_data);
          float v89_data = r2[5];
          float v90_data = ir3[5];
          ir3[5] = (v90_data + v89_data);
          float v92_data = r2[6];
          float v93_data = ir3[6];
          ir3[6] = (v93_data + v92_data);
          float v95_data = r2[7];
          float v96_data = ir3[7];
          ir3[7] = (v96_data + v95_data);
          float v98_data = r2[8];
          float v99_data = ir3[8];
          ir3[8] = (v99_data + v98_data);
          // r3 = ir3 + r1
          if (v18_g) {
            #pragma unroll
            for (int32_t v101_n1 = 0; v101_n1 < 9; ++v101_n1) {
              float v103_data = ir3[v101_n1];
              float v104_data = r1[v101_n1];
              r3[v101_n1] = (v104_data + v103_data);
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v17_lead < 1) {
            #pragma unroll
            for (int32_t v108_i1 = 0; v108_i1 < 9; ++v108_i1) {
              float v112_data = __builtin_nontemporal_load(&glb_m4[(v17_lead + v108_i1)]);
              r6[v108_i1] = v112_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[9]{};
          // ir5 = +(r4)
          // [(0, 4), (0, 9)] []
          float ir5[9]{};
          float v116_data = r4[0];
          float v117_data = ir5[0];
          ir5[0] = (v117_data + v116_data);
          float v119_data = r4[1];
          float v120_data = ir5[1];
          ir5[1] = (v120_data + v119_data);
          float v122_data = r4[2];
          float v123_data = ir5[2];
          ir5[2] = (v123_data + v122_data);
          float v125_data = r4[3];
          float v126_data = ir5[3];
          ir5[3] = (v126_data + v125_data);
          float v128_data = r4[4];
          float v129_data = ir5[4];
          ir5[4] = (v129_data + v128_data);
          float v131_data = r4[5];
          float v132_data = ir5[5];
          ir5[5] = (v132_data + v131_data);
          float v134_data = r4[6];
          float v135_data = ir5[6];
          ir5[6] = (v135_data + v134_data);
          float v137_data = r4[7];
          float v138_data = ir5[7];
          ir5[7] = (v138_data + v137_data);
          float v140_data = r4[8];
          float v141_data = ir5[8];
          ir5[8] = (v141_data + v140_data);
          // r5 = ir5 + r3
          if (v18_g) {
            #pragma unroll
            for (int32_t v143_n1 = 0; v143_n1 < 9; ++v143_n1) {
              float v145_data = ir5[v143_n1];
              float v146_data = r3[v143_n1];
              r5[v143_n1] = (v146_data + v145_data);
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // ir7 = +(r6)
          // [(0, 1), (0, 9)] []
          float ir7[9]{};
          float v150_data = r6[0];
          float v151_data = ir7[0];
          ir7[0] = (v151_data + v150_data);
          float v153_data = r6[1];
          float v154_data = ir7[1];
          ir7[1] = (v154_data + v153_data);
          float v156_data = r6[2];
          float v157_data = ir7[2];
          ir7[2] = (v157_data + v156_data);
          float v159_data = r6[3];
          float v160_data = ir7[3];
          ir7[3] = (v160_data + v159_data);
          float v162_data = r6[4];
          float v163_data = ir7[4];
          ir7[4] = (v163_data + v162_data);
          float v165_data = r6[5];
          float v166_data = ir7[5];
          ir7[5] = (v166_data + v165_data);
          float v168_data = r6[6];
          float v169_data = ir7[6];
          ir7[6] = (v169_data + v168_data);
          float v171_data = r6[7];
          float v172_data = ir7[7];
          ir7[7] = (v172_data + v171_data);
          float v174_data = r6[8];
          float v175_data = ir7[8];
          ir7[8] = (v175_data + v174_data);
          // r7 = ir7 + r5
          if (v18_g) {
            #pragma unroll
            for (int32_t v177_n1 = 0; v177_n1 < 9; ++v177_n1) {
              float v179_data = ir7[v177_n1];
              float v180_data = r5[v177_n1];
              r7[v177_n1] = (v180_data + v179_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v18_g) {
            #pragma unroll
            for (int32_t v182_i1 = 0; v182_i1 < 9; ++v182_i1) {
              float v184_data = r7[v182_i1];
              glb_m0[(v17_lead + (v182_i1 * 20))] = v184_data;
            }
          }
        }
      }
    }
  }
}

