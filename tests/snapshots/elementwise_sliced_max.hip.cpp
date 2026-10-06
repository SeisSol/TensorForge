// === base name ===
kernel_5f7ddce9decd53d0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5f7ddce9decd53d0 = {{32, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5f7ddce9decd53d0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5f7ddce9decd53d0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5f7ddce9decd53d0(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5f7ddce9decd53d0, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5f7ddce9decd53d0, block.x * block.y * block.z, 0));
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
void launcher_kernel_5f7ddce9decd53d0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5f7ddce9decd53d0(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5f7ddce9decd53d0), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5f7ddce9decd53d0, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5f7ddce9decd53d0(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (64 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 64×19(64×19) {0..64}×{0..19} strided
    //   m1 64×19(64×19) {0..64}×{0..19} strided
    // operations:
    //   m0[i,j]@{0..64}×{0..17} += m1[i,j]@{0..64}×{0..17}
    //   t = max(IA, I)
    //   m0[i,j]@{0..64}×{17..19} = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"IA","bbox":[[0,0],[64,19]],"name":"m0","ordered":false,"parts":1,"shape":[64,19],"variant":false},{"addressing":"strided","alias":"I","bbox":[[0,0],[64,19]],"name":"m1","ordered":false,"parts":1,"shape":[64,19],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[64,17]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,19]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,19]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,2]},"kind":"elementwise","op":"MAX","ops":[{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m0","offset":[0,17],"shape":[64,19]},{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m1","offset":[0,17],"shape":[64,19]}],"permute":[[0,1],[0,1]],"scalars":[],"target":[[0,1],[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m0","offset":[0,17],"shape":[64,19]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,2]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 1216 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 1216 + 0 + m1_extraOffset];
          float r0[38]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v20_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 2; ++v21_i0) {
            int32_t v24_lead = v20_lead + (v21_i0 * 32);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 19; ++v22_i1) {
              float v27_data = glb_m1[(v24_lead + (v22_i1 * 64))];
              r0[(v21_i0 + (v22_i1 * 2))] = v27_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[34]{};
          // r1 = +(r0) + None
          // [(0, 64), (0, 17)] []
          float v31_data = r0[0];
          float v32_data = r1[0];
          r1[0] = (v32_data + v31_data);
          float v34_data = r0[2];
          float v35_data = r1[2];
          r1[2] = (v35_data + v34_data);
          float v37_data = r0[4];
          float v38_data = r1[4];
          r1[4] = (v38_data + v37_data);
          float v40_data = r0[6];
          float v41_data = r1[6];
          r1[6] = (v41_data + v40_data);
          float v43_data = r0[8];
          float v44_data = r1[8];
          r1[8] = (v44_data + v43_data);
          float v46_data = r0[10];
          float v47_data = r1[10];
          r1[10] = (v47_data + v46_data);
          float v49_data = r0[12];
          float v50_data = r1[12];
          r1[12] = (v50_data + v49_data);
          float v52_data = r0[14];
          float v53_data = r1[14];
          r1[14] = (v53_data + v52_data);
          float v55_data = r0[16];
          float v56_data = r1[16];
          r1[16] = (v56_data + v55_data);
          float v58_data = r0[18];
          float v59_data = r1[18];
          r1[18] = (v59_data + v58_data);
          float v61_data = r0[20];
          float v62_data = r1[20];
          r1[20] = (v62_data + v61_data);
          float v64_data = r0[22];
          float v65_data = r1[22];
          r1[22] = (v65_data + v64_data);
          float v67_data = r0[24];
          float v68_data = r1[24];
          r1[24] = (v68_data + v67_data);
          float v70_data = r0[26];
          float v71_data = r1[26];
          r1[26] = (v71_data + v70_data);
          float v73_data = r0[28];
          float v74_data = r1[28];
          r1[28] = (v74_data + v73_data);
          float v76_data = r0[30];
          float v77_data = r1[30];
          r1[30] = (v77_data + v76_data);
          float v79_data = r0[32];
          float v80_data = r1[32];
          r1[32] = (v80_data + v79_data);
          float v82_data = r0[1];
          float v83_data = r1[1];
          r1[1] = (v83_data + v82_data);
          float v85_data = r0[3];
          float v86_data = r1[3];
          r1[3] = (v86_data + v85_data);
          float v88_data = r0[5];
          float v89_data = r1[5];
          r1[5] = (v89_data + v88_data);
          float v91_data = r0[7];
          float v92_data = r1[7];
          r1[7] = (v92_data + v91_data);
          float v94_data = r0[9];
          float v95_data = r1[9];
          r1[9] = (v95_data + v94_data);
          float v97_data = r0[11];
          float v98_data = r1[11];
          r1[11] = (v98_data + v97_data);
          float v100_data = r0[13];
          float v101_data = r1[13];
          r1[13] = (v101_data + v100_data);
          float v103_data = r0[15];
          float v104_data = r1[15];
          r1[15] = (v104_data + v103_data);
          float v106_data = r0[17];
          float v107_data = r1[17];
          r1[17] = (v107_data + v106_data);
          float v109_data = r0[19];
          float v110_data = r1[19];
          r1[19] = (v110_data + v109_data);
          float v112_data = r0[21];
          float v113_data = r1[21];
          r1[21] = (v113_data + v112_data);
          float v115_data = r0[23];
          float v116_data = r1[23];
          r1[23] = (v116_data + v115_data);
          float v118_data = r0[25];
          float v119_data = r1[25];
          r1[25] = (v119_data + v118_data);
          float v121_data = r0[27];
          float v122_data = r1[27];
          r1[27] = (v122_data + v121_data);
          float v124_data = r0[29];
          float v125_data = r1[29];
          r1[29] = (v125_data + v124_data);
          float v127_data = r0[31];
          float v128_data = r1[31];
          r1[31] = (v128_data + v127_data);
          float v130_data = r0[33];
          float v131_data = r1[33];
          r1[33] = (v131_data + v130_data);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v133_i0 = 0; v133_i0 < 2; ++v133_i0) {
            int32_t v139_lead = v20_lead + (v133_i0 * 32);
            #pragma unroll
            for (int32_t v134_i1 = 0; v134_i1 < 17; ++v134_i1) {
              float v137_data = r1[(v133_i0 + (v134_i1 * 2))];
              int32_t v141_a = v139_lead + (v134_i1 * 64);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v141_a], v137_data);
            }
          }
          float r2[4]{};
          // r2 = max(glb_m0, glb_m1)
          #pragma unroll
          for (int32_t v143_k0 = 0; v143_k0 < 2; ++v143_k0) {
            int32_t v146_lead = v20_lead + (v143_k0 * 32);
            #pragma unroll
            for (int32_t v144_k1 = 0; v144_k1 < 2; ++v144_k1) {
              int32_t v149_a = v146_lead + ((v144_k1 + 17) * 64);
              float v150_data = glb_m0[v149_a];
              float v151_data = glb_m1[v149_a];
              r2[(v143_k0 + (v144_k1 * 2))] = (fmaxf(v150_data, v151_data));
            }
          }
          float r3[4]{};
          // r3 = +(r2) + None
          // [(0, 64), (0, 2)] []
          float v156_data = r2[0];
          float v157_data = r3[0];
          r3[0] = (v157_data + v156_data);
          float v159_data = r2[2];
          float v160_data = r3[2];
          r3[2] = (v160_data + v159_data);
          float v162_data = r2[1];
          float v163_data = r3[1];
          r3[1] = (v163_data + v162_data);
          float v165_data = r2[3];
          float v166_data = r3[3];
          r3[3] = (v166_data + v165_data);
          // glb_m0 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v168_i0 = 0; v168_i0 < 2; ++v168_i0) {
            int32_t v174_lead = v20_lead + (v168_i0 * 32);
            #pragma unroll
            for (int32_t v169_i1 = 0; v169_i1 < 2; ++v169_i1) {
              float v172_data = r3[(v168_i0 + (v169_i1 * 2))];
              glb_m0[(v174_lead + ((v169_i1 + 17) * 64))] = v172_data;
            }
          }
        }
      }
    }
  }
}

