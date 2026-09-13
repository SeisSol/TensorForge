// === base name ===
kernel_78f5f3efeeb71371

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_78f5f3efeeb71371 = {{32, 8, 1}, 32, 32, 1, 8, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_78f5f3efeeb71371(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_78f5f3efeeb71371(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_78f5f3efeeb71371(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_78f5f3efeeb71371, block.x * block.y * block.z, 2560 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (2560 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_78f5f3efeeb71371, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (2560 * sizeof(float)));
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_78f5f3efeeb71371(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_78f5f3efeeb71371(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_78f5f3efeeb71371), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_78f5f3efeeb71371, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_78f5f3efeeb71371(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 10240 B shared, occupancy grid
    // operands:
    //   m0 32×9(32×9) {0..32}×{0..9} pointer_based
    //   m1 16×9(16×9) {0..16}×{0..9} pointer_based
    //   m2 16×9(16×9) {0..16}×{0..9} pointer_based
    //   m3 32×9(32×9) {0..32}×{0..9} pointer_based
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // operations:
    //   t0[i,j] = m0[i,j]
    //   t0[i,j] += m1[i,j]
    //   t0[i,j] += m2[i,j]
    //   m3[i,j] = t0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[320 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[320];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v5_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v5_batchId0][0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v5_batchId0][0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v5_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v25_lead = v21_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v25_lead + (v23_i1 * 32))]);
              r0[(v22_i0 + v23_i1)] = v28_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v31_g = v21_lead < 16;
          if (v31_g) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v32_i1 * 16))]);
              r2[v32_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v40_data = r0[0];
          float v41_data = r1[0];
          r1[0] = (v41_data + v40_data);
          float v43_data = r0[1];
          float v44_data = r1[1];
          r1[1] = (v44_data + v43_data);
          float v46_data = r0[2];
          float v47_data = r1[2];
          r1[2] = (v47_data + v46_data);
          float v49_data = r0[3];
          float v50_data = r1[3];
          r1[3] = (v50_data + v49_data);
          float v52_data = r0[4];
          float v53_data = r1[4];
          r1[4] = (v53_data + v52_data);
          float v55_data = r0[5];
          float v56_data = r1[5];
          r1[5] = (v56_data + v55_data);
          float v58_data = r0[6];
          float v59_data = r1[6];
          r1[6] = (v59_data + v58_data);
          float v61_data = r0[7];
          float v62_data = r1[7];
          r1[7] = (v62_data + v61_data);
          float v64_data = r0[8];
          float v65_data = r1[8];
          r1[8] = (v65_data + v64_data);
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
            int32_t v72_lead = v21_lead + (v67_i0 * 32);
            #pragma unroll
            for (int32_t v68_i1 = 0; v68_i1 < 9; ++v68_i1) {
              float v70_data = r1[(v67_i0 + v68_i1)];
              int32_t v74_a = v72_lead + (v68_i1 * 32);
              s0[(v74_a ^ ((v74_a >> 5) & 31))] = v70_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v31_g) {
            #pragma unroll
            for (int32_t v79_i1 = 0; v79_i1 < 9; ++v79_i1) {
              float v84_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v79_i1 * 16))]);
              r4[v79_i1] = v84_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          float v87_data = r2[0];
          float v88_data = r3[0];
          r3[0] = (v88_data + v87_data);
          float v90_data = r2[1];
          float v91_data = r3[1];
          r3[1] = (v91_data + v90_data);
          float v93_data = r2[2];
          float v94_data = r3[2];
          r3[2] = (v94_data + v93_data);
          float v96_data = r2[3];
          float v97_data = r3[3];
          r3[3] = (v97_data + v96_data);
          float v99_data = r2[4];
          float v100_data = r3[4];
          r3[4] = (v100_data + v99_data);
          float v102_data = r2[5];
          float v103_data = r3[5];
          r3[5] = (v103_data + v102_data);
          float v105_data = r2[6];
          float v106_data = r3[6];
          r3[6] = (v106_data + v105_data);
          float v108_data = r2[7];
          float v109_data = r3[7];
          r3[7] = (v109_data + v108_data);
          float v111_data = r2[8];
          float v112_data = r3[8];
          r3[8] = (v112_data + v111_data);
          // s0 = store{r>s}(localShrMem0, r3);
          if (v31_g) {
            #pragma unroll
            for (int32_t v114_i1 = 0; v114_i1 < 9; ++v114_i1) {
              float v116_data = r3[v114_i1];
              int32_t v120_a = v21_lead + (v114_i1 * 32);
              s0[(v120_a ^ ((v120_a >> 5) & 31))] = v116_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v21_lead < 9) {
            #pragma unroll
            for (int32_t v126_i1 = 0; v126_i1 < 9; ++v126_i1) {
              float v131_data = __builtin_nontemporal_load(&glb_m4[(v21_lead + (v126_i1 * 9))]);
              r6[v126_i1] = v131_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          float v134_data = r4[0];
          float v135_data = r5[0];
          r5[0] = (v135_data + v134_data);
          float v137_data = r4[1];
          float v138_data = r5[1];
          r5[1] = (v138_data + v137_data);
          float v140_data = r4[2];
          float v141_data = r5[2];
          r5[2] = (v141_data + v140_data);
          float v143_data = r4[3];
          float v144_data = r5[3];
          r5[3] = (v144_data + v143_data);
          float v146_data = r4[4];
          float v147_data = r5[4];
          r5[4] = (v147_data + v146_data);
          float v149_data = r4[5];
          float v150_data = r5[5];
          r5[5] = (v150_data + v149_data);
          float v152_data = r4[6];
          float v153_data = r5[6];
          r5[6] = (v153_data + v152_data);
          float v155_data = r4[7];
          float v156_data = r5[7];
          r5[7] = (v156_data + v155_data);
          float v158_data = r4[8];
          float v159_data = r5[8];
          r5[8] = (v159_data + v158_data);
          // s0 = store{r>s}(localShrMem0, r5);
          if (v31_g) {
            #pragma unroll
            for (int32_t v161_i1 = 0; v161_i1 < 9; ++v161_i1) {
              float v163_data = r5[v161_i1];
              int32_t v167_a = v21_lead + (v161_i1 * 32);
              s0[(v167_a ^ ((v167_a >> 5) & 31))] = v163_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v172_data = r6[0];
          float v173_data = r6[1];
          float v174_data = r6[2];
          float v175_data = r6[3];
          float v176_tp{};
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          tensorforge::transpose4x4b32(v176_tp, v177_tp, v178_tp, v179_tp, v172_data, v173_data, v174_data, v175_data);
          tensorforge::VectorT<float, 4> v180_acc{};
          int32_t v185_sw = (v21_lead >> 5) & 31;
          float v187_data = s0[(v21_lead ^ v185_sw)];
          int32_t v188_a = v21_lead + 32;
          int32_t v189_sw = v188_a >> 5;
          float v192_data = s0[(v188_a ^ (v189_sw & 31))];
          int32_t v193_a = v21_lead + 64;
          int32_t v194_sw = v193_a >> 5;
          float v197_data = s0[(v193_a ^ (v194_sw & 31))];
          int32_t v198_a = v21_lead + 96;
          int32_t v199_sw = v198_a >> 5;
          float v202_data = s0[(v198_a ^ (v199_sw & 31))];
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v187_data, v180_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v192_data, v203_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v197_data, v204_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v202_data, v205_acc, 3, 0, 0);
          int32_t v207_a = v21_lead + 128;
          int32_t v208_sw = v207_a >> 5;
          float v211_data = s0[(v207_a ^ (v208_sw & 31))];
          int32_t v212_a = v21_lead + 160;
          int32_t v213_sw = v212_a >> 5;
          float v216_data = s0[(v212_a ^ (v213_sw & 31))];
          int32_t v217_a = v21_lead + 192;
          int32_t v218_sw = v217_a >> 5;
          float v221_data = s0[(v217_a ^ (v218_sw & 31))];
          int32_t v222_a = v21_lead + 224;
          int32_t v223_sw = v222_a >> 5;
          float v226_data = s0[(v222_a ^ (v223_sw & 31))];
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v211_data, v206_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v216_data, v227_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v221_data, v228_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v226_data, v229_acc, 3, 1, 0);
          int32_t v231_a = v21_lead + 256;
          int32_t v232_sw = v231_a >> 5;
          float v235_data = s0[(v231_a ^ (v232_sw & 31))];
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v235_data, v230_acc, 3, 2, 0);
          r7[0] = (v237_acc[0]);
          r7[1] = (v237_acc[1]);
          r7[2] = (v237_acc[2]);
          r7[3] = (v237_acc[3]);
          float v242_data = r6[4];
          float v243_data = r6[5];
          float v244_data = r6[6];
          float v245_data = r6[7];
          float v246_tp{};
          float v247_tp{};
          float v248_tp{};
          float v249_tp{};
          tensorforge::transpose4x4b32(v246_tp, v247_tp, v248_tp, v249_tp, v242_data, v243_data, v244_data, v245_data);
          tensorforge::VectorT<float, 4> v250_acc{};
          float v257_data = s0[(v21_lead ^ v185_sw)];
          float v262_data = s0[(v188_a ^ (v189_sw & 31))];
          float v267_data = s0[(v193_a ^ (v194_sw & 31))];
          float v272_data = s0[(v198_a ^ (v199_sw & 31))];
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v257_data, v250_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v262_data, v273_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v267_data, v274_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v272_data, v275_acc, 3, 0, 0);
          float v281_data = s0[(v207_a ^ (v208_sw & 31))];
          float v286_data = s0[(v212_a ^ (v213_sw & 31))];
          float v291_data = s0[(v217_a ^ (v218_sw & 31))];
          float v296_data = s0[(v222_a ^ (v223_sw & 31))];
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v281_data, v276_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v286_data, v297_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v291_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v296_data, v299_acc, 3, 1, 0);
          float v305_data = s0[(v231_a ^ (v232_sw & 31))];
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v305_data, v300_acc, 3, 2, 0);
          r7[4] = (v307_acc[0]);
          r7[5] = (v307_acc[1]);
          r7[6] = (v307_acc[2]);
          r7[7] = (v307_acc[3]);
          float v318_data = s0[(v21_lead ^ v185_sw)];
          float v323_data = s0[(v188_a ^ (v189_sw & 31))];
          float v328_data = s0[(v193_a ^ (v194_sw & 31))];
          float v333_data = s0[(v198_a ^ (v199_sw & 31))];
          float v338_data = s0[(v207_a ^ (v208_sw & 31))];
          float v343_data = s0[(v212_a ^ (v213_sw & 31))];
          float v348_data = s0[(v217_a ^ (v218_sw & 31))];
          float v353_data = s0[(v222_a ^ (v223_sw & 31))];
          float v358_data = s0[(v231_a ^ (v232_sw & 31))];
          float v359_acc{};
          float v360_data = r6[8];
          float v361_bc = tensorforge::broadcast<32, 16, 0>(v360_data);
          tensorforge::fmacdpp16<0>(v359_acc, v361_bc, v318_data);
          tensorforge::fmacdpp16<1>(v359_acc, v361_bc, v323_data);
          tensorforge::fmacdpp16<2>(v359_acc, v361_bc, v328_data);
          tensorforge::fmacdpp16<3>(v359_acc, v361_bc, v333_data);
          tensorforge::fmacdpp16<4>(v359_acc, v361_bc, v338_data);
          tensorforge::fmacdpp16<5>(v359_acc, v361_bc, v343_data);
          tensorforge::fmacdpp16<6>(v359_acc, v361_bc, v348_data);
          tensorforge::fmacdpp16<7>(v359_acc, v361_bc, v353_data);
          tensorforge::fmacdpp16<8>(v359_acc, v361_bc, v358_data);
          r7[8] = v359_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v362_i0 = 0; v362_i0 < 1; ++v362_i0) {
            int32_t v367_lead = v21_lead + (v362_i0 * 32);
            #pragma unroll
            for (int32_t v363_i1 = 0; v363_i1 < 9; ++v363_i1) {
              float v365_data = r7[(v362_i0 + v363_i1)];
              glb_m3[(v367_lead + (v363_i1 * 32))] = v365_data;
            }
          }
        }
      }
    }
  }
}

