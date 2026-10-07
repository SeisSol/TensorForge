// === base name ===
kernel_de5c67279ba995c4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_de5c67279ba995c4 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_de5c67279ba995c4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_de5c67279ba995c4(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_de5c67279ba995c4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_de5c67279ba995c4, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_de5c67279ba995c4, block.x * block.y * block.z, 0));
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
void launcher_kernel_de5c67279ba995c4(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_de5c67279ba995c4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_de5c67279ba995c4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_de5c67279ba995c4, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_de5c67279ba995c4(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v7_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0][0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v7_batchId0][0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 32);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v27_lead + (v25_i1 * 32))]);
              r0[(v24_i0 + v25_i1)] = v30_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v61_g = v23_lead < 16;
          if (v61_g) {
            #pragma unroll
            for (int32_t v62_i1 = 0; v62_i1 < 9; ++v62_i1) {
              float v67_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v62_i1 * 16))]);
              r2[v62_i1] = v67_data;
            }
          }
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v33_data = r0[0];
          float v34_data = r1[0];
          r1[0] = (v34_data + v33_data);
          float v36_data = r0[1];
          float v37_data = r1[1];
          r1[1] = (v37_data + v36_data);
          float v39_data = r0[2];
          float v40_data = r1[2];
          r1[2] = (v40_data + v39_data);
          float v42_data = r0[3];
          float v43_data = r1[3];
          r1[3] = (v43_data + v42_data);
          float v45_data = r0[4];
          float v46_data = r1[4];
          r1[4] = (v46_data + v45_data);
          float v48_data = r0[5];
          float v49_data = r1[5];
          r1[5] = (v49_data + v48_data);
          float v51_data = r0[6];
          float v52_data = r1[6];
          r1[6] = (v52_data + v51_data);
          float v54_data = r0[7];
          float v55_data = r1[7];
          r1[7] = (v55_data + v54_data);
          float v57_data = r0[8];
          float v58_data = r1[8];
          r1[8] = (v58_data + v57_data);
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v61_g) {
            #pragma unroll
            for (int32_t v104_i1 = 0; v104_i1 < 9; ++v104_i1) {
              float v109_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v104_i1 * 16))]);
              r4[v104_i1] = v109_data;
            }
          }
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          float v71_data = r2[0];
          float v72_data = ir3[0];
          ir3[0] = (v72_data + v71_data);
          float v74_data = r2[1];
          float v75_data = ir3[1];
          ir3[1] = (v75_data + v74_data);
          float v77_data = r2[2];
          float v78_data = ir3[2];
          ir3[2] = (v78_data + v77_data);
          float v80_data = r2[3];
          float v81_data = ir3[3];
          ir3[3] = (v81_data + v80_data);
          float v83_data = r2[4];
          float v84_data = ir3[4];
          ir3[4] = (v84_data + v83_data);
          float v86_data = r2[5];
          float v87_data = ir3[5];
          ir3[5] = (v87_data + v86_data);
          float v89_data = r2[6];
          float v90_data = ir3[6];
          ir3[6] = (v90_data + v89_data);
          float v92_data = r2[7];
          float v93_data = ir3[7];
          ir3[7] = (v93_data + v92_data);
          float v95_data = r2[8];
          float v96_data = ir3[8];
          ir3[8] = (v96_data + v95_data);
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v98_n1 = 0; v98_n1 < 9; ++v98_n1) {
            float v100_data = ir3[v98_n1];
            float v101_data = r1[v98_n1];
            r3[v98_n1] = (v101_data + v100_data);
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v23_lead < 9) {
            #pragma unroll
            for (int32_t v147_i1 = 0; v147_i1 < 9; ++v147_i1) {
              float v152_data = __builtin_nontemporal_load(&glb_m4[(v23_lead + (v147_i1 * 9))]);
              r6[v147_i1] = v152_data;
            }
          }
          float r5[9]{};
          // ir5 = +(r4)
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          float v113_data = r4[0];
          float v114_data = ir5[0];
          ir5[0] = (v114_data + v113_data);
          float v116_data = r4[1];
          float v117_data = ir5[1];
          ir5[1] = (v117_data + v116_data);
          float v119_data = r4[2];
          float v120_data = ir5[2];
          ir5[2] = (v120_data + v119_data);
          float v122_data = r4[3];
          float v123_data = ir5[3];
          ir5[3] = (v123_data + v122_data);
          float v125_data = r4[4];
          float v126_data = ir5[4];
          ir5[4] = (v126_data + v125_data);
          float v128_data = r4[5];
          float v129_data = ir5[5];
          ir5[5] = (v129_data + v128_data);
          float v131_data = r4[6];
          float v132_data = ir5[6];
          ir5[6] = (v132_data + v131_data);
          float v134_data = r4[7];
          float v135_data = ir5[7];
          ir5[7] = (v135_data + v134_data);
          float v137_data = r4[8];
          float v138_data = ir5[8];
          ir5[8] = (v138_data + v137_data);
          // r5 = ir5 + r3
          #pragma unroll
          for (int32_t v140_n1 = 0; v140_n1 < 9; ++v140_n1) {
            float v142_data = ir5[v140_n1];
            float v143_data = r3[v140_n1];
            r5[v140_n1] = (v143_data + v142_data);
          }
          float r7[9]{};
          // r7 = +(r5 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v155_data = r6[0];
          float v156_data = r6[1];
          float v157_data = r6[2];
          float v158_data = r6[3];
          float v159_tp{};
          float v160_tp{};
          float v161_tp{};
          float v162_tp{};
          tensorforge::transpose4x4b32(v159_tp, v160_tp, v161_tp, v162_tp, v155_data, v156_data, v157_data, v158_data);
          tensorforge::VectorT<float, 4> v163_acc{};
          float v164_data = r5[0];
          float v165_data = r5[1];
          float v166_data = r5[2];
          float v167_data = r5[3];
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v164_data, v163_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v165_data, v168_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v166_data, v169_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v167_data, v170_acc, 3, 0, 0);
          float v172_data = r5[4];
          float v173_data = r5[5];
          float v174_data = r5[6];
          float v175_data = r5[7];
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v172_data, v171_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v160_tp, v173_data, v176_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v161_tp, v174_data, v177_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v162_tp, v175_data, v178_acc, 3, 1, 0);
          float v180_data = r5[8];
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v159_tp, v180_data, v179_acc, 3, 2, 0);
          r7[0] = (v182_acc[0]);
          r7[1] = (v182_acc[1]);
          r7[2] = (v182_acc[2]);
          r7[3] = (v182_acc[3]);
          float v187_data = r6[4];
          float v188_data = r6[5];
          float v189_data = r6[6];
          float v190_data = r6[7];
          float v191_tp{};
          float v192_tp{};
          float v193_tp{};
          float v194_tp{};
          tensorforge::transpose4x4b32(v191_tp, v192_tp, v193_tp, v194_tp, v187_data, v188_data, v189_data, v190_data);
          tensorforge::VectorT<float, 4> v195_acc{};
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v164_data, v195_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v165_data, v200_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v166_data, v201_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v167_data, v202_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v172_data, v203_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v173_data, v208_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v174_data, v209_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v175_data, v210_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v180_data, v211_acc, 3, 2, 0);
          r7[4] = (v214_acc[0]);
          r7[5] = (v214_acc[1]);
          r7[6] = (v214_acc[2]);
          r7[7] = (v214_acc[3]);
          float v228_acc{};
          float v229_data = r6[8];
          float v230_bc = tensorforge::broadcast<32, 16, 0>(v229_data);
          tensorforge::fmacdpp16<0>(v228_acc, v230_bc, v164_data);
          tensorforge::fmacdpp16<1>(v228_acc, v230_bc, v165_data);
          tensorforge::fmacdpp16<2>(v228_acc, v230_bc, v166_data);
          tensorforge::fmacdpp16<3>(v228_acc, v230_bc, v167_data);
          tensorforge::fmacdpp16<4>(v228_acc, v230_bc, v172_data);
          tensorforge::fmacdpp16<5>(v228_acc, v230_bc, v173_data);
          tensorforge::fmacdpp16<6>(v228_acc, v230_bc, v174_data);
          tensorforge::fmacdpp16<7>(v228_acc, v230_bc, v175_data);
          tensorforge::fmacdpp16<8>(v228_acc, v230_bc, v180_data);
          r7[8] = v228_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v231_i0 = 0; v231_i0 < 1; ++v231_i0) {
            int32_t v236_lead = v23_lead + (v231_i0 * 32);
            #pragma unroll
            for (int32_t v232_i1 = 0; v232_i1 < 9; ++v232_i1) {
              float v234_data = r7[(v231_i0 + v232_i1)];
              glb_m3[(v236_lead + (v232_i1 * 32))] = v234_data;
            }
          }
        }
      }
    }
  }
}

