// === base name ===
kernel_b685418ed7a09731

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b685418ed7a09731 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b685418ed7a09731(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b685418ed7a09731(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b685418ed7a09731(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b685418ed7a09731, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b685418ed7a09731, block.x * block.y * block.z, 0));
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
void launcher_kernel_b685418ed7a09731(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b685418ed7a09731(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b685418ed7a09731), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b685418ed7a09731, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b685418ed7a09731(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v1_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0][0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v1_batchId0][0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v17_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v21_lead = v17_lead + (v18_i0 * 32);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m0[(v21_lead + (v19_i1 * 32))]);
              r0[(v18_i0 + v19_i1)] = v24_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v27_g = v17_lead < 16;
          if (v27_g) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v28_i1 * 16))]);
              r2[v28_i1] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
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
          // r4 = load{g>r}(glb_m2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v64_i1 = 0; v64_i1 < 9; ++v64_i1) {
              float v69_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v64_i1 * 16))]);
              r4[v64_i1] = v69_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          float v73_data = r2[0];
          float v74_data = ir3[0];
          ir3[0] = (v74_data + v73_data);
          float v76_data = r2[1];
          float v77_data = ir3[1];
          ir3[1] = (v77_data + v76_data);
          float v79_data = r2[2];
          float v80_data = ir3[2];
          ir3[2] = (v80_data + v79_data);
          float v82_data = r2[3];
          float v83_data = ir3[3];
          ir3[3] = (v83_data + v82_data);
          float v85_data = r2[4];
          float v86_data = ir3[4];
          ir3[4] = (v86_data + v85_data);
          float v88_data = r2[5];
          float v89_data = ir3[5];
          ir3[5] = (v89_data + v88_data);
          float v91_data = r2[6];
          float v92_data = ir3[6];
          ir3[6] = (v92_data + v91_data);
          float v94_data = r2[7];
          float v95_data = ir3[7];
          ir3[7] = (v95_data + v94_data);
          float v97_data = r2[8];
          float v98_data = ir3[8];
          ir3[8] = (v98_data + v97_data);
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v100_n1 = 0; v100_n1 < 9; ++v100_n1) {
            float v102_data = ir3[v100_n1];
            float v103_data = r1[v100_n1];
            r3[v100_n1] = (v103_data + v102_data);
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v17_lead < 9) {
            #pragma unroll
            for (int32_t v107_i1 = 0; v107_i1 < 9; ++v107_i1) {
              float v112_data = __builtin_nontemporal_load(&glb_m4[(v17_lead + (v107_i1 * 9))]);
              r6[v107_i1] = v112_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // ir5 = +(r4)
          // [(0, 16), (0, 9)] []
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
          #pragma unroll
          for (int32_t v143_n1 = 0; v143_n1 < 9; ++v143_n1) {
            float v145_data = ir5[v143_n1];
            float v146_data = r3[v143_n1];
            r5[v143_n1] = (v146_data + v145_data);
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // r7 = +(r5 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float v149_data = r6[0];
          float v150_data = r6[1];
          float v151_data = r6[2];
          float v152_data = r6[3];
          float v153_tp{};
          float v154_tp{};
          float v155_tp{};
          float v156_tp{};
          tensorforge::transpose4x4b32(v153_tp, v154_tp, v155_tp, v156_tp, v149_data, v150_data, v151_data, v152_data);
          tensorforge::VectorT<float, 4> v157_acc{};
          float v158_data = r5[0];
          float v159_data = r5[1];
          float v160_data = r5[2];
          float v161_data = r5[3];
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v158_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v159_data, v162_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v160_data, v163_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v161_data, v164_acc, 3, 0, 0);
          float v166_data = r5[4];
          float v167_data = r5[5];
          float v168_data = r5[6];
          float v169_data = r5[7];
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v166_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v167_data, v170_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v168_data, v171_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v169_data, v172_acc, 3, 1, 0);
          float v174_data = r5[8];
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v174_data, v173_acc, 3, 2, 0);
          r7[0] = (v176_acc[0]);
          r7[1] = (v176_acc[1]);
          r7[2] = (v176_acc[2]);
          r7[3] = (v176_acc[3]);
          float v181_data = r6[4];
          float v182_data = r6[5];
          float v183_data = r6[6];
          float v184_data = r6[7];
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          float v188_tp{};
          tensorforge::transpose4x4b32(v185_tp, v186_tp, v187_tp, v188_tp, v181_data, v182_data, v183_data, v184_data);
          tensorforge::VectorT<float, 4> v189_acc{};
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v158_data, v189_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v159_data, v194_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v160_data, v195_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v161_data, v196_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v166_data, v197_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v167_data, v202_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v168_data, v203_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v169_data, v204_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v174_data, v205_acc, 3, 2, 0);
          r7[4] = (v208_acc[0]);
          r7[5] = (v208_acc[1]);
          r7[6] = (v208_acc[2]);
          r7[7] = (v208_acc[3]);
          float v222_acc{};
          float v223_data = r6[8];
          float v224_bc = tensorforge::broadcast<32, 16, 0>(v223_data);
          tensorforge::fmacdpp16<0>(v222_acc, v224_bc, v158_data);
          tensorforge::fmacdpp16<1>(v222_acc, v224_bc, v159_data);
          tensorforge::fmacdpp16<2>(v222_acc, v224_bc, v160_data);
          tensorforge::fmacdpp16<3>(v222_acc, v224_bc, v161_data);
          tensorforge::fmacdpp16<4>(v222_acc, v224_bc, v166_data);
          tensorforge::fmacdpp16<5>(v222_acc, v224_bc, v167_data);
          tensorforge::fmacdpp16<6>(v222_acc, v224_bc, v168_data);
          tensorforge::fmacdpp16<7>(v222_acc, v224_bc, v169_data);
          tensorforge::fmacdpp16<8>(v222_acc, v224_bc, v174_data);
          r7[8] = v222_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v225_i0 = 0; v225_i0 < 1; ++v225_i0) {
            int32_t v230_lead = v17_lead + (v225_i0 * 32);
            #pragma unroll
            for (int32_t v226_i1 = 0; v226_i1 < 9; ++v226_i1) {
              float v228_data = r7[(v225_i0 + v226_i1)];
              glb_m3[(v230_lead + (v226_i1 * 32))] = v228_data;
            }
          }
        }
      }
    }
  }
}

