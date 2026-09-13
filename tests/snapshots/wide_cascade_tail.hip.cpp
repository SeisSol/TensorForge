// === base name ===
kernel_c5313b40e874d286

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c5313b40e874d286 = {{32, 8, 1}, 32, 24, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c5313b40e874d286(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c5313b40e874d286(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c5313b40e874d286(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c5313b40e874d286, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c5313b40e874d286, block.x * block.y * block.z, 0));
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
void launcher_kernel_c5313b40e874d286(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c5313b40e874d286(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c5313b40e874d286), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_c5313b40e874d286, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c5313b40e874d286(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (24 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 24×9(24×9) {0..24}×{0..9} strided
    //   m1 24×24(24×24) {0..24}×{0..24} strided
    //   m2 24×9(24×9) {0..24}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[24,9]],"name":"m0","ordered":false,"parts":1,"shape":[24,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,24]],"name":"m1","ordered":false,"parts":1,"shape":[24,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,9]],"name":"m2","ordered":false,"parts":1,"shape":[24,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[24,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,24]},{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 216 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 576 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v15_lead = threadIdx.x % 32;
          bool v16_g = v15_lead < 24;
          if (v16_g) {
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 24; ++v17_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v15_lead + (v17_i1 * 24))]);
              r0[v17_i1] = v22_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          if (v16_g) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m2[(v15_lead + (v25_i1 * 24))]);
              r1[v25_i1] = v30_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float v33_data = r1[0];
          float v34_data = r1[1];
          float v35_data = r1[2];
          float v36_data = r1[3];
          float v37_tp{};
          float v38_tp{};
          float v39_tp{};
          float v40_tp{};
          tensorforge::transpose4x4b32(v37_tp, v38_tp, v39_tp, v40_tp, v33_data, v34_data, v35_data, v36_data);
          tensorforge::VectorT<float, 4> v41_acc{};
          float v42_data = r0[0];
          float v43_data = r0[1];
          float v44_data = r0[2];
          float v45_data = r0[3];
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v42_data, v41_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v43_data, v46_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v44_data, v47_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v45_data, v48_acc, 3, 0, 0);
          float v50_data = r0[4];
          float v51_data = r0[5];
          float v52_data = r0[6];
          float v53_data = r0[7];
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v50_data, v49_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v51_data, v54_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v52_data, v55_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v53_data, v56_acc, 3, 1, 0);
          float v58_data = r0[8];
          float v59_data = r0[9];
          float v60_data = r0[10];
          float v61_data = r0[11];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v58_data, v57_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v59_data, v62_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v60_data, v63_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v61_data, v64_acc, 3, 2, 0);
          float v66_data = r0[12];
          float v67_data = r0[13];
          float v68_data = r0[14];
          float v69_data = r0[15];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v66_data, v65_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v67_data, v70_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v68_data, v71_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v69_data, v72_acc, 3, 3, 0);
          float v74_data = r0[16];
          float v75_data = r0[17];
          float v76_data = r0[18];
          float v77_data = r0[19];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v74_data, v73_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v75_data, v78_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v76_data, v79_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v77_data, v80_acc, 3, 4, 0);
          float v82_data = r0[20];
          float v83_data = r0[21];
          float v84_data = r0[22];
          float v85_data = r0[23];
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v82_data, v81_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v83_data, v86_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v84_data, v87_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v85_data, v88_acc, 3, 5, 0);
          r2[0] = (v89_acc[0]);
          r2[1] = (v89_acc[1]);
          r2[2] = (v89_acc[2]);
          r2[3] = (v89_acc[3]);
          float v94_data = r1[4];
          float v95_data = r1[5];
          float v96_data = r1[6];
          float v97_data = r1[7];
          float v98_tp{};
          float v99_tp{};
          float v100_tp{};
          float v101_tp{};
          tensorforge::transpose4x4b32(v98_tp, v99_tp, v100_tp, v101_tp, v94_data, v95_data, v96_data, v97_data);
          tensorforge::VectorT<float, 4> v102_acc{};
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v42_data, v102_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v43_data, v107_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v44_data, v108_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v45_data, v109_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v50_data, v110_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v51_data, v115_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v52_data, v116_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v53_data, v117_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v58_data, v118_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v59_data, v123_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v60_data, v124_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v61_data, v125_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v126_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v131_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v132_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v69_data, v133_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v134_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v139_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v140_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v77_data, v141_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v142_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v147_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v148_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v85_data, v149_acc, 3, 5, 0);
          r2[4] = (v150_acc[0]);
          r2[5] = (v150_acc[1]);
          r2[6] = (v150_acc[2]);
          r2[7] = (v150_acc[3]);
          float v179_acc{};
          float v180_data = r1[8];
          float v181_bc = tensorforge::broadcast<32, 16, 0>(v180_data);
          tensorforge::fmacdpp16<0>(v179_acc, v181_bc, v42_data);
          tensorforge::fmacdpp16<1>(v179_acc, v181_bc, v43_data);
          tensorforge::fmacdpp16<2>(v179_acc, v181_bc, v44_data);
          tensorforge::fmacdpp16<3>(v179_acc, v181_bc, v45_data);
          tensorforge::fmacdpp16<4>(v179_acc, v181_bc, v50_data);
          tensorforge::fmacdpp16<5>(v179_acc, v181_bc, v51_data);
          tensorforge::fmacdpp16<6>(v179_acc, v181_bc, v52_data);
          tensorforge::fmacdpp16<7>(v179_acc, v181_bc, v53_data);
          tensorforge::fmacdpp16<8>(v179_acc, v181_bc, v58_data);
          tensorforge::fmacdpp16<9>(v179_acc, v181_bc, v59_data);
          tensorforge::fmacdpp16<10>(v179_acc, v181_bc, v60_data);
          tensorforge::fmacdpp16<11>(v179_acc, v181_bc, v61_data);
          tensorforge::fmacdpp16<12>(v179_acc, v181_bc, v66_data);
          tensorforge::fmacdpp16<13>(v179_acc, v181_bc, v67_data);
          tensorforge::fmacdpp16<14>(v179_acc, v181_bc, v68_data);
          tensorforge::fmacdpp16<15>(v179_acc, v181_bc, v69_data);
          float v182_bc = tensorforge::broadcast<32, 16, 1>(v180_data);
          tensorforge::fmacdpp16<0>(v179_acc, v182_bc, v74_data);
          tensorforge::fmacdpp16<1>(v179_acc, v182_bc, v75_data);
          tensorforge::fmacdpp16<2>(v179_acc, v182_bc, v76_data);
          tensorforge::fmacdpp16<3>(v179_acc, v182_bc, v77_data);
          tensorforge::fmacdpp16<4>(v179_acc, v182_bc, v82_data);
          tensorforge::fmacdpp16<5>(v179_acc, v182_bc, v83_data);
          tensorforge::fmacdpp16<6>(v179_acc, v182_bc, v84_data);
          tensorforge::fmacdpp16<7>(v179_acc, v182_bc, v85_data);
          r2[8] = v179_acc;
          // glb_m0 = store{r>g}(r2);
          if (v16_g) {
            #pragma unroll
            for (int32_t v183_i1 = 0; v183_i1 < 9; ++v183_i1) {
              float v185_data = r2[v183_i1];
              glb_m0[(v15_lead + (v183_i1 * 24))] = v185_data;
            }
          }
        }
      }
    }
  }
}

