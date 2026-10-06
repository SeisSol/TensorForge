// === base name ===
kernel_0980b8e4fab71f91

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0980b8e4fab71f91 = {{32, 8, 1}, 32, 24, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0980b8e4fab71f91(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0980b8e4fab71f91(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0980b8e4fab71f91(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0980b8e4fab71f91, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0980b8e4fab71f91, block.x * block.y * block.z, 0));
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
void launcher_kernel_0980b8e4fab71f91(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0980b8e4fab71f91(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0980b8e4fab71f91), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0980b8e4fab71f91, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0980b8e4fab71f91(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (24 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 24×9(24×9) {0..24}×{0..9} strided
    //   m1 24×24(24×24) {0..24}×{0..24} strided
    //   m2 24×9(24×9) {0..24}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[24,9]],"name":"m0","ordered":false,"parts":1,"shape":[24,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,24]],"name":"m1","ordered":false,"parts":1,"shape":[24,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,9]],"name":"m2","ordered":false,"parts":1,"shape":[24,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[24,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,24]},{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 216 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 576 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v21_lead = threadIdx.x % 32;
          bool v22_g = v21_lead < 24;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 24; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v23_i1 * 24))]);
              r0[v23_i1] = v28_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v31_i1 * 24))]);
              r1[v31_i1] = v36_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float v39_data = r1[0];
          float v40_data = r1[1];
          float v41_data = r1[2];
          float v42_data = r1[3];
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          float v46_tp{};
          tensorforge::transpose4x4b32(v43_tp, v44_tp, v45_tp, v46_tp, v39_data, v40_data, v41_data, v42_data);
          tensorforge::VectorT<float, 4> v47_acc{};
          float v48_data = r0[0];
          float v49_data = r0[1];
          float v50_data = r0[2];
          float v51_data = r0[3];
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v47_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 3, 0, 0);
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v55_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 3, 1, 0);
          float v64_data = r0[8];
          float v65_data = r0[9];
          float v66_data = r0[10];
          float v67_data = r0[11];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v64_data, v63_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v68_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v69_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v70_acc, 3, 2, 0);
          float v72_data = r0[12];
          float v73_data = r0[13];
          float v74_data = r0[14];
          float v75_data = r0[15];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v72_data, v71_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v76_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v77_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v78_acc, 3, 3, 0);
          float v80_data = r0[16];
          float v81_data = r0[17];
          float v82_data = r0[18];
          float v83_data = r0[19];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v80_data, v79_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v81_data, v84_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v82_data, v85_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v83_data, v86_acc, 3, 4, 0);
          float v88_data = r0[20];
          float v89_data = r0[21];
          float v90_data = r0[22];
          float v91_data = r0[23];
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v88_data, v87_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v89_data, v92_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v90_data, v93_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v91_data, v94_acc, 3, 5, 0);
          r2[0] = (v95_acc[0]);
          r2[1] = (v95_acc[1]);
          r2[2] = (v95_acc[2]);
          r2[3] = (v95_acc[3]);
          float v100_data = r1[4];
          float v101_data = r1[5];
          float v102_data = r1[6];
          float v103_data = r1[7];
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          float v107_tp{};
          tensorforge::transpose4x4b32(v104_tp, v105_tp, v106_tp, v107_tp, v100_data, v101_data, v102_data, v103_data);
          tensorforge::VectorT<float, 4> v108_acc{};
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v48_data, v108_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v49_data, v113_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v50_data, v114_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v51_data, v115_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v56_data, v116_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v57_data, v121_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v58_data, v122_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v59_data, v123_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v64_data, v124_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v65_data, v129_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v66_data, v130_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v67_data, v131_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v72_data, v132_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v73_data, v137_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v74_data, v138_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v75_data, v139_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v80_data, v140_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v81_data, v145_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v82_data, v146_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v83_data, v147_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v88_data, v148_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v89_data, v153_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v90_data, v154_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v91_data, v155_acc, 3, 5, 0);
          r2[4] = (v156_acc[0]);
          r2[5] = (v156_acc[1]);
          r2[6] = (v156_acc[2]);
          r2[7] = (v156_acc[3]);
          float v185_acc{};
          float v186_data = r1[8];
          float v187_bc = tensorforge::broadcast<32, 16, 0>(v186_data);
          tensorforge::fmacdpp16<0>(v185_acc, v187_bc, v48_data);
          tensorforge::fmacdpp16<1>(v185_acc, v187_bc, v49_data);
          tensorforge::fmacdpp16<2>(v185_acc, v187_bc, v50_data);
          tensorforge::fmacdpp16<3>(v185_acc, v187_bc, v51_data);
          tensorforge::fmacdpp16<4>(v185_acc, v187_bc, v56_data);
          tensorforge::fmacdpp16<5>(v185_acc, v187_bc, v57_data);
          tensorforge::fmacdpp16<6>(v185_acc, v187_bc, v58_data);
          tensorforge::fmacdpp16<7>(v185_acc, v187_bc, v59_data);
          tensorforge::fmacdpp16<8>(v185_acc, v187_bc, v64_data);
          tensorforge::fmacdpp16<9>(v185_acc, v187_bc, v65_data);
          tensorforge::fmacdpp16<10>(v185_acc, v187_bc, v66_data);
          tensorforge::fmacdpp16<11>(v185_acc, v187_bc, v67_data);
          tensorforge::fmacdpp16<12>(v185_acc, v187_bc, v72_data);
          tensorforge::fmacdpp16<13>(v185_acc, v187_bc, v73_data);
          tensorforge::fmacdpp16<14>(v185_acc, v187_bc, v74_data);
          tensorforge::fmacdpp16<15>(v185_acc, v187_bc, v75_data);
          float v188_bc = tensorforge::broadcast<32, 16, 1>(v186_data);
          tensorforge::fmacdpp16<0>(v185_acc, v188_bc, v80_data);
          tensorforge::fmacdpp16<1>(v185_acc, v188_bc, v81_data);
          tensorforge::fmacdpp16<2>(v185_acc, v188_bc, v82_data);
          tensorforge::fmacdpp16<3>(v185_acc, v188_bc, v83_data);
          tensorforge::fmacdpp16<4>(v185_acc, v188_bc, v88_data);
          tensorforge::fmacdpp16<5>(v185_acc, v188_bc, v89_data);
          tensorforge::fmacdpp16<6>(v185_acc, v188_bc, v90_data);
          tensorforge::fmacdpp16<7>(v185_acc, v188_bc, v91_data);
          r2[8] = v185_acc;
          // glb_m0 = store{r>g}(r2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v189_i1 = 0; v189_i1 < 9; ++v189_i1) {
              float v191_data = r2[v189_i1];
              glb_m0[(v21_lead + (v189_i1 * 24))] = v191_data;
            }
          }
        }
      }
    }
  }
}

