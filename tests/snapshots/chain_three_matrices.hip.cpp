// === base name ===
kernel_f430cf2afec1bb5e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f430cf2afec1bb5e = {{32, 8, 1}, 32, 56, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f430cf2afec1bb5e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f430cf2afec1bb5e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f430cf2afec1bb5e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f430cf2afec1bb5e, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_f430cf2afec1bb5e, block.x * block.y * block.z, 0));
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
void launcher_kernel_f430cf2afec1bb5e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f430cf2afec1bb5e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_f430cf2afec1bb5e), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_f430cf2afec1bb5e, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f430cf2afec1bb5e(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 56×9(56×9) {0..56}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 56×9(56×9) {0..56}×{0..9} strided
    //   m3 56×56(56×56) {0..56}×{0..56} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[56,9]],"name":"m0","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[56,9]],"name":"m2","ordered":false,"parts":1,"shape":[56,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[56,56]],"name":"m3","ordered":false,"parts":1,"shape":[56,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[56,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[56,56]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[56,56]},{"addressing":"pointer_based","bbox":[[0,0],[56,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[56,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 504 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 81 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 504 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 3136 + 0 + m3_extraOffset];
          float r0[18]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 9; ++v24_i1) {
              float v29_data = __builtin_nontemporal_load(&glb_m0[(v26_lead + (v24_i1 * 56))]);
              r0[(v23_i0 + (v24_i1 * 2))] = v29_data;
            }
          }
          bool v32_g = v22_lead < 24;
          if (v32_g) {
            int32_t v35_lead = v22_lead + 32_i32;
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m0[(v35_lead + (v33_i1 * 56))]);
              r0[(1 + (v33_i1 * 2))] = v38_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m1);
          if (v22_lead < 9) {
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 9; ++v43_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m1[(v22_lead + (v43_i1 * 9))]);
              r1[v43_i1] = v48_data;
            }
          }
          float r3[112]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v186_i0 = 0; v186_i0 < 1; ++v186_i0) {
            int32_t v189_lead = v22_lead + (v186_i0 * 32);
            #pragma unroll
            for (int32_t v187_i1 = 0; v187_i1 < 56; ++v187_i1) {
              float v192_data = __builtin_nontemporal_load(&glb_m3[(v189_lead + (v187_i1 * 56))]);
              r3[(v186_i0 + (v187_i1 * 2))] = v192_data;
            }
          }
          if (v32_g) {
            int32_t v197_lead = v22_lead + 32_i32;
            #pragma unroll
            for (int32_t v195_i1 = 0; v195_i1 < 56; ++v195_i1) {
              float v200_data = __builtin_nontemporal_load(&glb_m3[(v197_lead + (v195_i1 * 56))]);
              r3[(1 + (v195_i1 * 2))] = v200_data;
            }
          }
          float r2[18]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 9)] [(0, 9)]
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
          float v61_data = r0[2];
          float v62_data = r0[4];
          float v63_data = r0[6];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v59_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 3, 0, 0);
          float v68_data = r0[8];
          float v69_data = r0[10];
          float v70_data = r0[12];
          float v71_data = r0[14];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 3, 1, 0);
          float v76_data = r0[16];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v75_acc, 3, 2, 0);
          r2[0] = (v78_acc[0]);
          r2[2] = (v78_acc[1]);
          r2[4] = (v78_acc[2]);
          r2[6] = (v78_acc[3]);
          tensorforge::VectorT<float, 4> v83_acc{};
          float v84_data = r0[1];
          float v85_data = r0[3];
          float v86_data = r0[5];
          float v87_data = r0[7];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v83_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v85_data, v88_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v86_data, v89_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v87_data, v90_acc, 3, 0, 0);
          float v92_data = r0[9];
          float v93_data = r0[11];
          float v94_data = r0[13];
          float v95_data = r0[15];
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v92_data, v91_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v93_data, v96_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v94_data, v97_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v98_acc, 3, 1, 0);
          float v100_data = r0[17];
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v100_data, v99_acc, 3, 2, 0);
          r2[1] = (v102_acc[0]);
          r2[3] = (v102_acc[1]);
          r2[5] = (v102_acc[2]);
          r2[7] = (v102_acc[3]);
          float v107_data = r1[4];
          float v108_data = r1[5];
          float v109_data = r1[6];
          float v110_data = r1[7];
          float v111_tp{};
          float v112_tp{};
          float v113_tp{};
          float v114_tp{};
          tensorforge::transpose4x4b32(v111_tp, v112_tp, v113_tp, v114_tp, v107_data, v108_data, v109_data, v110_data);
          tensorforge::VectorT<float, 4> v115_acc{};
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v60_data, v115_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v61_data, v120_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v62_data, v121_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v63_data, v122_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v68_data, v123_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v69_data, v128_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v70_data, v129_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v71_data, v130_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v76_data, v131_acc, 3, 2, 0);
          r2[8] = (v134_acc[0]);
          r2[10] = (v134_acc[1]);
          r2[12] = (v134_acc[2]);
          r2[14] = (v134_acc[3]);
          tensorforge::VectorT<float, 4> v139_acc{};
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v84_data, v139_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v85_data, v144_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v86_data, v145_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v87_data, v146_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v92_data, v147_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v93_data, v152_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v94_data, v153_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v95_data, v154_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v100_data, v155_acc, 3, 2, 0);
          r2[9] = (v158_acc[0]);
          r2[11] = (v158_acc[1]);
          r2[13] = (v158_acc[2]);
          r2[15] = (v158_acc[3]);
          float v181_acc{};
          float v182_acc{};
          float v183_data = r1[8];
          float v184_bc = tensorforge::broadcast<32, 16, 0>(v183_data);
          tensorforge::fmacdpp16<0>(v181_acc, v184_bc, v60_data);
          tensorforge::fmacdpp16<0>(v182_acc, v184_bc, v84_data);
          tensorforge::fmacdpp16<1>(v181_acc, v184_bc, v61_data);
          tensorforge::fmacdpp16<1>(v182_acc, v184_bc, v85_data);
          tensorforge::fmacdpp16<2>(v181_acc, v184_bc, v62_data);
          tensorforge::fmacdpp16<2>(v182_acc, v184_bc, v86_data);
          tensorforge::fmacdpp16<3>(v181_acc, v184_bc, v63_data);
          tensorforge::fmacdpp16<3>(v182_acc, v184_bc, v87_data);
          tensorforge::fmacdpp16<4>(v181_acc, v184_bc, v68_data);
          tensorforge::fmacdpp16<4>(v182_acc, v184_bc, v92_data);
          tensorforge::fmacdpp16<5>(v181_acc, v184_bc, v69_data);
          tensorforge::fmacdpp16<5>(v182_acc, v184_bc, v93_data);
          tensorforge::fmacdpp16<6>(v181_acc, v184_bc, v70_data);
          tensorforge::fmacdpp16<6>(v182_acc, v184_bc, v94_data);
          tensorforge::fmacdpp16<7>(v181_acc, v184_bc, v71_data);
          tensorforge::fmacdpp16<7>(v182_acc, v184_bc, v95_data);
          tensorforge::fmacdpp16<8>(v181_acc, v184_bc, v76_data);
          tensorforge::fmacdpp16<8>(v182_acc, v184_bc, v100_data);
          r2[16] = v181_acc;
          r2[17] = v182_acc;
          float r4[18]{};
          // r4 = +(r3 * r2) + None
          // [(0, 56), (0, 9)] [(0, 56)]
          float v204_data = r2[0];
          float v205_data = r2[2];
          float v206_data = r2[4];
          float v207_data = r2[6];
          float v208_tp{};
          float v209_tp{};
          float v210_tp{};
          float v211_tp{};
          tensorforge::transpose4x4b32(v208_tp, v209_tp, v210_tp, v211_tp, v204_data, v205_data, v206_data, v207_data);
          float v212_data = r2[1];
          float v213_data = r2[3];
          float v214_data = r2[5];
          float v215_data = r2[7];
          float v216_tp{};
          float v217_tp{};
          float v218_tp{};
          float v219_tp{};
          tensorforge::transpose4x4b32(v216_tp, v217_tp, v218_tp, v219_tp, v212_data, v213_data, v214_data, v215_data);
          tensorforge::VectorT<float, 4> v220_acc{};
          float v221_data = r3[0];
          float v222_data = r3[2];
          float v223_data = r3[4];
          float v224_data = r3[6];
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v221_data, v220_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v222_data, v225_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v223_data, v226_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v224_data, v227_acc, 3, 0, 0);
          float v229_data = r3[8];
          float v230_data = r3[10];
          float v231_data = r3[12];
          float v232_data = r3[14];
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v229_data, v228_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v230_data, v233_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v231_data, v234_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v232_data, v235_acc, 3, 1, 0);
          float v237_data = r3[16];
          float v238_data = r3[18];
          float v239_data = r3[20];
          float v240_data = r3[22];
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v237_data, v236_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v238_data, v241_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v239_data, v242_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v240_data, v243_acc, 3, 2, 0);
          float v245_data = r3[24];
          float v246_data = r3[26];
          float v247_data = r3[28];
          float v248_data = r3[30];
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v245_data, v244_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v246_data, v249_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v247_data, v250_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v248_data, v251_acc, 3, 3, 0);
          float v253_data = r3[32];
          float v254_data = r3[34];
          float v255_data = r3[36];
          float v256_data = r3[38];
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v253_data, v252_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v254_data, v257_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v255_data, v258_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v256_data, v259_acc, 3, 4, 0);
          float v261_data = r3[40];
          float v262_data = r3[42];
          float v263_data = r3[44];
          float v264_data = r3[46];
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v261_data, v260_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v262_data, v265_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v263_data, v266_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v264_data, v267_acc, 3, 5, 0);
          float v269_data = r3[48];
          float v270_data = r3[50];
          float v271_data = r3[52];
          float v272_data = r3[54];
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v269_data, v268_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v270_data, v273_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v271_data, v274_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v272_data, v275_acc, 3, 6, 0);
          float v277_data = r3[56];
          float v278_data = r3[58];
          float v279_data = r3[60];
          float v280_data = r3[62];
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v277_data, v276_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v278_data, v281_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v279_data, v282_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v280_data, v283_acc, 3, 7, 0);
          float v285_data = r3[64];
          float v286_data = r3[66];
          float v287_data = r3[68];
          float v288_data = r3[70];
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v285_data, v284_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v286_data, v289_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v287_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v288_data, v291_acc, 3, 0, 0);
          float v293_data = r3[72];
          float v294_data = r3[74];
          float v295_data = r3[76];
          float v296_data = r3[78];
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v293_data, v292_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v294_data, v297_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v295_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v296_data, v299_acc, 3, 1, 0);
          float v301_data = r3[80];
          float v302_data = r3[82];
          float v303_data = r3[84];
          float v304_data = r3[86];
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v301_data, v300_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v302_data, v305_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v303_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v304_data, v307_acc, 3, 2, 0);
          float v309_data = r3[88];
          float v310_data = r3[90];
          float v311_data = r3[92];
          float v312_data = r3[94];
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v309_data, v308_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v310_data, v313_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v311_data, v314_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v312_data, v315_acc, 3, 3, 0);
          float v317_data = r3[96];
          float v318_data = r3[98];
          float v319_data = r3[100];
          float v320_data = r3[102];
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v317_data, v316_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v318_data, v321_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v319_data, v322_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v320_data, v323_acc, 3, 4, 0);
          float v325_data = r3[104];
          float v326_data = r3[106];
          float v327_data = r3[108];
          float v328_data = r3[110];
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v325_data, v324_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v326_data, v329_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v327_data, v330_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v328_data, v331_acc, 3, 5, 0);
          r4[0] = (v332_acc[0]);
          r4[2] = (v332_acc[1]);
          r4[4] = (v332_acc[2]);
          r4[6] = (v332_acc[3]);
          tensorforge::VectorT<float, 4> v337_acc{};
          float v338_data = r3[1];
          float v339_data = r3[3];
          float v340_data = r3[5];
          float v341_data = r3[7];
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v338_data, v337_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v339_data, v342_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v340_data, v343_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v341_data, v344_acc, 3, 0, 0);
          float v346_data = r3[9];
          float v347_data = r3[11];
          float v348_data = r3[13];
          float v349_data = r3[15];
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v346_data, v345_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v347_data, v350_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v348_data, v351_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v349_data, v352_acc, 3, 1, 0);
          float v354_data = r3[17];
          float v355_data = r3[19];
          float v356_data = r3[21];
          float v357_data = r3[23];
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v354_data, v353_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v355_data, v358_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v356_data, v359_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v357_data, v360_acc, 3, 2, 0);
          float v362_data = r3[25];
          float v363_data = r3[27];
          float v364_data = r3[29];
          float v365_data = r3[31];
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v362_data, v361_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v363_data, v366_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v364_data, v367_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v365_data, v368_acc, 3, 3, 0);
          float v370_data = r3[33];
          float v371_data = r3[35];
          float v372_data = r3[37];
          float v373_data = r3[39];
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v370_data, v369_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v371_data, v374_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v372_data, v375_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v373_data, v376_acc, 3, 4, 0);
          float v378_data = r3[41];
          float v379_data = r3[43];
          float v380_data = r3[45];
          float v381_data = r3[47];
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v378_data, v377_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v379_data, v382_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v380_data, v383_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v381_data, v384_acc, 3, 5, 0);
          float v386_data = r3[49];
          float v387_data = r3[51];
          float v388_data = r3[53];
          float v389_data = r3[55];
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v386_data, v385_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v387_data, v390_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v388_data, v391_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v389_data, v392_acc, 3, 6, 0);
          float v394_data = r3[57];
          float v395_data = r3[59];
          float v396_data = r3[61];
          float v397_data = r3[63];
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v394_data, v393_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v395_data, v398_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v396_data, v399_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v397_data, v400_acc, 3, 7, 0);
          float v402_data = r3[65];
          float v403_data = r3[67];
          float v404_data = r3[69];
          float v405_data = r3[71];
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v402_data, v401_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v403_data, v406_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v404_data, v407_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v405_data, v408_acc, 3, 0, 0);
          float v410_data = r3[73];
          float v411_data = r3[75];
          float v412_data = r3[77];
          float v413_data = r3[79];
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v410_data, v409_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v411_data, v414_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v412_data, v415_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v413_data, v416_acc, 3, 1, 0);
          float v418_data = r3[81];
          float v419_data = r3[83];
          float v420_data = r3[85];
          float v421_data = r3[87];
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v418_data, v417_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v419_data, v422_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v420_data, v423_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v421_data, v424_acc, 3, 2, 0);
          float v426_data = r3[89];
          float v427_data = r3[91];
          float v428_data = r3[93];
          float v429_data = r3[95];
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v426_data, v425_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v427_data, v430_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v428_data, v431_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v429_data, v432_acc, 3, 3, 0);
          float v434_data = r3[97];
          float v435_data = r3[99];
          float v436_data = r3[101];
          float v437_data = r3[103];
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v434_data, v433_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v435_data, v438_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v436_data, v439_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v437_data, v440_acc, 3, 4, 0);
          float v442_data = r3[105];
          float v443_data = r3[107];
          float v444_data = r3[109];
          float v445_data = r3[111];
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v442_data, v441_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v443_data, v446_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v444_data, v447_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v445_data, v448_acc, 3, 5, 0);
          r4[1] = (v449_acc[0]);
          r4[3] = (v449_acc[1]);
          r4[5] = (v449_acc[2]);
          r4[7] = (v449_acc[3]);
          float v454_data = r2[8];
          float v455_data = r2[10];
          float v456_data = r2[12];
          float v457_data = r2[14];
          float v458_tp{};
          float v459_tp{};
          float v460_tp{};
          float v461_tp{};
          tensorforge::transpose4x4b32(v458_tp, v459_tp, v460_tp, v461_tp, v454_data, v455_data, v456_data, v457_data);
          float v462_data = r2[9];
          float v463_data = r2[11];
          float v464_data = r2[13];
          float v465_data = r2[15];
          float v466_tp{};
          float v467_tp{};
          float v468_tp{};
          float v469_tp{};
          tensorforge::transpose4x4b32(v466_tp, v467_tp, v468_tp, v469_tp, v462_data, v463_data, v464_data, v465_data);
          tensorforge::VectorT<float, 4> v470_acc{};
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v221_data, v470_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v222_data, v475_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v223_data, v476_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v224_data, v477_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v229_data, v478_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v230_data, v483_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v231_data, v484_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v232_data, v485_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v237_data, v486_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v238_data, v491_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v239_data, v492_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v240_data, v493_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v245_data, v494_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v246_data, v499_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v247_data, v500_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v248_data, v501_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v253_data, v502_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v254_data, v507_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v255_data, v508_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v256_data, v509_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v261_data, v510_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v262_data, v515_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v263_data, v516_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v264_data, v517_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v269_data, v518_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v270_data, v523_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v271_data, v524_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v272_data, v525_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v277_data, v526_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v278_data, v531_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v279_data, v532_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v280_data, v533_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v285_data, v534_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v286_data, v539_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v541_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v287_data, v540_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v288_data, v541_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v547_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v293_data, v542_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v548_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v294_data, v547_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v295_data, v548_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v296_data, v549_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v301_data, v550_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v302_data, v555_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v303_data, v556_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v304_data, v557_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v309_data, v558_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v310_data, v563_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v565_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v311_data, v564_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v312_data, v565_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v571_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v317_data, v566_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v318_data, v571_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v319_data, v572_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v320_data, v573_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v579_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v325_data, v574_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v326_data, v579_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v327_data, v580_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v328_data, v581_acc, 3, 5, 0);
          r4[8] = (v582_acc[0]);
          r4[10] = (v582_acc[1]);
          r4[12] = (v582_acc[2]);
          r4[14] = (v582_acc[3]);
          tensorforge::VectorT<float, 4> v587_acc{};
          tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v338_data, v587_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v593_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v339_data, v592_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v340_data, v593_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v341_data, v594_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v346_data, v595_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v347_data, v600_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v348_data, v601_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v603_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v349_data, v602_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v354_data, v603_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v355_data, v608_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v356_data, v609_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v357_data, v610_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v362_data, v611_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v363_data, v616_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v364_data, v617_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v365_data, v618_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v370_data, v619_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v625_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v371_data, v624_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v372_data, v625_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v373_data, v626_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v632_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v378_data, v627_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v633_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v379_data, v632_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v634_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v380_data, v633_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v381_data, v634_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v386_data, v635_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v641_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v387_data, v640_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v388_data, v641_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v643_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v389_data, v642_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v394_data, v643_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v395_data, v648_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v396_data, v649_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v397_data, v650_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v402_data, v651_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v403_data, v656_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v404_data, v657_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v405_data, v658_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v410_data, v659_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v411_data, v664_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v412_data, v665_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v413_data, v666_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v418_data, v667_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v673_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v419_data, v672_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v420_data, v673_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v421_data, v674_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v680_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v426_data, v675_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v427_data, v680_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v428_data, v681_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v429_data, v682_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v434_data, v683_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v689_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v435_data, v688_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v436_data, v689_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v437_data, v690_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v466_tp, v442_data, v691_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v697_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v467_tp, v443_data, v696_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v468_tp, v444_data, v697_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v469_tp, v445_data, v698_acc, 3, 5, 0);
          r4[9] = (v699_acc[0]);
          r4[11] = (v699_acc[1]);
          r4[13] = (v699_acc[2]);
          r4[15] = (v699_acc[3]);
          float v816_acc{};
          float v817_acc{};
          float v818_data = r2[16];
          float v819_data = r2[17];
          float v820_bc = tensorforge::broadcast<32, 16, 0>(v818_data);
          tensorforge::fmacdpp16<0>(v816_acc, v820_bc, v221_data);
          tensorforge::fmacdpp16<0>(v817_acc, v820_bc, v338_data);
          tensorforge::fmacdpp16<1>(v816_acc, v820_bc, v222_data);
          tensorforge::fmacdpp16<1>(v817_acc, v820_bc, v339_data);
          tensorforge::fmacdpp16<2>(v816_acc, v820_bc, v223_data);
          tensorforge::fmacdpp16<2>(v817_acc, v820_bc, v340_data);
          tensorforge::fmacdpp16<3>(v816_acc, v820_bc, v224_data);
          tensorforge::fmacdpp16<3>(v817_acc, v820_bc, v341_data);
          tensorforge::fmacdpp16<4>(v816_acc, v820_bc, v229_data);
          tensorforge::fmacdpp16<4>(v817_acc, v820_bc, v346_data);
          tensorforge::fmacdpp16<5>(v816_acc, v820_bc, v230_data);
          tensorforge::fmacdpp16<5>(v817_acc, v820_bc, v347_data);
          tensorforge::fmacdpp16<6>(v816_acc, v820_bc, v231_data);
          tensorforge::fmacdpp16<6>(v817_acc, v820_bc, v348_data);
          tensorforge::fmacdpp16<7>(v816_acc, v820_bc, v232_data);
          tensorforge::fmacdpp16<7>(v817_acc, v820_bc, v349_data);
          tensorforge::fmacdpp16<8>(v816_acc, v820_bc, v237_data);
          tensorforge::fmacdpp16<8>(v817_acc, v820_bc, v354_data);
          tensorforge::fmacdpp16<9>(v816_acc, v820_bc, v238_data);
          tensorforge::fmacdpp16<9>(v817_acc, v820_bc, v355_data);
          tensorforge::fmacdpp16<10>(v816_acc, v820_bc, v239_data);
          tensorforge::fmacdpp16<10>(v817_acc, v820_bc, v356_data);
          tensorforge::fmacdpp16<11>(v816_acc, v820_bc, v240_data);
          tensorforge::fmacdpp16<11>(v817_acc, v820_bc, v357_data);
          tensorforge::fmacdpp16<12>(v816_acc, v820_bc, v245_data);
          tensorforge::fmacdpp16<12>(v817_acc, v820_bc, v362_data);
          tensorforge::fmacdpp16<13>(v816_acc, v820_bc, v246_data);
          tensorforge::fmacdpp16<13>(v817_acc, v820_bc, v363_data);
          tensorforge::fmacdpp16<14>(v816_acc, v820_bc, v247_data);
          tensorforge::fmacdpp16<14>(v817_acc, v820_bc, v364_data);
          tensorforge::fmacdpp16<15>(v816_acc, v820_bc, v248_data);
          tensorforge::fmacdpp16<15>(v817_acc, v820_bc, v365_data);
          float v821_bc = tensorforge::broadcast<32, 16, 1>(v818_data);
          tensorforge::fmacdpp16<0>(v816_acc, v821_bc, v253_data);
          tensorforge::fmacdpp16<0>(v817_acc, v821_bc, v370_data);
          tensorforge::fmacdpp16<1>(v816_acc, v821_bc, v254_data);
          tensorforge::fmacdpp16<1>(v817_acc, v821_bc, v371_data);
          tensorforge::fmacdpp16<2>(v816_acc, v821_bc, v255_data);
          tensorforge::fmacdpp16<2>(v817_acc, v821_bc, v372_data);
          tensorforge::fmacdpp16<3>(v816_acc, v821_bc, v256_data);
          tensorforge::fmacdpp16<3>(v817_acc, v821_bc, v373_data);
          tensorforge::fmacdpp16<4>(v816_acc, v821_bc, v261_data);
          tensorforge::fmacdpp16<4>(v817_acc, v821_bc, v378_data);
          tensorforge::fmacdpp16<5>(v816_acc, v821_bc, v262_data);
          tensorforge::fmacdpp16<5>(v817_acc, v821_bc, v379_data);
          tensorforge::fmacdpp16<6>(v816_acc, v821_bc, v263_data);
          tensorforge::fmacdpp16<6>(v817_acc, v821_bc, v380_data);
          tensorforge::fmacdpp16<7>(v816_acc, v821_bc, v264_data);
          tensorforge::fmacdpp16<7>(v817_acc, v821_bc, v381_data);
          tensorforge::fmacdpp16<8>(v816_acc, v821_bc, v269_data);
          tensorforge::fmacdpp16<8>(v817_acc, v821_bc, v386_data);
          tensorforge::fmacdpp16<9>(v816_acc, v821_bc, v270_data);
          tensorforge::fmacdpp16<9>(v817_acc, v821_bc, v387_data);
          tensorforge::fmacdpp16<10>(v816_acc, v821_bc, v271_data);
          tensorforge::fmacdpp16<10>(v817_acc, v821_bc, v388_data);
          tensorforge::fmacdpp16<11>(v816_acc, v821_bc, v272_data);
          tensorforge::fmacdpp16<11>(v817_acc, v821_bc, v389_data);
          tensorforge::fmacdpp16<12>(v816_acc, v821_bc, v277_data);
          tensorforge::fmacdpp16<12>(v817_acc, v821_bc, v394_data);
          tensorforge::fmacdpp16<13>(v816_acc, v821_bc, v278_data);
          tensorforge::fmacdpp16<13>(v817_acc, v821_bc, v395_data);
          tensorforge::fmacdpp16<14>(v816_acc, v821_bc, v279_data);
          tensorforge::fmacdpp16<14>(v817_acc, v821_bc, v396_data);
          tensorforge::fmacdpp16<15>(v816_acc, v821_bc, v280_data);
          tensorforge::fmacdpp16<15>(v817_acc, v821_bc, v397_data);
          float v822_bc = tensorforge::broadcast<32, 16, 0>(v819_data);
          tensorforge::fmacdpp16<0>(v816_acc, v822_bc, v285_data);
          tensorforge::fmacdpp16<0>(v817_acc, v822_bc, v402_data);
          tensorforge::fmacdpp16<1>(v816_acc, v822_bc, v286_data);
          tensorforge::fmacdpp16<1>(v817_acc, v822_bc, v403_data);
          tensorforge::fmacdpp16<2>(v816_acc, v822_bc, v287_data);
          tensorforge::fmacdpp16<2>(v817_acc, v822_bc, v404_data);
          tensorforge::fmacdpp16<3>(v816_acc, v822_bc, v288_data);
          tensorforge::fmacdpp16<3>(v817_acc, v822_bc, v405_data);
          tensorforge::fmacdpp16<4>(v816_acc, v822_bc, v293_data);
          tensorforge::fmacdpp16<4>(v817_acc, v822_bc, v410_data);
          tensorforge::fmacdpp16<5>(v816_acc, v822_bc, v294_data);
          tensorforge::fmacdpp16<5>(v817_acc, v822_bc, v411_data);
          tensorforge::fmacdpp16<6>(v816_acc, v822_bc, v295_data);
          tensorforge::fmacdpp16<6>(v817_acc, v822_bc, v412_data);
          tensorforge::fmacdpp16<7>(v816_acc, v822_bc, v296_data);
          tensorforge::fmacdpp16<7>(v817_acc, v822_bc, v413_data);
          tensorforge::fmacdpp16<8>(v816_acc, v822_bc, v301_data);
          tensorforge::fmacdpp16<8>(v817_acc, v822_bc, v418_data);
          tensorforge::fmacdpp16<9>(v816_acc, v822_bc, v302_data);
          tensorforge::fmacdpp16<9>(v817_acc, v822_bc, v419_data);
          tensorforge::fmacdpp16<10>(v816_acc, v822_bc, v303_data);
          tensorforge::fmacdpp16<10>(v817_acc, v822_bc, v420_data);
          tensorforge::fmacdpp16<11>(v816_acc, v822_bc, v304_data);
          tensorforge::fmacdpp16<11>(v817_acc, v822_bc, v421_data);
          tensorforge::fmacdpp16<12>(v816_acc, v822_bc, v309_data);
          tensorforge::fmacdpp16<12>(v817_acc, v822_bc, v426_data);
          tensorforge::fmacdpp16<13>(v816_acc, v822_bc, v310_data);
          tensorforge::fmacdpp16<13>(v817_acc, v822_bc, v427_data);
          tensorforge::fmacdpp16<14>(v816_acc, v822_bc, v311_data);
          tensorforge::fmacdpp16<14>(v817_acc, v822_bc, v428_data);
          tensorforge::fmacdpp16<15>(v816_acc, v822_bc, v312_data);
          tensorforge::fmacdpp16<15>(v817_acc, v822_bc, v429_data);
          float v823_bc = tensorforge::broadcast<32, 16, 1>(v819_data);
          tensorforge::fmacdpp16<0>(v816_acc, v823_bc, v317_data);
          tensorforge::fmacdpp16<0>(v817_acc, v823_bc, v434_data);
          tensorforge::fmacdpp16<1>(v816_acc, v823_bc, v318_data);
          tensorforge::fmacdpp16<1>(v817_acc, v823_bc, v435_data);
          tensorforge::fmacdpp16<2>(v816_acc, v823_bc, v319_data);
          tensorforge::fmacdpp16<2>(v817_acc, v823_bc, v436_data);
          tensorforge::fmacdpp16<3>(v816_acc, v823_bc, v320_data);
          tensorforge::fmacdpp16<3>(v817_acc, v823_bc, v437_data);
          tensorforge::fmacdpp16<4>(v816_acc, v823_bc, v325_data);
          tensorforge::fmacdpp16<4>(v817_acc, v823_bc, v442_data);
          tensorforge::fmacdpp16<5>(v816_acc, v823_bc, v326_data);
          tensorforge::fmacdpp16<5>(v817_acc, v823_bc, v443_data);
          tensorforge::fmacdpp16<6>(v816_acc, v823_bc, v327_data);
          tensorforge::fmacdpp16<6>(v817_acc, v823_bc, v444_data);
          tensorforge::fmacdpp16<7>(v816_acc, v823_bc, v328_data);
          tensorforge::fmacdpp16<7>(v817_acc, v823_bc, v445_data);
          r4[16] = v816_acc;
          r4[17] = v817_acc;
          // glb_m2 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v824_i0 = 0; v824_i0 < 1; ++v824_i0) {
            int32_t v830_lead = v22_lead + (v824_i0 * 32);
            #pragma unroll
            for (int32_t v825_i1 = 0; v825_i1 < 9; ++v825_i1) {
              float v828_data = r4[(v824_i0 + (v825_i1 * 2))];
              glb_m2[(v830_lead + (v825_i1 * 56))] = v828_data;
            }
          }
          if (v32_g) {
            int32_t v838_lead = v22_lead + 32_i32;
            #pragma unroll
            for (int32_t v833_i1 = 0; v833_i1 < 9; ++v833_i1) {
              float v836_data = r4[(1 + (v833_i1 * 2))];
              glb_m2[(v838_lead + (v833_i1 * 56))] = v836_data;
            }
          }
        }
      }
    }
  }
}

