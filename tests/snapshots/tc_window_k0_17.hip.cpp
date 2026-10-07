// === base name ===
kernel_11a4d2eeade82ad4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_11a4d2eeade82ad4 = {{16, 16, 1}, 16, 16, 1, 16, 2304, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_11a4d2eeade82ad4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_11a4d2eeade82ad4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_11a4d2eeade82ad4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_11a4d2eeade82ad4, block.x * block.y * block.z, 576 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (576 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_11a4d2eeade82ad4, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (576 * sizeof(float)));
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
  config.sharedMemBytes = 576 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_11a4d2eeade82ad4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_11a4d2eeade82ad4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_11a4d2eeade82ad4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_11a4d2eeade82ad4, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_11a4d2eeade82ad4(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2304 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×17) {0..16}×{0..17} none
    //   m2 20×9(17×9) {0..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":576}],"shared_bytes":2304,"shared_elements":576,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 320];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 16) {
        float v10_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v10_ld;
      }
      __syncthreads();
      size_t v12_batchIdLane0 = threadIdx.y % 4;
      int32_t v28_lead = threadIdx.x % 16;
      bool v38_g = v28_lead < 1;
      int32_t v69_a = v28_lead + ((threadIdx.y % 4) * 16);
      int32_t v76_a = v69_a + 64;
      int32_t v82_a = v69_a + 128;
      int32_t v88_a = v69_a + 192;
      int32_t v94_a = v28_lead + 256;
      int32_t v158_a = v28_lead + 16;
      int32_t v160_a = v28_lead + 32;
      int32_t v162_a = v28_lead + 48;
      int32_t v164_a = v28_lead + 64;
      int32_t v166_a = v28_lead + 80;
      int32_t v168_a = v28_lead + 96;
      int32_t v170_a = v28_lead + 112;
      int32_t v172_a = v28_lead + 128;
      int32_t v174_a = v28_lead + 144;
      int32_t v176_a = v28_lead + 160;
      int32_t v178_a = v28_lead + 176;
      int32_t v180_a = v28_lead + 192;
      int32_t v182_a = v28_lead + 208;
      int32_t v184_a = v28_lead + 224;
      int32_t v186_a = v28_lead + 240;
      for (size_t v13_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v13_batchIdGroup0 < numElements0; v13_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v14_row = v13_batchIdGroup0 + v12_batchIdLane0;
        const bool batchIdActive0 = v14_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v14_row]));
        size_t v16_batchId0 = batchIdActive0 ? v14_row : v13_batchIdGroup0;
        size_t v17_ahead1 = v16_batchId0 + (gridDim.x * blockDim.y);
        size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v16_batchId0 * 144 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v16_batchId0 * 153 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
          int32_t v32_lead = v28_lead + (v29_i0 * 16);
          #pragma unroll
          for (int32_t v30_i1 = 0; v30_i1 < 9; ++v30_i1) {
            float v35_data = __builtin_nontemporal_load(&glb_m2[(v32_lead + (v30_i1 * 17))]);
            r0[(v29_i0 + (v30_i1 * 2))] = v35_data;
          }
        }
        if (v38_g) {
          int32_t v41_lead = v28_lead + 16_i32;
          #pragma unroll
          for (int32_t v39_i1 = 0; v39_i1 < 9; ++v39_i1) {
            float v44_data = __builtin_nontemporal_load(&glb_m2[(v41_lead + (v39_i1 * 17))]);
            r0[(1 + (v39_i1 * 2))] = v44_data;
          }
        }
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 9)] [(0, 17)]
        float v48_data = r0[0];
        float v49_data = r0[2];
        float v50_data = r0[4];
        float v51_data = r0[6];
        float v52_tp{};
        float v53_tp{};
        float v54_tp{};
        float v55_tp{};
        tensorforge::transpose4x4b32(v52_tp, v53_tp, v54_tp, v55_tp, v48_data, v49_data, v50_data, v51_data);
        float v56_data = r0[1];
        float v57_data = r0[3];
        float v58_data = r0[5];
        float v59_data = r0[7];
        float v60_tp{};
        float v61_tp{};
        float v62_tp{};
        float v63_tp{};
        tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v56_data, v57_data, v58_data, v59_data);
        tensorforge::VectorT<float, 4> v64_acc{};
        float v71_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v71_data, v64_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v71_data, v72_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v71_data, v73_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v71_data, v74_acc, 2, 0, 7);
        float v77_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v77_data, v75_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v77_data, v78_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v77_data, v79_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v77_data, v80_acc, 2, 1, 7);
        float v83_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v83_data, v81_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v83_data, v84_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v83_data, v85_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v83_data, v86_acc, 2, 2, 7);
        float v89_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v89_data, v87_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v89_data, v90_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v89_data, v91_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v89_data, v92_acc, 2, 3, 7);
        float v95_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v95_data, v93_acc, 2, 0, 0);
        r1[0] = (v96_acc[0]);
        r1[1] = (v96_acc[1]);
        r1[2] = (v96_acc[2]);
        r1[3] = (v96_acc[3]);
        float v101_data = r0[8];
        float v102_data = r0[10];
        float v103_data = r0[12];
        float v104_data = r0[14];
        float v105_tp{};
        float v106_tp{};
        float v107_tp{};
        float v108_tp{};
        tensorforge::transpose4x4b32(v105_tp, v106_tp, v107_tp, v108_tp, v101_data, v102_data, v103_data, v104_data);
        float v109_data = r0[9];
        float v110_data = r0[11];
        float v111_data = r0[13];
        float v112_data = r0[15];
        float v113_tp{};
        float v114_tp{};
        float v115_tp{};
        float v116_tp{};
        tensorforge::transpose4x4b32(v113_tp, v114_tp, v115_tp, v116_tp, v109_data, v110_data, v111_data, v112_data);
        tensorforge::VectorT<float, 4> v117_acc{};
        float v124_data = glb_m1[v69_a];
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v124_data, v117_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v124_data, v125_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v124_data, v126_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v124_data, v127_acc, 2, 0, 7);
        float v130_data = glb_m1[v76_a];
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v130_data, v128_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v130_data, v131_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v130_data, v132_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v130_data, v133_acc, 2, 1, 7);
        float v136_data = glb_m1[v82_a];
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v136_data, v134_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v136_data, v137_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v136_data, v138_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v136_data, v139_acc, 2, 2, 7);
        float v142_data = glb_m1[v88_a];
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v142_data, v140_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v142_data, v143_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v142_data, v144_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v142_data, v145_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v95_data, v146_acc, 2, 0, 0);
        r1[4] = (v149_acc[0]);
        r1[5] = (v149_acc[1]);
        r1[6] = (v149_acc[2]);
        r1[7] = (v149_acc[3]);
        float v157_data = glb_m1[v28_lead];
        float v159_data = glb_m1[v158_a];
        float v161_data = glb_m1[v160_a];
        float v163_data = glb_m1[v162_a];
        float v165_data = glb_m1[v164_a];
        float v167_data = glb_m1[v166_a];
        float v169_data = glb_m1[v168_a];
        float v171_data = glb_m1[v170_a];
        float v173_data = glb_m1[v172_a];
        float v175_data = glb_m1[v174_a];
        float v177_data = glb_m1[v176_a];
        float v179_data = glb_m1[v178_a];
        float v181_data = glb_m1[v180_a];
        float v183_data = glb_m1[v182_a];
        float v185_data = glb_m1[v184_a];
        float v187_data = glb_m1[v186_a];
        float v190_acc{};
        float v191_data = r0[16];
        float v192_data = r0[17];
        tensorforge::fmacdpp16<0>(v190_acc, v191_data, v157_data);
        tensorforge::fmacdpp16<1>(v190_acc, v191_data, v159_data);
        tensorforge::fmacdpp16<2>(v190_acc, v191_data, v161_data);
        tensorforge::fmacdpp16<3>(v190_acc, v191_data, v163_data);
        tensorforge::fmacdpp16<4>(v190_acc, v191_data, v165_data);
        tensorforge::fmacdpp16<5>(v190_acc, v191_data, v167_data);
        tensorforge::fmacdpp16<6>(v190_acc, v191_data, v169_data);
        tensorforge::fmacdpp16<7>(v190_acc, v191_data, v171_data);
        tensorforge::fmacdpp16<8>(v190_acc, v191_data, v173_data);
        tensorforge::fmacdpp16<9>(v190_acc, v191_data, v175_data);
        tensorforge::fmacdpp16<10>(v190_acc, v191_data, v177_data);
        tensorforge::fmacdpp16<11>(v190_acc, v191_data, v179_data);
        tensorforge::fmacdpp16<12>(v190_acc, v191_data, v181_data);
        tensorforge::fmacdpp16<13>(v190_acc, v191_data, v183_data);
        tensorforge::fmacdpp16<14>(v190_acc, v191_data, v185_data);
        tensorforge::fmacdpp16<15>(v190_acc, v191_data, v187_data);
        tensorforge::fmacdpp16<0>(v190_acc, v192_data, v95_data);
        r1[8] = v190_acc;
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v193_i0 = 0; v193_i0 < 1; ++v193_i0) {
          #pragma unroll
          for (int32_t v194_i1 = 0; v194_i1 < 9; ++v194_i1) {
            float v196_data = r1[(v193_i0 + v194_i1)];
            if (batchIdActive0) {
              glb_m0[((v28_lead + (v193_i0 * 16)) + (v194_i1 * 16))] = v196_data;
            }
          }
        }
      }
    }
  }
}

