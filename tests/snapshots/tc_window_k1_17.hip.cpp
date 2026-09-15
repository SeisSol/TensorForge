// === base name ===
kernel_082ef619972bdeb4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_082ef619972bdeb4 = {{16, 16, 1}, 16, 16, 1, 16, 2304, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_082ef619972bdeb4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_082ef619972bdeb4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_082ef619972bdeb4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_082ef619972bdeb4, block.x * block.y * block.z, 576 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (576 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_082ef619972bdeb4, block.x * block.y * block.z, 0));
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
void launcher_kernel_082ef619972bdeb4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_082ef619972bdeb4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_082ef619972bdeb4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_082ef619972bdeb4, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_082ef619972bdeb4(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2304 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×17) {0..16}×{1..18} none
    //   m2 20×9(17×9) {1..18}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":576}],"shared_bytes":2304,"shared_elements":576,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 320];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 16) {
        float v6_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v6_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v8_batchIdLane0 = threadIdx.y % 4;
      int32_t v25_lead = threadIdx.x % 16;
      bool v26_g = v25_lead >= 1;
      bool v36_g = v25_lead < 2;
      int32_t v68_a = v25_lead + ((threadIdx.y % 4) * 16);
      int32_t v75_a = v68_a + 64;
      int32_t v81_a = v68_a + 128;
      int32_t v87_a = v68_a + 192;
      int32_t v93_a = v25_lead + 256;
      int32_t v157_a = v25_lead + 16;
      int32_t v159_a = v25_lead + 32;
      int32_t v161_a = v25_lead + 48;
      int32_t v163_a = v25_lead + 64;
      int32_t v165_a = v25_lead + 80;
      int32_t v167_a = v25_lead + 96;
      int32_t v169_a = v25_lead + 112;
      int32_t v171_a = v25_lead + 128;
      int32_t v173_a = v25_lead + 144;
      int32_t v175_a = v25_lead + 160;
      int32_t v177_a = v25_lead + 176;
      int32_t v179_a = v25_lead + 192;
      int32_t v181_a = v25_lead + 208;
      int32_t v183_a = v25_lead + 224;
      int32_t v185_a = v25_lead + 240;
      for (size_t v9_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v9_batchIdGroup0 < numElements0; v9_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v10_row = v9_batchIdGroup0 + v8_batchIdLane0;
        const bool batchIdActive0 = v10_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v10_row]));
        size_t v12_batchId0 = batchIdActive0 ? v10_row : v9_batchIdGroup0;
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v16_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v12_batchId0 * 144 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v12_batchId0 * 153 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        if (v26_g) {
          int32_t v30_a = v25_lead - 1;
          #pragma unroll
          for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
            float v33_data = __builtin_nontemporal_load(&glb_m2[(v30_a + (v27_i1 * 17))]);
            r0[(v27_i1 * 2)] = v33_data;
          }
        }
        if (v36_g) {
          int32_t v40_a = (v25_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v37_i1 = 0; v37_i1 < 9; ++v37_i1) {
            float v43_data = __builtin_nontemporal_load(&glb_m2[(v40_a + (v37_i1 * 17))]);
            r0[(1 + (v37_i1 * 2))] = v43_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 9)] [(1, 18)]
        float v47_data = r0[0];
        float v48_data = r0[2];
        float v49_data = r0[4];
        float v50_data = r0[6];
        float v51_tp{};
        float v52_tp{};
        float v53_tp{};
        float v54_tp{};
        tensorforge::transpose4x4b32(v51_tp, v52_tp, v53_tp, v54_tp, v47_data, v48_data, v49_data, v50_data);
        float v55_data = r0[1];
        float v56_data = r0[3];
        float v57_data = r0[5];
        float v58_data = r0[7];
        float v59_tp{};
        float v60_tp{};
        float v61_tp{};
        float v62_tp{};
        tensorforge::transpose4x4b32(v59_tp, v60_tp, v61_tp, v62_tp, v55_data, v56_data, v57_data, v58_data);
        tensorforge::VectorT<float, 4> v63_acc{};
        float v70_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v70_data, v63_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v70_data, v71_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v70_data, v72_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v70_data, v73_acc, 2, 1, 7);
        float v76_data = glb_m1[v75_a];
        tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v76_data, v74_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v76_data, v77_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v76_data, v78_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v76_data, v79_acc, 2, 2, 7);
        float v82_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v82_data, v80_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v82_data, v83_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v82_data, v84_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v82_data, v85_acc, 2, 3, 7);
        float v88_data = glb_m1[v87_a];
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v88_data, v86_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v88_data, v89_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v88_data, v90_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v88_data, v91_acc, 2, 0, 7);
        float v94_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v94_data, v92_acc, 2, 0, 0);
        r1[0] = (v95_acc[0]);
        r1[1] = (v95_acc[1]);
        r1[2] = (v95_acc[2]);
        r1[3] = (v95_acc[3]);
        float v100_data = r0[8];
        float v101_data = r0[10];
        float v102_data = r0[12];
        float v103_data = r0[14];
        float v104_tp{};
        float v105_tp{};
        float v106_tp{};
        float v107_tp{};
        tensorforge::transpose4x4b32(v104_tp, v105_tp, v106_tp, v107_tp, v100_data, v101_data, v102_data, v103_data);
        float v108_data = r0[9];
        float v109_data = r0[11];
        float v110_data = r0[13];
        float v111_data = r0[15];
        float v112_tp{};
        float v113_tp{};
        float v114_tp{};
        float v115_tp{};
        tensorforge::transpose4x4b32(v112_tp, v113_tp, v114_tp, v115_tp, v108_data, v109_data, v110_data, v111_data);
        tensorforge::VectorT<float, 4> v116_acc{};
        float v123_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v123_data, v116_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v123_data, v124_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v123_data, v125_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v123_data, v126_acc, 2, 1, 7);
        float v129_data = glb_m1[v75_a];
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v129_data, v127_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v129_data, v130_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v129_data, v131_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v129_data, v132_acc, 2, 2, 7);
        float v135_data = glb_m1[v81_a];
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v135_data, v133_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v135_data, v136_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v135_data, v137_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v135_data, v138_acc, 2, 3, 7);
        float v141_data = glb_m1[v87_a];
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v141_data, v139_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v141_data, v142_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v141_data, v143_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v141_data, v144_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v94_data, v145_acc, 2, 0, 0);
        r1[4] = (v148_acc[0]);
        r1[5] = (v148_acc[1]);
        r1[6] = (v148_acc[2]);
        r1[7] = (v148_acc[3]);
        float v156_data = glb_m1[v25_lead];
        float v158_data = glb_m1[v157_a];
        float v160_data = glb_m1[v159_a];
        float v162_data = glb_m1[v161_a];
        float v164_data = glb_m1[v163_a];
        float v166_data = glb_m1[v165_a];
        float v168_data = glb_m1[v167_a];
        float v170_data = glb_m1[v169_a];
        float v172_data = glb_m1[v171_a];
        float v174_data = glb_m1[v173_a];
        float v176_data = glb_m1[v175_a];
        float v178_data = glb_m1[v177_a];
        float v180_data = glb_m1[v179_a];
        float v182_data = glb_m1[v181_a];
        float v184_data = glb_m1[v183_a];
        float v186_data = glb_m1[v185_a];
        float v189_acc{};
        float v190_data = r0[16];
        float v191_data = r0[17];
        tensorforge::fmacdpp16<1>(v189_acc, v190_data, v156_data);
        tensorforge::fmacdpp16<2>(v189_acc, v190_data, v158_data);
        tensorforge::fmacdpp16<3>(v189_acc, v190_data, v160_data);
        tensorforge::fmacdpp16<4>(v189_acc, v190_data, v162_data);
        tensorforge::fmacdpp16<5>(v189_acc, v190_data, v164_data);
        tensorforge::fmacdpp16<6>(v189_acc, v190_data, v166_data);
        tensorforge::fmacdpp16<7>(v189_acc, v190_data, v168_data);
        tensorforge::fmacdpp16<8>(v189_acc, v190_data, v170_data);
        tensorforge::fmacdpp16<9>(v189_acc, v190_data, v172_data);
        tensorforge::fmacdpp16<10>(v189_acc, v190_data, v174_data);
        tensorforge::fmacdpp16<11>(v189_acc, v190_data, v176_data);
        tensorforge::fmacdpp16<12>(v189_acc, v190_data, v178_data);
        tensorforge::fmacdpp16<13>(v189_acc, v190_data, v180_data);
        tensorforge::fmacdpp16<14>(v189_acc, v190_data, v182_data);
        tensorforge::fmacdpp16<15>(v189_acc, v190_data, v184_data);
        tensorforge::fmacdpp16<0>(v189_acc, v191_data, v186_data);
        tensorforge::fmacdpp16<1>(v189_acc, v191_data, v94_data);
        r1[8] = v189_acc;
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v192_i0 = 0; v192_i0 < 1; ++v192_i0) {
          #pragma unroll
          for (int32_t v193_i1 = 0; v193_i1 < 9; ++v193_i1) {
            float v195_data = r1[(v192_i0 + v193_i1)];
            if (batchIdActive0) {
              glb_m0[((v25_lead + (v192_i0 * 16)) + (v193_i1 * 16))] = v195_data;
            }
          }
        }
      }
    }
  }
}

