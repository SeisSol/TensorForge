// === base name ===
kernel_0f20aae960e2ee39

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0f20aae960e2ee39 = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0f20aae960e2ee39(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0f20aae960e2ee39(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0f20aae960e2ee39(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0f20aae960e2ee39, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0f20aae960e2ee39, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (640 * sizeof(float)));
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
  config.sharedMemBytes = 640 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0f20aae960e2ee39(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0f20aae960e2ee39(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0f20aae960e2ee39), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0f20aae960e2ee39, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0f20aae960e2ee39(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 2560 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 16×20(10×17) {0..10}×{1..18} none
    //   m2 20×9(17×9) {1..18}×{0..9} strided
    //   m3 16×20(10×18) {0..10}×{1..19} none
    //   m4 20×9(18×9) {1..19}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 384];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v12_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[192];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 180) {
        float v15_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v15_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v17_batchIdLane0 = threadIdx.y % 4;
      int32_t v34_lead = threadIdx.x % 16;
      bool v35_g = v34_lead >= 1;
      bool v45_g = v34_lead < 2;
      bool v65_g = v34_lead < 3;
      int32_t v97_a = v34_lead + ((threadIdx.y % 4) * 10);
      int32_t v104_a = v97_a + 40;
      int32_t v110_a = v97_a + 80;
      int32_t v116_a = v97_a + 120;
      int32_t v122_a = v34_lead + 160;
      int32_t v186_a = v34_lead + 10;
      int32_t v188_a = v34_lead + 20;
      int32_t v190_a = v34_lead + 30;
      int32_t v192_a = v34_lead + 40;
      int32_t v194_a = v34_lead + 50;
      int32_t v196_a = v34_lead + 60;
      int32_t v198_a = v34_lead + 70;
      int32_t v200_a = v34_lead + 80;
      int32_t v202_a = v34_lead + 90;
      int32_t v204_a = v34_lead + 100;
      int32_t v206_a = v34_lead + 110;
      int32_t v208_a = v34_lead + 120;
      int32_t v210_a = v34_lead + 130;
      int32_t v212_a = v34_lead + 140;
      int32_t v214_a = v34_lead + 150;
      int32_t v272_a = v34_lead + 170;
      bool v376_g = v34_lead < 10;
      for (size_t v18_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v18_batchIdGroup0 < numElements0; v18_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v19_row = v18_batchIdGroup0 + v17_batchIdLane0;
        const bool batchIdActive0 = v19_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v19_row]));
        size_t v21_batchId0 = batchIdActive0 ? v19_row : v18_batchIdGroup0;
        size_t v22_ahead1 = v21_batchId0 + (gridDim.x * blockDim.y);
        size_t v24_batchId1 = (v22_ahead1 < numElements0) ? v22_ahead1 : v21_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v21_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v21_batchId0 * 153 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v21_batchId0 * 162 + 0 + m4_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        if (v35_g) {
          int32_t v39_a = v34_lead - 1;
          #pragma unroll
          for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
            float v42_data = __builtin_nontemporal_load(&glb_m2[(v39_a + (v36_i1 * 17))]);
            r0[(v36_i1 * 2)] = v42_data;
          }
        }
        if (v45_g) {
          int32_t v49_a = (v34_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v46_i1 = 0; v46_i1 < 9; ++v46_i1) {
            float v52_data = __builtin_nontemporal_load(&glb_m2[(v49_a + (v46_i1 * 17))]);
            r0[(1 + (v46_i1 * 2))] = v52_data;
          }
        }
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        if (v35_g) {
          int32_t v59_a = v34_lead - 1;
          #pragma unroll
          for (int32_t v56_i1 = 0; v56_i1 < 9; ++v56_i1) {
            float v62_data = __builtin_nontemporal_load(&glb_m4[(v59_a + (v56_i1 * 18))]);
            r2[(v56_i1 * 2)] = v62_data;
          }
        }
        if (v65_g) {
          int32_t v69_a = (v34_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v66_i1 = 0; v66_i1 < 9; ++v66_i1) {
            float v72_data = __builtin_nontemporal_load(&glb_m4[(v69_a + (v66_i1 * 18))]);
            r2[(1 + (v66_i1 * 2))] = v72_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(1, 18)]
        float v76_data = r0[0];
        float v77_data = r0[2];
        float v78_data = r0[4];
        float v79_data = r0[6];
        float v80_tp{};
        float v81_tp{};
        float v82_tp{};
        float v83_tp{};
        tensorforge::transpose4x4b32(v80_tp, v81_tp, v82_tp, v83_tp, v76_data, v77_data, v78_data, v79_data);
        float v84_data = r0[1];
        float v85_data = r0[3];
        float v86_data = r0[5];
        float v87_data = r0[7];
        float v88_tp{};
        float v89_tp{};
        float v90_tp{};
        float v91_tp{};
        tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v84_data, v85_data, v86_data, v87_data);
        tensorforge::VectorT<float, 4> v92_acc{};
        float v99_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v99_data, v92_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v99_data, v100_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v99_data, v101_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v99_data, v102_acc, 2, 1, 7);
        float v105_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v105_data, v103_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v105_data, v106_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v105_data, v107_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v105_data, v108_acc, 2, 2, 7);
        float v111_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v111_data, v109_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v111_data, v112_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v111_data, v113_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v111_data, v114_acc, 2, 3, 7);
        float v117_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v117_data, v115_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v117_data, v118_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v117_data, v119_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v117_data, v120_acc, 2, 0, 7);
        float v123_data = glb_m1[v122_a];
        tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v123_data, v121_acc, 2, 0, 0);
        r1[0] = (v124_acc[0]);
        r1[1] = (v124_acc[1]);
        r1[2] = (v124_acc[2]);
        r1[3] = (v124_acc[3]);
        float v129_data = r0[8];
        float v130_data = r0[10];
        float v131_data = r0[12];
        float v132_data = r0[14];
        float v133_tp{};
        float v134_tp{};
        float v135_tp{};
        float v136_tp{};
        tensorforge::transpose4x4b32(v133_tp, v134_tp, v135_tp, v136_tp, v129_data, v130_data, v131_data, v132_data);
        float v137_data = r0[9];
        float v138_data = r0[11];
        float v139_data = r0[13];
        float v140_data = r0[15];
        float v141_tp{};
        float v142_tp{};
        float v143_tp{};
        float v144_tp{};
        tensorforge::transpose4x4b32(v141_tp, v142_tp, v143_tp, v144_tp, v137_data, v138_data, v139_data, v140_data);
        tensorforge::VectorT<float, 4> v145_acc{};
        float v152_data = glb_m1[v97_a];
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v152_data, v145_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v152_data, v153_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v152_data, v154_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v152_data, v155_acc, 2, 1, 7);
        float v158_data = glb_m1[v104_a];
        tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v158_data, v156_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v158_data, v159_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v158_data, v160_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v158_data, v161_acc, 2, 2, 7);
        float v164_data = glb_m1[v110_a];
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v164_data, v162_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v164_data, v165_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v164_data, v166_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v164_data, v167_acc, 2, 3, 7);
        float v170_data = glb_m1[v116_a];
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v170_data, v168_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v170_data, v171_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v170_data, v172_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v170_data, v173_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v123_data, v174_acc, 2, 0, 0);
        r1[4] = (v177_acc[0]);
        r1[5] = (v177_acc[1]);
        r1[6] = (v177_acc[2]);
        r1[7] = (v177_acc[3]);
        float v185_data = glb_m1[v34_lead];
        float v187_data = glb_m1[v186_a];
        float v189_data = glb_m1[v188_a];
        float v191_data = glb_m1[v190_a];
        float v193_data = glb_m1[v192_a];
        float v195_data = glb_m1[v194_a];
        float v197_data = glb_m1[v196_a];
        float v199_data = glb_m1[v198_a];
        float v201_data = glb_m1[v200_a];
        float v203_data = glb_m1[v202_a];
        float v205_data = glb_m1[v204_a];
        float v207_data = glb_m1[v206_a];
        float v209_data = glb_m1[v208_a];
        float v211_data = glb_m1[v210_a];
        float v213_data = glb_m1[v212_a];
        float v215_data = glb_m1[v214_a];
        float v218_acc{};
        float v219_data = r0[16];
        float v220_data = r0[17];
        tensorforge::fmacdpp16<1>(v218_acc, v219_data, v185_data);
        tensorforge::fmacdpp16<2>(v218_acc, v219_data, v187_data);
        tensorforge::fmacdpp16<3>(v218_acc, v219_data, v189_data);
        tensorforge::fmacdpp16<4>(v218_acc, v219_data, v191_data);
        tensorforge::fmacdpp16<5>(v218_acc, v219_data, v193_data);
        tensorforge::fmacdpp16<6>(v218_acc, v219_data, v195_data);
        tensorforge::fmacdpp16<7>(v218_acc, v219_data, v197_data);
        tensorforge::fmacdpp16<8>(v218_acc, v219_data, v199_data);
        tensorforge::fmacdpp16<9>(v218_acc, v219_data, v201_data);
        tensorforge::fmacdpp16<10>(v218_acc, v219_data, v203_data);
        tensorforge::fmacdpp16<11>(v218_acc, v219_data, v205_data);
        tensorforge::fmacdpp16<12>(v218_acc, v219_data, v207_data);
        tensorforge::fmacdpp16<13>(v218_acc, v219_data, v209_data);
        tensorforge::fmacdpp16<14>(v218_acc, v219_data, v211_data);
        tensorforge::fmacdpp16<15>(v218_acc, v219_data, v213_data);
        tensorforge::fmacdpp16<0>(v218_acc, v220_data, v215_data);
        tensorforge::fmacdpp16<1>(v218_acc, v220_data, v123_data);
        r1[8] = v218_acc;
        // wait(r2 = load{g>r}(glb_m4););
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(1, 19)]
        float ir3[9]{};
        float v223_data = r2[0];
        float v224_data = r2[2];
        float v225_data = r2[4];
        float v226_data = r2[6];
        float v227_tp{};
        float v228_tp{};
        float v229_tp{};
        float v230_tp{};
        tensorforge::transpose4x4b32(v227_tp, v228_tp, v229_tp, v230_tp, v223_data, v224_data, v225_data, v226_data);
        float v231_data = r2[1];
        float v232_data = r2[3];
        float v233_data = r2[5];
        float v234_data = r2[7];
        float v235_tp{};
        float v236_tp{};
        float v237_tp{};
        float v238_tp{};
        tensorforge::transpose4x4b32(v235_tp, v236_tp, v237_tp, v238_tp, v231_data, v232_data, v233_data, v234_data);
        tensorforge::VectorT<float, 4> v239_acc{};
        float v246_data = glb_m3[v97_a];
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v246_data, v239_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v246_data, v247_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v246_data, v248_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v246_data, v249_acc, 2, 1, 7);
        float v252_data = glb_m3[v104_a];
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v252_data, v250_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v252_data, v253_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v252_data, v254_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v252_data, v255_acc, 2, 2, 7);
        float v258_data = glb_m3[v110_a];
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v258_data, v256_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v258_data, v259_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v258_data, v260_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v258_data, v261_acc, 2, 3, 7);
        float v264_data = glb_m3[v116_a];
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v264_data, v262_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v264_data, v265_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v264_data, v266_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v264_data, v267_acc, 2, 0, 7);
        float v270_data = glb_m3[v122_a];
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v270_data, v268_acc, 2, 0, 0);
        float v273_data = glb_m3[v272_a];
        tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v273_data, v271_acc, 2, 0, 0);
        ir3[0] = (v274_acc[0]);
        ir3[1] = (v274_acc[1]);
        ir3[2] = (v274_acc[2]);
        ir3[3] = (v274_acc[3]);
        float v279_data = r2[8];
        float v280_data = r2[10];
        float v281_data = r2[12];
        float v282_data = r2[14];
        float v283_tp{};
        float v284_tp{};
        float v285_tp{};
        float v286_tp{};
        tensorforge::transpose4x4b32(v283_tp, v284_tp, v285_tp, v286_tp, v279_data, v280_data, v281_data, v282_data);
        float v287_data = r2[9];
        float v288_data = r2[11];
        float v289_data = r2[13];
        float v290_data = r2[15];
        float v291_tp{};
        float v292_tp{};
        float v293_tp{};
        float v294_tp{};
        tensorforge::transpose4x4b32(v291_tp, v292_tp, v293_tp, v294_tp, v287_data, v288_data, v289_data, v290_data);
        tensorforge::VectorT<float, 4> v295_acc{};
        float v302_data = glb_m3[v97_a];
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v302_data, v295_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v302_data, v303_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v302_data, v304_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v302_data, v305_acc, 2, 1, 7);
        float v308_data = glb_m3[v104_a];
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v308_data, v306_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v308_data, v309_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v308_data, v310_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v308_data, v311_acc, 2, 2, 7);
        float v314_data = glb_m3[v110_a];
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v314_data, v312_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v314_data, v315_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v314_data, v316_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v314_data, v317_acc, 2, 3, 7);
        float v320_data = glb_m3[v116_a];
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v320_data, v318_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v285_tp, v320_data, v321_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v320_data, v322_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v320_data, v323_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v270_data, v324_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v273_data, v327_acc, 2, 0, 0);
        ir3[4] = (v330_acc[0]);
        ir3[5] = (v330_acc[1]);
        ir3[6] = (v330_acc[2]);
        ir3[7] = (v330_acc[3]);
        float v338_data = glb_m3[v34_lead];
        float v340_data = glb_m3[v186_a];
        float v342_data = glb_m3[v188_a];
        float v344_data = glb_m3[v190_a];
        float v346_data = glb_m3[v192_a];
        float v348_data = glb_m3[v194_a];
        float v350_data = glb_m3[v196_a];
        float v352_data = glb_m3[v198_a];
        float v354_data = glb_m3[v200_a];
        float v356_data = glb_m3[v202_a];
        float v358_data = glb_m3[v204_a];
        float v360_data = glb_m3[v206_a];
        float v362_data = glb_m3[v208_a];
        float v364_data = glb_m3[v210_a];
        float v366_data = glb_m3[v212_a];
        float v368_data = glb_m3[v214_a];
        float v373_acc{};
        float v374_data = r2[16];
        float v375_data = r2[17];
        tensorforge::fmacdpp16<1>(v373_acc, v374_data, v338_data);
        tensorforge::fmacdpp16<2>(v373_acc, v374_data, v340_data);
        tensorforge::fmacdpp16<3>(v373_acc, v374_data, v342_data);
        tensorforge::fmacdpp16<4>(v373_acc, v374_data, v344_data);
        tensorforge::fmacdpp16<5>(v373_acc, v374_data, v346_data);
        tensorforge::fmacdpp16<6>(v373_acc, v374_data, v348_data);
        tensorforge::fmacdpp16<7>(v373_acc, v374_data, v350_data);
        tensorforge::fmacdpp16<8>(v373_acc, v374_data, v352_data);
        tensorforge::fmacdpp16<9>(v373_acc, v374_data, v354_data);
        tensorforge::fmacdpp16<10>(v373_acc, v374_data, v356_data);
        tensorforge::fmacdpp16<11>(v373_acc, v374_data, v358_data);
        tensorforge::fmacdpp16<12>(v373_acc, v374_data, v360_data);
        tensorforge::fmacdpp16<13>(v373_acc, v374_data, v362_data);
        tensorforge::fmacdpp16<14>(v373_acc, v374_data, v364_data);
        tensorforge::fmacdpp16<15>(v373_acc, v374_data, v366_data);
        tensorforge::fmacdpp16<0>(v373_acc, v375_data, v368_data);
        tensorforge::fmacdpp16<1>(v373_acc, v375_data, v270_data);
        tensorforge::fmacdpp16<2>(v373_acc, v375_data, v273_data);
        ir3[8] = v373_acc;
        // r3 = ir3 + r1
        if (v376_g) {
          #pragma unroll
          for (int32_t v377_n1 = 0; v377_n1 < 9; ++v377_n1) {
            float v379_data = ir3[v377_n1];
            float v380_data = r1[v377_n1];
            r3[v377_n1] = (v380_data + v379_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v376_g) {
          #pragma unroll
          for (int32_t v383_i1 = 0; v383_i1 < 9; ++v383_i1) {
            float v385_data = r3[v383_i1];
            if (batchIdActive0) {
              glb_m0[(v34_lead + (v383_i1 * 10))] = v385_data;
            }
          }
        }
      }
    }
  }
}

