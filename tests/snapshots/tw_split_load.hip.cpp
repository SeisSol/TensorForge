// === base name ===
kernel_29851ff1548dce82

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_29851ff1548dce82 = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_29851ff1548dce82(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_29851ff1548dce82(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_29851ff1548dce82(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_29851ff1548dce82, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_29851ff1548dce82, block.x * block.y * block.z, 0));
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
void launcher_kernel_29851ff1548dce82(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_29851ff1548dce82(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_29851ff1548dce82), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_29851ff1548dce82, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_29851ff1548dce82(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 2560 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 10×17(10×17) {0..10}×{0..17} none
    //   m2 17×9(17×9) {0..17}×{0..9} strided
    //   m3 10×17(10×17) {0..10}×{0..17} none
    //   m4 17×9(17×9) {0..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v15_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v15_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v17_batchIdLane0 = threadIdx.y % 4;
      int32_t v34_lead = threadIdx.x % 16;
      bool v44_g = v34_lead < 1;
      int32_t v93_a = v34_lead + ((threadIdx.y % 4) * 10);
      int32_t v100_a = v93_a + 40;
      int32_t v106_a = v93_a + 80;
      int32_t v112_a = v93_a + 120;
      int32_t v118_a = v34_lead + 160;
      int32_t v182_a = v34_lead + 10;
      int32_t v184_a = v34_lead + 20;
      int32_t v186_a = v34_lead + 30;
      int32_t v188_a = v34_lead + 40;
      int32_t v190_a = v34_lead + 50;
      int32_t v192_a = v34_lead + 60;
      int32_t v194_a = v34_lead + 70;
      int32_t v196_a = v34_lead + 80;
      int32_t v198_a = v34_lead + 90;
      int32_t v200_a = v34_lead + 100;
      int32_t v202_a = v34_lead + 110;
      int32_t v204_a = v34_lead + 120;
      int32_t v206_a = v34_lead + 130;
      int32_t v208_a = v34_lead + 140;
      int32_t v210_a = v34_lead + 150;
      bool v364_g = v34_lead < 10;
      for (size_t v18_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v18_batchIdGroup0 < numElements0; v18_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v19_row = v18_batchIdGroup0 + v17_batchIdLane0;
        const bool batchIdActive0 = v19_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v19_row]));
        size_t v21_batchId0 = batchIdActive0 ? v19_row : v18_batchIdGroup0;
        size_t v22_ahead1 = v21_batchId0 + (gridDim.x * blockDim.y);
        size_t v24_batchId1 = (v22_ahead1 < numElements0) ? v22_ahead1 : v21_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v21_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v21_batchId0 * 153 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v21_batchId0 * 153 + 0 + m4_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
          int32_t v38_lead = v34_lead + (v35_i0 * 16);
          #pragma unroll
          for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
            float v41_data = __builtin_nontemporal_load(&glb_m2[(v38_lead + (v36_i1 * 17))]);
            r0[(v35_i0 + (v36_i1 * 2))] = v41_data;
          }
        }
        if (v44_g) {
          int32_t v47_lead = v34_lead + 16_i32;
          #pragma unroll
          for (int32_t v45_i1 = 0; v45_i1 < 9; ++v45_i1) {
            float v50_data = __builtin_nontemporal_load(&glb_m2[(v47_lead + (v45_i1 * 17))]);
            r0[(1 + (v45_i1 * 2))] = v50_data;
          }
        }
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v54_i0 = 0; v54_i0 < 1; ++v54_i0) {
          int32_t v57_lead = v34_lead + (v54_i0 * 16);
          #pragma unroll
          for (int32_t v55_i1 = 0; v55_i1 < 9; ++v55_i1) {
            float v60_data = __builtin_nontemporal_load(&glb_m4[(v57_lead + (v55_i1 * 17))]);
            r2[(v54_i0 + (v55_i1 * 2))] = v60_data;
          }
        }
        if (v44_g) {
          int32_t v65_lead = v34_lead + 16_i32;
          #pragma unroll
          for (int32_t v63_i1 = 0; v63_i1 < 9; ++v63_i1) {
            float v68_data = __builtin_nontemporal_load(&glb_m4[(v65_lead + (v63_i1 * 17))]);
            r2[(1 + (v63_i1 * 2))] = v68_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(0, 17)]
        float v72_data = r0[0];
        float v73_data = r0[2];
        float v74_data = r0[4];
        float v75_data = r0[6];
        float v76_tp{};
        float v77_tp{};
        float v78_tp{};
        float v79_tp{};
        tensorforge::transpose4x4b32(v76_tp, v77_tp, v78_tp, v79_tp, v72_data, v73_data, v74_data, v75_data);
        float v80_data = r0[1];
        float v81_data = r0[3];
        float v82_data = r0[5];
        float v83_data = r0[7];
        float v84_tp{};
        float v85_tp{};
        float v86_tp{};
        float v87_tp{};
        tensorforge::transpose4x4b32(v84_tp, v85_tp, v86_tp, v87_tp, v80_data, v81_data, v82_data, v83_data);
        tensorforge::VectorT<float, 4> v88_acc{};
        float v95_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v95_data, v88_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v95_data, v96_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v95_data, v97_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v95_data, v98_acc, 2, 0, 7);
        float v101_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v101_data, v99_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v101_data, v102_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v101_data, v103_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v101_data, v104_acc, 2, 1, 7);
        float v107_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v107_data, v105_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v107_data, v108_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v107_data, v109_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v107_data, v110_acc, 2, 2, 7);
        float v113_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v113_data, v111_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v113_data, v114_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v113_data, v115_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v113_data, v116_acc, 2, 3, 7);
        float v119_data = glb_m1[v118_a];
        tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v119_data, v117_acc, 2, 0, 0);
        r1[0] = (v120_acc[0]);
        r1[1] = (v120_acc[1]);
        r1[2] = (v120_acc[2]);
        r1[3] = (v120_acc[3]);
        float v125_data = r0[8];
        float v126_data = r0[10];
        float v127_data = r0[12];
        float v128_data = r0[14];
        float v129_tp{};
        float v130_tp{};
        float v131_tp{};
        float v132_tp{};
        tensorforge::transpose4x4b32(v129_tp, v130_tp, v131_tp, v132_tp, v125_data, v126_data, v127_data, v128_data);
        float v133_data = r0[9];
        float v134_data = r0[11];
        float v135_data = r0[13];
        float v136_data = r0[15];
        float v137_tp{};
        float v138_tp{};
        float v139_tp{};
        float v140_tp{};
        tensorforge::transpose4x4b32(v137_tp, v138_tp, v139_tp, v140_tp, v133_data, v134_data, v135_data, v136_data);
        tensorforge::VectorT<float, 4> v141_acc{};
        float v148_data = glb_m1[v93_a];
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v148_data, v141_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v148_data, v149_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v148_data, v150_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v148_data, v151_acc, 2, 0, 7);
        float v154_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v154_data, v152_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v154_data, v155_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v154_data, v156_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v154_data, v157_acc, 2, 1, 7);
        float v160_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v160_data, v158_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v160_data, v161_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v160_data, v162_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v160_data, v163_acc, 2, 2, 7);
        float v166_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v166_data, v164_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v166_data, v167_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v166_data, v168_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v166_data, v169_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v119_data, v170_acc, 2, 0, 0);
        r1[4] = (v173_acc[0]);
        r1[5] = (v173_acc[1]);
        r1[6] = (v173_acc[2]);
        r1[7] = (v173_acc[3]);
        float v181_data = glb_m1[v34_lead];
        float v183_data = glb_m1[v182_a];
        float v185_data = glb_m1[v184_a];
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
        float v214_acc{};
        float v215_data = r0[16];
        float v216_data = r0[17];
        tensorforge::fmacdpp16<0>(v214_acc, v215_data, v181_data);
        tensorforge::fmacdpp16<1>(v214_acc, v215_data, v183_data);
        tensorforge::fmacdpp16<2>(v214_acc, v215_data, v185_data);
        tensorforge::fmacdpp16<3>(v214_acc, v215_data, v187_data);
        tensorforge::fmacdpp16<4>(v214_acc, v215_data, v189_data);
        tensorforge::fmacdpp16<5>(v214_acc, v215_data, v191_data);
        tensorforge::fmacdpp16<6>(v214_acc, v215_data, v193_data);
        tensorforge::fmacdpp16<7>(v214_acc, v215_data, v195_data);
        tensorforge::fmacdpp16<8>(v214_acc, v215_data, v197_data);
        tensorforge::fmacdpp16<9>(v214_acc, v215_data, v199_data);
        tensorforge::fmacdpp16<10>(v214_acc, v215_data, v201_data);
        tensorforge::fmacdpp16<11>(v214_acc, v215_data, v203_data);
        tensorforge::fmacdpp16<12>(v214_acc, v215_data, v205_data);
        tensorforge::fmacdpp16<13>(v214_acc, v215_data, v207_data);
        tensorforge::fmacdpp16<14>(v214_acc, v215_data, v209_data);
        tensorforge::fmacdpp16<15>(v214_acc, v215_data, v211_data);
        tensorforge::fmacdpp16<0>(v214_acc, v216_data, v119_data);
        r1[8] = v214_acc;
        // wait(r2 = load{g>r}(glb_m4););
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(0, 17)]
        float ir3[9]{};
        float v219_data = r2[0];
        float v220_data = r2[2];
        float v221_data = r2[4];
        float v222_data = r2[6];
        float v223_tp{};
        float v224_tp{};
        float v225_tp{};
        float v226_tp{};
        tensorforge::transpose4x4b32(v223_tp, v224_tp, v225_tp, v226_tp, v219_data, v220_data, v221_data, v222_data);
        float v227_data = r2[1];
        float v228_data = r2[3];
        float v229_data = r2[5];
        float v230_data = r2[7];
        float v231_tp{};
        float v232_tp{};
        float v233_tp{};
        float v234_tp{};
        tensorforge::transpose4x4b32(v231_tp, v232_tp, v233_tp, v234_tp, v227_data, v228_data, v229_data, v230_data);
        tensorforge::VectorT<float, 4> v235_acc{};
        float v242_data = glb_m3[v93_a];
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v242_data, v235_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v242_data, v243_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v242_data, v244_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v242_data, v245_acc, 2, 0, 7);
        float v248_data = glb_m3[v100_a];
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v248_data, v246_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v248_data, v249_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v248_data, v250_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v248_data, v251_acc, 2, 1, 7);
        float v254_data = glb_m3[v106_a];
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v254_data, v252_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v254_data, v255_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v254_data, v256_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v254_data, v257_acc, 2, 2, 7);
        float v260_data = glb_m3[v112_a];
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v223_tp, v260_data, v258_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v224_tp, v260_data, v261_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v260_data, v262_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v226_tp, v260_data, v263_acc, 2, 3, 7);
        float v266_data = glb_m3[v118_a];
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v266_data, v264_acc, 2, 0, 0);
        ir3[0] = (v267_acc[0]);
        ir3[1] = (v267_acc[1]);
        ir3[2] = (v267_acc[2]);
        ir3[3] = (v267_acc[3]);
        float v272_data = r2[8];
        float v273_data = r2[10];
        float v274_data = r2[12];
        float v275_data = r2[14];
        float v276_tp{};
        float v277_tp{};
        float v278_tp{};
        float v279_tp{};
        tensorforge::transpose4x4b32(v276_tp, v277_tp, v278_tp, v279_tp, v272_data, v273_data, v274_data, v275_data);
        float v280_data = r2[9];
        float v281_data = r2[11];
        float v282_data = r2[13];
        float v283_data = r2[15];
        float v284_tp{};
        float v285_tp{};
        float v286_tp{};
        float v287_tp{};
        tensorforge::transpose4x4b32(v284_tp, v285_tp, v286_tp, v287_tp, v280_data, v281_data, v282_data, v283_data);
        tensorforge::VectorT<float, 4> v288_acc{};
        float v295_data = glb_m3[v93_a];
        tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v295_data, v288_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v295_data, v296_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v295_data, v297_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v295_data, v298_acc, 2, 0, 7);
        float v301_data = glb_m3[v100_a];
        tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v301_data, v299_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v301_data, v302_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v301_data, v303_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v301_data, v304_acc, 2, 1, 7);
        float v307_data = glb_m3[v106_a];
        tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v307_data, v305_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v307_data, v308_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v307_data, v309_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v307_data, v310_acc, 2, 2, 7);
        float v313_data = glb_m3[v112_a];
        tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v313_data, v311_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v313_data, v314_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v313_data, v315_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v313_data, v316_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v266_data, v317_acc, 2, 0, 0);
        ir3[4] = (v320_acc[0]);
        ir3[5] = (v320_acc[1]);
        ir3[6] = (v320_acc[2]);
        ir3[7] = (v320_acc[3]);
        float v328_data = glb_m3[v34_lead];
        float v330_data = glb_m3[v182_a];
        float v332_data = glb_m3[v184_a];
        float v334_data = glb_m3[v186_a];
        float v336_data = glb_m3[v188_a];
        float v338_data = glb_m3[v190_a];
        float v340_data = glb_m3[v192_a];
        float v342_data = glb_m3[v194_a];
        float v344_data = glb_m3[v196_a];
        float v346_data = glb_m3[v198_a];
        float v348_data = glb_m3[v200_a];
        float v350_data = glb_m3[v202_a];
        float v352_data = glb_m3[v204_a];
        float v354_data = glb_m3[v206_a];
        float v356_data = glb_m3[v208_a];
        float v358_data = glb_m3[v210_a];
        float v361_acc{};
        float v362_data = r2[16];
        float v363_data = r2[17];
        tensorforge::fmacdpp16<0>(v361_acc, v362_data, v328_data);
        tensorforge::fmacdpp16<1>(v361_acc, v362_data, v330_data);
        tensorforge::fmacdpp16<2>(v361_acc, v362_data, v332_data);
        tensorforge::fmacdpp16<3>(v361_acc, v362_data, v334_data);
        tensorforge::fmacdpp16<4>(v361_acc, v362_data, v336_data);
        tensorforge::fmacdpp16<5>(v361_acc, v362_data, v338_data);
        tensorforge::fmacdpp16<6>(v361_acc, v362_data, v340_data);
        tensorforge::fmacdpp16<7>(v361_acc, v362_data, v342_data);
        tensorforge::fmacdpp16<8>(v361_acc, v362_data, v344_data);
        tensorforge::fmacdpp16<9>(v361_acc, v362_data, v346_data);
        tensorforge::fmacdpp16<10>(v361_acc, v362_data, v348_data);
        tensorforge::fmacdpp16<11>(v361_acc, v362_data, v350_data);
        tensorforge::fmacdpp16<12>(v361_acc, v362_data, v352_data);
        tensorforge::fmacdpp16<13>(v361_acc, v362_data, v354_data);
        tensorforge::fmacdpp16<14>(v361_acc, v362_data, v356_data);
        tensorforge::fmacdpp16<15>(v361_acc, v362_data, v358_data);
        tensorforge::fmacdpp16<0>(v361_acc, v363_data, v266_data);
        ir3[8] = v361_acc;
        // r3 = ir3 + r1
        if (v364_g) {
          #pragma unroll
          for (int32_t v365_n1 = 0; v365_n1 < 9; ++v365_n1) {
            float v367_data = ir3[v365_n1];
            float v368_data = r1[v365_n1];
            r3[v365_n1] = (v368_data + v367_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v364_g) {
          #pragma unroll
          for (int32_t v371_i1 = 0; v371_i1 < 9; ++v371_i1) {
            float v373_data = r3[v371_i1];
            if (batchIdActive0) {
              glb_m0[(v34_lead + (v371_i1 * 10))] = v373_data;
            }
          }
        }
      }
    }
  }
}

