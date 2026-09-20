// === base name ===
kernel_b5cb40f375f7d206

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b5cb40f375f7d206 = {{16, 16, 1}, 16, 10, 1, 16, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b5cb40f375f7d206(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b5cb40f375f7d206(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b5cb40f375f7d206(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b5cb40f375f7d206, block.x * block.y * block.z, 640 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (640 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b5cb40f375f7d206, block.x * block.y * block.z, 0));
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
void launcher_kernel_b5cb40f375f7d206(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b5cb40f375f7d206(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b5cb40f375f7d206), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_b5cb40f375f7d206, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b5cb40f375f7d206(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 384];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[192];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 170) {
        float v8_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v8_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v10_batchIdLane0 = threadIdx.y % 4;
      int32_t v28_lead = threadIdx.x % 16;
      bool v38_g = v28_lead < 1;
      int32_t v87_a = v28_lead + ((threadIdx.y % 4) * 10);
      int32_t v94_a = v87_a + 40;
      int32_t v100_a = v87_a + 80;
      int32_t v106_a = v87_a + 120;
      int32_t v112_a = v28_lead + 160;
      int32_t v176_a = v28_lead + 10;
      int32_t v178_a = v28_lead + 20;
      int32_t v180_a = v28_lead + 30;
      int32_t v182_a = v28_lead + 40;
      int32_t v184_a = v28_lead + 50;
      int32_t v186_a = v28_lead + 60;
      int32_t v188_a = v28_lead + 70;
      int32_t v190_a = v28_lead + 80;
      int32_t v192_a = v28_lead + 90;
      int32_t v194_a = v28_lead + 100;
      int32_t v196_a = v28_lead + 110;
      int32_t v198_a = v28_lead + 120;
      int32_t v200_a = v28_lead + 130;
      int32_t v202_a = v28_lead + 140;
      int32_t v204_a = v28_lead + 150;
      bool v358_g = v28_lead < 10;
      for (size_t v11_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v11_batchIdGroup0 < numElements0; v11_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v12_row = v11_batchIdGroup0 + v10_batchIdLane0;
        const bool batchIdActive0 = v12_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v12_row]));
        size_t v14_batchId0 = batchIdActive0 ? v12_row : v11_batchIdGroup0;
        size_t v15_ahead1 = v14_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v15_ahead1 < numElements0) ? v15_ahead1 : v14_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v14_batchId0 * 90 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v14_batchId0 * 153 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v14_batchId0 * 153 + 0 + m4_extraOffset];
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
        float r2[18]{};
        // r2 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v48_i0 = 0; v48_i0 < 1; ++v48_i0) {
          int32_t v51_lead = v28_lead + (v48_i0 * 16);
          #pragma unroll
          for (int32_t v49_i1 = 0; v49_i1 < 9; ++v49_i1) {
            float v54_data = __builtin_nontemporal_load(&glb_m4[(v51_lead + (v49_i1 * 17))]);
            r2[(v48_i0 + (v49_i1 * 2))] = v54_data;
          }
        }
        if (v38_g) {
          int32_t v59_lead = v28_lead + 16_i32;
          #pragma unroll
          for (int32_t v57_i1 = 0; v57_i1 < 9; ++v57_i1) {
            float v62_data = __builtin_nontemporal_load(&glb_m4[(v59_lead + (v57_i1 * 17))]);
            r2[(1 + (v57_i1 * 2))] = v62_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 10), (0, 9)] [(0, 17)]
        float v66_data = r0[0];
        float v67_data = r0[2];
        float v68_data = r0[4];
        float v69_data = r0[6];
        float v70_tp{};
        float v71_tp{};
        float v72_tp{};
        float v73_tp{};
        tensorforge::transpose4x4b32(v70_tp, v71_tp, v72_tp, v73_tp, v66_data, v67_data, v68_data, v69_data);
        float v74_data = r0[1];
        float v75_data = r0[3];
        float v76_data = r0[5];
        float v77_data = r0[7];
        float v78_tp{};
        float v79_tp{};
        float v80_tp{};
        float v81_tp{};
        tensorforge::transpose4x4b32(v78_tp, v79_tp, v80_tp, v81_tp, v74_data, v75_data, v76_data, v77_data);
        tensorforge::VectorT<float, 4> v82_acc{};
        float v89_data = glb_m1[v87_a];
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v89_data, v82_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v89_data, v90_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v89_data, v91_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v89_data, v92_acc, 2, 0, 7);
        float v95_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v95_data, v93_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v95_data, v96_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v95_data, v97_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v95_data, v98_acc, 2, 1, 7);
        float v101_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v101_data, v99_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v101_data, v102_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v101_data, v103_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v101_data, v104_acc, 2, 2, 7);
        float v107_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v107_data, v105_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v107_data, v108_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v107_data, v109_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v107_data, v110_acc, 2, 3, 7);
        float v113_data = glb_m1[v112_a];
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v113_data, v111_acc, 2, 0, 0);
        r1[0] = (v114_acc[0]);
        r1[1] = (v114_acc[1]);
        r1[2] = (v114_acc[2]);
        r1[3] = (v114_acc[3]);
        float v119_data = r0[8];
        float v120_data = r0[10];
        float v121_data = r0[12];
        float v122_data = r0[14];
        float v123_tp{};
        float v124_tp{};
        float v125_tp{};
        float v126_tp{};
        tensorforge::transpose4x4b32(v123_tp, v124_tp, v125_tp, v126_tp, v119_data, v120_data, v121_data, v122_data);
        float v127_data = r0[9];
        float v128_data = r0[11];
        float v129_data = r0[13];
        float v130_data = r0[15];
        float v131_tp{};
        float v132_tp{};
        float v133_tp{};
        float v134_tp{};
        tensorforge::transpose4x4b32(v131_tp, v132_tp, v133_tp, v134_tp, v127_data, v128_data, v129_data, v130_data);
        tensorforge::VectorT<float, 4> v135_acc{};
        float v142_data = glb_m1[v87_a];
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v142_data, v135_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v142_data, v143_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v142_data, v144_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v142_data, v145_acc, 2, 0, 7);
        float v148_data = glb_m1[v94_a];
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v148_data, v146_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v148_data, v149_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v148_data, v150_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v148_data, v151_acc, 2, 1, 7);
        float v154_data = glb_m1[v100_a];
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v154_data, v152_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v154_data, v155_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v154_data, v156_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v154_data, v157_acc, 2, 2, 7);
        float v160_data = glb_m1[v106_a];
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v160_data, v158_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v160_data, v161_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v160_data, v162_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v160_data, v163_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v113_data, v164_acc, 2, 0, 0);
        r1[4] = (v167_acc[0]);
        r1[5] = (v167_acc[1]);
        r1[6] = (v167_acc[2]);
        r1[7] = (v167_acc[3]);
        float v175_data = glb_m1[v28_lead];
        float v177_data = glb_m1[v176_a];
        float v179_data = glb_m1[v178_a];
        float v181_data = glb_m1[v180_a];
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
        float v208_acc{};
        float v209_data = r0[16];
        float v210_data = r0[17];
        tensorforge::fmacdpp16<0>(v208_acc, v209_data, v175_data);
        tensorforge::fmacdpp16<1>(v208_acc, v209_data, v177_data);
        tensorforge::fmacdpp16<2>(v208_acc, v209_data, v179_data);
        tensorforge::fmacdpp16<3>(v208_acc, v209_data, v181_data);
        tensorforge::fmacdpp16<4>(v208_acc, v209_data, v183_data);
        tensorforge::fmacdpp16<5>(v208_acc, v209_data, v185_data);
        tensorforge::fmacdpp16<6>(v208_acc, v209_data, v187_data);
        tensorforge::fmacdpp16<7>(v208_acc, v209_data, v189_data);
        tensorforge::fmacdpp16<8>(v208_acc, v209_data, v191_data);
        tensorforge::fmacdpp16<9>(v208_acc, v209_data, v193_data);
        tensorforge::fmacdpp16<10>(v208_acc, v209_data, v195_data);
        tensorforge::fmacdpp16<11>(v208_acc, v209_data, v197_data);
        tensorforge::fmacdpp16<12>(v208_acc, v209_data, v199_data);
        tensorforge::fmacdpp16<13>(v208_acc, v209_data, v201_data);
        tensorforge::fmacdpp16<14>(v208_acc, v209_data, v203_data);
        tensorforge::fmacdpp16<15>(v208_acc, v209_data, v205_data);
        tensorforge::fmacdpp16<0>(v208_acc, v210_data, v113_data);
        r1[8] = v208_acc;
        // wait(r2 = load{g>r}(glb_m4););
        float r3[9]{};
        // ir3 = +(glb_m3 * r2)
        // [(0, 10), (0, 9)] [(0, 17)]
        float ir3[9]{};
        float v213_data = r2[0];
        float v214_data = r2[2];
        float v215_data = r2[4];
        float v216_data = r2[6];
        float v217_tp{};
        float v218_tp{};
        float v219_tp{};
        float v220_tp{};
        tensorforge::transpose4x4b32(v217_tp, v218_tp, v219_tp, v220_tp, v213_data, v214_data, v215_data, v216_data);
        float v221_data = r2[1];
        float v222_data = r2[3];
        float v223_data = r2[5];
        float v224_data = r2[7];
        float v225_tp{};
        float v226_tp{};
        float v227_tp{};
        float v228_tp{};
        tensorforge::transpose4x4b32(v225_tp, v226_tp, v227_tp, v228_tp, v221_data, v222_data, v223_data, v224_data);
        tensorforge::VectorT<float, 4> v229_acc{};
        float v236_data = glb_m3[v87_a];
        tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v236_data, v229_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v236_data, v237_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v236_data, v238_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v236_data, v239_acc, 2, 0, 7);
        float v242_data = glb_m3[v94_a];
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v242_data, v240_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v242_data, v243_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v242_data, v244_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v242_data, v245_acc, 2, 1, 7);
        float v248_data = glb_m3[v100_a];
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v248_data, v246_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v248_data, v249_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v248_data, v250_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v248_data, v251_acc, 2, 2, 7);
        float v254_data = glb_m3[v106_a];
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v254_data, v252_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v254_data, v255_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v219_tp, v254_data, v256_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v220_tp, v254_data, v257_acc, 2, 3, 7);
        float v260_data = glb_m3[v112_a];
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v225_tp, v260_data, v258_acc, 2, 0, 0);
        ir3[0] = (v261_acc[0]);
        ir3[1] = (v261_acc[1]);
        ir3[2] = (v261_acc[2]);
        ir3[3] = (v261_acc[3]);
        float v266_data = r2[8];
        float v267_data = r2[10];
        float v268_data = r2[12];
        float v269_data = r2[14];
        float v270_tp{};
        float v271_tp{};
        float v272_tp{};
        float v273_tp{};
        tensorforge::transpose4x4b32(v270_tp, v271_tp, v272_tp, v273_tp, v266_data, v267_data, v268_data, v269_data);
        float v274_data = r2[9];
        float v275_data = r2[11];
        float v276_data = r2[13];
        float v277_data = r2[15];
        float v278_tp{};
        float v279_tp{};
        float v280_tp{};
        float v281_tp{};
        tensorforge::transpose4x4b32(v278_tp, v279_tp, v280_tp, v281_tp, v274_data, v275_data, v276_data, v277_data);
        tensorforge::VectorT<float, 4> v282_acc{};
        float v289_data = glb_m3[v87_a];
        tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v289_data, v282_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v289_data, v290_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v289_data, v291_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v289_data, v292_acc, 2, 0, 7);
        float v295_data = glb_m3[v94_a];
        tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v295_data, v293_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v295_data, v296_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v295_data, v297_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v295_data, v298_acc, 2, 1, 7);
        float v301_data = glb_m3[v100_a];
        tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v301_data, v299_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v301_data, v302_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v301_data, v303_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v301_data, v304_acc, 2, 2, 7);
        float v307_data = glb_m3[v106_a];
        tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v307_data, v305_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v307_data, v308_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v307_data, v309_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v307_data, v310_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v260_data, v311_acc, 2, 0, 0);
        ir3[4] = (v314_acc[0]);
        ir3[5] = (v314_acc[1]);
        ir3[6] = (v314_acc[2]);
        ir3[7] = (v314_acc[3]);
        float v322_data = glb_m3[v28_lead];
        float v324_data = glb_m3[v176_a];
        float v326_data = glb_m3[v178_a];
        float v328_data = glb_m3[v180_a];
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
        float v355_acc{};
        float v356_data = r2[16];
        float v357_data = r2[17];
        tensorforge::fmacdpp16<0>(v355_acc, v356_data, v322_data);
        tensorforge::fmacdpp16<1>(v355_acc, v356_data, v324_data);
        tensorforge::fmacdpp16<2>(v355_acc, v356_data, v326_data);
        tensorforge::fmacdpp16<3>(v355_acc, v356_data, v328_data);
        tensorforge::fmacdpp16<4>(v355_acc, v356_data, v330_data);
        tensorforge::fmacdpp16<5>(v355_acc, v356_data, v332_data);
        tensorforge::fmacdpp16<6>(v355_acc, v356_data, v334_data);
        tensorforge::fmacdpp16<7>(v355_acc, v356_data, v336_data);
        tensorforge::fmacdpp16<8>(v355_acc, v356_data, v338_data);
        tensorforge::fmacdpp16<9>(v355_acc, v356_data, v340_data);
        tensorforge::fmacdpp16<10>(v355_acc, v356_data, v342_data);
        tensorforge::fmacdpp16<11>(v355_acc, v356_data, v344_data);
        tensorforge::fmacdpp16<12>(v355_acc, v356_data, v346_data);
        tensorforge::fmacdpp16<13>(v355_acc, v356_data, v348_data);
        tensorforge::fmacdpp16<14>(v355_acc, v356_data, v350_data);
        tensorforge::fmacdpp16<15>(v355_acc, v356_data, v352_data);
        tensorforge::fmacdpp16<0>(v355_acc, v357_data, v260_data);
        ir3[8] = v355_acc;
        // r3 = ir3 + r1
        if (v358_g) {
          #pragma unroll
          for (int32_t v359_n1 = 0; v359_n1 < 9; ++v359_n1) {
            float v361_data = ir3[v359_n1];
            float v362_data = r1[v359_n1];
            r3[v359_n1] = (v362_data + v361_data);
          }
        }
        // glb_m0 = store{r>g}(r3);
        if (v358_g) {
          #pragma unroll
          for (int32_t v365_i1 = 0; v365_i1 < 9; ++v365_i1) {
            float v367_data = r3[v365_i1];
            if (batchIdActive0) {
              glb_m0[(v28_lead + (v365_i1 * 10))] = v367_data;
            }
          }
        }
      }
    }
  }
}

