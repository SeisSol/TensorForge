// === base name ===
kernel_5d50b1568f36b787

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5d50b1568f36b787 = {{16, 16, 1}, 16, 16, 1, 16, 3328, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5d50b1568f36b787(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5d50b1568f36b787(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5d50b1568f36b787(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5d50b1568f36b787, block.x * block.y * block.z, 832 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (832 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5d50b1568f36b787, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (832 * sizeof(float)));
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
  config.sharedMemBytes = 832 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5d50b1568f36b787(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5d50b1568f36b787(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5d50b1568f36b787), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5d50b1568f36b787, grid, block, config.sharedMemBytes, stream, m0Arg, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5d50b1568f36b787(tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m0, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 3328 B shared, occupancy grid
    // operands:
    //   m0 16×20(16×17) {0..16}×{1..18} none
    //   m1 20×9(17×9) {1..18}×{0..9} strided
    //   m2 16×9(16×9) {0..16}×{0..9} strided
    //   m3 16×20(16×15) {0..16}×{1..16} none
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":832}],"shared_bytes":3328,"shared_elements":832,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 576];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m0[0];
      float * __restrict__ glb_m0 = &totalShrMem[0];
      // glb_m0 = load{g>s}(ptr_glb_m0[0, 1])
      float v12_ld = __builtin_nontemporal_load(&ptr_glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 16) {
        float v13_ld = __builtin_nontemporal_load(&ptr_glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256]);
        glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v13_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[320];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 240) {
        float v16_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v16_ld;
      }
      // wait(glb_m0 = load{g>s}(ptr_glb_m0[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v18_batchIdLane0 = threadIdx.y % 4;
      int32_t v34_lead = threadIdx.x % 16;
      bool v35_g = v34_lead >= 1;
      bool v45_g = v34_lead < 2;
      int32_t v77_a = v34_lead + ((threadIdx.y % 4) * 16);
      int32_t v84_a = v77_a + 64;
      int32_t v90_a = v77_a + 128;
      int32_t v96_a = v77_a + 192;
      int32_t v102_a = v34_lead + 256;
      int32_t v166_a = v34_lead + 16;
      int32_t v168_a = v34_lead + 32;
      int32_t v170_a = v34_lead + 48;
      int32_t v172_a = v34_lead + 64;
      int32_t v174_a = v34_lead + 80;
      int32_t v176_a = v34_lead + 96;
      int32_t v178_a = v34_lead + 112;
      int32_t v180_a = v34_lead + 128;
      int32_t v182_a = v34_lead + 144;
      int32_t v184_a = v34_lead + 160;
      int32_t v186_a = v34_lead + 176;
      int32_t v188_a = v34_lead + 192;
      int32_t v190_a = v34_lead + 208;
      int32_t v192_a = v34_lead + 224;
      int32_t v194_a = v34_lead + 240;
      for (size_t v19_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v19_batchIdGroup0 < numElements0; v19_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v20_row = v19_batchIdGroup0 + v18_batchIdLane0;
        const bool batchIdActive0 = v20_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v20_row]));
        size_t v22_batchId0 = batchIdActive0 ? v20_row : v19_batchIdGroup0;
        size_t v23_ahead1 = v22_batchId0 + (gridDim.x * blockDim.y);
        size_t v25_batchId1 = (v23_ahead1 < numElements0) ? v23_ahead1 : v22_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v22_batchId0 * 153 + 0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v22_batchId0 * 144 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m1);
        if (v35_g) {
          int32_t v39_a = v34_lead - 1;
          #pragma unroll
          for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
            float v42_data = __builtin_nontemporal_load(&glb_m1[(v39_a + (v36_i1 * 17))]);
            r0[(v36_i1 * 2)] = v42_data;
          }
        }
        if (v45_g) {
          int32_t v49_a = (v34_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v46_i1 = 0; v46_i1 < 9; ++v46_i1) {
            float v52_data = __builtin_nontemporal_load(&glb_m1[(v49_a + (v46_i1 * 17))]);
            r0[(1 + (v46_i1 * 2))] = v52_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m1););
        float r1[9]{};
        // r1 = +(glb_m0 * r0) + None
        // [(0, 16), (0, 9)] [(1, 18)]
        float v56_data = r0[0];
        float v57_data = r0[2];
        float v58_data = r0[4];
        float v59_data = r0[6];
        float v60_tp{};
        float v61_tp{};
        float v62_tp{};
        float v63_tp{};
        tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v56_data, v57_data, v58_data, v59_data);
        float v64_data = r0[1];
        float v65_data = r0[3];
        float v66_data = r0[5];
        float v67_data = r0[7];
        float v68_tp{};
        float v69_tp{};
        float v70_tp{};
        float v71_tp{};
        tensorforge::transpose4x4b32(v68_tp, v69_tp, v70_tp, v71_tp, v64_data, v65_data, v66_data, v67_data);
        tensorforge::VectorT<float, 4> v72_acc{};
        float v79_data = glb_m0[v77_a];
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v79_data, v72_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v79_data, v80_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v79_data, v81_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v79_data, v82_acc, 2, 1, 7);
        float v85_data = glb_m0[v84_a];
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v85_data, v83_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v85_data, v86_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v85_data, v87_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v85_data, v88_acc, 2, 2, 7);
        float v91_data = glb_m0[v90_a];
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v91_data, v89_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v91_data, v92_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v91_data, v93_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v91_data, v94_acc, 2, 3, 7);
        float v97_data = glb_m0[v96_a];
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v97_data, v95_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v97_data, v98_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v97_data, v99_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v97_data, v100_acc, 2, 0, 7);
        float v103_data = glb_m0[v102_a];
        tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v103_data, v101_acc, 2, 0, 0);
        r1[0] = (v104_acc[0]);
        r1[1] = (v104_acc[1]);
        r1[2] = (v104_acc[2]);
        r1[3] = (v104_acc[3]);
        float v109_data = r0[8];
        float v110_data = r0[10];
        float v111_data = r0[12];
        float v112_data = r0[14];
        float v113_tp{};
        float v114_tp{};
        float v115_tp{};
        float v116_tp{};
        tensorforge::transpose4x4b32(v113_tp, v114_tp, v115_tp, v116_tp, v109_data, v110_data, v111_data, v112_data);
        float v117_data = r0[9];
        float v118_data = r0[11];
        float v119_data = r0[13];
        float v120_data = r0[15];
        float v121_tp{};
        float v122_tp{};
        float v123_tp{};
        float v124_tp{};
        tensorforge::transpose4x4b32(v121_tp, v122_tp, v123_tp, v124_tp, v117_data, v118_data, v119_data, v120_data);
        tensorforge::VectorT<float, 4> v125_acc{};
        float v132_data = glb_m0[v77_a];
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v132_data, v125_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v132_data, v133_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v132_data, v134_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v132_data, v135_acc, 2, 1, 7);
        float v138_data = glb_m0[v84_a];
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v138_data, v136_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v138_data, v139_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v138_data, v140_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v138_data, v141_acc, 2, 2, 7);
        float v144_data = glb_m0[v90_a];
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v144_data, v142_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v144_data, v145_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v144_data, v146_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v144_data, v147_acc, 2, 3, 7);
        float v150_data = glb_m0[v96_a];
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v150_data, v148_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v150_data, v151_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v150_data, v152_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v150_data, v153_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v103_data, v154_acc, 2, 0, 0);
        r1[4] = (v157_acc[0]);
        r1[5] = (v157_acc[1]);
        r1[6] = (v157_acc[2]);
        r1[7] = (v157_acc[3]);
        float v165_data = glb_m0[v34_lead];
        float v167_data = glb_m0[v166_a];
        float v169_data = glb_m0[v168_a];
        float v171_data = glb_m0[v170_a];
        float v173_data = glb_m0[v172_a];
        float v175_data = glb_m0[v174_a];
        float v177_data = glb_m0[v176_a];
        float v179_data = glb_m0[v178_a];
        float v181_data = glb_m0[v180_a];
        float v183_data = glb_m0[v182_a];
        float v185_data = glb_m0[v184_a];
        float v187_data = glb_m0[v186_a];
        float v189_data = glb_m0[v188_a];
        float v191_data = glb_m0[v190_a];
        float v193_data = glb_m0[v192_a];
        float v195_data = glb_m0[v194_a];
        float v198_acc{};
        float v199_data = r0[16];
        float v200_data = r0[17];
        tensorforge::fmacdpp16<1>(v198_acc, v199_data, v165_data);
        tensorforge::fmacdpp16<2>(v198_acc, v199_data, v167_data);
        tensorforge::fmacdpp16<3>(v198_acc, v199_data, v169_data);
        tensorforge::fmacdpp16<4>(v198_acc, v199_data, v171_data);
        tensorforge::fmacdpp16<5>(v198_acc, v199_data, v173_data);
        tensorforge::fmacdpp16<6>(v198_acc, v199_data, v175_data);
        tensorforge::fmacdpp16<7>(v198_acc, v199_data, v177_data);
        tensorforge::fmacdpp16<8>(v198_acc, v199_data, v179_data);
        tensorforge::fmacdpp16<9>(v198_acc, v199_data, v181_data);
        tensorforge::fmacdpp16<10>(v198_acc, v199_data, v183_data);
        tensorforge::fmacdpp16<11>(v198_acc, v199_data, v185_data);
        tensorforge::fmacdpp16<12>(v198_acc, v199_data, v187_data);
        tensorforge::fmacdpp16<13>(v198_acc, v199_data, v189_data);
        tensorforge::fmacdpp16<14>(v198_acc, v199_data, v191_data);
        tensorforge::fmacdpp16<15>(v198_acc, v199_data, v193_data);
        tensorforge::fmacdpp16<0>(v198_acc, v200_data, v195_data);
        tensorforge::fmacdpp16<1>(v198_acc, v200_data, v103_data);
        r1[8] = v198_acc;
        float r2[9]{};
        // r2 = +(glb_m3 * r1) + None
        // [(0, 16), (0, 9)] [(1, 16)]
        float v202_data = r1[0];
        float v203_data = r1[1];
        float v204_data = r1[2];
        float v205_data = r1[3];
        float v206_tp{};
        float v207_tp{};
        float v208_tp{};
        float v209_tp{};
        tensorforge::transpose4x4b32(v206_tp, v207_tp, v208_tp, v209_tp, v202_data, v203_data, v204_data, v205_data);
        tensorforge::VectorT<float, 4> v210_acc{};
        float v217_data = glb_m3[v77_a];
        tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v217_data, v210_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v217_data, v218_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v217_data, v219_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v217_data, v220_acc, 2, 1, 7);
        float v223_data = glb_m3[v84_a];
        tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v223_data, v221_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v223_data, v224_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v223_data, v225_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v223_data, v226_acc, 2, 2, 7);
        float v229_data = glb_m3[v90_a];
        tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v229_data, v227_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v229_data, v230_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v229_data, v231_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v229_data, v232_acc, 2, 3, 7);
        float v235_data = glb_m3[v188_a];
        tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v235_data, v233_acc, 2, 3, 0);
        float v238_data = glb_m3[v190_a];
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v238_data, v236_acc, 2, 3, 0);
        float v241_data = glb_m3[v192_a];
        tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v241_data, v239_acc, 2, 3, 0);
        r2[0] = (v242_acc[0]);
        r2[1] = (v242_acc[1]);
        r2[2] = (v242_acc[2]);
        r2[3] = (v242_acc[3]);
        float v247_data = r1[4];
        float v248_data = r1[5];
        float v249_data = r1[6];
        float v250_data = r1[7];
        float v251_tp{};
        float v252_tp{};
        float v253_tp{};
        float v254_tp{};
        tensorforge::transpose4x4b32(v251_tp, v252_tp, v253_tp, v254_tp, v247_data, v248_data, v249_data, v250_data);
        tensorforge::VectorT<float, 4> v255_acc{};
        float v262_data = glb_m3[v77_a];
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v262_data, v255_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v262_data, v263_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v262_data, v264_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v262_data, v265_acc, 2, 1, 7);
        float v268_data = glb_m3[v84_a];
        tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v268_data, v266_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v268_data, v269_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v268_data, v270_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v268_data, v271_acc, 2, 2, 7);
        float v274_data = glb_m3[v90_a];
        tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v274_data, v272_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v274_data, v275_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v274_data, v276_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v274_data, v277_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v235_data, v278_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v253_tp, v238_data, v281_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v241_data, v284_acc, 2, 3, 0);
        r2[4] = (v287_acc[0]);
        r2[5] = (v287_acc[1]);
        r2[6] = (v287_acc[2]);
        r2[7] = (v287_acc[3]);
        float v295_data = glb_m3[v34_lead];
        float v297_data = glb_m3[v166_a];
        float v299_data = glb_m3[v168_a];
        float v301_data = glb_m3[v170_a];
        float v303_data = glb_m3[v172_a];
        float v305_data = glb_m3[v174_a];
        float v307_data = glb_m3[v176_a];
        float v309_data = glb_m3[v178_a];
        float v311_data = glb_m3[v180_a];
        float v313_data = glb_m3[v182_a];
        float v315_data = glb_m3[v184_a];
        float v317_data = glb_m3[v186_a];
        float v324_acc{};
        float v325_data = r1[8];
        tensorforge::fmacdpp16<1>(v324_acc, v325_data, v295_data);
        tensorforge::fmacdpp16<2>(v324_acc, v325_data, v297_data);
        tensorforge::fmacdpp16<3>(v324_acc, v325_data, v299_data);
        tensorforge::fmacdpp16<4>(v324_acc, v325_data, v301_data);
        tensorforge::fmacdpp16<5>(v324_acc, v325_data, v303_data);
        tensorforge::fmacdpp16<6>(v324_acc, v325_data, v305_data);
        tensorforge::fmacdpp16<7>(v324_acc, v325_data, v307_data);
        tensorforge::fmacdpp16<8>(v324_acc, v325_data, v309_data);
        tensorforge::fmacdpp16<9>(v324_acc, v325_data, v311_data);
        tensorforge::fmacdpp16<10>(v324_acc, v325_data, v313_data);
        tensorforge::fmacdpp16<11>(v324_acc, v325_data, v315_data);
        tensorforge::fmacdpp16<12>(v324_acc, v325_data, v317_data);
        tensorforge::fmacdpp16<13>(v324_acc, v325_data, v235_data);
        tensorforge::fmacdpp16<14>(v324_acc, v325_data, v238_data);
        tensorforge::fmacdpp16<15>(v324_acc, v325_data, v241_data);
        r2[8] = v324_acc;
        // glb_m2 = store{r>g}(r2);
        #pragma unroll
        for (int32_t v326_i0 = 0; v326_i0 < 1; ++v326_i0) {
          #pragma unroll
          for (int32_t v327_i1 = 0; v327_i1 < 9; ++v327_i1) {
            float v329_data = r2[(v326_i0 + v327_i1)];
            if (batchIdActive0) {
              glb_m2[((v34_lead + (v326_i0 * 16)) + (v327_i1 * 16))] = v329_data;
            }
          }
        }
      }
    }
  }
}

