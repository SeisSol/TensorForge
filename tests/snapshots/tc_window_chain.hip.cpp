// === base name ===
kernel_0f98dc9f5d208804

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0f98dc9f5d208804 = {{16, 16, 1}, 16, 16, 1, 16, 3328, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0f98dc9f5d208804(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0f98dc9f5d208804(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0f98dc9f5d208804(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0f98dc9f5d208804, block.x * block.y * block.z, 832 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (832 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_0f98dc9f5d208804, block.x * block.y * block.z, 0));
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
void launcher_kernel_0f98dc9f5d208804(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0f98dc9f5d208804(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0f98dc9f5d208804), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m3;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_0f98dc9f5d208804, grid, block, config.sharedMemBytes, stream, m0Arg, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0f98dc9f5d208804(tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m0, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m3, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m0[0];
      float * __restrict__ glb_m0 = &totalShrMem[0];
      // glb_m0 = load{g>s}(ptr_glb_m0[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 16) {
        float v10_ld = __builtin_nontemporal_load(&ptr_glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256]);
        glb_m0[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v10_ld;
      }
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m3[0];
      float * __restrict__ glb_m3 = &totalShrMem[320];
      // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 240) {
        float v13_ld = __builtin_nontemporal_load(&ptr_glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m3[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v13_ld;
      }
      // wait(glb_m0 = load{g>s}(ptr_glb_m0[0, 1]));
      // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
      __syncthreads();
      size_t v15_batchIdLane0 = threadIdx.y % 4;
      int32_t v31_lead = threadIdx.x % 16;
      bool v32_g = v31_lead >= 1;
      bool v42_g = v31_lead < 2;
      int32_t v74_a = v31_lead + ((threadIdx.y % 4) * 16);
      int32_t v81_a = v74_a + 64;
      int32_t v87_a = v74_a + 128;
      int32_t v93_a = v74_a + 192;
      int32_t v99_a = v31_lead + 256;
      int32_t v163_a = v31_lead + 16;
      int32_t v165_a = v31_lead + 32;
      int32_t v167_a = v31_lead + 48;
      int32_t v169_a = v31_lead + 64;
      int32_t v171_a = v31_lead + 80;
      int32_t v173_a = v31_lead + 96;
      int32_t v175_a = v31_lead + 112;
      int32_t v177_a = v31_lead + 128;
      int32_t v179_a = v31_lead + 144;
      int32_t v181_a = v31_lead + 160;
      int32_t v183_a = v31_lead + 176;
      int32_t v185_a = v31_lead + 192;
      int32_t v187_a = v31_lead + 208;
      int32_t v189_a = v31_lead + 224;
      int32_t v191_a = v31_lead + 240;
      for (size_t v16_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v16_batchIdGroup0 < numElements0; v16_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v17_row = v16_batchIdGroup0 + v15_batchIdLane0;
        const bool batchIdActive0 = v17_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v17_row]));
        size_t v19_batchId0 = batchIdActive0 ? v17_row : v16_batchIdGroup0;
        size_t v20_ahead1 = v19_batchId0 + (gridDim.x * blockDim.y);
        size_t v22_batchId1 = (v20_ahead1 < numElements0) ? v20_ahead1 : v19_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v19_batchId0 * 153 + 0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v19_batchId0 * 144 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m1);
        if (v32_g) {
          int32_t v36_a = v31_lead - 1;
          #pragma unroll
          for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
            float v39_data = __builtin_nontemporal_load(&glb_m1[(v36_a + (v33_i1 * 17))]);
            r0[(v33_i1 * 2)] = v39_data;
          }
        }
        if (v42_g) {
          int32_t v46_a = (v31_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v43_i1 = 0; v43_i1 < 9; ++v43_i1) {
            float v49_data = __builtin_nontemporal_load(&glb_m1[(v46_a + (v43_i1 * 17))]);
            r0[(1 + (v43_i1 * 2))] = v49_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m1););
        float r1[9]{};
        // r1 = +(glb_m0 * r0) + None
        // [(0, 16), (0, 9)] [(1, 18)]
        float v53_data = r0[0];
        float v54_data = r0[2];
        float v55_data = r0[4];
        float v56_data = r0[6];
        float v57_tp{};
        float v58_tp{};
        float v59_tp{};
        float v60_tp{};
        tensorforge::transpose4x4b32(v57_tp, v58_tp, v59_tp, v60_tp, v53_data, v54_data, v55_data, v56_data);
        float v61_data = r0[1];
        float v62_data = r0[3];
        float v63_data = r0[5];
        float v64_data = r0[7];
        float v65_tp{};
        float v66_tp{};
        float v67_tp{};
        float v68_tp{};
        tensorforge::transpose4x4b32(v65_tp, v66_tp, v67_tp, v68_tp, v61_data, v62_data, v63_data, v64_data);
        tensorforge::VectorT<float, 4> v69_acc{};
        float v76_data = glb_m0[v74_a];
        tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v76_data, v69_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v76_data, v77_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v76_data, v78_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v76_data, v79_acc, 2, 1, 7);
        float v82_data = glb_m0[v81_a];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v82_data, v80_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v82_data, v83_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v82_data, v84_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v82_data, v85_acc, 2, 2, 7);
        float v88_data = glb_m0[v87_a];
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v88_data, v86_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v88_data, v89_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v88_data, v90_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v88_data, v91_acc, 2, 3, 7);
        float v94_data = glb_m0[v93_a];
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v94_data, v92_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v94_data, v95_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v94_data, v96_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v94_data, v97_acc, 2, 0, 7);
        float v100_data = glb_m0[v99_a];
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v100_data, v98_acc, 2, 0, 0);
        r1[0] = (v101_acc[0]);
        r1[1] = (v101_acc[1]);
        r1[2] = (v101_acc[2]);
        r1[3] = (v101_acc[3]);
        float v106_data = r0[8];
        float v107_data = r0[10];
        float v108_data = r0[12];
        float v109_data = r0[14];
        float v110_tp{};
        float v111_tp{};
        float v112_tp{};
        float v113_tp{};
        tensorforge::transpose4x4b32(v110_tp, v111_tp, v112_tp, v113_tp, v106_data, v107_data, v108_data, v109_data);
        float v114_data = r0[9];
        float v115_data = r0[11];
        float v116_data = r0[13];
        float v117_data = r0[15];
        float v118_tp{};
        float v119_tp{};
        float v120_tp{};
        float v121_tp{};
        tensorforge::transpose4x4b32(v118_tp, v119_tp, v120_tp, v121_tp, v114_data, v115_data, v116_data, v117_data);
        tensorforge::VectorT<float, 4> v122_acc{};
        float v129_data = glb_m0[v74_a];
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v129_data, v122_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v129_data, v130_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v129_data, v131_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v129_data, v132_acc, 2, 1, 7);
        float v135_data = glb_m0[v81_a];
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v135_data, v133_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v135_data, v136_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v135_data, v137_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v135_data, v138_acc, 2, 2, 7);
        float v141_data = glb_m0[v87_a];
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v141_data, v139_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v141_data, v142_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v141_data, v143_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v141_data, v144_acc, 2, 3, 7);
        float v147_data = glb_m0[v93_a];
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v147_data, v145_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v147_data, v148_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v147_data, v149_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v147_data, v150_acc, 2, 0, 7);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v100_data, v151_acc, 2, 0, 0);
        r1[4] = (v154_acc[0]);
        r1[5] = (v154_acc[1]);
        r1[6] = (v154_acc[2]);
        r1[7] = (v154_acc[3]);
        float v162_data = glb_m0[v31_lead];
        float v164_data = glb_m0[v163_a];
        float v166_data = glb_m0[v165_a];
        float v168_data = glb_m0[v167_a];
        float v170_data = glb_m0[v169_a];
        float v172_data = glb_m0[v171_a];
        float v174_data = glb_m0[v173_a];
        float v176_data = glb_m0[v175_a];
        float v178_data = glb_m0[v177_a];
        float v180_data = glb_m0[v179_a];
        float v182_data = glb_m0[v181_a];
        float v184_data = glb_m0[v183_a];
        float v186_data = glb_m0[v185_a];
        float v188_data = glb_m0[v187_a];
        float v190_data = glb_m0[v189_a];
        float v192_data = glb_m0[v191_a];
        float v195_acc{};
        float v196_data = r0[16];
        float v197_data = r0[17];
        tensorforge::fmacdpp16<1>(v195_acc, v196_data, v162_data);
        tensorforge::fmacdpp16<2>(v195_acc, v196_data, v164_data);
        tensorforge::fmacdpp16<3>(v195_acc, v196_data, v166_data);
        tensorforge::fmacdpp16<4>(v195_acc, v196_data, v168_data);
        tensorforge::fmacdpp16<5>(v195_acc, v196_data, v170_data);
        tensorforge::fmacdpp16<6>(v195_acc, v196_data, v172_data);
        tensorforge::fmacdpp16<7>(v195_acc, v196_data, v174_data);
        tensorforge::fmacdpp16<8>(v195_acc, v196_data, v176_data);
        tensorforge::fmacdpp16<9>(v195_acc, v196_data, v178_data);
        tensorforge::fmacdpp16<10>(v195_acc, v196_data, v180_data);
        tensorforge::fmacdpp16<11>(v195_acc, v196_data, v182_data);
        tensorforge::fmacdpp16<12>(v195_acc, v196_data, v184_data);
        tensorforge::fmacdpp16<13>(v195_acc, v196_data, v186_data);
        tensorforge::fmacdpp16<14>(v195_acc, v196_data, v188_data);
        tensorforge::fmacdpp16<15>(v195_acc, v196_data, v190_data);
        tensorforge::fmacdpp16<0>(v195_acc, v197_data, v192_data);
        tensorforge::fmacdpp16<1>(v195_acc, v197_data, v100_data);
        r1[8] = v195_acc;
        float r2[9]{};
        // r2 = +(glb_m3 * r1) + None
        // [(0, 16), (0, 9)] [(1, 16)]
        float v199_data = r1[0];
        float v200_data = r1[1];
        float v201_data = r1[2];
        float v202_data = r1[3];
        float v203_tp{};
        float v204_tp{};
        float v205_tp{};
        float v206_tp{};
        tensorforge::transpose4x4b32(v203_tp, v204_tp, v205_tp, v206_tp, v199_data, v200_data, v201_data, v202_data);
        tensorforge::VectorT<float, 4> v207_acc{};
        float v214_data = glb_m3[v74_a];
        tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v214_data, v207_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v214_data, v215_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v214_data, v216_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v214_data, v217_acc, 2, 1, 7);
        float v220_data = glb_m3[v81_a];
        tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v220_data, v218_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v220_data, v221_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v220_data, v222_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v220_data, v223_acc, 2, 2, 7);
        float v226_data = glb_m3[v87_a];
        tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v226_data, v224_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v226_data, v227_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v226_data, v228_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v226_data, v229_acc, 2, 3, 7);
        float v232_data = glb_m3[v185_a];
        tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v232_data, v230_acc, 2, 3, 0);
        float v235_data = glb_m3[v187_a];
        tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v235_data, v233_acc, 2, 3, 0);
        float v238_data = glb_m3[v189_a];
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v238_data, v236_acc, 2, 3, 0);
        r2[0] = (v239_acc[0]);
        r2[1] = (v239_acc[1]);
        r2[2] = (v239_acc[2]);
        r2[3] = (v239_acc[3]);
        float v244_data = r1[4];
        float v245_data = r1[5];
        float v246_data = r1[6];
        float v247_data = r1[7];
        float v248_tp{};
        float v249_tp{};
        float v250_tp{};
        float v251_tp{};
        tensorforge::transpose4x4b32(v248_tp, v249_tp, v250_tp, v251_tp, v244_data, v245_data, v246_data, v247_data);
        tensorforge::VectorT<float, 4> v252_acc{};
        float v259_data = glb_m3[v74_a];
        tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v259_data, v252_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v259_data, v260_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v259_data, v261_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v259_data, v262_acc, 2, 1, 7);
        float v265_data = glb_m3[v81_a];
        tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v265_data, v263_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v265_data, v266_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v265_data, v267_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v265_data, v268_acc, 2, 2, 7);
        float v271_data = glb_m3[v87_a];
        tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v271_data, v269_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v271_data, v272_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v271_data, v273_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v271_data, v274_acc, 2, 3, 7);
        tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v232_data, v275_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v235_data, v278_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v238_data, v281_acc, 2, 3, 0);
        r2[4] = (v284_acc[0]);
        r2[5] = (v284_acc[1]);
        r2[6] = (v284_acc[2]);
        r2[7] = (v284_acc[3]);
        float v292_data = glb_m3[v31_lead];
        float v294_data = glb_m3[v163_a];
        float v296_data = glb_m3[v165_a];
        float v298_data = glb_m3[v167_a];
        float v300_data = glb_m3[v169_a];
        float v302_data = glb_m3[v171_a];
        float v304_data = glb_m3[v173_a];
        float v306_data = glb_m3[v175_a];
        float v308_data = glb_m3[v177_a];
        float v310_data = glb_m3[v179_a];
        float v312_data = glb_m3[v181_a];
        float v314_data = glb_m3[v183_a];
        float v321_acc{};
        float v322_data = r1[8];
        tensorforge::fmacdpp16<1>(v321_acc, v322_data, v292_data);
        tensorforge::fmacdpp16<2>(v321_acc, v322_data, v294_data);
        tensorforge::fmacdpp16<3>(v321_acc, v322_data, v296_data);
        tensorforge::fmacdpp16<4>(v321_acc, v322_data, v298_data);
        tensorforge::fmacdpp16<5>(v321_acc, v322_data, v300_data);
        tensorforge::fmacdpp16<6>(v321_acc, v322_data, v302_data);
        tensorforge::fmacdpp16<7>(v321_acc, v322_data, v304_data);
        tensorforge::fmacdpp16<8>(v321_acc, v322_data, v306_data);
        tensorforge::fmacdpp16<9>(v321_acc, v322_data, v308_data);
        tensorforge::fmacdpp16<10>(v321_acc, v322_data, v310_data);
        tensorforge::fmacdpp16<11>(v321_acc, v322_data, v312_data);
        tensorforge::fmacdpp16<12>(v321_acc, v322_data, v314_data);
        tensorforge::fmacdpp16<13>(v321_acc, v322_data, v232_data);
        tensorforge::fmacdpp16<14>(v321_acc, v322_data, v235_data);
        tensorforge::fmacdpp16<15>(v321_acc, v322_data, v238_data);
        r2[8] = v321_acc;
        // glb_m2 = store{r>g}(r2);
        #pragma unroll
        for (int32_t v323_i0 = 0; v323_i0 < 1; ++v323_i0) {
          #pragma unroll
          for (int32_t v324_i1 = 0; v324_i1 < 9; ++v324_i1) {
            float v326_data = r2[(v323_i0 + v324_i1)];
            if (batchIdActive0) {
              glb_m2[((v31_lead + (v323_i0 * 16)) + (v324_i1 * 16))] = v326_data;
            }
          }
        }
      }
    }
  }
}

