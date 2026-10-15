// === base name ===
kernel_28d503b182e5c905

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_28d503b182e5c905 = {{32, 8, 1}, 32, 56, 1, 8, 8192, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_28d503b182e5c905(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_28d503b182e5c905(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_28d503b182e5c905(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_28d503b182e5c905, block.x * block.y * block.z, 2048 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (2048 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_28d503b182e5c905, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (2048 * sizeof(float)));
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
  config.sharedMemBytes = 2048 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_28d503b182e5c905(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_28d503b182e5c905(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_28d503b182e5c905), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_28d503b182e5c905, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_28d503b182e5c905(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 8192 B shared, occupancy grid
    // operands:
    //   m0 56×18(56×18) {0..56}×{0..18} strided
    //   m1 56×32(56×32) {0..56}×{0..32} none ordered
    //   m2 32×18(32×18) {0..32}×{0..18} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2048}],"shared_bytes":8192,"shared_elements":2048,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,18]],"name":"m0","ordered":false,"parts":1,"shape":[56,18],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,32]],"name":"m1","ordered":true,"parts":1,"shape":[56,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[32,18]],"name":"m2","ordered":false,"parts":1,"shape":[32,18],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,18]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,32]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,32]},{"addressing":"strided","bbox":[[0,0],[32,18]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,18]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 2048];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 4> v9_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      tensorforge::VectorT<float, 4> v10_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v10_ld;
      __syncthreads();
      size_t v12_batchIdLane0 = threadIdx.y % 2;
      size_t v15_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v31_lead = threadIdx.x % 32;
      int32_t v51_r = (threadIdx.y % 2) * 2;
      int32_t v55_vlead = (v31_lead + (v51_r * 32)) * 4;
      int32_t v77_vlead = (v31_lead + ((v51_r + 4) * 32)) * 4;
      int32_t v99_vlead = (v31_lead + ((v51_r + 8) * 32)) * 4;
      int32_t v121_vlead = (v31_lead + ((v51_r + 12) * 32)) * 4;
      int32_t v150_vlead = (v31_lead + ((v51_r + 1) * 32)) * 4;
      int32_t v172_vlead = (v31_lead + ((v51_r + 5) * 32)) * 4;
      int32_t v194_vlead = (v31_lead + ((v51_r + 9) * 32)) * 4;
      int32_t v216_vlead = (v31_lead + ((v51_r + 13) * 32)) * 4;
      bool v1035_g = v31_lead < 24;
      float r0[18]{};
      size_t v1047_row = v15_batchIdGroup0First + v12_batchIdLane0;
      const bool peel_batchIdActive0 = v1047_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1047_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v1047_row : v15_batchIdGroup0First) * 576 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v1055_i0 = 0; v1055_i0 < 1; ++v1055_i0) {
        int32_t v1057_lead = v31_lead + (v1055_i0 * 32);
        #pragma unroll
        for (int32_t v1058_i1 = 0; v1058_i1 < 18; ++v1058_i1) {
          float v1061_data = __builtin_nontemporal_load(&peel_glb_m2[(v1057_lead + (v1058_i1 * 32))]);
          r0[(v1055_i0 + v1058_i1)] = v1061_data;
        }
      }
      for (size_t v16_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v16_batchIdGroup0 < numElements0; v16_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v1044_groupNextAhead = v16_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v1046_groupNext = (v1044_groupNextAhead < numElements0) ? v1044_groupNextAhead : v16_batchIdGroup0;
        size_t v1051_row = v1046_groupNext + v12_batchIdLane0;
        const bool wrap_batchIdActive0 = v1051_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1051_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v1051_row : v1046_groupNext) * 576 + 0 + m2_extraOffset];
        size_t v17_row = v16_batchIdGroup0 + v12_batchIdLane0;
        const bool batchIdActive0 = v17_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v17_row]));
        size_t v19_batchId0 = batchIdActive0 ? v17_row : v16_batchIdGroup0;
        size_t v20_ahead1 = v19_batchId0 + (gridDim.x * blockDim.y);
        size_t v22_batchId1 = (v20_ahead1 < numElements0) ? v20_ahead1 : v19_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v19_batchId0 * 1008 + 0 + m0_extraOffset];
        float r1[36]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 56), (0, 18)] [(0, 32)]
        float v41_data = r0[0];
        float v42_data = r0[1];
        float v43_data = r0[2];
        float v44_data = r0[3];
        float v45_tp{};
        float v46_tp{};
        float v47_tp{};
        float v48_tp{};
        tensorforge::transpose4x4b32(v45_tp, v46_tp, v47_tp, v48_tp, v41_data, v42_data, v43_data, v44_data);
        tensorforge::VectorT<float, 4> v49_acc{};
        tensorforge::VectorT<float, 4> v57_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v55_vlead];
        float v58_el = v57_data[0];
        tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_el, v49_acc, 3, 0, 1);
        float v60_el = v57_data[1];
        tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v60_el, v59_acc, 3, 0, 1);
        float v62_el = v57_data[2];
        tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v62_el, v61_acc, 3, 0, 1);
        float v64_el = v57_data[3];
        tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v64_el, v63_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_el, v65_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v60_el, v67_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v62_el, v69_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v64_el, v71_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v79_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v77_vlead];
        float v80_el = v79_data[0];
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v80_el, v73_acc, 3, 2, 1);
        float v82_el = v79_data[1];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v82_el, v81_acc, 3, 2, 1);
        float v84_el = v79_data[2];
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v84_el, v83_acc, 3, 2, 1);
        float v86_el = v79_data[3];
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v86_el, v85_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v80_el, v87_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v82_el, v89_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v84_el, v91_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v86_el, v93_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v101_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v99_vlead];
        float v102_el = v101_data[0];
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v102_el, v95_acc, 3, 4, 1);
        float v104_el = v101_data[1];
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v104_el, v103_acc, 3, 4, 1);
        float v106_el = v101_data[2];
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v106_el, v105_acc, 3, 4, 1);
        float v108_el = v101_data[3];
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v108_el, v107_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v102_el, v109_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v104_el, v111_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v106_el, v113_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v108_el, v115_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v123_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v121_vlead];
        float v124_el = v123_data[0];
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v124_el, v117_acc, 3, 6, 1);
        float v126_el = v123_data[1];
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v126_el, v125_acc, 3, 6, 1);
        float v128_el = v123_data[2];
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v128_el, v127_acc, 3, 6, 1);
        float v130_el = v123_data[3];
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v130_el, v129_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v124_el, v131_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v126_el, v133_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v128_el, v135_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v130_el, v137_acc, 3, 7, 2);
        r1[0] = (v139_acc[0]);
        r1[2] = (v139_acc[1]);
        r1[4] = (v139_acc[2]);
        r1[6] = (v139_acc[3]);
        tensorforge::VectorT<float, 4> v144_acc{};
        tensorforge::VectorT<float, 4> v152_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v150_vlead];
        float v153_el = v152_data[0];
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v153_el, v144_acc, 3, 0, 1);
        float v155_el = v152_data[1];
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v155_el, v154_acc, 3, 0, 1);
        float v157_el = v152_data[2];
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v157_el, v156_acc, 3, 0, 1);
        float v159_el = v152_data[3];
        tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v159_el, v158_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v153_el, v160_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v155_el, v162_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v157_el, v164_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v159_el, v166_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v174_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v172_vlead];
        float v175_el = v174_data[0];
        tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v175_el, v168_acc, 3, 2, 1);
        float v177_el = v174_data[1];
        tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v177_el, v176_acc, 3, 2, 1);
        float v179_el = v174_data[2];
        tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v179_el, v178_acc, 3, 2, 1);
        float v181_el = v174_data[3];
        tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v181_el, v180_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v175_el, v182_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v177_el, v184_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v179_el, v186_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v181_el, v188_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v196_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v194_vlead];
        float v197_el = v196_data[0];
        tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v197_el, v190_acc, 3, 4, 1);
        float v199_el = v196_data[1];
        tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v199_el, v198_acc, 3, 4, 1);
        float v201_el = v196_data[2];
        tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v201_el, v200_acc, 3, 4, 1);
        float v203_el = v196_data[3];
        tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v203_el, v202_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v197_el, v204_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v199_el, v206_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v201_el, v208_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v203_el, v210_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v218_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v216_vlead];
        float v219_el = v218_data[0];
        tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v219_el, v212_acc, 3, 6, 1);
        float v221_el = v218_data[1];
        tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v221_el, v220_acc, 3, 6, 1);
        float v223_el = v218_data[2];
        tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v223_el, v222_acc, 3, 6, 1);
        float v225_el = v218_data[3];
        tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v225_el, v224_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v219_el, v226_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v221_el, v228_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v223_el, v230_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v225_el, v232_acc, 3, 7, 2);
        r1[1] = (v234_acc[0]);
        r1[3] = (v234_acc[1]);
        r1[5] = (v234_acc[2]);
        r1[7] = (v234_acc[3]);
        float v239_data = r0[4];
        float v240_data = r0[5];
        float v241_data = r0[6];
        float v242_data = r0[7];
        float v243_tp{};
        float v244_tp{};
        float v245_tp{};
        float v246_tp{};
        tensorforge::transpose4x4b32(v243_tp, v244_tp, v245_tp, v246_tp, v239_data, v240_data, v241_data, v242_data);
        tensorforge::VectorT<float, 4> v247_acc{};
        tensorforge::VectorT<float, 4> v255_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v55_vlead];
        float v256_el = v255_data[0];
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v256_el, v247_acc, 3, 0, 1);
        float v258_el = v255_data[1];
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v258_el, v257_acc, 3, 0, 1);
        float v260_el = v255_data[2];
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v260_el, v259_acc, 3, 0, 1);
        float v262_el = v255_data[3];
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v262_el, v261_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v256_el, v263_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v258_el, v265_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v260_el, v267_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v262_el, v269_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v277_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v77_vlead];
        float v278_el = v277_data[0];
        tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v278_el, v271_acc, 3, 2, 1);
        float v280_el = v277_data[1];
        tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v280_el, v279_acc, 3, 2, 1);
        float v282_el = v277_data[2];
        tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v282_el, v281_acc, 3, 2, 1);
        float v284_el = v277_data[3];
        tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v284_el, v283_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v278_el, v285_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v280_el, v287_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v282_el, v289_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v284_el, v291_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v299_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v99_vlead];
        float v300_el = v299_data[0];
        tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v300_el, v293_acc, 3, 4, 1);
        float v302_el = v299_data[1];
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v302_el, v301_acc, 3, 4, 1);
        float v304_el = v299_data[2];
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v304_el, v303_acc, 3, 4, 1);
        float v306_el = v299_data[3];
        tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v306_el, v305_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v300_el, v307_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v302_el, v309_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v304_el, v311_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v306_el, v313_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v321_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v121_vlead];
        float v322_el = v321_data[0];
        tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v322_el, v315_acc, 3, 6, 1);
        float v324_el = v321_data[1];
        tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v324_el, v323_acc, 3, 6, 1);
        float v326_el = v321_data[2];
        tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v326_el, v325_acc, 3, 6, 1);
        float v328_el = v321_data[3];
        tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v328_el, v327_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v322_el, v329_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v324_el, v331_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v326_el, v333_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v328_el, v335_acc, 3, 7, 2);
        r1[8] = (v337_acc[0]);
        r1[10] = (v337_acc[1]);
        r1[12] = (v337_acc[2]);
        r1[14] = (v337_acc[3]);
        tensorforge::VectorT<float, 4> v342_acc{};
        tensorforge::VectorT<float, 4> v350_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v150_vlead];
        float v351_el = v350_data[0];
        tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v351_el, v342_acc, 3, 0, 1);
        float v353_el = v350_data[1];
        tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v353_el, v352_acc, 3, 0, 1);
        float v355_el = v350_data[2];
        tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v355_el, v354_acc, 3, 0, 1);
        float v357_el = v350_data[3];
        tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v357_el, v356_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v351_el, v358_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v353_el, v360_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v355_el, v362_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v357_el, v364_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v372_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v172_vlead];
        float v373_el = v372_data[0];
        tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v373_el, v366_acc, 3, 2, 1);
        float v375_el = v372_data[1];
        tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v375_el, v374_acc, 3, 2, 1);
        float v377_el = v372_data[2];
        tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v377_el, v376_acc, 3, 2, 1);
        float v379_el = v372_data[3];
        tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v379_el, v378_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v373_el, v380_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v375_el, v382_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v377_el, v384_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v379_el, v386_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v394_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v194_vlead];
        float v395_el = v394_data[0];
        tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v395_el, v388_acc, 3, 4, 1);
        float v397_el = v394_data[1];
        tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v397_el, v396_acc, 3, 4, 1);
        float v399_el = v394_data[2];
        tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v399_el, v398_acc, 3, 4, 1);
        float v401_el = v394_data[3];
        tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v401_el, v400_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v395_el, v402_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v397_el, v404_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v399_el, v406_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v401_el, v408_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v416_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v216_vlead];
        float v417_el = v416_data[0];
        tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v417_el, v410_acc, 3, 6, 1);
        float v419_el = v416_data[1];
        tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v419_el, v418_acc, 3, 6, 1);
        float v421_el = v416_data[2];
        tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v421_el, v420_acc, 3, 6, 1);
        float v423_el = v416_data[3];
        tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v423_el, v422_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v417_el, v424_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v419_el, v426_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v421_el, v428_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v423_el, v430_acc, 3, 7, 2);
        r1[9] = (v432_acc[0]);
        r1[11] = (v432_acc[1]);
        r1[13] = (v432_acc[2]);
        r1[15] = (v432_acc[3]);
        float v437_data = r0[8];
        float v438_data = r0[9];
        float v439_data = r0[10];
        float v440_data = r0[11];
        float v441_tp{};
        float v442_tp{};
        float v443_tp{};
        float v444_tp{};
        tensorforge::transpose4x4b32(v441_tp, v442_tp, v443_tp, v444_tp, v437_data, v438_data, v439_data, v440_data);
        tensorforge::VectorT<float, 4> v445_acc{};
        tensorforge::VectorT<float, 4> v453_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v55_vlead];
        float v454_el = v453_data[0];
        tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v454_el, v445_acc, 3, 0, 1);
        float v456_el = v453_data[1];
        tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v456_el, v455_acc, 3, 0, 1);
        float v458_el = v453_data[2];
        tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v458_el, v457_acc, 3, 0, 1);
        float v460_el = v453_data[3];
        tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v460_el, v459_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v454_el, v461_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v456_el, v463_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v458_el, v465_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v460_el, v467_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v475_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v77_vlead];
        float v476_el = v475_data[0];
        tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v476_el, v469_acc, 3, 2, 1);
        float v478_el = v475_data[1];
        tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v478_el, v477_acc, 3, 2, 1);
        float v480_el = v475_data[2];
        tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v480_el, v479_acc, 3, 2, 1);
        float v482_el = v475_data[3];
        tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v482_el, v481_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v476_el, v483_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v478_el, v485_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v480_el, v487_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v482_el, v489_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v497_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v99_vlead];
        float v498_el = v497_data[0];
        tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v498_el, v491_acc, 3, 4, 1);
        float v500_el = v497_data[1];
        tensorforge::VectorT<float, 4> v501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v500_el, v499_acc, 3, 4, 1);
        float v502_el = v497_data[2];
        tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v502_el, v501_acc, 3, 4, 1);
        float v504_el = v497_data[3];
        tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v504_el, v503_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v498_el, v505_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v500_el, v507_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v502_el, v509_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v504_el, v511_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v519_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v121_vlead];
        float v520_el = v519_data[0];
        tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v520_el, v513_acc, 3, 6, 1);
        float v522_el = v519_data[1];
        tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v522_el, v521_acc, 3, 6, 1);
        float v524_el = v519_data[2];
        tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v524_el, v523_acc, 3, 6, 1);
        float v526_el = v519_data[3];
        tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v526_el, v525_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v520_el, v527_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v522_el, v529_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v533_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v524_el, v531_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v526_el, v533_acc, 3, 7, 2);
        r1[16] = (v535_acc[0]);
        r1[18] = (v535_acc[1]);
        r1[20] = (v535_acc[2]);
        r1[22] = (v535_acc[3]);
        tensorforge::VectorT<float, 4> v540_acc{};
        tensorforge::VectorT<float, 4> v548_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v150_vlead];
        float v549_el = v548_data[0];
        tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v549_el, v540_acc, 3, 0, 1);
        float v551_el = v548_data[1];
        tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v551_el, v550_acc, 3, 0, 1);
        float v553_el = v548_data[2];
        tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v553_el, v552_acc, 3, 0, 1);
        float v555_el = v548_data[3];
        tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v555_el, v554_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v549_el, v556_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v551_el, v558_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v553_el, v560_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v555_el, v562_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v570_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v172_vlead];
        float v571_el = v570_data[0];
        tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v571_el, v564_acc, 3, 2, 1);
        float v573_el = v570_data[1];
        tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v573_el, v572_acc, 3, 2, 1);
        float v575_el = v570_data[2];
        tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v575_el, v574_acc, 3, 2, 1);
        float v577_el = v570_data[3];
        tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v577_el, v576_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v571_el, v578_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v573_el, v580_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v575_el, v582_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v577_el, v584_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v592_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v194_vlead];
        float v593_el = v592_data[0];
        tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v593_el, v586_acc, 3, 4, 1);
        float v595_el = v592_data[1];
        tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v595_el, v594_acc, 3, 4, 1);
        float v597_el = v592_data[2];
        tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v597_el, v596_acc, 3, 4, 1);
        float v599_el = v592_data[3];
        tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v599_el, v598_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v593_el, v600_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v595_el, v602_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v597_el, v604_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v599_el, v606_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v614_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v216_vlead];
        float v615_el = v614_data[0];
        tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v615_el, v608_acc, 3, 6, 1);
        float v617_el = v614_data[1];
        tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v617_el, v616_acc, 3, 6, 1);
        float v619_el = v614_data[2];
        tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v619_el, v618_acc, 3, 6, 1);
        float v621_el = v614_data[3];
        tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v621_el, v620_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v615_el, v622_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v617_el, v624_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v443_tp, v619_el, v626_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v444_tp, v621_el, v628_acc, 3, 7, 2);
        r1[17] = (v630_acc[0]);
        r1[19] = (v630_acc[1]);
        r1[21] = (v630_acc[2]);
        r1[23] = (v630_acc[3]);
        float v635_data = r0[12];
        float v636_data = r0[13];
        float v637_data = r0[14];
        float v638_data = r0[15];
        float v639_tp{};
        float v640_tp{};
        float v641_tp{};
        float v642_tp{};
        tensorforge::transpose4x4b32(v639_tp, v640_tp, v641_tp, v642_tp, v635_data, v636_data, v637_data, v638_data);
        tensorforge::VectorT<float, 4> v643_acc{};
        tensorforge::VectorT<float, 4> v651_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v55_vlead];
        float v652_el = v651_data[0];
        tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v652_el, v643_acc, 3, 0, 1);
        float v654_el = v651_data[1];
        tensorforge::VectorT<float, 4> v655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v654_el, v653_acc, 3, 0, 1);
        float v656_el = v651_data[2];
        tensorforge::VectorT<float, 4> v657_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v656_el, v655_acc, 3, 0, 1);
        float v658_el = v651_data[3];
        tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v658_el, v657_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v652_el, v659_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v654_el, v661_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v656_el, v663_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v658_el, v665_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v673_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v77_vlead];
        float v674_el = v673_data[0];
        tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v674_el, v667_acc, 3, 2, 1);
        float v676_el = v673_data[1];
        tensorforge::VectorT<float, 4> v677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v676_el, v675_acc, 3, 2, 1);
        float v678_el = v673_data[2];
        tensorforge::VectorT<float, 4> v679_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v678_el, v677_acc, 3, 2, 1);
        float v680_el = v673_data[3];
        tensorforge::VectorT<float, 4> v681_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v680_el, v679_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v674_el, v681_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v676_el, v683_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v687_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v678_el, v685_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v689_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v680_el, v687_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v695_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v99_vlead];
        float v696_el = v695_data[0];
        tensorforge::VectorT<float, 4> v697_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v696_el, v689_acc, 3, 4, 1);
        float v698_el = v695_data[1];
        tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v698_el, v697_acc, 3, 4, 1);
        float v700_el = v695_data[2];
        tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v700_el, v699_acc, 3, 4, 1);
        float v702_el = v695_data[3];
        tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v702_el, v701_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v696_el, v703_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v698_el, v705_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v700_el, v707_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v702_el, v709_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v717_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v121_vlead];
        float v718_el = v717_data[0];
        tensorforge::VectorT<float, 4> v719_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v718_el, v711_acc, 3, 6, 1);
        float v720_el = v717_data[1];
        tensorforge::VectorT<float, 4> v721_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v720_el, v719_acc, 3, 6, 1);
        float v722_el = v717_data[2];
        tensorforge::VectorT<float, 4> v723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v722_el, v721_acc, 3, 6, 1);
        float v724_el = v717_data[3];
        tensorforge::VectorT<float, 4> v725_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v724_el, v723_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v727_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v718_el, v725_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v729_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v720_el, v727_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v731_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v722_el, v729_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v733_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v724_el, v731_acc, 3, 7, 2);
        r1[24] = (v733_acc[0]);
        r1[26] = (v733_acc[1]);
        r1[28] = (v733_acc[2]);
        r1[30] = (v733_acc[3]);
        tensorforge::VectorT<float, 4> v738_acc{};
        tensorforge::VectorT<float, 4> v746_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v150_vlead];
        float v747_el = v746_data[0];
        tensorforge::VectorT<float, 4> v748_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v747_el, v738_acc, 3, 0, 1);
        float v749_el = v746_data[1];
        tensorforge::VectorT<float, 4> v750_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v749_el, v748_acc, 3, 0, 1);
        float v751_el = v746_data[2];
        tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v751_el, v750_acc, 3, 0, 1);
        float v753_el = v746_data[3];
        tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v753_el, v752_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v756_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v747_el, v754_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v749_el, v756_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v751_el, v758_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v762_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v753_el, v760_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v768_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v172_vlead];
        float v769_el = v768_data[0];
        tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v769_el, v762_acc, 3, 2, 1);
        float v771_el = v768_data[1];
        tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v771_el, v770_acc, 3, 2, 1);
        float v773_el = v768_data[2];
        tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v773_el, v772_acc, 3, 2, 1);
        float v775_el = v768_data[3];
        tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v775_el, v774_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v769_el, v776_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v771_el, v778_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v773_el, v780_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v775_el, v782_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v790_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v194_vlead];
        float v791_el = v790_data[0];
        tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v791_el, v784_acc, 3, 4, 1);
        float v793_el = v790_data[1];
        tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v793_el, v792_acc, 3, 4, 1);
        float v795_el = v790_data[2];
        tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v795_el, v794_acc, 3, 4, 1);
        float v797_el = v790_data[3];
        tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v797_el, v796_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v800_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v791_el, v798_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v802_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v793_el, v800_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v804_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v795_el, v802_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v806_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v797_el, v804_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v812_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v216_vlead];
        float v813_el = v812_data[0];
        tensorforge::VectorT<float, 4> v814_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v813_el, v806_acc, 3, 6, 1);
        float v815_el = v812_data[1];
        tensorforge::VectorT<float, 4> v816_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v815_el, v814_acc, 3, 6, 1);
        float v817_el = v812_data[2];
        tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v817_el, v816_acc, 3, 6, 1);
        float v819_el = v812_data[3];
        tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v819_el, v818_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v639_tp, v813_el, v820_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v824_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v640_tp, v815_el, v822_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v641_tp, v817_el, v824_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v642_tp, v819_el, v826_acc, 3, 7, 2);
        r1[25] = (v828_acc[0]);
        r1[27] = (v828_acc[1]);
        r1[29] = (v828_acc[2]);
        r1[31] = (v828_acc[3]);
        float v833_data = r0[16];
        float v834_data = r0[17];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v1063_i0 = 0; v1063_i0 < 1; ++v1063_i0) {
          int32_t v1065_lead = v31_lead + (v1063_i0 * 32);
          #pragma unroll
          for (int32_t v1066_i1 = 0; v1066_i1 < 18; ++v1066_i1) {
            float v1069_data = __builtin_nontemporal_load(&wrap_glb_m2[(v1065_lead + (v1066_i1 * 32))]);
            r0[(v1063_i0 + v1066_i1)] = v1069_data;
          }
        }
        float v836_tp{};
        float v837_tp{};
        float v838_tp{};
        float v839_tp{};
        tensorforge::transpose4x4b32(v836_tp, v837_tp, v838_tp, v839_tp, v833_data, v834_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v840_acc{};
        tensorforge::VectorT<float, 4> v848_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v55_vlead];
        float v849_el = v848_data[0];
        tensorforge::VectorT<float, 4> v850_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v849_el, v840_acc, 3, 0, 1);
        float v851_el = v848_data[1];
        tensorforge::VectorT<float, 4> v852_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v851_el, v850_acc, 3, 0, 1);
        float v853_el = v848_data[2];
        tensorforge::VectorT<float, 4> v854_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v853_el, v852_acc, 3, 0, 1);
        float v855_el = v848_data[3];
        tensorforge::VectorT<float, 4> v856_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v855_el, v854_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v858_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v849_el, v856_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v860_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v851_el, v858_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v862_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v853_el, v860_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v864_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v855_el, v862_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v870_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v77_vlead];
        float v871_el = v870_data[0];
        tensorforge::VectorT<float, 4> v872_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v871_el, v864_acc, 3, 2, 1);
        float v873_el = v870_data[1];
        tensorforge::VectorT<float, 4> v874_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v873_el, v872_acc, 3, 2, 1);
        float v875_el = v870_data[2];
        tensorforge::VectorT<float, 4> v876_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v875_el, v874_acc, 3, 2, 1);
        float v877_el = v870_data[3];
        tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v877_el, v876_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v880_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v871_el, v878_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v882_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v873_el, v880_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v875_el, v882_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v877_el, v884_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v892_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v99_vlead];
        float v893_el = v892_data[0];
        tensorforge::VectorT<float, 4> v894_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v893_el, v886_acc, 3, 4, 1);
        float v895_el = v892_data[1];
        tensorforge::VectorT<float, 4> v896_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v895_el, v894_acc, 3, 4, 1);
        float v897_el = v892_data[2];
        tensorforge::VectorT<float, 4> v898_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v897_el, v896_acc, 3, 4, 1);
        float v899_el = v892_data[3];
        tensorforge::VectorT<float, 4> v900_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v899_el, v898_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v902_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v893_el, v900_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v895_el, v902_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v906_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v897_el, v904_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v908_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v899_el, v906_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v914_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v121_vlead];
        float v915_el = v914_data[0];
        tensorforge::VectorT<float, 4> v916_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v915_el, v908_acc, 3, 6, 1);
        float v917_el = v914_data[1];
        tensorforge::VectorT<float, 4> v918_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v917_el, v916_acc, 3, 6, 1);
        float v919_el = v914_data[2];
        tensorforge::VectorT<float, 4> v920_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v919_el, v918_acc, 3, 6, 1);
        float v921_el = v914_data[3];
        tensorforge::VectorT<float, 4> v922_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v921_el, v920_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v924_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v915_el, v922_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v926_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v917_el, v924_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v928_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v919_el, v926_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v930_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v921_el, v928_acc, 3, 7, 2);
        r1[32] = (v930_acc[0]);
        r1[34] = (v930_acc[1]);
        tensorforge::VectorT<float, 4> v933_acc{};
        tensorforge::VectorT<float, 4> v941_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v150_vlead];
        float v942_el = v941_data[0];
        tensorforge::VectorT<float, 4> v943_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v942_el, v933_acc, 3, 0, 1);
        float v944_el = v941_data[1];
        tensorforge::VectorT<float, 4> v945_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v944_el, v943_acc, 3, 0, 1);
        float v946_el = v941_data[2];
        tensorforge::VectorT<float, 4> v947_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v946_el, v945_acc, 3, 0, 1);
        float v948_el = v941_data[3];
        tensorforge::VectorT<float, 4> v949_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v948_el, v947_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v951_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v942_el, v949_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v953_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v944_el, v951_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v955_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v946_el, v953_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v957_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v948_el, v955_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v963_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v172_vlead];
        float v964_el = v963_data[0];
        tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v964_el, v957_acc, 3, 2, 1);
        float v966_el = v963_data[1];
        tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v966_el, v965_acc, 3, 2, 1);
        float v968_el = v963_data[2];
        tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v968_el, v967_acc, 3, 2, 1);
        float v970_el = v963_data[3];
        tensorforge::VectorT<float, 4> v971_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v970_el, v969_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v973_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v964_el, v971_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v975_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v966_el, v973_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v977_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v968_el, v975_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v979_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v970_el, v977_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v985_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v194_vlead];
        float v986_el = v985_data[0];
        tensorforge::VectorT<float, 4> v987_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v986_el, v979_acc, 3, 4, 1);
        float v988_el = v985_data[1];
        tensorforge::VectorT<float, 4> v989_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v988_el, v987_acc, 3, 4, 1);
        float v990_el = v985_data[2];
        tensorforge::VectorT<float, 4> v991_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v990_el, v989_acc, 3, 4, 1);
        float v992_el = v985_data[3];
        tensorforge::VectorT<float, 4> v993_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v992_el, v991_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v995_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v986_el, v993_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v997_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v988_el, v995_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v999_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v990_el, v997_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1001_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v992_el, v999_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1007_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v216_vlead];
        float v1008_el = v1007_data[0];
        tensorforge::VectorT<float, 4> v1009_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v1008_el, v1001_acc, 3, 6, 1);
        float v1010_el = v1007_data[1];
        tensorforge::VectorT<float, 4> v1011_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v1010_el, v1009_acc, 3, 6, 1);
        float v1012_el = v1007_data[2];
        tensorforge::VectorT<float, 4> v1013_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v1012_el, v1011_acc, 3, 6, 1);
        float v1014_el = v1007_data[3];
        tensorforge::VectorT<float, 4> v1015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v1014_el, v1013_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v1017_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v836_tp, v1008_el, v1015_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1019_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v837_tp, v1010_el, v1017_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1021_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v838_tp, v1012_el, v1019_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1023_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v839_tp, v1014_el, v1021_acc, 3, 7, 2);
        r1[33] = (v1023_acc[0]);
        r1[35] = (v1023_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v1026_i0 = 0; v1026_i0 < 1; ++v1026_i0) {
          #pragma unroll
          for (int32_t v1027_i1 = 0; v1027_i1 < 18; ++v1027_i1) {
            float v1030_data = r1[(v1026_i0 + (v1027_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v31_lead + (v1026_i0 * 32)) + (v1027_i1 * 56))] = v1030_data;
            }
          }
        }
        if (v1035_g) {
          #pragma unroll
          for (int32_t v1036_i1 = 0; v1036_i1 < 18; ++v1036_i1) {
            float v1039_data = r1[(1 + (v1036_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v31_lead + 32_i32) + (v1036_i1 * 56))] = v1039_data;
            }
          }
        }
      }
    }
  }
}

