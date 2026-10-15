// === base name ===
kernel_5b8af33f12cd7b49

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5b8af33f12cd7b49 = {{32, 8, 1}, 32, 56, 1, 8, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5b8af33f12cd7b49(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5b8af33f12cd7b49(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5b8af33f12cd7b49(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5b8af33f12cd7b49, block.x * block.y * block.z, 1024 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1024 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5b8af33f12cd7b49, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (1024 * sizeof(float)));
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
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5b8af33f12cd7b49(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5b8af33f12cd7b49(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5b8af33f12cd7b49), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5b8af33f12cd7b49, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5b8af33f12cd7b49(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=2,trip_reads=1
    // launch: 32 lanes (56 active) x 8 per block = block 32x8x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 56×6(56×6) {0..56}×{0..6} strided
    //   m1 56×16(56×16) {0..56}×{0..16} none ordered
    //   m2 16×6(16×6) {0..16}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,6]],"name":"m0","ordered":false,"parts":1,"shape":[56,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,16]],"name":"m1","ordered":true,"parts":1,"shape":[56,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,6]],"name":"m2","ordered":false,"parts":1,"shape":[16,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,16]},{"addressing":"strided","bbox":[[0,0],[16,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 1024];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 4> v9_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 2;
      size_t v14_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v30_lead = threadIdx.x % 32;
      bool v36_g = v30_lead < 16;
      int32_t v50_r = (threadIdx.y % 2) * 2;
      int32_t v54_vlead = (v30_lead + (v50_r * 32)) * 4;
      int32_t v76_vlead = (v30_lead + ((v50_r + 4) * 32)) * 4;
      int32_t v105_vlead = (v30_lead + ((v50_r + 1) * 32)) * 4;
      int32_t v127_vlead = (v30_lead + ((v50_r + 5) * 32)) * 4;
      bool v264_g = v30_lead < 24;
      float r0[6]{};
      size_t v276_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdActive0 = v276_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v276_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v276_row : v14_batchIdGroup0First) * 96 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v284_i1 = 0; v284_i1 < 6; ++v284_i1) {
        float v287_data = v36_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v30_lead + (v284_i1 * 16))])) : (0.0f);
        r0[v284_i1] = v287_data;
      }
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v273_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v275_groupNext = (v273_groupNextAhead < numElements0) ? v273_groupNextAhead : v15_batchIdGroup0;
        size_t v280_row = v275_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdActive0 = v280_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v280_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v280_row : v275_groupNext) * 96 + 0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 336 + 0 + m0_extraOffset];
        float r1[12]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 56), (0, 6)] [(0, 16)]
        float v40_data = r0[0];
        float v41_data = r0[1];
        float v42_data = r0[2];
        float v43_data = r0[3];
        float v44_tp{};
        float v45_tp{};
        float v46_tp{};
        float v47_tp{};
        tensorforge::transpose4x4b32(v44_tp, v45_tp, v46_tp, v47_tp, v40_data, v41_data, v42_data, v43_data);
        tensorforge::VectorT<float, 4> v48_acc{};
        tensorforge::VectorT<float, 4> v56_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v54_vlead];
        float v57_el = v56_data[0];
        tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_el, v48_acc, 3, 0, 1);
        float v59_el = v56_data[1];
        tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v59_el, v58_acc, 3, 0, 1);
        float v61_el = v56_data[2];
        tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v61_el, v60_acc, 3, 0, 1);
        float v63_el = v56_data[3];
        tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v63_el, v62_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_el, v64_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v59_el, v66_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v61_el, v68_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v63_el, v70_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v78_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v76_vlead];
        float v79_el = v78_data[0];
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v79_el, v72_acc, 3, 2, 1);
        float v81_el = v78_data[1];
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v81_el, v80_acc, 3, 2, 1);
        float v83_el = v78_data[2];
        tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v83_el, v82_acc, 3, 2, 1);
        float v85_el = v78_data[3];
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v85_el, v84_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v79_el, v86_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v81_el, v88_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v83_el, v90_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v85_el, v92_acc, 3, 3, 2);
        r1[0] = (v94_acc[0]);
        r1[2] = (v94_acc[1]);
        r1[4] = (v94_acc[2]);
        r1[6] = (v94_acc[3]);
        tensorforge::VectorT<float, 4> v99_acc{};
        tensorforge::VectorT<float, 4> v107_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v105_vlead];
        float v108_el = v107_data[0];
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v108_el, v99_acc, 3, 0, 1);
        float v110_el = v107_data[1];
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v110_el, v109_acc, 3, 0, 1);
        float v112_el = v107_data[2];
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v112_el, v111_acc, 3, 0, 1);
        float v114_el = v107_data[3];
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v114_el, v113_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v108_el, v115_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v110_el, v117_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v112_el, v119_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v114_el, v121_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v129_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v127_vlead];
        float v130_el = v129_data[0];
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v130_el, v123_acc, 3, 2, 1);
        float v132_el = v129_data[1];
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v132_el, v131_acc, 3, 2, 1);
        float v134_el = v129_data[2];
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v134_el, v133_acc, 3, 2, 1);
        float v136_el = v129_data[3];
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v136_el, v135_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v130_el, v137_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v132_el, v139_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v134_el, v141_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v136_el, v143_acc, 3, 3, 2);
        r1[1] = (v145_acc[0]);
        r1[3] = (v145_acc[1]);
        r1[5] = (v145_acc[2]);
        r1[7] = (v145_acc[3]);
        float v150_data = r0[4];
        float v151_data = r0[5];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v288_i1 = 0; v288_i1 < 6; ++v288_i1) {
          float v291_data = v36_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v30_lead + (v288_i1 * 16))])) : (0.0f);
          r0[v288_i1] = v291_data;
        }
        float v153_tp{};
        float v154_tp{};
        float v155_tp{};
        float v156_tp{};
        tensorforge::transpose4x4b32(v153_tp, v154_tp, v155_tp, v156_tp, v150_data, v151_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v157_acc{};
        tensorforge::VectorT<float, 4> v165_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v54_vlead];
        float v166_el = v165_data[0];
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v166_el, v157_acc, 3, 0, 1);
        float v168_el = v165_data[1];
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v168_el, v167_acc, 3, 0, 1);
        float v170_el = v165_data[2];
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v170_el, v169_acc, 3, 0, 1);
        float v172_el = v165_data[3];
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v172_el, v171_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v166_el, v173_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v168_el, v175_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v170_el, v177_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v172_el, v179_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v187_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v76_vlead];
        float v188_el = v187_data[0];
        tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v188_el, v181_acc, 3, 2, 1);
        float v190_el = v187_data[1];
        tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v190_el, v189_acc, 3, 2, 1);
        float v192_el = v187_data[2];
        tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v192_el, v191_acc, 3, 2, 1);
        float v194_el = v187_data[3];
        tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v194_el, v193_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v188_el, v195_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v190_el, v197_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v192_el, v199_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v194_el, v201_acc, 3, 3, 2);
        r1[8] = (v203_acc[0]);
        r1[10] = (v203_acc[1]);
        tensorforge::VectorT<float, 4> v206_acc{};
        tensorforge::VectorT<float, 4> v214_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v105_vlead];
        float v215_el = v214_data[0];
        tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v215_el, v206_acc, 3, 0, 1);
        float v217_el = v214_data[1];
        tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v217_el, v216_acc, 3, 0, 1);
        float v219_el = v214_data[2];
        tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v219_el, v218_acc, 3, 0, 1);
        float v221_el = v214_data[3];
        tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v221_el, v220_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v215_el, v222_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v217_el, v224_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v219_el, v226_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v221_el, v228_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v236_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v127_vlead];
        float v237_el = v236_data[0];
        tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v237_el, v230_acc, 3, 2, 1);
        float v239_el = v236_data[1];
        tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v239_el, v238_acc, 3, 2, 1);
        float v241_el = v236_data[2];
        tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v241_el, v240_acc, 3, 2, 1);
        float v243_el = v236_data[3];
        tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v243_el, v242_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v237_el, v244_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v239_el, v246_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v155_tp, v241_el, v248_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v156_tp, v243_el, v250_acc, 3, 3, 2);
        r1[9] = (v252_acc[0]);
        r1[11] = (v252_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v255_i0 = 0; v255_i0 < 1; ++v255_i0) {
          #pragma unroll
          for (int32_t v256_i1 = 0; v256_i1 < 6; ++v256_i1) {
            float v259_data = r1[(v255_i0 + (v256_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v30_lead + (v255_i0 * 32)) + (v256_i1 * 56))] = v259_data;
            }
          }
        }
        if (v264_g) {
          #pragma unroll
          for (int32_t v265_i1 = 0; v265_i1 < 6; ++v265_i1) {
            float v268_data = r1[(1 + (v265_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v30_lead + 32_i32) + (v265_i1 * 56))] = v268_data;
            }
          }
        }
      }
    }
  }
}

