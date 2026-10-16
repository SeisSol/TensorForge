// === base name ===
kernel_80290a7c1fab2bb8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_80290a7c1fab2bb8 = {{32, 8, 1}, 32, 32, 1, 8, 6144, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_80290a7c1fab2bb8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_80290a7c1fab2bb8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_80290a7c1fab2bb8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_80290a7c1fab2bb8, block.x * block.y * block.z, 1536 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (1536 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_80290a7c1fab2bb8, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (1536 * sizeof(float)));
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
  config.sharedMemBytes = 1536 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_80290a7c1fab2bb8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_80290a7c1fab2bb8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_80290a7c1fab2bb8), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_80290a7c1fab2bb8, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_80290a7c1fab2bb8(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=bits
    // launch: 32 lanes x 8 per block = block 32x8x1, 6144 B shared, occupancy grid
    // operands:
    //   m0 32×6(32×6) {0..32}×{0..6} strided
    //   m1 32×48(32×48) {0..32}×{0..48} none ordered
    //   m2 48×6(48×6) {0..48}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1536}],"shared_bytes":6144,"shared_elements":1536,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[32,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[32,48]],"name":"m1","ordered":true,"parts":1,"shape":[32,48],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[48,6]],"name":"m2","ordered":false,"parts":1,"shape":[48,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[32,48]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,48]},{"addressing":"strided","bbox":[[0,0],[48,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[48,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 1536];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 4> v9_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 4>*)&glb_m1[0 + 0 + 4 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      tensorforge::VectorT<float, 2> v10_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024]);
      *(tensorforge::VectorT<float, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 1024] = v10_ld;
      __syncthreads();
      size_t v12_batchIdLane0 = threadIdx.y % 2;
      size_t v15_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v31_lead = threadIdx.x % 32;
      int32_t v43_lead = v31_lead + 32_i32;
      bool v46_g = v31_lead < 16;
      int32_t v68_m = threadIdx.y % 2;
      int32_t v73_vlead = (v31_lead + (v68_m * 32)) * 4;
      int32_t v95_vlead = (v31_lead + ((v68_m + 2) * 32)) * 4;
      int32_t v117_vlead = (v31_lead + ((v68_m + 4) * 32)) * 4;
      int32_t v139_vlead = (v31_lead + ((v68_m + 6) * 32)) * 4;
      int32_t v161_vlead = (v31_lead + ((v68_m + 8) * 32)) * 4;
      int32_t v183_vlead = (v31_lead + ((v68_m + 10) * 32)) * 4;
      float r0[12]{};
      size_t v367_row = v15_batchIdGroup0First + v12_batchIdLane0;
      const bool peel_batchIdActive0 = v367_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v367_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v367_row : v15_batchIdGroup0First) * 288 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v375_i0 = 0; v375_i0 < 1; ++v375_i0) {
        int32_t v377_lead = v31_lead + (v375_i0 * 32);
        #pragma unroll
        for (int32_t v378_i1 = 0; v378_i1 < 6; ++v378_i1) {
          float v381_data = __builtin_nontemporal_load(&peel_glb_m2[(v377_lead + (v378_i1 * 48))]);
          r0[(v375_i0 + (v378_i1 * 2))] = v381_data;
        }
      }
      #pragma unroll
      for (int32_t v384_i1 = 0; v384_i1 < 6; ++v384_i1) {
        float v387_data = v46_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v43_lead + (v384_i1 * 48))])) : (0.0f);
        r0[(1 + (v384_i1 * 2))] = v387_data;
      }
      for (size_t v16_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v16_batchIdGroup0 < numElements0; v16_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v364_groupNextAhead = v16_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v366_groupNext = (v364_groupNextAhead < numElements0) ? v364_groupNextAhead : v16_batchIdGroup0;
        size_t v371_row = v366_groupNext + v12_batchIdLane0;
        const bool wrap_batchIdActive0 = v371_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v371_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v371_row : v366_groupNext) * 288 + 0 + m2_extraOffset];
        size_t v17_row = v16_batchIdGroup0 + v12_batchIdLane0;
        const bool batchIdActive0 = v17_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v17_row]));
        size_t v19_batchId0 = batchIdActive0 ? v17_row : v16_batchIdGroup0;
        size_t v20_ahead1 = v19_batchId0 + (gridDim.x * blockDim.y);
        size_t v22_batchId1 = (v20_ahead1 < numElements0) ? v20_ahead1 : v19_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v19_batchId0 * 192 + 0 + m0_extraOffset];
        float r1[6]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 32), (0, 6)] [(0, 48)]
        float v51_data = r0[0];
        float v52_data = r0[2];
        float v53_data = r0[4];
        float v54_data = r0[6];
        float v55_tp{};
        float v56_tp{};
        float v57_tp{};
        float v58_tp{};
        tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
        float v59_data = r0[1];
        float v60_data = r0[3];
        float v61_data = r0[5];
        float v62_data = r0[7];
        float v63_tp{};
        float v64_tp{};
        float v65_tp{};
        float v66_tp{};
        tensorforge::transpose4x4b32(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
        tensorforge::VectorT<float, 4> v67_acc{};
        tensorforge::VectorT<float, 4> v75_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v73_vlead];
        float v76_el = v75_data[0];
        tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_el, v67_acc, 3, 0, 1);
        float v78_el = v75_data[1];
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v78_el, v77_acc, 3, 0, 1);
        float v80_el = v75_data[2];
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v80_el, v79_acc, 3, 0, 1);
        float v82_el = v75_data[3];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v82_el, v81_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_el, v83_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v78_el, v85_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v80_el, v87_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v82_el, v89_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v97_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v95_vlead];
        float v98_el = v97_data[0];
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v98_el, v91_acc, 3, 2, 1);
        float v100_el = v97_data[1];
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v100_el, v99_acc, 3, 2, 1);
        float v102_el = v97_data[2];
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v102_el, v101_acc, 3, 2, 1);
        float v104_el = v97_data[3];
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v104_el, v103_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v98_el, v105_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v100_el, v107_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v102_el, v109_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v104_el, v111_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v119_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v117_vlead];
        float v120_el = v119_data[0];
        tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v120_el, v113_acc, 3, 4, 1);
        float v122_el = v119_data[1];
        tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v122_el, v121_acc, 3, 4, 1);
        float v124_el = v119_data[2];
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v124_el, v123_acc, 3, 4, 1);
        float v126_el = v119_data[3];
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v126_el, v125_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v120_el, v127_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v122_el, v129_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v124_el, v131_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v126_el, v133_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v141_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v139_vlead];
        float v142_el = v141_data[0];
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v142_el, v135_acc, 3, 6, 1);
        float v144_el = v141_data[1];
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v144_el, v143_acc, 3, 6, 1);
        float v146_el = v141_data[2];
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v146_el, v145_acc, 3, 6, 1);
        float v148_el = v141_data[3];
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v148_el, v147_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v142_el, v149_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v144_el, v151_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v146_el, v153_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v148_el, v155_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v163_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v161_vlead];
        float v164_el = v163_data[0];
        tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v164_el, v157_acc, 3, 0, 1);
        float v166_el = v163_data[1];
        tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v166_el, v165_acc, 3, 0, 1);
        float v168_el = v163_data[2];
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v168_el, v167_acc, 3, 0, 1);
        float v170_el = v163_data[3];
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v170_el, v169_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v164_el, v171_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v166_el, v173_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v168_el, v175_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v170_el, v177_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v185_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v183_vlead];
        float v186_el = v185_data[0];
        tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v186_el, v179_acc, 3, 2, 1);
        float v188_el = v185_data[1];
        tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v188_el, v187_acc, 3, 2, 1);
        float v190_el = v185_data[2];
        tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v190_el, v189_acc, 3, 2, 1);
        float v192_el = v185_data[3];
        tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v192_el, v191_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v186_el, v193_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v188_el, v195_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v190_el, v197_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v192_el, v199_acc, 3, 3, 2);
        r1[0] = (v201_acc[0]);
        r1[1] = (v201_acc[1]);
        r1[2] = (v201_acc[2]);
        r1[3] = (v201_acc[3]);
        float v206_data = r0[8];
        float v207_data = r0[10];
        float v209_tp{};
        float v210_tp{};
        float v211_tp{};
        float v212_tp{};
        tensorforge::transpose4x4b32(v209_tp, v210_tp, v211_tp, v212_tp, v206_data, v207_data, 0.0f, 0.0f);
        float v213_data = r0[9];
        float v214_data = r0[11];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v390_i0 = 0; v390_i0 < 1; ++v390_i0) {
          int32_t v392_lead = v31_lead + (v390_i0 * 32);
          #pragma unroll
          for (int32_t v393_i1 = 0; v393_i1 < 6; ++v393_i1) {
            float v396_data = __builtin_nontemporal_load(&wrap_glb_m2[(v392_lead + (v393_i1 * 48))]);
            r0[(v390_i0 + (v393_i1 * 2))] = v396_data;
          }
        }
        #pragma unroll
        for (int32_t v399_i1 = 0; v399_i1 < 6; ++v399_i1) {
          float v402_data = v46_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v43_lead + (v399_i1 * 48))])) : (0.0f);
          r0[(1 + (v399_i1 * 2))] = v402_data;
        }
        float v215_tp{};
        float v216_tp{};
        float v217_tp{};
        float v218_tp{};
        tensorforge::transpose4x4b32(v215_tp, v216_tp, v217_tp, v218_tp, v213_data, v214_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v219_acc{};
        tensorforge::VectorT<float, 4> v227_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v73_vlead];
        float v228_el = v227_data[0];
        tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v228_el, v219_acc, 3, 0, 1);
        float v230_el = v227_data[1];
        tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v230_el, v229_acc, 3, 0, 1);
        float v232_el = v227_data[2];
        tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v232_el, v231_acc, 3, 0, 1);
        float v234_el = v227_data[3];
        tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v234_el, v233_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v228_el, v235_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v230_el, v237_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v232_el, v239_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v234_el, v241_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v249_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v95_vlead];
        float v250_el = v249_data[0];
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v250_el, v243_acc, 3, 2, 1);
        float v252_el = v249_data[1];
        tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v252_el, v251_acc, 3, 2, 1);
        float v254_el = v249_data[2];
        tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v254_el, v253_acc, 3, 2, 1);
        float v256_el = v249_data[3];
        tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v256_el, v255_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v250_el, v257_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v252_el, v259_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v254_el, v261_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v256_el, v263_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v271_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v117_vlead];
        float v272_el = v271_data[0];
        tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v272_el, v265_acc, 3, 4, 1);
        float v274_el = v271_data[1];
        tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v274_el, v273_acc, 3, 4, 1);
        float v276_el = v271_data[2];
        tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v276_el, v275_acc, 3, 4, 1);
        float v278_el = v271_data[3];
        tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v278_el, v277_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v272_el, v279_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v274_el, v281_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v276_el, v283_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v278_el, v285_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v293_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v139_vlead];
        float v294_el = v293_data[0];
        tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v294_el, v287_acc, 3, 6, 1);
        float v296_el = v293_data[1];
        tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v296_el, v295_acc, 3, 6, 1);
        float v298_el = v293_data[2];
        tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v298_el, v297_acc, 3, 6, 1);
        float v300_el = v293_data[3];
        tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v300_el, v299_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v294_el, v301_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v296_el, v303_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v298_el, v305_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v300_el, v307_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v315_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v161_vlead];
        float v316_el = v315_data[0];
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v316_el, v309_acc, 3, 0, 1);
        float v318_el = v315_data[1];
        tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v318_el, v317_acc, 3, 0, 1);
        float v320_el = v315_data[2];
        tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v320_el, v319_acc, 3, 0, 1);
        float v322_el = v315_data[3];
        tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v322_el, v321_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v316_el, v323_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v318_el, v325_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v320_el, v327_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v322_el, v329_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v337_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v183_vlead];
        float v338_el = v337_data[0];
        tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v338_el, v331_acc, 3, 2, 1);
        float v340_el = v337_data[1];
        tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v340_el, v339_acc, 3, 2, 1);
        float v342_el = v337_data[2];
        tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v342_el, v341_acc, 3, 2, 1);
        float v344_el = v337_data[3];
        tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v344_el, v343_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v215_tp, v338_el, v345_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v216_tp, v340_el, v347_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v217_tp, v342_el, v349_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v218_tp, v344_el, v351_acc, 3, 3, 2);
        r1[4] = (v353_acc[0]);
        r1[5] = (v353_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v356_i0 = 0; v356_i0 < 1; ++v356_i0) {
          #pragma unroll
          for (int32_t v357_i1 = 0; v357_i1 < 6; ++v357_i1) {
            float v359_data = r1[(v356_i0 + v357_i1)];
            if (batchIdActive0) {
              glb_m0[((v31_lead + (v356_i0 * 32)) + (v357_i1 * 32))] = v359_data;
            }
          }
        }
      }
    }
  }
}

