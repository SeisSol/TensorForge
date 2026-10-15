// === base name ===
kernel_043d88a46de15752

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_043d88a46de15752 = {{32, 8, 1}, 32, 64, 1, 8, 24576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_043d88a46de15752(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_043d88a46de15752(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_043d88a46de15752(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_043d88a46de15752, block.x * block.y * block.z, 6144 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6144 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_043d88a46de15752, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (6144 * sizeof(float)));
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
  config.sharedMemBytes = 6144 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_043d88a46de15752(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_043d88a46de15752(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_043d88a46de15752), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_043d88a46de15752, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_043d88a46de15752(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=rows
    // launch: 32 lanes (64 active) x 8 per block = block 32x8x1, 24576 B shared, occupancy grid
    // operands:
    //   m0 64×6(64×6) {0..64}×{0..6} strided
    //   m1 64×96(64×96) {0..64}×{0..96} none ordered
    //   m2 96×6(96×6) {0..96}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":6144}],"shared_bytes":24576,"shared_elements":6144,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,96]],"name":"m1","ordered":true,"parts":1,"shape":[64,96],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[96,6]],"name":"m2","ordered":false,"parts":1,"shape":[96,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,96]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,96]},{"addressing":"strided","bbox":[[0,0],[96,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[96,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 6144];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      int32_t v14_h = (threadIdx.x + (threadIdx.y * 32)) * 4;
      #pragma unroll
      for (int32_t v13_i = 0; v13_i < 24; v13_i += 4) {
        int32_t v16_a = v14_h + (v13_i * 256);
        tensorforge::VectorT<float, 4> v17_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 4>*)&ptr_glb_m1[v16_a]);
        *(tensorforge::VectorT<float, 4>*)&glb_m1[v16_a] = v17_ld;
      }
      __syncthreads();
      size_t v19_batchIdLane0 = threadIdx.y % 2;
      size_t v22_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v38_lead = threadIdx.x % 32;
      int32_t v75_r = (threadIdx.y % 2) * 2;
      int32_t v79_vlead = (v38_lead + (v75_r * 32)) * 4;
      int32_t v101_vlead = (v38_lead + ((v75_r + 4) * 32)) * 4;
      int32_t v123_vlead = (v38_lead + ((v75_r + 8) * 32)) * 4;
      int32_t v145_vlead = (v38_lead + ((v75_r + 12) * 32)) * 4;
      int32_t v167_vlead = (v38_lead + ((v75_r + 16) * 32)) * 4;
      int32_t v189_vlead = (v38_lead + ((v75_r + 20) * 32)) * 4;
      int32_t v211_vlead = (v38_lead + ((v75_r + 24) * 32)) * 4;
      int32_t v233_vlead = (v38_lead + ((v75_r + 28) * 32)) * 4;
      int32_t v255_vlead = (v38_lead + ((v75_r + 32) * 32)) * 4;
      int32_t v277_vlead = (v38_lead + ((v75_r + 36) * 32)) * 4;
      int32_t v299_vlead = (v38_lead + ((v75_r + 40) * 32)) * 4;
      int32_t v321_vlead = (v38_lead + ((v75_r + 44) * 32)) * 4;
      int32_t v350_vlead = (v38_lead + ((v75_r + 1) * 32)) * 4;
      int32_t v372_vlead = (v38_lead + ((v75_r + 5) * 32)) * 4;
      int32_t v394_vlead = (v38_lead + ((v75_r + 9) * 32)) * 4;
      int32_t v416_vlead = (v38_lead + ((v75_r + 13) * 32)) * 4;
      int32_t v438_vlead = (v38_lead + ((v75_r + 17) * 32)) * 4;
      int32_t v460_vlead = (v38_lead + ((v75_r + 21) * 32)) * 4;
      int32_t v482_vlead = (v38_lead + ((v75_r + 25) * 32)) * 4;
      int32_t v504_vlead = (v38_lead + ((v75_r + 29) * 32)) * 4;
      int32_t v526_vlead = (v38_lead + ((v75_r + 33) * 32)) * 4;
      int32_t v548_vlead = (v38_lead + ((v75_r + 37) * 32)) * 4;
      int32_t v570_vlead = (v38_lead + ((v75_r + 41) * 32)) * 4;
      int32_t v592_vlead = (v38_lead + ((v75_r + 45) * 32)) * 4;
      float r0[18]{};
      size_t v1184_row = v22_batchIdGroup0First + v19_batchIdLane0;
      const bool peel_batchIdActive0 = v1184_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1184_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v1184_row : v22_batchIdGroup0First) * 576 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v1192_i0 = 0; v1192_i0 < 3; ++v1192_i0) {
        int32_t v1194_lead = v38_lead + (v1192_i0 * 32);
        #pragma unroll
        for (int32_t v1195_i1 = 0; v1195_i1 < 6; ++v1195_i1) {
          float v1198_data = __builtin_nontemporal_load(&peel_glb_m2[(v1194_lead + (v1195_i1 * 96))]);
          r0[(v1192_i0 + (v1195_i1 * 3))] = v1198_data;
        }
      }
      for (size_t v23_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v23_batchIdGroup0 < numElements0; v23_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v1181_groupNextAhead = v23_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v1183_groupNext = (v1181_groupNextAhead < numElements0) ? v1181_groupNextAhead : v23_batchIdGroup0;
        size_t v1188_row = v1183_groupNext + v19_batchIdLane0;
        const bool wrap_batchIdActive0 = v1188_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1188_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v1188_row : v1183_groupNext) * 576 + 0 + m2_extraOffset];
        size_t v24_row = v23_batchIdGroup0 + v19_batchIdLane0;
        const bool batchIdActive0 = v24_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v24_row]));
        size_t v26_batchId0 = batchIdActive0 ? v24_row : v23_batchIdGroup0;
        size_t v27_ahead1 = v26_batchId0 + (gridDim.x * blockDim.y);
        size_t v29_batchId1 = (v27_ahead1 < numElements0) ? v27_ahead1 : v26_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v26_batchId0 * 384 + 0 + m0_extraOffset];
        float r1[12]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 64), (0, 6)] [(0, 96)]
        float v49_data = r0[0];
        float v50_data = r0[3];
        float v51_data = r0[6];
        float v52_data = r0[9];
        float v53_tp{};
        float v54_tp{};
        float v55_tp{};
        float v56_tp{};
        tensorforge::transpose4x4b32(v53_tp, v54_tp, v55_tp, v56_tp, v49_data, v50_data, v51_data, v52_data);
        float v57_data = r0[1];
        float v58_data = r0[4];
        float v59_data = r0[7];
        float v60_data = r0[10];
        float v61_tp{};
        float v62_tp{};
        float v63_tp{};
        float v64_tp{};
        tensorforge::transpose4x4b32(v61_tp, v62_tp, v63_tp, v64_tp, v57_data, v58_data, v59_data, v60_data);
        float v65_data = r0[2];
        float v66_data = r0[5];
        float v67_data = r0[8];
        float v68_data = r0[11];
        float v69_tp{};
        float v70_tp{};
        float v71_tp{};
        float v72_tp{};
        tensorforge::transpose4x4b32(v69_tp, v70_tp, v71_tp, v72_tp, v65_data, v66_data, v67_data, v68_data);
        tensorforge::VectorT<float, 4> v73_acc{};
        tensorforge::VectorT<float, 4> v81_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v79_vlead];
        float v82_el = v81_data[0];
        tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v82_el, v73_acc, 3, 0, 1);
        float v84_el = v81_data[1];
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v84_el, v83_acc, 3, 0, 1);
        float v86_el = v81_data[2];
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v86_el, v85_acc, 3, 0, 1);
        float v88_el = v81_data[3];
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v88_el, v87_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v82_el, v89_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v84_el, v91_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v86_el, v93_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v88_el, v95_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v103_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v101_vlead];
        float v104_el = v103_data[0];
        tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v104_el, v97_acc, 3, 2, 1);
        float v106_el = v103_data[1];
        tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v106_el, v105_acc, 3, 2, 1);
        float v108_el = v103_data[2];
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v108_el, v107_acc, 3, 2, 1);
        float v110_el = v103_data[3];
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v110_el, v109_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v104_el, v111_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v106_el, v113_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v108_el, v115_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v110_el, v117_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v125_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v123_vlead];
        float v126_el = v125_data[0];
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v126_el, v119_acc, 3, 4, 1);
        float v128_el = v125_data[1];
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v128_el, v127_acc, 3, 4, 1);
        float v130_el = v125_data[2];
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v130_el, v129_acc, 3, 4, 1);
        float v132_el = v125_data[3];
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v132_el, v131_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v126_el, v133_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v128_el, v135_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v130_el, v137_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v132_el, v139_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v147_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v145_vlead];
        float v148_el = v147_data[0];
        tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v148_el, v141_acc, 3, 6, 1);
        float v150_el = v147_data[1];
        tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v150_el, v149_acc, 3, 6, 1);
        float v152_el = v147_data[2];
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v152_el, v151_acc, 3, 6, 1);
        float v154_el = v147_data[3];
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v154_el, v153_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v148_el, v155_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v150_el, v157_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v152_el, v159_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v154_el, v161_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v169_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v167_vlead];
        float v170_el = v169_data[0];
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v170_el, v163_acc, 3, 0, 1);
        float v172_el = v169_data[1];
        tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v172_el, v171_acc, 3, 0, 1);
        float v174_el = v169_data[2];
        tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v174_el, v173_acc, 3, 0, 1);
        float v176_el = v169_data[3];
        tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v176_el, v175_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v170_el, v177_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v172_el, v179_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v174_el, v181_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v176_el, v183_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v191_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v189_vlead];
        float v192_el = v191_data[0];
        tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v192_el, v185_acc, 3, 2, 1);
        float v194_el = v191_data[1];
        tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v194_el, v193_acc, 3, 2, 1);
        float v196_el = v191_data[2];
        tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v196_el, v195_acc, 3, 2, 1);
        float v198_el = v191_data[3];
        tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v198_el, v197_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v192_el, v199_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v194_el, v201_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v196_el, v203_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v198_el, v205_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v213_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v211_vlead];
        float v214_el = v213_data[0];
        tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v214_el, v207_acc, 3, 4, 1);
        float v216_el = v213_data[1];
        tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v216_el, v215_acc, 3, 4, 1);
        float v218_el = v213_data[2];
        tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v218_el, v217_acc, 3, 4, 1);
        float v220_el = v213_data[3];
        tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v220_el, v219_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v214_el, v221_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v216_el, v223_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v218_el, v225_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v220_el, v227_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v235_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v233_vlead];
        float v236_el = v235_data[0];
        tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v236_el, v229_acc, 3, 6, 1);
        float v238_el = v235_data[1];
        tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v238_el, v237_acc, 3, 6, 1);
        float v240_el = v235_data[2];
        tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v240_el, v239_acc, 3, 6, 1);
        float v242_el = v235_data[3];
        tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v242_el, v241_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v236_el, v243_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v238_el, v245_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v240_el, v247_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v242_el, v249_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v257_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v255_vlead];
        float v258_el = v257_data[0];
        tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v258_el, v251_acc, 3, 0, 1);
        float v260_el = v257_data[1];
        tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v260_el, v259_acc, 3, 0, 1);
        float v262_el = v257_data[2];
        tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v262_el, v261_acc, 3, 0, 1);
        float v264_el = v257_data[3];
        tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v264_el, v263_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v258_el, v265_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v260_el, v267_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v262_el, v269_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v264_el, v271_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v279_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v277_vlead];
        float v280_el = v279_data[0];
        tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v280_el, v273_acc, 3, 2, 1);
        float v282_el = v279_data[1];
        tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v282_el, v281_acc, 3, 2, 1);
        float v284_el = v279_data[2];
        tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v284_el, v283_acc, 3, 2, 1);
        float v286_el = v279_data[3];
        tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v286_el, v285_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v280_el, v287_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v282_el, v289_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v284_el, v291_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v286_el, v293_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v301_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v299_vlead];
        float v302_el = v301_data[0];
        tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v302_el, v295_acc, 3, 4, 1);
        float v304_el = v301_data[1];
        tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v304_el, v303_acc, 3, 4, 1);
        float v306_el = v301_data[2];
        tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v306_el, v305_acc, 3, 4, 1);
        float v308_el = v301_data[3];
        tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v308_el, v307_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v302_el, v309_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v304_el, v311_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v306_el, v313_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v308_el, v315_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v323_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v321_vlead];
        float v324_el = v323_data[0];
        tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v324_el, v317_acc, 3, 6, 1);
        float v326_el = v323_data[1];
        tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v326_el, v325_acc, 3, 6, 1);
        float v328_el = v323_data[2];
        tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v328_el, v327_acc, 3, 6, 1);
        float v330_el = v323_data[3];
        tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v330_el, v329_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v324_el, v331_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v326_el, v333_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v328_el, v335_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v330_el, v337_acc, 3, 7, 2);
        r1[0] = (v339_acc[0]);
        r1[2] = (v339_acc[1]);
        r1[4] = (v339_acc[2]);
        r1[6] = (v339_acc[3]);
        tensorforge::VectorT<float, 4> v344_acc{};
        tensorforge::VectorT<float, 4> v352_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v350_vlead];
        float v353_el = v352_data[0];
        tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v353_el, v344_acc, 3, 0, 1);
        float v355_el = v352_data[1];
        tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v355_el, v354_acc, 3, 0, 1);
        float v357_el = v352_data[2];
        tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v357_el, v356_acc, 3, 0, 1);
        float v359_el = v352_data[3];
        tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v359_el, v358_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v353_el, v360_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v355_el, v362_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v357_el, v364_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v359_el, v366_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v374_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v372_vlead];
        float v375_el = v374_data[0];
        tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v375_el, v368_acc, 3, 2, 1);
        float v377_el = v374_data[1];
        tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v377_el, v376_acc, 3, 2, 1);
        float v379_el = v374_data[2];
        tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v379_el, v378_acc, 3, 2, 1);
        float v381_el = v374_data[3];
        tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v381_el, v380_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v375_el, v382_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v377_el, v384_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v379_el, v386_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v381_el, v388_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v396_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v394_vlead];
        float v397_el = v396_data[0];
        tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v397_el, v390_acc, 3, 4, 1);
        float v399_el = v396_data[1];
        tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v399_el, v398_acc, 3, 4, 1);
        float v401_el = v396_data[2];
        tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v401_el, v400_acc, 3, 4, 1);
        float v403_el = v396_data[3];
        tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v403_el, v402_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v397_el, v404_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v399_el, v406_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v401_el, v408_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v403_el, v410_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v418_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v416_vlead];
        float v419_el = v418_data[0];
        tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v419_el, v412_acc, 3, 6, 1);
        float v421_el = v418_data[1];
        tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v421_el, v420_acc, 3, 6, 1);
        float v423_el = v418_data[2];
        tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v423_el, v422_acc, 3, 6, 1);
        float v425_el = v418_data[3];
        tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v425_el, v424_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v419_el, v426_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v421_el, v428_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v423_el, v430_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v425_el, v432_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v440_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v438_vlead];
        float v441_el = v440_data[0];
        tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v441_el, v434_acc, 3, 0, 1);
        float v443_el = v440_data[1];
        tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v443_el, v442_acc, 3, 0, 1);
        float v445_el = v440_data[2];
        tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v445_el, v444_acc, 3, 0, 1);
        float v447_el = v440_data[3];
        tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v447_el, v446_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v441_el, v448_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v443_el, v450_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v445_el, v452_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v447_el, v454_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v462_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v460_vlead];
        float v463_el = v462_data[0];
        tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v463_el, v456_acc, 3, 2, 1);
        float v465_el = v462_data[1];
        tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v465_el, v464_acc, 3, 2, 1);
        float v467_el = v462_data[2];
        tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v467_el, v466_acc, 3, 2, 1);
        float v469_el = v462_data[3];
        tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v469_el, v468_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v463_el, v470_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v465_el, v472_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v467_el, v474_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v469_el, v476_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v484_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v482_vlead];
        float v485_el = v484_data[0];
        tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v485_el, v478_acc, 3, 4, 1);
        float v487_el = v484_data[1];
        tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v487_el, v486_acc, 3, 4, 1);
        float v489_el = v484_data[2];
        tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v489_el, v488_acc, 3, 4, 1);
        float v491_el = v484_data[3];
        tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v491_el, v490_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v485_el, v492_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v487_el, v494_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v489_el, v496_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v491_el, v498_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v506_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v504_vlead];
        float v507_el = v506_data[0];
        tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v507_el, v500_acc, 3, 6, 1);
        float v509_el = v506_data[1];
        tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v509_el, v508_acc, 3, 6, 1);
        float v511_el = v506_data[2];
        tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v511_el, v510_acc, 3, 6, 1);
        float v513_el = v506_data[3];
        tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v513_el, v512_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v507_el, v514_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v509_el, v516_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v511_el, v518_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v513_el, v520_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v528_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v526_vlead];
        float v529_el = v528_data[0];
        tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v529_el, v522_acc, 3, 0, 1);
        float v531_el = v528_data[1];
        tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v531_el, v530_acc, 3, 0, 1);
        float v533_el = v528_data[2];
        tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v533_el, v532_acc, 3, 0, 1);
        float v535_el = v528_data[3];
        tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v535_el, v534_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v529_el, v536_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v531_el, v538_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v533_el, v540_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v535_el, v542_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v550_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v548_vlead];
        float v551_el = v550_data[0];
        tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v551_el, v544_acc, 3, 2, 1);
        float v553_el = v550_data[1];
        tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v553_el, v552_acc, 3, 2, 1);
        float v555_el = v550_data[2];
        tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v555_el, v554_acc, 3, 2, 1);
        float v557_el = v550_data[3];
        tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v557_el, v556_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v551_el, v558_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v553_el, v560_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v555_el, v562_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v557_el, v564_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v572_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v570_vlead];
        float v573_el = v572_data[0];
        tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v573_el, v566_acc, 3, 4, 1);
        float v575_el = v572_data[1];
        tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v575_el, v574_acc, 3, 4, 1);
        float v577_el = v572_data[2];
        tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v577_el, v576_acc, 3, 4, 1);
        float v579_el = v572_data[3];
        tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v579_el, v578_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v573_el, v580_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v575_el, v582_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v577_el, v584_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v588_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v579_el, v586_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v594_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v592_vlead];
        float v595_el = v594_data[0];
        tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v595_el, v588_acc, 3, 6, 1);
        float v597_el = v594_data[1];
        tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v597_el, v596_acc, 3, 6, 1);
        float v599_el = v594_data[2];
        tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v599_el, v598_acc, 3, 6, 1);
        float v601_el = v594_data[3];
        tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v601_el, v600_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v595_el, v602_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v597_el, v604_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v599_el, v606_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v601_el, v608_acc, 3, 7, 2);
        r1[1] = (v610_acc[0]);
        r1[3] = (v610_acc[1]);
        r1[5] = (v610_acc[2]);
        r1[7] = (v610_acc[3]);
        float v615_data = r0[12];
        float v616_data = r0[15];
        float v618_tp{};
        float v619_tp{};
        float v620_tp{};
        float v621_tp{};
        tensorforge::transpose4x4b32(v618_tp, v619_tp, v620_tp, v621_tp, v615_data, v616_data, 0.0f, 0.0f);
        float v622_data = r0[13];
        float v623_data = r0[16];
        float v624_tp{};
        float v625_tp{};
        float v626_tp{};
        float v627_tp{};
        tensorforge::transpose4x4b32(v624_tp, v625_tp, v626_tp, v627_tp, v622_data, v623_data, 0.0f, 0.0f);
        float v628_data = r0[14];
        float v629_data = r0[17];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v1201_i0 = 0; v1201_i0 < 3; ++v1201_i0) {
          int32_t v1203_lead = v38_lead + (v1201_i0 * 32);
          #pragma unroll
          for (int32_t v1204_i1 = 0; v1204_i1 < 6; ++v1204_i1) {
            float v1207_data = __builtin_nontemporal_load(&wrap_glb_m2[(v1203_lead + (v1204_i1 * 96))]);
            r0[(v1201_i0 + (v1204_i1 * 3))] = v1207_data;
          }
        }
        float v630_tp{};
        float v631_tp{};
        float v632_tp{};
        float v633_tp{};
        tensorforge::transpose4x4b32(v630_tp, v631_tp, v632_tp, v633_tp, v628_data, v629_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v634_acc{};
        tensorforge::VectorT<float, 4> v642_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v79_vlead];
        float v643_el = v642_data[0];
        tensorforge::VectorT<float, 4> v644_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v643_el, v634_acc, 3, 0, 1);
        float v645_el = v642_data[1];
        tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v645_el, v644_acc, 3, 0, 1);
        float v647_el = v642_data[2];
        tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v647_el, v646_acc, 3, 0, 1);
        float v649_el = v642_data[3];
        tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v649_el, v648_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v643_el, v650_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v645_el, v652_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v647_el, v654_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v649_el, v656_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v664_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v101_vlead];
        float v665_el = v664_data[0];
        tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v665_el, v658_acc, 3, 2, 1);
        float v667_el = v664_data[1];
        tensorforge::VectorT<float, 4> v668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v667_el, v666_acc, 3, 2, 1);
        float v669_el = v664_data[2];
        tensorforge::VectorT<float, 4> v670_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v669_el, v668_acc, 3, 2, 1);
        float v671_el = v664_data[3];
        tensorforge::VectorT<float, 4> v672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v671_el, v670_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v665_el, v672_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v667_el, v674_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v678_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v669_el, v676_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v680_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v671_el, v678_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v686_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v123_vlead];
        float v687_el = v686_data[0];
        tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v687_el, v680_acc, 3, 4, 1);
        float v689_el = v686_data[1];
        tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v689_el, v688_acc, 3, 4, 1);
        float v691_el = v686_data[2];
        tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v691_el, v690_acc, 3, 4, 1);
        float v693_el = v686_data[3];
        tensorforge::VectorT<float, 4> v694_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v693_el, v692_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v687_el, v694_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v689_el, v696_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v691_el, v698_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v693_el, v700_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v708_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v145_vlead];
        float v709_el = v708_data[0];
        tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v709_el, v702_acc, 3, 6, 1);
        float v711_el = v708_data[1];
        tensorforge::VectorT<float, 4> v712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v711_el, v710_acc, 3, 6, 1);
        float v713_el = v708_data[2];
        tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v713_el, v712_acc, 3, 6, 1);
        float v715_el = v708_data[3];
        tensorforge::VectorT<float, 4> v716_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v715_el, v714_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v718_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v709_el, v716_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v720_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v711_el, v718_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v713_el, v720_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v724_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v715_el, v722_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v730_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v167_vlead];
        float v731_el = v730_data[0];
        tensorforge::VectorT<float, 4> v732_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v731_el, v724_acc, 3, 0, 1);
        float v733_el = v730_data[1];
        tensorforge::VectorT<float, 4> v734_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v733_el, v732_acc, 3, 0, 1);
        float v735_el = v730_data[2];
        tensorforge::VectorT<float, 4> v736_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v735_el, v734_acc, 3, 0, 1);
        float v737_el = v730_data[3];
        tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v737_el, v736_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v740_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v731_el, v738_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v742_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v733_el, v740_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v744_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v735_el, v742_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v737_el, v744_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v752_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v189_vlead];
        float v753_el = v752_data[0];
        tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v753_el, v746_acc, 3, 2, 1);
        float v755_el = v752_data[1];
        tensorforge::VectorT<float, 4> v756_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v755_el, v754_acc, 3, 2, 1);
        float v757_el = v752_data[2];
        tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v757_el, v756_acc, 3, 2, 1);
        float v759_el = v752_data[3];
        tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v759_el, v758_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v762_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v753_el, v760_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v764_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v755_el, v762_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v766_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v757_el, v764_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v759_el, v766_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v774_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v211_vlead];
        float v775_el = v774_data[0];
        tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v775_el, v768_acc, 3, 4, 1);
        float v777_el = v774_data[1];
        tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v777_el, v776_acc, 3, 4, 1);
        float v779_el = v774_data[2];
        tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v779_el, v778_acc, 3, 4, 1);
        float v781_el = v774_data[3];
        tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v781_el, v780_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v784_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v775_el, v782_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v777_el, v784_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v788_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v779_el, v786_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v781_el, v788_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v796_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v233_vlead];
        float v797_el = v796_data[0];
        tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v797_el, v790_acc, 3, 6, 1);
        float v799_el = v796_data[1];
        tensorforge::VectorT<float, 4> v800_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v799_el, v798_acc, 3, 6, 1);
        float v801_el = v796_data[2];
        tensorforge::VectorT<float, 4> v802_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v801_el, v800_acc, 3, 6, 1);
        float v803_el = v796_data[3];
        tensorforge::VectorT<float, 4> v804_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v803_el, v802_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v806_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v797_el, v804_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v808_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v799_el, v806_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v810_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v801_el, v808_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v803_el, v810_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v818_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v255_vlead];
        float v819_el = v818_data[0];
        tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v819_el, v812_acc, 3, 0, 1);
        float v821_el = v818_data[1];
        tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v821_el, v820_acc, 3, 0, 1);
        float v823_el = v818_data[2];
        tensorforge::VectorT<float, 4> v824_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v823_el, v822_acc, 3, 0, 1);
        float v825_el = v818_data[3];
        tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v825_el, v824_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v819_el, v826_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v821_el, v828_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v832_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v823_el, v830_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v834_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v825_el, v832_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v840_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v277_vlead];
        float v841_el = v840_data[0];
        tensorforge::VectorT<float, 4> v842_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v841_el, v834_acc, 3, 2, 1);
        float v843_el = v840_data[1];
        tensorforge::VectorT<float, 4> v844_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v843_el, v842_acc, 3, 2, 1);
        float v845_el = v840_data[2];
        tensorforge::VectorT<float, 4> v846_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v845_el, v844_acc, 3, 2, 1);
        float v847_el = v840_data[3];
        tensorforge::VectorT<float, 4> v848_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v847_el, v846_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v850_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v841_el, v848_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v852_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v843_el, v850_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v854_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v845_el, v852_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v856_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v847_el, v854_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v862_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v299_vlead];
        float v863_el = v862_data[0];
        tensorforge::VectorT<float, 4> v864_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v863_el, v856_acc, 3, 4, 1);
        float v865_el = v862_data[1];
        tensorforge::VectorT<float, 4> v866_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v865_el, v864_acc, 3, 4, 1);
        float v867_el = v862_data[2];
        tensorforge::VectorT<float, 4> v868_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v867_el, v866_acc, 3, 4, 1);
        float v869_el = v862_data[3];
        tensorforge::VectorT<float, 4> v870_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v869_el, v868_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v872_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v863_el, v870_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v874_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v865_el, v872_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v876_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v867_el, v874_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v869_el, v876_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v884_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v321_vlead];
        float v885_el = v884_data[0];
        tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v885_el, v878_acc, 3, 6, 1);
        float v887_el = v884_data[1];
        tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v887_el, v886_acc, 3, 6, 1);
        float v889_el = v884_data[2];
        tensorforge::VectorT<float, 4> v890_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v889_el, v888_acc, 3, 6, 1);
        float v891_el = v884_data[3];
        tensorforge::VectorT<float, 4> v892_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v891_el, v890_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v894_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v885_el, v892_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v896_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v887_el, v894_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v898_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v889_el, v896_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v900_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v891_el, v898_acc, 3, 7, 2);
        r1[8] = (v900_acc[0]);
        r1[10] = (v900_acc[1]);
        tensorforge::VectorT<float, 4> v903_acc{};
        tensorforge::VectorT<float, 4> v911_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v350_vlead];
        float v912_el = v911_data[0];
        tensorforge::VectorT<float, 4> v913_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v912_el, v903_acc, 3, 0, 1);
        float v914_el = v911_data[1];
        tensorforge::VectorT<float, 4> v915_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v914_el, v913_acc, 3, 0, 1);
        float v916_el = v911_data[2];
        tensorforge::VectorT<float, 4> v917_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v916_el, v915_acc, 3, 0, 1);
        float v918_el = v911_data[3];
        tensorforge::VectorT<float, 4> v919_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v918_el, v917_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v921_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v912_el, v919_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v923_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v914_el, v921_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v925_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v916_el, v923_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v927_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v918_el, v925_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v933_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v372_vlead];
        float v934_el = v933_data[0];
        tensorforge::VectorT<float, 4> v935_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v934_el, v927_acc, 3, 2, 1);
        float v936_el = v933_data[1];
        tensorforge::VectorT<float, 4> v937_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v936_el, v935_acc, 3, 2, 1);
        float v938_el = v933_data[2];
        tensorforge::VectorT<float, 4> v939_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v938_el, v937_acc, 3, 2, 1);
        float v940_el = v933_data[3];
        tensorforge::VectorT<float, 4> v941_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v940_el, v939_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v943_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v934_el, v941_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v945_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v936_el, v943_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v947_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v938_el, v945_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v949_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v940_el, v947_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v955_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v394_vlead];
        float v956_el = v955_data[0];
        tensorforge::VectorT<float, 4> v957_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v956_el, v949_acc, 3, 4, 1);
        float v958_el = v955_data[1];
        tensorforge::VectorT<float, 4> v959_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v958_el, v957_acc, 3, 4, 1);
        float v960_el = v955_data[2];
        tensorforge::VectorT<float, 4> v961_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v960_el, v959_acc, 3, 4, 1);
        float v962_el = v955_data[3];
        tensorforge::VectorT<float, 4> v963_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v962_el, v961_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v956_el, v963_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v958_el, v965_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v960_el, v967_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v971_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v962_el, v969_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v977_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v416_vlead];
        float v978_el = v977_data[0];
        tensorforge::VectorT<float, 4> v979_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v978_el, v971_acc, 3, 6, 1);
        float v980_el = v977_data[1];
        tensorforge::VectorT<float, 4> v981_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v980_el, v979_acc, 3, 6, 1);
        float v982_el = v977_data[2];
        tensorforge::VectorT<float, 4> v983_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v982_el, v981_acc, 3, 6, 1);
        float v984_el = v977_data[3];
        tensorforge::VectorT<float, 4> v985_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v984_el, v983_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v987_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v618_tp, v978_el, v985_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v989_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v619_tp, v980_el, v987_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v991_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v620_tp, v982_el, v989_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v993_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v621_tp, v984_el, v991_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v999_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v438_vlead];
        float v1000_el = v999_data[0];
        tensorforge::VectorT<float, 4> v1001_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1000_el, v993_acc, 3, 0, 1);
        float v1002_el = v999_data[1];
        tensorforge::VectorT<float, 4> v1003_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1002_el, v1001_acc, 3, 0, 1);
        float v1004_el = v999_data[2];
        tensorforge::VectorT<float, 4> v1005_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1004_el, v1003_acc, 3, 0, 1);
        float v1006_el = v999_data[3];
        tensorforge::VectorT<float, 4> v1007_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1006_el, v1005_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v1009_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1000_el, v1007_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1011_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1002_el, v1009_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1013_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1004_el, v1011_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1006_el, v1013_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1021_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v460_vlead];
        float v1022_el = v1021_data[0];
        tensorforge::VectorT<float, 4> v1023_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1022_el, v1015_acc, 3, 2, 1);
        float v1024_el = v1021_data[1];
        tensorforge::VectorT<float, 4> v1025_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1024_el, v1023_acc, 3, 2, 1);
        float v1026_el = v1021_data[2];
        tensorforge::VectorT<float, 4> v1027_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1026_el, v1025_acc, 3, 2, 1);
        float v1028_el = v1021_data[3];
        tensorforge::VectorT<float, 4> v1029_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1028_el, v1027_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v1031_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1022_el, v1029_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1033_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1024_el, v1031_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1035_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1026_el, v1033_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1037_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1028_el, v1035_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1043_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v482_vlead];
        float v1044_el = v1043_data[0];
        tensorforge::VectorT<float, 4> v1045_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1044_el, v1037_acc, 3, 4, 1);
        float v1046_el = v1043_data[1];
        tensorforge::VectorT<float, 4> v1047_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1046_el, v1045_acc, 3, 4, 1);
        float v1048_el = v1043_data[2];
        tensorforge::VectorT<float, 4> v1049_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1048_el, v1047_acc, 3, 4, 1);
        float v1050_el = v1043_data[3];
        tensorforge::VectorT<float, 4> v1051_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1050_el, v1049_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v1053_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1044_el, v1051_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1055_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1046_el, v1053_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1048_el, v1055_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1059_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1050_el, v1057_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1065_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v504_vlead];
        float v1066_el = v1065_data[0];
        tensorforge::VectorT<float, 4> v1067_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1066_el, v1059_acc, 3, 6, 1);
        float v1068_el = v1065_data[1];
        tensorforge::VectorT<float, 4> v1069_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1068_el, v1067_acc, 3, 6, 1);
        float v1070_el = v1065_data[2];
        tensorforge::VectorT<float, 4> v1071_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1070_el, v1069_acc, 3, 6, 1);
        float v1072_el = v1065_data[3];
        tensorforge::VectorT<float, 4> v1073_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1072_el, v1071_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v1075_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v624_tp, v1066_el, v1073_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1077_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v625_tp, v1068_el, v1075_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1079_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v626_tp, v1070_el, v1077_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1081_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v627_tp, v1072_el, v1079_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1087_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v526_vlead];
        float v1088_el = v1087_data[0];
        tensorforge::VectorT<float, 4> v1089_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1088_el, v1081_acc, 3, 0, 1);
        float v1090_el = v1087_data[1];
        tensorforge::VectorT<float, 4> v1091_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1090_el, v1089_acc, 3, 0, 1);
        float v1092_el = v1087_data[2];
        tensorforge::VectorT<float, 4> v1093_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1092_el, v1091_acc, 3, 0, 1);
        float v1094_el = v1087_data[3];
        tensorforge::VectorT<float, 4> v1095_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1094_el, v1093_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v1097_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1088_el, v1095_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1099_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1090_el, v1097_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1092_el, v1099_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1094_el, v1101_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v1109_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v548_vlead];
        float v1110_el = v1109_data[0];
        tensorforge::VectorT<float, 4> v1111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1110_el, v1103_acc, 3, 2, 1);
        float v1112_el = v1109_data[1];
        tensorforge::VectorT<float, 4> v1113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1112_el, v1111_acc, 3, 2, 1);
        float v1114_el = v1109_data[2];
        tensorforge::VectorT<float, 4> v1115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1114_el, v1113_acc, 3, 2, 1);
        float v1116_el = v1109_data[3];
        tensorforge::VectorT<float, 4> v1117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1116_el, v1115_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v1119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1110_el, v1117_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1112_el, v1119_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1114_el, v1121_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1116_el, v1123_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v1131_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v570_vlead];
        float v1132_el = v1131_data[0];
        tensorforge::VectorT<float, 4> v1133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1132_el, v1125_acc, 3, 4, 1);
        float v1134_el = v1131_data[1];
        tensorforge::VectorT<float, 4> v1135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1134_el, v1133_acc, 3, 4, 1);
        float v1136_el = v1131_data[2];
        tensorforge::VectorT<float, 4> v1137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1136_el, v1135_acc, 3, 4, 1);
        float v1138_el = v1131_data[3];
        tensorforge::VectorT<float, 4> v1139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1138_el, v1137_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v1141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1132_el, v1139_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1134_el, v1141_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1136_el, v1143_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1138_el, v1145_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1153_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v592_vlead];
        float v1154_el = v1153_data[0];
        tensorforge::VectorT<float, 4> v1155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1154_el, v1147_acc, 3, 6, 1);
        float v1156_el = v1153_data[1];
        tensorforge::VectorT<float, 4> v1157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1156_el, v1155_acc, 3, 6, 1);
        float v1158_el = v1153_data[2];
        tensorforge::VectorT<float, 4> v1159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1158_el, v1157_acc, 3, 6, 1);
        float v1160_el = v1153_data[3];
        tensorforge::VectorT<float, 4> v1161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1160_el, v1159_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v1163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v630_tp, v1154_el, v1161_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v631_tp, v1156_el, v1163_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v632_tp, v1158_el, v1165_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v633_tp, v1160_el, v1167_acc, 3, 7, 2);
        r1[9] = (v1169_acc[0]);
        r1[11] = (v1169_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v1172_i0 = 0; v1172_i0 < 2; ++v1172_i0) {
          #pragma unroll
          for (int32_t v1173_i1 = 0; v1173_i1 < 6; ++v1173_i1) {
            float v1176_data = r1[(v1172_i0 + (v1173_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v38_lead + (v1172_i0 * 32)) + (v1173_i1 * 64))] = v1176_data;
            }
          }
        }
      }
    }
  }
}

