// === base name ===
kernel_067280b8365da08a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_067280b8365da08a = {{32, 8, 1}, 32, 64, 1, 8, 24576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_067280b8365da08a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_067280b8365da08a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_067280b8365da08a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_067280b8365da08a, block.x * block.y * block.z, 6144 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (6144 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_067280b8365da08a, block.x * block.y * block.z, 0));
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
void launcher_kernel_067280b8365da08a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_067280b8365da08a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_067280b8365da08a), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_067280b8365da08a, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_067280b8365da08a(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=runs
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
      int32_t v366_vlead = (v38_lead + ((v75_r + 5) * 32)) * 4;
      int32_t v382_vlead = (v38_lead + ((v75_r + 9) * 32)) * 4;
      int32_t v396_vlead = (v38_lead + ((v75_r + 13) * 32)) * 4;
      int32_t v412_vlead = (v38_lead + ((v75_r + 17) * 32)) * 4;
      int32_t v428_vlead = (v38_lead + ((v75_r + 21) * 32)) * 4;
      int32_t v446_vlead = (v38_lead + ((v75_r + 25) * 32)) * 4;
      int32_t v460_vlead = (v38_lead + ((v75_r + 29) * 32)) * 4;
      int32_t v476_vlead = (v38_lead + ((v75_r + 33) * 32)) * 4;
      int32_t v494_vlead = (v38_lead + ((v75_r + 37) * 32)) * 4;
      int32_t v508_vlead = (v38_lead + ((v75_r + 41) * 32)) * 4;
      int32_t v522_vlead = (v38_lead + ((v75_r + 45) * 32)) * 4;
      float r0[18]{};
      size_t v1036_row = v22_batchIdGroup0First + v19_batchIdLane0;
      const bool peel_batchIdActive0 = v1036_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1036_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v1036_row : v22_batchIdGroup0First) * 576 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v1044_i0 = 0; v1044_i0 < 3; ++v1044_i0) {
        int32_t v1046_lead = v38_lead + (v1044_i0 * 32);
        #pragma unroll
        for (int32_t v1047_i1 = 0; v1047_i1 < 6; ++v1047_i1) {
          float v1050_data = __builtin_nontemporal_load(&peel_glb_m2[(v1046_lead + (v1047_i1 * 96))]);
          r0[(v1044_i0 + (v1047_i1 * 3))] = v1050_data;
        }
      }
      for (size_t v23_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v23_batchIdGroup0 < numElements0; v23_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v1033_groupNextAhead = v23_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v1035_groupNext = (v1033_groupNextAhead < numElements0) ? v1033_groupNextAhead : v23_batchIdGroup0;
        size_t v1040_row = v1035_groupNext + v19_batchIdLane0;
        const bool wrap_batchIdActive0 = v1040_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v1040_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v1040_row : v1035_groupNext) * 576 + 0 + m2_extraOffset];
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
        float v353_el = v352_data[2];
        tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v353_el, v344_acc, 3, 0, 1);
        float v355_el = v352_data[3];
        tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v355_el, v354_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, (v352_data[0]), v356_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v353_el, v358_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v355_el, v360_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v368_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v366_vlead];
        float v369_el = v368_data[0];
        tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v369_el, v362_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, (v368_data[2]), v370_acc, 3, 2, 1);
        float v373_el = v368_data[3];
        tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v373_el, v372_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v369_el, v374_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v373_el, v376_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v384_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v382_vlead];
        float v385_el = v384_data[0];
        tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v385_el, v378_acc, 3, 4, 1);
        float v387_el = v384_data[3];
        tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v387_el, v386_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v385_el, v388_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v387_el, v390_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v398_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v396_vlead];
        float v399_el = v398_data[0];
        tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v399_el, v392_acc, 3, 6, 1);
        float v401_el = v398_data[2];
        tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v401_el, v400_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, (v398_data[3]), v402_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v399_el, v404_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v401_el, v406_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v414_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v412_vlead];
        float v415_el = v414_data[0];
        tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v415_el, v408_acc, 3, 0, 1);
        float v417_el = v414_data[1];
        tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v417_el, v416_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v415_el, v418_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v417_el, v420_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, (v414_data[3]), v422_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v430_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v428_vlead];
        float v431_el = v430_data[0];
        tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v431_el, v424_acc, 3, 2, 1);
        float v433_el = v430_data[1];
        tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v433_el, v432_acc, 3, 2, 1);
        float v435_el = v430_data[3];
        tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v435_el, v434_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v431_el, v436_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v433_el, v438_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v435_el, v440_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v448_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v446_vlead];
        float v449_el = v448_data[0];
        tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v449_el, v442_acc, 3, 4, 1);
        float v451_el = v448_data[1];
        tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v451_el, v450_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v449_el, v452_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v451_el, v454_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v462_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v460_vlead];
        float v463_el = v462_data[0];
        tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v463_el, v456_acc, 3, 6, 1);
        float v465_el = v462_data[1];
        tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v465_el, v464_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v463_el, v466_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v465_el, v468_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, (v462_data[2]), v470_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v478_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v476_vlead];
        tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, (v478_data[0]), v472_acc, 3, 0, 1);
        float v481_el = v478_data[1];
        tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v481_el, v480_acc, 3, 0, 1);
        float v483_el = v478_data[2];
        tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v483_el, v482_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v481_el, v484_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v483_el, v486_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, (v478_data[3]), v488_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v496_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v494_vlead];
        float v497_el = v496_data[1];
        tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v497_el, v490_acc, 3, 2, 1);
        float v499_el = v496_data[2];
        tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v499_el, v498_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v497_el, v500_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v499_el, v502_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v510_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v508_vlead];
        tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, (v510_data[1]), v504_acc, 3, 4, 1);
        float v513_el = v510_data[2];
        tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v513_el, v512_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, (v510_data[0]), v514_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v513_el, v516_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v524_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v522_vlead];
        tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, (v524_data[0]), v518_acc, 3, 6, 1);
        float v527_el = v524_data[2];
        tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v527_el, v526_acc, 3, 6, 1);
        float v529_el = v524_data[3];
        tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v529_el, v528_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, (v524_data[1]), v530_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v71_tp, v527_el, v532_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v529_el, v534_acc, 3, 7, 2);
        r1[1] = (v536_acc[0]);
        r1[3] = (v536_acc[1]);
        r1[5] = (v536_acc[2]);
        r1[7] = (v536_acc[3]);
        float v541_data = r0[12];
        float v542_data = r0[15];
        float v544_tp{};
        float v545_tp{};
        float v546_tp{};
        float v547_tp{};
        tensorforge::transpose4x4b32(v544_tp, v545_tp, v546_tp, v547_tp, v541_data, v542_data, 0.0f, 0.0f);
        float v548_data = r0[13];
        float v549_data = r0[16];
        float v550_tp{};
        float v551_tp{};
        float v552_tp{};
        float v553_tp{};
        tensorforge::transpose4x4b32(v550_tp, v551_tp, v552_tp, v553_tp, v548_data, v549_data, 0.0f, 0.0f);
        float v554_data = r0[14];
        float v555_data = r0[17];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v1053_i0 = 0; v1053_i0 < 3; ++v1053_i0) {
          int32_t v1055_lead = v38_lead + (v1053_i0 * 32);
          #pragma unroll
          for (int32_t v1056_i1 = 0; v1056_i1 < 6; ++v1056_i1) {
            float v1059_data = __builtin_nontemporal_load(&wrap_glb_m2[(v1055_lead + (v1056_i1 * 96))]);
            r0[(v1053_i0 + (v1056_i1 * 3))] = v1059_data;
          }
        }
        float v556_tp{};
        float v557_tp{};
        float v558_tp{};
        float v559_tp{};
        tensorforge::transpose4x4b32(v556_tp, v557_tp, v558_tp, v559_tp, v554_data, v555_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v560_acc{};
        tensorforge::VectorT<float, 4> v568_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v79_vlead];
        float v569_el = v568_data[0];
        tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v569_el, v560_acc, 3, 0, 1);
        float v571_el = v568_data[1];
        tensorforge::VectorT<float, 4> v572_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v571_el, v570_acc, 3, 0, 1);
        float v573_el = v568_data[2];
        tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v573_el, v572_acc, 3, 0, 1);
        float v575_el = v568_data[3];
        tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v575_el, v574_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v569_el, v576_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v580_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v571_el, v578_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v573_el, v580_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v575_el, v582_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v590_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v101_vlead];
        float v591_el = v590_data[0];
        tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v591_el, v584_acc, 3, 2, 1);
        float v593_el = v590_data[1];
        tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v593_el, v592_acc, 3, 2, 1);
        float v595_el = v590_data[2];
        tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v595_el, v594_acc, 3, 2, 1);
        float v597_el = v590_data[3];
        tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v597_el, v596_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v591_el, v598_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v602_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v593_el, v600_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v604_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v595_el, v602_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v597_el, v604_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v612_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v123_vlead];
        float v613_el = v612_data[0];
        tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v613_el, v606_acc, 3, 4, 1);
        float v615_el = v612_data[1];
        tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v615_el, v614_acc, 3, 4, 1);
        float v617_el = v612_data[2];
        tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v617_el, v616_acc, 3, 4, 1);
        float v619_el = v612_data[3];
        tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v619_el, v618_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v613_el, v620_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v624_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v615_el, v622_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v626_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v617_el, v624_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v619_el, v626_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v634_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v145_vlead];
        float v635_el = v634_data[0];
        tensorforge::VectorT<float, 4> v636_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v635_el, v628_acc, 3, 6, 1);
        float v637_el = v634_data[1];
        tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v637_el, v636_acc, 3, 6, 1);
        float v639_el = v634_data[2];
        tensorforge::VectorT<float, 4> v640_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v639_el, v638_acc, 3, 6, 1);
        float v641_el = v634_data[3];
        tensorforge::VectorT<float, 4> v642_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v641_el, v640_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v644_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v635_el, v642_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v545_tp, v637_el, v644_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v639_el, v646_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v641_el, v648_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v656_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v167_vlead];
        float v657_el = v656_data[0];
        tensorforge::VectorT<float, 4> v658_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v657_el, v650_acc, 3, 0, 1);
        float v659_el = v656_data[1];
        tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v659_el, v658_acc, 3, 0, 1);
        float v661_el = v656_data[2];
        tensorforge::VectorT<float, 4> v662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v661_el, v660_acc, 3, 0, 1);
        float v663_el = v656_data[3];
        tensorforge::VectorT<float, 4> v664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v663_el, v662_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v657_el, v664_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v659_el, v666_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v670_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v661_el, v668_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v663_el, v670_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v678_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v189_vlead];
        float v679_el = v678_data[0];
        tensorforge::VectorT<float, 4> v680_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v679_el, v672_acc, 3, 2, 1);
        float v681_el = v678_data[1];
        tensorforge::VectorT<float, 4> v682_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v681_el, v680_acc, 3, 2, 1);
        float v683_el = v678_data[2];
        tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v683_el, v682_acc, 3, 2, 1);
        float v685_el = v678_data[3];
        tensorforge::VectorT<float, 4> v686_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v685_el, v684_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v688_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v679_el, v686_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v690_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v681_el, v688_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v683_el, v690_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v694_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v685_el, v692_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v700_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v211_vlead];
        float v701_el = v700_data[0];
        tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v701_el, v694_acc, 3, 4, 1);
        float v703_el = v700_data[1];
        tensorforge::VectorT<float, 4> v704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v703_el, v702_acc, 3, 4, 1);
        float v705_el = v700_data[2];
        tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v705_el, v704_acc, 3, 4, 1);
        float v707_el = v700_data[3];
        tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v707_el, v706_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v701_el, v708_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v703_el, v710_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v705_el, v712_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v716_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v707_el, v714_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v722_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v233_vlead];
        float v723_el = v722_data[0];
        tensorforge::VectorT<float, 4> v724_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v723_el, v716_acc, 3, 6, 1);
        float v725_el = v722_data[1];
        tensorforge::VectorT<float, 4> v726_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v725_el, v724_acc, 3, 6, 1);
        float v727_el = v722_data[2];
        tensorforge::VectorT<float, 4> v728_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v727_el, v726_acc, 3, 6, 1);
        float v729_el = v722_data[3];
        tensorforge::VectorT<float, 4> v730_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v729_el, v728_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v732_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v723_el, v730_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v734_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v725_el, v732_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v736_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v727_el, v734_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v738_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v729_el, v736_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v744_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v255_vlead];
        float v745_el = v744_data[0];
        tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v745_el, v738_acc, 3, 0, 1);
        float v747_el = v744_data[1];
        tensorforge::VectorT<float, 4> v748_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v747_el, v746_acc, 3, 0, 1);
        float v749_el = v744_data[2];
        tensorforge::VectorT<float, 4> v750_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v749_el, v748_acc, 3, 0, 1);
        float v751_el = v744_data[3];
        tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v751_el, v750_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v745_el, v752_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v756_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v747_el, v754_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v758_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v749_el, v756_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v751_el, v758_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v766_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v277_vlead];
        float v767_el = v766_data[0];
        tensorforge::VectorT<float, 4> v768_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v767_el, v760_acc, 3, 2, 1);
        float v769_el = v766_data[1];
        tensorforge::VectorT<float, 4> v770_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v769_el, v768_acc, 3, 2, 1);
        float v771_el = v766_data[2];
        tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v771_el, v770_acc, 3, 2, 1);
        float v773_el = v766_data[3];
        tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v773_el, v772_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v767_el, v774_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v778_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v769_el, v776_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v780_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v771_el, v778_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v782_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v773_el, v780_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v788_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v299_vlead];
        float v789_el = v788_data[0];
        tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v789_el, v782_acc, 3, 4, 1);
        float v791_el = v788_data[1];
        tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v791_el, v790_acc, 3, 4, 1);
        float v793_el = v788_data[2];
        tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v793_el, v792_acc, 3, 4, 1);
        float v795_el = v788_data[3];
        tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v795_el, v794_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v789_el, v796_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v800_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v791_el, v798_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v802_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v793_el, v800_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v804_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v795_el, v802_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v810_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v321_vlead];
        float v811_el = v810_data[0];
        tensorforge::VectorT<float, 4> v812_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v811_el, v804_acc, 3, 6, 1);
        float v813_el = v810_data[1];
        tensorforge::VectorT<float, 4> v814_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v813_el, v812_acc, 3, 6, 1);
        float v815_el = v810_data[2];
        tensorforge::VectorT<float, 4> v816_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v815_el, v814_acc, 3, 6, 1);
        float v817_el = v810_data[3];
        tensorforge::VectorT<float, 4> v818_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v817_el, v816_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v820_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, v811_el, v818_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v822_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v813_el, v820_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v824_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v815_el, v822_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v826_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v817_el, v824_acc, 3, 7, 2);
        r1[8] = (v826_acc[0]);
        r1[10] = (v826_acc[1]);
        tensorforge::VectorT<float, 4> v829_acc{};
        tensorforge::VectorT<float, 4> v837_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v350_vlead];
        float v838_el = v837_data[2];
        tensorforge::VectorT<float, 4> v839_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v838_el, v829_acc, 3, 0, 1);
        float v840_el = v837_data[3];
        tensorforge::VectorT<float, 4> v841_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v840_el, v839_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v843_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, (v837_data[0]), v841_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v845_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v838_el, v843_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v847_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v840_el, v845_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v853_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v366_vlead];
        float v854_el = v853_data[0];
        tensorforge::VectorT<float, 4> v855_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v854_el, v847_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v857_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, (v853_data[2]), v855_acc, 3, 2, 1);
        float v858_el = v853_data[3];
        tensorforge::VectorT<float, 4> v859_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v858_el, v857_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v861_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v854_el, v859_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v863_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v858_el, v861_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v869_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v382_vlead];
        float v870_el = v869_data[0];
        tensorforge::VectorT<float, 4> v871_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v870_el, v863_acc, 3, 4, 1);
        float v872_el = v869_data[3];
        tensorforge::VectorT<float, 4> v873_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v872_el, v871_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v875_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v870_el, v873_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v877_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, v872_el, v875_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v883_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v396_vlead];
        float v884_el = v883_data[0];
        tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v884_el, v877_acc, 3, 6, 1);
        float v886_el = v883_data[2];
        tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v886_el, v885_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v889_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v547_tp, (v883_data[3]), v887_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v891_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v544_tp, v884_el, v889_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v893_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v546_tp, v886_el, v891_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v899_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v412_vlead];
        float v900_el = v899_data[0];
        tensorforge::VectorT<float, 4> v901_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v900_el, v893_acc, 3, 0, 1);
        float v902_el = v899_data[1];
        tensorforge::VectorT<float, 4> v903_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v902_el, v901_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v905_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v900_el, v903_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v902_el, v905_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v909_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, (v899_data[3]), v907_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v915_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v428_vlead];
        float v916_el = v915_data[0];
        tensorforge::VectorT<float, 4> v917_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v916_el, v909_acc, 3, 2, 1);
        float v918_el = v915_data[1];
        tensorforge::VectorT<float, 4> v919_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v918_el, v917_acc, 3, 2, 1);
        float v920_el = v915_data[3];
        tensorforge::VectorT<float, 4> v921_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v920_el, v919_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v923_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v916_el, v921_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v925_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v918_el, v923_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v927_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v553_tp, v920_el, v925_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v933_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v446_vlead];
        float v934_el = v933_data[0];
        tensorforge::VectorT<float, 4> v935_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v934_el, v927_acc, 3, 4, 1);
        float v936_el = v933_data[1];
        tensorforge::VectorT<float, 4> v937_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v936_el, v935_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v939_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v934_el, v937_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v941_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v936_el, v939_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v947_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v460_vlead];
        float v948_el = v947_data[0];
        tensorforge::VectorT<float, 4> v949_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v948_el, v941_acc, 3, 6, 1);
        float v950_el = v947_data[1];
        tensorforge::VectorT<float, 4> v951_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v950_el, v949_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v953_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v948_el, v951_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v955_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v950_el, v953_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v957_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, (v947_data[2]), v955_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v963_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v476_vlead];
        tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, (v963_data[0]), v957_acc, 3, 0, 1);
        float v966_el = v963_data[1];
        tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v966_el, v965_acc, 3, 0, 1);
        float v968_el = v963_data[2];
        tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v968_el, v967_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v971_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v966_el, v969_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v973_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v968_el, v971_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v975_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, (v963_data[3]), v973_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v981_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v494_vlead];
        float v982_el = v981_data[1];
        tensorforge::VectorT<float, 4> v983_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v982_el, v975_acc, 3, 2, 1);
        float v984_el = v981_data[2];
        tensorforge::VectorT<float, 4> v985_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v984_el, v983_acc, 3, 2, 1);
        tensorforge::VectorT<float, 4> v987_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, v982_el, v985_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v989_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v984_el, v987_acc, 3, 3, 2);
        tensorforge::VectorT<float, 4> v995_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v508_vlead];
        tensorforge::VectorT<float, 4> v997_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, (v995_data[1]), v989_acc, 3, 4, 1);
        float v998_el = v995_data[2];
        tensorforge::VectorT<float, 4> v999_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v998_el, v997_acc, 3, 4, 1);
        tensorforge::VectorT<float, 4> v1001_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, (v995_data[0]), v999_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1003_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v998_el, v1001_acc, 3, 5, 2);
        tensorforge::VectorT<float, 4> v1009_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v522_vlead];
        tensorforge::VectorT<float, 4> v1011_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v556_tp, (v1009_data[0]), v1003_acc, 3, 6, 1);
        float v1012_el = v1009_data[2];
        tensorforge::VectorT<float, 4> v1013_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v1012_el, v1011_acc, 3, 6, 1);
        float v1014_el = v1009_data[3];
        tensorforge::VectorT<float, 4> v1015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v1014_el, v1013_acc, 3, 6, 1);
        tensorforge::VectorT<float, 4> v1017_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v557_tp, (v1009_data[1]), v1015_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1019_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v558_tp, v1012_el, v1017_acc, 3, 7, 2);
        tensorforge::VectorT<float, 4> v1021_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v559_tp, v1014_el, v1019_acc, 3, 7, 2);
        r1[9] = (v1021_acc[0]);
        r1[11] = (v1021_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v1024_i0 = 0; v1024_i0 < 2; ++v1024_i0) {
          #pragma unroll
          for (int32_t v1025_i1 = 0; v1025_i1 < 6; ++v1025_i1) {
            float v1028_data = r1[(v1024_i0 + (v1025_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v38_lead + (v1024_i0 * 32)) + (v1025_i1 * 64))] = v1028_data;
            }
          }
        }
      }
    }
  }
}

