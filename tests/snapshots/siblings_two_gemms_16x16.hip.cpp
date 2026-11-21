// === base name ===
kernel_98f42065c3d8eded

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_98f42065c3d8eded = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 2};
tensorforge::LaunchConfig launch_config_kernel_98f42065c3d8eded(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_98f42065c3d8eded(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_98f42065c3d8eded(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_98f42065c3d8eded, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_98f42065c3d8eded, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_98f42065c3d8eded(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_98f42065c3d8eded(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_98f42065c3d8eded), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags1Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags1;
  hipLaunchKernelGGL(kernel_kernel_98f42065c3d8eded, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_98f42065c3d8eded(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} strided
    //   m5 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   siblings
    //   m3[i,j] = m4[i,k] × m5[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0},{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"siblings"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      size_t v8_batchIdLane0 = threadIdx.y % 2;
      size_t v11_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v31_lead = threadIdx.x % 32;
      bool v37_g = v31_lead < 16;
      float r0[16]{};
      size_t v307_row = v11_batchIdGroup0First + v8_batchIdLane0;
      const bool peel_batchIdActive0 = v307_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v307_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[(peel_batchIdActive0 ? v307_row : v11_batchIdGroup0First) * 256 + 0 + m1_extraOffset];
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v315_i1 = 0; v315_i1 < 16; ++v315_i1) {
        float v318_data = v37_g ? (__builtin_nontemporal_load(&peel_glb_m1[(v31_lead + (v315_i1 * 16))])) : (0.0f);
        r0[v315_i1] = v318_data;
      }
      for (size_t v12_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v304_groupNextAhead = v12_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v306_groupNext = (v304_groupNextAhead < numElements0) ? v304_groupNextAhead : v12_batchIdGroup0;
        size_t v311_row = v306_groupNext + v8_batchIdLane0;
        const bool wrap_batchIdActive0 = v311_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v311_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[(wrap_batchIdActive0 ? v311_row : v306_groupNext) * 256 + 0 + m1_extraOffset];
        size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
        const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
        size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v15_batchId0 * 256 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v15_batchId0 * 256 + 0 + m3_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v15_batchId0 * 256 + 0 + m4_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v15_batchId0 * 256 + 0 + m5_extraOffset];
        float r1[16]{};
        // r1 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v41_i1 = 0; v41_i1 < 16; ++v41_i1) {
          float v47_data = v37_g ? (__builtin_nontemporal_load(&glb_m2[(v31_lead + (v41_i1 * 16))])) : (0.0f);
          r1[v41_i1] = v47_data;
        }
        float r2[16]{};
        // ir2 = +(r0 * r1)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir2[16]{};
        float v51_data = r1[0];
        float v52_data = r1[1];
        float v53_data = r1[2];
        float v54_data = r1[3];
        float v55_data = r1[4];
        float v56_data = r1[5];
        float v57_data = r1[6];
        float v58_data = r1[7];
        float v59_data = r1[8];
        float v60_data = r1[9];
        float v61_data = r1[10];
        float v62_data = r1[11];
        float v63_data = r1[12];
        float v64_data = r1[13];
        float v65_data = r1[14];
        float v66_data = r1[15];
        tensorforge::transpose16x16b32(v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data);
        tensorforge::VectorT<float, 16> v67_acc{};
        float v68_data = r0[0];
        float v69_data = r0[1];
        float v70_data = r0[2];
        float v71_data = r0[3];
        float v72_data = r0[4];
        float v73_data = r0[5];
        float v74_data = r0[6];
        float v75_data = r0[7];
        float v76_data = r0[8];
        float v77_data = r0[9];
        float v78_data = r0[10];
        float v79_data = r0[11];
        float v80_data = r0[12];
        float v81_data = r0[13];
        float v82_data = r0[14];
        float v83_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m1);
        #pragma unroll
        for (int32_t v319_i1 = 0; v319_i1 < 16; ++v319_i1) {
          float v322_data = v37_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v31_lead + (v319_i1 * 16))])) : (0.0f);
          r0[v319_i1] = v322_data;
        }
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v67_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v90_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v91_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v92_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v93_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v94_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v95_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v81_data, v96_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v98_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v82_data, v97_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v83_data, v98_acc, 1, 0, 0);
        float v100_el = v99_acc[0];
        float v102_el = v99_acc[4];
        float v103_sw = tensorforge::swap<32>(v102_el);
        float v105_el = v99_acc[8];
        float v108_el = v99_acc[12];
        float v109_sw = tensorforge::swap<32>(v108_el);
        ir2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v109_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v103_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v100_el, v100_el))))))));
        float v112_el = v99_acc[1];
        float v114_el = v99_acc[5];
        float v115_sw = tensorforge::swap<32>(v114_el);
        float v117_el = v99_acc[9];
        float v120_el = v99_acc[13];
        float v121_sw = tensorforge::swap<32>(v120_el);
        ir2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v121_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v115_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v112_el, v112_el))))))));
        float v124_el = v99_acc[2];
        float v126_el = v99_acc[6];
        float v127_sw = tensorforge::swap<32>(v126_el);
        float v129_el = v99_acc[10];
        float v132_el = v99_acc[14];
        float v133_sw = tensorforge::swap<32>(v132_el);
        ir2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v124_el, v124_el))))))));
        float v136_el = v99_acc[3];
        float v138_el = v99_acc[7];
        float v139_sw = tensorforge::swap<32>(v138_el);
        float v141_el = v99_acc[11];
        float v144_el = v99_acc[15];
        float v145_sw = tensorforge::swap<32>(v144_el);
        ir2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v136_el, v136_el))))))));
        float v149_sw = tensorforge::swap<32>(v100_el);
        float v154_sw = tensorforge::swap<32>(v105_el);
        ir2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v108_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v154_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v102_el, (tensorforge::dppUpdate<228, 1, 15, false>(v149_sw, v149_sw))))))));
        float v161_sw = tensorforge::swap<32>(v112_el);
        float v166_sw = tensorforge::swap<32>(v117_el);
        ir2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v120_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v166_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v114_el, (tensorforge::dppUpdate<228, 1, 15, false>(v161_sw, v161_sw))))))));
        float v173_sw = tensorforge::swap<32>(v124_el);
        float v178_sw = tensorforge::swap<32>(v129_el);
        ir2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v178_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v126_el, (tensorforge::dppUpdate<228, 1, 15, false>(v173_sw, v173_sw))))))));
        float v185_sw = tensorforge::swap<32>(v136_el);
        float v190_sw = tensorforge::swap<32>(v141_el);
        ir2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v190_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v138_el, (tensorforge::dppUpdate<228, 1, 15, false>(v185_sw, v185_sw))))))));
        float v197_sw = tensorforge::swap<64>(v100_el);
        ir2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v109_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v105_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v103_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v197_sw, v197_sw))))))));
        float v209_sw = tensorforge::swap<64>(v112_el);
        ir2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v121_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v117_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v115_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v209_sw, v209_sw))))))));
        float v221_sw = tensorforge::swap<64>(v124_el);
        ir2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v129_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v221_sw, v221_sw))))))));
        float v233_sw = tensorforge::swap<64>(v136_el);
        ir2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v141_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
        float v246_sw = tensorforge::swap<64>(v149_sw);
        ir2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v108_el, (tensorforge::dppUpdate<228, 4, 15, false>(v154_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v102_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v246_sw, v246_sw))))))));
        float v258_sw = tensorforge::swap<64>(v161_sw);
        ir2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v120_el, (tensorforge::dppUpdate<228, 4, 15, false>(v166_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v114_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v258_sw, v258_sw))))))));
        float v270_sw = tensorforge::swap<64>(v173_sw);
        ir2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v132_el, (tensorforge::dppUpdate<228, 4, 15, false>(v178_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v270_sw, v270_sw))))))));
        float v282_sw = tensorforge::swap<64>(v185_sw);
        ir2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v144_el, (tensorforge::dppUpdate<228, 4, 15, false>(v190_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v282_sw, v282_sw))))))));
        // r2 = ir2
        // glb_m0 = store{r>g}(r2);
        if (v37_g) {
          #pragma unroll
          for (int32_t v297_i1 = 0; v297_i1 < 16; ++v297_i1) {
            float v299_data = ir2[v297_i1];
            if (batchIdActive0) {
              glb_m0[(v31_lead + (v297_i1 * 16))] = v299_data;
            }
          }
        }
      }
    }
    {
      __syncthreads();
      size_t v654_batchIdLane1 = threadIdx.y % 2;
      size_t v657_batchIdGroup1First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements1) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v677_lead = threadIdx.x % 32;
      bool v683_g = v677_lead < 16;
      float r0[16]{};
      size_t v953_row = v657_batchIdGroup1First + v654_batchIdLane1;
      const bool peel_batchIdActive1 = v953_row < numElements1 && (flags1 == nullptr || static_cast<bool>(flags1[v953_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[(peel_batchIdActive1 ? v953_row : v657_batchIdGroup1First) * 256 + 0 + m4_extraOffset];
      // r0 = load{g>r}(peel_glb_m4);
      #pragma unroll
      for (int32_t v961_i1 = 0; v961_i1 < 16; ++v961_i1) {
        float v964_data = v683_g ? (__builtin_nontemporal_load(&peel_glb_m4[(v677_lead + (v961_i1 * 16))])) : (0.0f);
        r0[v961_i1] = v964_data;
      }
      for (size_t v658_batchIdGroup1 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v658_batchIdGroup1 < numElements1; v658_batchIdGroup1 += (gridDim.x * blockDim.y)) {
        size_t v950_groupNextAhead = v658_batchIdGroup1 + (gridDim.x * blockDim.y);
        size_t v952_groupNext = (v950_groupNextAhead < numElements1) ? v950_groupNextAhead : v658_batchIdGroup1;
        size_t v957_row = v952_groupNext + v654_batchIdLane1;
        const bool wrap_batchIdActive1 = v957_row < numElements1 && (flags1 == nullptr || static_cast<bool>(flags1[v957_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[(wrap_batchIdActive1 ? v957_row : v952_groupNext) * 256 + 0 + m4_extraOffset];
        size_t v659_row = v658_batchIdGroup1 + v654_batchIdLane1;
        const bool batchIdActive1 = v659_row < numElements1 && (flags1 == nullptr || static_cast<bool>(flags1[v659_row]));
        size_t v661_batchId0 = batchIdActive1 ? v659_row : v658_batchIdGroup1;
        size_t v662_ahead1 = v661_batchId0 + (gridDim.x * blockDim.y);
        size_t v664_batchId1 = (v662_ahead1 < numElements1) ? v662_ahead1 : v661_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v661_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v661_batchId0 * 256 + 0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v661_batchId0 * 256 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v661_batchId0 * 256 + 0 + m3_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v661_batchId0 * 256 + 0 + m5_extraOffset];
        float r1[16]{};
        // r1 = load{g>r}(glb_m5);
        #pragma unroll
        for (int32_t v687_i1 = 0; v687_i1 < 16; ++v687_i1) {
          float v693_data = v683_g ? (__builtin_nontemporal_load(&glb_m5[(v677_lead + (v687_i1 * 16))])) : (0.0f);
          r1[v687_i1] = v693_data;
        }
        float r2[16]{};
        // ir2 = +(r0 * r1)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir2[16]{};
        float v697_data = r1[0];
        float v698_data = r1[1];
        float v699_data = r1[2];
        float v700_data = r1[3];
        float v701_data = r1[4];
        float v702_data = r1[5];
        float v703_data = r1[6];
        float v704_data = r1[7];
        float v705_data = r1[8];
        float v706_data = r1[9];
        float v707_data = r1[10];
        float v708_data = r1[11];
        float v709_data = r1[12];
        float v710_data = r1[13];
        float v711_data = r1[14];
        float v712_data = r1[15];
        tensorforge::transpose16x16b32(v697_data, v698_data, v699_data, v700_data, v701_data, v702_data, v703_data, v704_data, v705_data, v706_data, v707_data, v708_data, v709_data, v710_data, v711_data, v712_data);
        tensorforge::VectorT<float, 16> v713_acc{};
        float v714_data = r0[0];
        float v715_data = r0[1];
        float v716_data = r0[2];
        float v717_data = r0[3];
        float v718_data = r0[4];
        float v719_data = r0[5];
        float v720_data = r0[6];
        float v721_data = r0[7];
        float v722_data = r0[8];
        float v723_data = r0[9];
        float v724_data = r0[10];
        float v725_data = r0[11];
        float v726_data = r0[12];
        float v727_data = r0[13];
        float v728_data = r0[14];
        float v729_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m4);
        #pragma unroll
        for (int32_t v965_i1 = 0; v965_i1 < 16; ++v965_i1) {
          float v968_data = v683_g ? (__builtin_nontemporal_load(&wrap_glb_m4[(v677_lead + (v965_i1 * 16))])) : (0.0f);
          r0[v965_i1] = v968_data;
        }
        tensorforge::VectorT<float, 16> v730_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v697_data, v714_data, v713_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v731_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v698_data, v715_data, v730_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v732_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v699_data, v716_data, v731_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v733_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v700_data, v717_data, v732_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v734_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v701_data, v718_data, v733_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v735_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v702_data, v719_data, v734_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v736_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v703_data, v720_data, v735_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v737_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v704_data, v721_data, v736_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v738_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v705_data, v722_data, v737_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v739_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v706_data, v723_data, v738_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v740_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v707_data, v724_data, v739_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v741_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v708_data, v725_data, v740_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v742_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v709_data, v726_data, v741_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v743_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v710_data, v727_data, v742_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v744_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v711_data, v728_data, v743_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v745_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v712_data, v729_data, v744_acc, 1, 0, 0);
        float v746_el = v745_acc[0];
        float v748_el = v745_acc[4];
        float v749_sw = tensorforge::swap<32>(v748_el);
        float v751_el = v745_acc[8];
        float v754_el = v745_acc[12];
        float v755_sw = tensorforge::swap<32>(v754_el);
        ir2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v755_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v751_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v749_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v746_el, v746_el))))))));
        float v758_el = v745_acc[1];
        float v760_el = v745_acc[5];
        float v761_sw = tensorforge::swap<32>(v760_el);
        float v763_el = v745_acc[9];
        float v766_el = v745_acc[13];
        float v767_sw = tensorforge::swap<32>(v766_el);
        ir2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v767_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v763_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v761_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v758_el, v758_el))))))));
        float v770_el = v745_acc[2];
        float v772_el = v745_acc[6];
        float v773_sw = tensorforge::swap<32>(v772_el);
        float v775_el = v745_acc[10];
        float v778_el = v745_acc[14];
        float v779_sw = tensorforge::swap<32>(v778_el);
        ir2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v779_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v775_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v773_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v770_el, v770_el))))))));
        float v782_el = v745_acc[3];
        float v784_el = v745_acc[7];
        float v785_sw = tensorforge::swap<32>(v784_el);
        float v787_el = v745_acc[11];
        float v790_el = v745_acc[15];
        float v791_sw = tensorforge::swap<32>(v790_el);
        ir2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v791_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v787_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v785_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v782_el, v782_el))))))));
        float v795_sw = tensorforge::swap<32>(v746_el);
        float v800_sw = tensorforge::swap<32>(v751_el);
        ir2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v754_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v800_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v748_el, (tensorforge::dppUpdate<228, 1, 15, false>(v795_sw, v795_sw))))))));
        float v807_sw = tensorforge::swap<32>(v758_el);
        float v812_sw = tensorforge::swap<32>(v763_el);
        ir2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v766_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v812_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v760_el, (tensorforge::dppUpdate<228, 1, 15, false>(v807_sw, v807_sw))))))));
        float v819_sw = tensorforge::swap<32>(v770_el);
        float v824_sw = tensorforge::swap<32>(v775_el);
        ir2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v778_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v824_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v772_el, (tensorforge::dppUpdate<228, 1, 15, false>(v819_sw, v819_sw))))))));
        float v831_sw = tensorforge::swap<32>(v782_el);
        float v836_sw = tensorforge::swap<32>(v787_el);
        ir2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v790_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v836_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v784_el, (tensorforge::dppUpdate<228, 1, 15, false>(v831_sw, v831_sw))))))));
        float v843_sw = tensorforge::swap<64>(v746_el);
        ir2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v755_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v751_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v749_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v843_sw, v843_sw))))))));
        float v855_sw = tensorforge::swap<64>(v758_el);
        ir2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v767_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v763_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v761_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v855_sw, v855_sw))))))));
        float v867_sw = tensorforge::swap<64>(v770_el);
        ir2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v779_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v775_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v773_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v867_sw, v867_sw))))))));
        float v879_sw = tensorforge::swap<64>(v782_el);
        ir2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v791_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v787_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v785_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v879_sw, v879_sw))))))));
        float v892_sw = tensorforge::swap<64>(v795_sw);
        ir2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v754_el, (tensorforge::dppUpdate<228, 4, 15, false>(v800_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v748_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v892_sw, v892_sw))))))));
        float v904_sw = tensorforge::swap<64>(v807_sw);
        ir2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v766_el, (tensorforge::dppUpdate<228, 4, 15, false>(v812_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v760_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v904_sw, v904_sw))))))));
        float v916_sw = tensorforge::swap<64>(v819_sw);
        ir2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v778_el, (tensorforge::dppUpdate<228, 4, 15, false>(v824_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v772_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v916_sw, v916_sw))))))));
        float v928_sw = tensorforge::swap<64>(v831_sw);
        ir2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v790_el, (tensorforge::dppUpdate<228, 4, 15, false>(v836_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v784_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v928_sw, v928_sw))))))));
        // r2 = ir2
        // glb_m3 = store{r>g}(r2);
        if (v683_g) {
          #pragma unroll
          for (int32_t v943_i1 = 0; v943_i1 < 16; ++v943_i1) {
            float v945_data = ir2[v943_i1];
            if (batchIdActive1) {
              glb_m3[(v677_lead + (v943_i1 * 16))] = v945_data;
            }
          }
        }
      }
    }
  }
}

