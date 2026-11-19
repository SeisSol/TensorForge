// === base name ===
kernel_ecd81b1de69792a4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ecd81b1de69792a4 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ecd81b1de69792a4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ecd81b1de69792a4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ecd81b1de69792a4(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ecd81b1de69792a4, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_ecd81b1de69792a4, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ecd81b1de69792a4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ecd81b1de69792a4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ecd81b1de69792a4), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  const float *const * m2Arg = (const float *const *)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_ecd81b1de69792a4, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m1_stride, m2Arg, m2_extraOffset, m2_slot, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ecd81b1de69792a4(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, size_t m1_stride, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided(stride=300, offset=8)
    //   m1 16×16(16×16) {0..16}×{0..16} indexed(stride=runtime)
    //   m2 16×16(16×16) {0..16}×{0..16} indexed_pointer(offset=4, slots=4, slot=runtime)
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided(stride=300, offset=8)","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided(stride=300, offset=8)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      size_t v8_batchIdLane0 = threadIdx.y % 4;
      size_t v11_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v29_lead = threadIdx.x % 16;
      float r0[16]{};
      size_t v305_row = v11_batchIdGroup0First + v8_batchIdLane0;
      const bool peel_batchIdActive0 = v305_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v305_row]));
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[(peel_batchIdActive0 ? v305_row : v11_batchIdGroup0First)]);
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v315_i0 = 0; v315_i0 < 1; ++v315_i0) {
        int32_t v317_lead = v29_lead + (v315_i0 * 16);
        #pragma unroll
        for (int32_t v318_i1 = 0; v318_i1 < 16; ++v318_i1) {
          float v321_data = __builtin_nontemporal_load(&peel_glb_m1[(v317_lead + (v318_i1 * 16))]);
          r0[(v315_i0 + v318_i1)] = v321_data;
        }
      }
      size_t v331_elementIndex = peel_elementIndex;
      for (size_t v12_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v302_groupNextAhead = v12_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v304_groupNext = (v302_groupNextAhead < numElements0) ? v302_groupNextAhead : v12_batchIdGroup0;
        size_t v310_row = v304_groupNext + v8_batchIdLane0;
        const bool wrap_batchIdActive0 = v310_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v310_row]));
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[(wrap_batchIdActive0 ? v310_row : v304_groupNext)]);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
        size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
        const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
        size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 300 + 0 + 8 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v331_elementIndex * 4 + m2_slot][0 + 4 + m2_extraOffset];
        float r1[16]{};
        // r1 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
          int32_t v42_lead = v29_lead + (v39_i0 * 16);
          #pragma unroll
          for (int32_t v40_i1 = 0; v40_i1 < 16; ++v40_i1) {
            float v45_data = __builtin_nontemporal_load(&glb_m2[(v42_lead + (v40_i1 * 16))]);
            r1[(v39_i0 + v40_i1)] = v45_data;
          }
        }
        float r2[16]{};
        // ir2 = +(r0 * r1)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir2[16]{};
        float v49_data = r1[0];
        float v50_data = r1[1];
        float v51_data = r1[2];
        float v52_data = r1[3];
        float v53_data = r1[4];
        float v54_data = r1[5];
        float v55_data = r1[6];
        float v56_data = r1[7];
        float v57_data = r1[8];
        float v58_data = r1[9];
        float v59_data = r1[10];
        float v60_data = r1[11];
        float v61_data = r1[12];
        float v62_data = r1[13];
        float v63_data = r1[14];
        float v64_data = r1[15];
        tensorforge::transpose16x16b32(v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data);
        tensorforge::VectorT<float, 16> v65_acc{};
        float v66_data = r0[0];
        float v67_data = r0[1];
        float v68_data = r0[2];
        float v69_data = r0[3];
        float v70_data = r0[4];
        float v71_data = r0[5];
        float v72_data = r0[6];
        float v73_data = r0[7];
        float v74_data = r0[8];
        float v75_data = r0[9];
        float v76_data = r0[10];
        float v77_data = r0[11];
        float v78_data = r0[12];
        float v79_data = r0[13];
        float v80_data = r0[14];
        float v81_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m1);
        #pragma unroll
        for (int32_t v323_i0 = 0; v323_i0 < 1; ++v323_i0) {
          int32_t v325_lead = v29_lead + (v323_i0 * 16);
          #pragma unroll
          for (int32_t v326_i1 = 0; v326_i1 < 16; ++v326_i1) {
            float v329_data = __builtin_nontemporal_load(&wrap_glb_m1[(v325_lead + (v326_i1 * 16))]);
            r0[(v323_i0 + v326_i1)] = v329_data;
          }
        }
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v65_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v90_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v91_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v92_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v93_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v94_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v95_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v81_data, v96_acc, 0, 0, 0);
        float v98_el = v97_acc[0];
        float v100_el = v97_acc[4];
        float v101_sw = tensorforge::swap<32>(v100_el);
        float v103_el = v97_acc[8];
        float v106_el = v97_acc[12];
        float v107_sw = tensorforge::swap<32>(v106_el);
        ir2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v107_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v101_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v98_el, v98_el))))))));
        float v110_el = v97_acc[1];
        float v112_el = v97_acc[5];
        float v113_sw = tensorforge::swap<32>(v112_el);
        float v115_el = v97_acc[9];
        float v118_el = v97_acc[13];
        float v119_sw = tensorforge::swap<32>(v118_el);
        ir2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v119_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v113_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v110_el, v110_el))))))));
        float v122_el = v97_acc[2];
        float v124_el = v97_acc[6];
        float v125_sw = tensorforge::swap<32>(v124_el);
        float v127_el = v97_acc[10];
        float v130_el = v97_acc[14];
        float v131_sw = tensorforge::swap<32>(v130_el);
        ir2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v131_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v125_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v122_el, v122_el))))))));
        float v134_el = v97_acc[3];
        float v136_el = v97_acc[7];
        float v137_sw = tensorforge::swap<32>(v136_el);
        float v139_el = v97_acc[11];
        float v142_el = v97_acc[15];
        float v143_sw = tensorforge::swap<32>(v142_el);
        ir2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v143_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v139_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v137_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v134_el, v134_el))))))));
        float v147_sw = tensorforge::swap<32>(v98_el);
        float v152_sw = tensorforge::swap<32>(v103_el);
        ir2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v152_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v100_el, (tensorforge::dppUpdate<228, 1, 15, false>(v147_sw, v147_sw))))))));
        float v159_sw = tensorforge::swap<32>(v110_el);
        float v164_sw = tensorforge::swap<32>(v115_el);
        ir2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v164_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v112_el, (tensorforge::dppUpdate<228, 1, 15, false>(v159_sw, v159_sw))))))));
        float v171_sw = tensorforge::swap<32>(v122_el);
        float v176_sw = tensorforge::swap<32>(v127_el);
        ir2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v176_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_el, (tensorforge::dppUpdate<228, 1, 15, false>(v171_sw, v171_sw))))))));
        float v183_sw = tensorforge::swap<32>(v134_el);
        float v188_sw = tensorforge::swap<32>(v139_el);
        ir2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v142_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v188_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v136_el, (tensorforge::dppUpdate<228, 1, 15, false>(v183_sw, v183_sw))))))));
        float v195_sw = tensorforge::swap<64>(v98_el);
        ir2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v107_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v103_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v101_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v195_sw, v195_sw))))))));
        float v207_sw = tensorforge::swap<64>(v110_el);
        ir2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v119_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v115_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v113_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v207_sw, v207_sw))))))));
        float v219_sw = tensorforge::swap<64>(v122_el);
        ir2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v131_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v127_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v125_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v219_sw, v219_sw))))))));
        float v231_sw = tensorforge::swap<64>(v134_el);
        ir2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v143_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v139_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v137_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v231_sw, v231_sw))))))));
        float v244_sw = tensorforge::swap<64>(v147_sw);
        ir2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v106_el, (tensorforge::dppUpdate<228, 4, 15, false>(v152_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v244_sw, v244_sw))))))));
        float v256_sw = tensorforge::swap<64>(v159_sw);
        ir2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v118_el, (tensorforge::dppUpdate<228, 4, 15, false>(v164_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v256_sw, v256_sw))))))));
        float v268_sw = tensorforge::swap<64>(v171_sw);
        ir2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_el, (tensorforge::dppUpdate<228, 4, 15, false>(v176_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v268_sw, v268_sw))))))));
        float v280_sw = tensorforge::swap<64>(v183_sw);
        ir2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v142_el, (tensorforge::dppUpdate<228, 4, 15, false>(v188_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v136_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v280_sw, v280_sw))))))));
        // r2 = ir2
        // glb_m0 = store{r>g}(r2);
        #pragma unroll
        for (int32_t v294_i0 = 0; v294_i0 < 1; ++v294_i0) {
          #pragma unroll
          for (int32_t v295_i1 = 0; v295_i1 < 16; ++v295_i1) {
            float v297_data = ir2[v295_i1];
            if (batchIdActive0) {
              glb_m0[((v29_lead + (v294_i0 * 16)) + (v295_i1 * 16))] = v297_data;
            }
          }
        }
        v331_elementIndex = wrap_elementIndex;
      }
    }
  }
}

