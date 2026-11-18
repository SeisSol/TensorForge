// === base name ===
kernel_b8c2fbc340b3bec9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b8c2fbc340b3bec9 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b8c2fbc340b3bec9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b8c2fbc340b3bec9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b8c2fbc340b3bec9(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b8c2fbc340b3bec9, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b8c2fbc340b3bec9, block.x * block.y * block.z, 0));
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
void launcher_kernel_b8c2fbc340b3bec9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b8c2fbc340b3bec9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b8c2fbc340b3bec9), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  const float *const * m2Arg = (const float *const *)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b8c2fbc340b3bec9, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m1_stride, m2Arg, m2_extraOffset, m2_slot, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b8c2fbc340b3bec9(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, size_t m1_stride, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
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
      int32_t v28_lead = threadIdx.x % 16;
      float r0[16]{};
      size_t v304_row = v11_batchIdGroup0First + v8_batchIdLane0;
      const bool peel_batchIdActive0 = v304_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v304_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[static_cast<size_t>(elementIndices[(peel_batchIdActive0 ? v304_row : v11_batchIdGroup0First)]) * m1_stride + 0 + m1_extraOffset];
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v312_i0 = 0; v312_i0 < 1; ++v312_i0) {
        int32_t v314_lead = v28_lead + (v312_i0 * 16);
        #pragma unroll
        for (int32_t v315_i1 = 0; v315_i1 < 16; ++v315_i1) {
          float v318_data = __builtin_nontemporal_load(&peel_glb_m1[(v314_lead + (v315_i1 * 16))]);
          r0[(v312_i0 + v315_i1)] = v318_data;
        }
      }
      for (size_t v12_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v301_groupNextAhead = v12_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v303_groupNext = (v301_groupNextAhead < numElements0) ? v301_groupNextAhead : v12_batchIdGroup0;
        size_t v308_row = v303_groupNext + v8_batchIdLane0;
        const bool wrap_batchIdActive0 = v308_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v308_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[static_cast<size_t>(elementIndices[(wrap_batchIdActive0 ? v308_row : v303_groupNext)]) * m1_stride + 0 + m1_extraOffset];
        size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
        const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
        size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 300 + 0 + 8 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[v15_batchId0]) * 4 + m2_slot][0 + 4 + m2_extraOffset];
        float r1[16]{};
        // r1 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v38_i0 = 0; v38_i0 < 1; ++v38_i0) {
          int32_t v41_lead = v28_lead + (v38_i0 * 16);
          #pragma unroll
          for (int32_t v39_i1 = 0; v39_i1 < 16; ++v39_i1) {
            float v44_data = __builtin_nontemporal_load(&glb_m2[(v41_lead + (v39_i1 * 16))]);
            r1[(v38_i0 + v39_i1)] = v44_data;
          }
        }
        float r2[16]{};
        // ir2 = +(r0 * r1)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir2[16]{};
        float v48_data = r1[0];
        float v49_data = r1[1];
        float v50_data = r1[2];
        float v51_data = r1[3];
        float v52_data = r1[4];
        float v53_data = r1[5];
        float v54_data = r1[6];
        float v55_data = r1[7];
        float v56_data = r1[8];
        float v57_data = r1[9];
        float v58_data = r1[10];
        float v59_data = r1[11];
        float v60_data = r1[12];
        float v61_data = r1[13];
        float v62_data = r1[14];
        float v63_data = r1[15];
        tensorforge::transpose16x16b32(v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data);
        tensorforge::VectorT<float, 16> v64_acc{};
        float v65_data = r0[0];
        float v66_data = r0[1];
        float v67_data = r0[2];
        float v68_data = r0[3];
        float v69_data = r0[4];
        float v70_data = r0[5];
        float v71_data = r0[6];
        float v72_data = r0[7];
        float v73_data = r0[8];
        float v74_data = r0[9];
        float v75_data = r0[10];
        float v76_data = r0[11];
        float v77_data = r0[12];
        float v78_data = r0[13];
        float v79_data = r0[14];
        float v80_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m1);
        #pragma unroll
        for (int32_t v320_i0 = 0; v320_i0 < 1; ++v320_i0) {
          int32_t v322_lead = v28_lead + (v320_i0 * 16);
          #pragma unroll
          for (int32_t v323_i1 = 0; v323_i1 < 16; ++v323_i1) {
            float v326_data = __builtin_nontemporal_load(&wrap_glb_m1[(v322_lead + (v323_i1 * 16))]);
            r0[(v320_i0 + v323_i1)] = v326_data;
          }
        }
        tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v64_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v81_acc, 0, 0, 0);
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
        float v97_el = v96_acc[0];
        float v99_el = v96_acc[4];
        float v100_sw = tensorforge::swap<32>(v99_el);
        float v102_el = v96_acc[8];
        float v105_el = v96_acc[12];
        float v106_sw = tensorforge::swap<32>(v105_el);
        ir2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v106_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v102_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v100_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v97_el, v97_el))))))));
        float v109_el = v96_acc[1];
        float v111_el = v96_acc[5];
        float v112_sw = tensorforge::swap<32>(v111_el);
        float v114_el = v96_acc[9];
        float v117_el = v96_acc[13];
        float v118_sw = tensorforge::swap<32>(v117_el);
        ir2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v118_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v114_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v112_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v109_el, v109_el))))))));
        float v121_el = v96_acc[2];
        float v123_el = v96_acc[6];
        float v124_sw = tensorforge::swap<32>(v123_el);
        float v126_el = v96_acc[10];
        float v129_el = v96_acc[14];
        float v130_sw = tensorforge::swap<32>(v129_el);
        ir2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v121_el, v121_el))))))));
        float v133_el = v96_acc[3];
        float v135_el = v96_acc[7];
        float v136_sw = tensorforge::swap<32>(v135_el);
        float v138_el = v96_acc[11];
        float v141_el = v96_acc[15];
        float v142_sw = tensorforge::swap<32>(v141_el);
        ir2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v142_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v133_el, v133_el))))))));
        float v146_sw = tensorforge::swap<32>(v97_el);
        float v151_sw = tensorforge::swap<32>(v102_el);
        ir2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v99_el, (tensorforge::dppUpdate<228, 1, 15, false>(v146_sw, v146_sw))))))));
        float v158_sw = tensorforge::swap<32>(v109_el);
        float v163_sw = tensorforge::swap<32>(v114_el);
        ir2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v111_el, (tensorforge::dppUpdate<228, 1, 15, false>(v158_sw, v158_sw))))))));
        float v170_sw = tensorforge::swap<32>(v121_el);
        float v175_sw = tensorforge::swap<32>(v126_el);
        ir2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v175_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v123_el, (tensorforge::dppUpdate<228, 1, 15, false>(v170_sw, v170_sw))))))));
        float v182_sw = tensorforge::swap<32>(v133_el);
        float v187_sw = tensorforge::swap<32>(v138_el);
        ir2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v187_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v135_el, (tensorforge::dppUpdate<228, 1, 15, false>(v182_sw, v182_sw))))))));
        float v194_sw = tensorforge::swap<64>(v97_el);
        ir2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v106_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v102_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v100_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v194_sw, v194_sw))))))));
        float v206_sw = tensorforge::swap<64>(v109_el);
        ir2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v118_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v114_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v112_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v206_sw, v206_sw))))))));
        float v218_sw = tensorforge::swap<64>(v121_el);
        ir2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v126_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v218_sw, v218_sw))))))));
        float v230_sw = tensorforge::swap<64>(v133_el);
        ir2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v142_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v138_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v230_sw, v230_sw))))))));
        float v243_sw = tensorforge::swap<64>(v146_sw);
        ir2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v105_el, (tensorforge::dppUpdate<228, 4, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v99_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v243_sw, v243_sw))))))));
        float v255_sw = tensorforge::swap<64>(v158_sw);
        ir2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v117_el, (tensorforge::dppUpdate<228, 4, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v255_sw, v255_sw))))))));
        float v267_sw = tensorforge::swap<64>(v170_sw);
        ir2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v129_el, (tensorforge::dppUpdate<228, 4, 15, false>(v175_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v267_sw, v267_sw))))))));
        float v279_sw = tensorforge::swap<64>(v182_sw);
        ir2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v141_el, (tensorforge::dppUpdate<228, 4, 15, false>(v187_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v279_sw, v279_sw))))))));
        // r2 = ir2
        // glb_m0 = store{r>g}(r2);
        #pragma unroll
        for (int32_t v293_i0 = 0; v293_i0 < 1; ++v293_i0) {
          #pragma unroll
          for (int32_t v294_i1 = 0; v294_i1 < 16; ++v294_i1) {
            float v296_data = ir2[v294_i1];
            if (batchIdActive0) {
              glb_m0[((v28_lead + (v293_i0 * 16)) + (v294_i1 * 16))] = v296_data;
            }
          }
        }
      }
    }
  }
}

