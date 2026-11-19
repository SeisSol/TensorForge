// === base name ===
kernel_a8b6427d391693e3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a8b6427d391693e3 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a8b6427d391693e3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a8b6427d391693e3(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a8b6427d391693e3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a8b6427d391693e3, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_a8b6427d391693e3, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(float)));
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
  config.grid[0] = std::min(gridsize, static_cast<size_t>((numElements0 + 1) / 2));
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a8b6427d391693e3(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a8b6427d391693e3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a8b6427d391693e3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  const float *const * m2Arg = (const float *const *)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_a8b6427d391693e3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a8b6427d391693e3(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: stack_elements=2
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed(stride=136, offset=4)
    //   m1 16×16(16×16) {0..16}×{0..16} none ordered
    //   m2 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed_pointer(slots=3, slot=2)
    // operations:
    //   m0[i,j,l] = m1[i,k] × m2[k,j,l]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed(stride=136, offset=4)","alias":"C","bbox":[[0,0,0],[16,8,2]],"name":"m0","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":true,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(slots=3, slot=2)","alias":"B","bbox":[[0,0,0],[16,8,2]],"name":"m2","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed(stride=136, offset=4)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m0","offset":[0,0,0],"shape":[16,8,2]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(slots=3, slot=2)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m2","offset":[0,0,0],"shape":[16,8,2]}],"permute":[[0,1],[0,1,2]],"target":[[0,-1],[-1,1,2]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 4;
      size_t v14_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < ((numElements0 + 1) / 2)) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v36_lead = threadIdx.x % 16;
      int32_t v73_vlead = (v36_lead + ((threadIdx.y % 4) * 16)) * 4;
      float r0[16]{};
      size_t v322_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdStack0_0 = 2 * v322_row + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v322_row + 0]));
      const bool peel_batchIdStack0_1 = 2 * v322_row + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v322_row + 1]));
      const bool peel_batchIdActive0 = peel_batchIdStack0_0 || peel_batchIdStack0_1;
      size_t v326_batchId0 = peel_batchIdActive0 ? v322_row : v14_batchIdGroup0First;
      const size_t peel_elementIndex_s0 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v326_batchId0 + 0 : (peel_batchIdStack0_0 ? 2 * v326_batchId0 + 0 : peel_batchIdStack0_1 ? 2 * v326_batchId0 + 1 : 2 * v326_batchId0))]);
      const size_t peel_elementIndex_s1 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v326_batchId0 + 1 : (peel_batchIdStack0_0 ? 2 * v326_batchId0 + 0 : peel_batchIdStack0_1 ? 2 * v326_batchId0 + 1 : 2 * v326_batchId0))]);
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2_s0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[peel_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2_s1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[peel_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
      // r0 = load{g>r}(glb_m2);
      #pragma unroll
      for (int32_t v340_i0 = 0; v340_i0 < 1; ++v340_i0) {
        int32_t v342_lead = v36_lead + (v340_i0 * 16);
        #pragma unroll
        for (int32_t v343_i1 = 0; v343_i1 < 8; ++v343_i1) {
          int32_t v345_a = v342_lead + (v343_i1 * 16);
          int32_t v346_a = v340_i0 + v343_i1;
          #pragma unroll
          for (int32_t v347_i2 = 0; v347_i2 < 2; ++v347_i2) {
            tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2_se0 = v347_i2 == 0 ? peel_glb_m2_s0 : peel_glb_m2_s1;
            float v349_data = __builtin_nontemporal_load(&glb_m2_se0[v345_a]);
            r0[(v346_a + (v347_i2 * 8))] = v349_data;
          }
        }
      }
      size_t v364_elementIndex_s0 = peel_elementIndex_s0;
      size_t v366_elementIndex_s1 = peel_elementIndex_s1;
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < ((numElements0 + 1) / 2); v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v319_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v321_groupNext = (v319_groupNextAhead < ((numElements0 + 1) / 2)) ? v319_groupNextAhead : v15_batchIdGroup0;
        size_t v331_row = v321_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdStack0_0 = 2 * v331_row + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v331_row + 0]));
        const bool wrap_batchIdStack0_1 = 2 * v331_row + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v331_row + 1]));
        const bool wrap_batchIdActive0 = wrap_batchIdStack0_0 || wrap_batchIdStack0_1;
        size_t v335_batchId0 = wrap_batchIdActive0 ? v331_row : v321_groupNext;
        const size_t wrap_elementIndex_s0 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v335_batchId0 + 0 : (wrap_batchIdStack0_0 ? 2 * v335_batchId0 + 0 : wrap_batchIdStack0_1 ? 2 * v335_batchId0 + 1 : 2 * v335_batchId0))]);
        const size_t wrap_elementIndex_s1 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v335_batchId0 + 1 : (wrap_batchIdStack0_0 ? 2 * v335_batchId0 + 0 : wrap_batchIdStack0_1 ? 2 * v335_batchId0 + 1 : 2 * v335_batchId0))]);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2_s0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[wrap_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2_s1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[wrap_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdStack0_0 = 2 * v16_row + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v16_row + 0]));
        const bool batchIdStack0_1 = 2 * v16_row + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v16_row + 1]));
        const bool batchIdActive0 = batchIdStack0_0 || batchIdStack0_1;
        size_t v20_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v21_ahead1 = v20_batchId0 + (gridDim.x * blockDim.y);
        size_t v23_batchId1 = (v21_ahead1 < ((numElements0 + 1) / 2)) ? v21_ahead1 : v20_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> const glb_m0_s0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v364_elementIndex_s0 * 136 + 0 + 4 + m0_extraOffset];
        tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> const glb_m0_s1 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v366_elementIndex_s1 * 136 + 0 + 4 + m0_extraOffset];
        float r1[16]{};
        // ir1 = +(glb_m1 * r0)
        // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
        float ir1[16]{};
        float v51_data = r0[0];
        float v52_data = r0[1];
        float v53_data = r0[2];
        float v54_data = r0[3];
        float v55_data = r0[4];
        float v56_data = r0[5];
        float v57_data = r0[6];
        float v58_data = r0[7];
        float v59_data = r0[8];
        float v60_data = r0[9];
        float v61_data = r0[10];
        float v62_data = r0[11];
        float v63_data = r0[12];
        float v64_data = r0[13];
        float v65_data = r0[14];
        float v66_data = r0[15];
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v352_i0 = 0; v352_i0 < 1; ++v352_i0) {
          int32_t v354_lead = v36_lead + (v352_i0 * 16);
          #pragma unroll
          for (int32_t v355_i1 = 0; v355_i1 < 8; ++v355_i1) {
            int32_t v357_a = v354_lead + (v355_i1 * 16);
            int32_t v358_a = v352_i0 + v355_i1;
            #pragma unroll
            for (int32_t v359_i2 = 0; v359_i2 < 2; ++v359_i2) {
              tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2_se0 = v359_i2 == 0 ? wrap_glb_m2_s0 : wrap_glb_m2_s1;
              float v361_data = __builtin_nontemporal_load(&glb_m2_se0[v357_a]);
              r0[(v358_a + (v359_i2 * 8))] = v361_data;
            }
          }
        }
        tensorforge::transpose16x16b32(v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data);
        tensorforge::VectorT<float, 16> v67_acc{};
        tensorforge::VectorT<float, 4> v75_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v73_vlead];
        float v76_el = v75_data[0];
        tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v76_el, v67_acc, 0, 0, 4);
        float v78_el = v75_data[1];
        tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v78_el, v77_acc, 0, 0, 4);
        float v80_el = v75_data[2];
        tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v80_el, v79_acc, 0, 0, 4);
        float v82_el = v75_data[3];
        tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v82_el, v81_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v76_el, v83_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v78_el, v85_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v80_el, v87_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v82_el, v89_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_el, v91_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v78_el, v93_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v80_el, v95_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v82_el, v97_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v101_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v76_el, v99_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v103_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v78_el, v101_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v105_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v80_el, v103_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v107_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v82_el, v105_acc, 0, 0, 7);
        float v108_el = v107_acc[0];
        float v110_el = v107_acc[4];
        float v111_sw = tensorforge::swap<32>(v110_el);
        float v113_el = v107_acc[8];
        float v116_el = v107_acc[12];
        float v117_sw = tensorforge::swap<32>(v116_el);
        ir1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v117_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v113_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v111_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v108_el, v108_el))))))));
        float v120_el = v107_acc[1];
        float v122_el = v107_acc[5];
        float v123_sw = tensorforge::swap<32>(v122_el);
        float v125_el = v107_acc[9];
        float v128_el = v107_acc[13];
        float v129_sw = tensorforge::swap<32>(v128_el);
        ir1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v129_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v125_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v123_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v120_el, v120_el))))))));
        float v132_el = v107_acc[2];
        float v134_el = v107_acc[6];
        float v135_sw = tensorforge::swap<32>(v134_el);
        float v137_el = v107_acc[10];
        float v140_el = v107_acc[14];
        float v141_sw = tensorforge::swap<32>(v140_el);
        ir1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v141_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v137_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v135_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v132_el, v132_el))))))));
        float v144_el = v107_acc[3];
        float v146_el = v107_acc[7];
        float v147_sw = tensorforge::swap<32>(v146_el);
        float v149_el = v107_acc[11];
        float v152_el = v107_acc[15];
        float v153_sw = tensorforge::swap<32>(v152_el);
        ir1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v153_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v149_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v147_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v144_el, v144_el))))))));
        float v157_sw = tensorforge::swap<32>(v108_el);
        float v162_sw = tensorforge::swap<32>(v113_el);
        ir1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v116_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v162_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v110_el, (tensorforge::dppUpdate<228, 1, 15, false>(v157_sw, v157_sw))))))));
        float v169_sw = tensorforge::swap<32>(v120_el);
        float v174_sw = tensorforge::swap<32>(v125_el);
        ir1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v128_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v174_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v122_el, (tensorforge::dppUpdate<228, 1, 15, false>(v169_sw, v169_sw))))))));
        float v181_sw = tensorforge::swap<32>(v132_el);
        float v186_sw = tensorforge::swap<32>(v137_el);
        ir1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v140_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v186_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v134_el, (tensorforge::dppUpdate<228, 1, 15, false>(v181_sw, v181_sw))))))));
        float v193_sw = tensorforge::swap<32>(v144_el);
        float v198_sw = tensorforge::swap<32>(v149_el);
        ir1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v152_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v198_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v146_el, (tensorforge::dppUpdate<228, 1, 15, false>(v193_sw, v193_sw))))))));
        float v205_sw = tensorforge::swap<64>(v108_el);
        ir1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v117_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v113_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v111_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v205_sw, v205_sw))))))));
        float v217_sw = tensorforge::swap<64>(v120_el);
        ir1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v129_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v125_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v123_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v217_sw, v217_sw))))))));
        float v229_sw = tensorforge::swap<64>(v132_el);
        ir1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v141_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v137_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v135_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v229_sw, v229_sw))))))));
        float v241_sw = tensorforge::swap<64>(v144_el);
        ir1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v153_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v149_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v147_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v241_sw, v241_sw))))))));
        float v254_sw = tensorforge::swap<64>(v157_sw);
        ir1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v116_el, (tensorforge::dppUpdate<228, 4, 15, false>(v162_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v110_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v254_sw, v254_sw))))))));
        float v266_sw = tensorforge::swap<64>(v169_sw);
        ir1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v128_el, (tensorforge::dppUpdate<228, 4, 15, false>(v174_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v122_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v266_sw, v266_sw))))))));
        float v278_sw = tensorforge::swap<64>(v181_sw);
        ir1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v140_el, (tensorforge::dppUpdate<228, 4, 15, false>(v186_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v134_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v278_sw, v278_sw))))))));
        float v290_sw = tensorforge::swap<64>(v193_sw);
        ir1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v152_el, (tensorforge::dppUpdate<228, 4, 15, false>(v198_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v146_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v290_sw, v290_sw))))))));
        // r1 = ir1
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v307_i0 = 0; v307_i0 < 1; ++v307_i0) {
          #pragma unroll
          for (int32_t v308_i1 = 0; v308_i1 < 8; ++v308_i1) {
            #pragma unroll
            for (int32_t v309_i2 = 0; v309_i2 < 2; ++v309_i2) {
              float v313_data = ir1[(v308_i1 + (v309_i2 * 8))];
              if ((v309_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> const glb_m0_se0 = v309_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                glb_m0_se0[((v36_lead + (v307_i0 * 16)) + (v308_i1 * 16))] = v313_data;
              }
            }
          }
        }
        v364_elementIndex_s0 = wrap_elementIndex_s0;
        v366_elementIndex_s1 = wrap_elementIndex_s1;
      }
    }
  }
}

