// === base name ===
kernel_5af9f3c703cd3027

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5af9f3c703cd3027 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5af9f3c703cd3027(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5af9f3c703cd3027(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5af9f3c703cd3027(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5af9f3c703cd3027, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_5af9f3c703cd3027, block.x * block.y * block.z, 0));
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
void launcher_kernel_5af9f3c703cd3027(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5af9f3c703cd3027(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5af9f3c703cd3027), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  const float *const * m2Arg = (const float *const *)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_5af9f3c703cd3027, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5af9f3c703cd3027(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
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
      int32_t v34_lead = threadIdx.x % 16;
      int32_t v71_vlead = (v34_lead + ((threadIdx.y % 4) * 16)) * 4;
      float r0[16]{};
      size_t v320_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdStack0_0 = 2 * v320_row + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v320_row + 0]));
      const bool peel_batchIdStack0_1 = 2 * v320_row + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v320_row + 1]));
      const bool peel_batchIdActive0 = peel_batchIdStack0_0 || peel_batchIdStack0_1;
      size_t v324_batchId0 = peel_batchIdActive0 ? v320_row : v14_batchIdGroup0First;
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2_s0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v324_batchId0 + 0 : (peel_batchIdStack0_0 ? 2 * v324_batchId0 + 0 : peel_batchIdStack0_1 ? 2 * v324_batchId0 + 1 : 2 * v324_batchId0))]) * 3 + 2][0 + m2_extraOffset];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2_s1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v324_batchId0 + 1 : (peel_batchIdStack0_0 ? 2 * v324_batchId0 + 0 : peel_batchIdStack0_1 ? 2 * v324_batchId0 + 1 : 2 * v324_batchId0))]) * 3 + 2][0 + m2_extraOffset];
      // r0 = load{g>r}(glb_m2);
      #pragma unroll
      for (int32_t v334_i0 = 0; v334_i0 < 1; ++v334_i0) {
        int32_t v336_lead = v34_lead + (v334_i0 * 16);
        #pragma unroll
        for (int32_t v337_i1 = 0; v337_i1 < 8; ++v337_i1) {
          int32_t v339_a = v336_lead + (v337_i1 * 16);
          int32_t v340_a = v334_i0 + v337_i1;
          #pragma unroll
          for (int32_t v341_i2 = 0; v341_i2 < 2; ++v341_i2) {
            tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2_se0 = v341_i2 == 0 ? peel_glb_m2_s0 : peel_glb_m2_s1;
            float v343_data = __builtin_nontemporal_load(&glb_m2_se0[v339_a]);
            r0[(v340_a + (v341_i2 * 8))] = v343_data;
          }
        }
      }
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < ((numElements0 + 1) / 2); v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v317_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v319_groupNext = (v317_groupNextAhead < ((numElements0 + 1) / 2)) ? v317_groupNextAhead : v15_batchIdGroup0;
        size_t v327_row = v319_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdStack0_0 = 2 * v327_row + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v327_row + 0]));
        const bool wrap_batchIdStack0_1 = 2 * v327_row + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v327_row + 1]));
        const bool wrap_batchIdActive0 = wrap_batchIdStack0_0 || wrap_batchIdStack0_1;
        size_t v331_batchId0 = wrap_batchIdActive0 ? v327_row : v319_groupNext;
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2_s0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v331_batchId0 + 0 : (wrap_batchIdStack0_0 ? 2 * v331_batchId0 + 0 : wrap_batchIdStack0_1 ? 2 * v331_batchId0 + 1 : 2 * v331_batchId0))]) * 3 + 2][0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2_s1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v331_batchId0 + 1 : (wrap_batchIdStack0_0 ? 2 * v331_batchId0 + 0 : wrap_batchIdStack0_1 ? 2 * v331_batchId0 + 1 : 2 * v331_batchId0))]) * 3 + 2][0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdStack0_0 = 2 * v16_row + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v16_row + 0]));
        const bool batchIdStack0_1 = 2 * v16_row + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v16_row + 1]));
        const bool batchIdActive0 = batchIdStack0_0 || batchIdStack0_1;
        size_t v20_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v21_ahead1 = v20_batchId0 + (gridDim.x * blockDim.y);
        size_t v23_batchId1 = (v21_ahead1 < ((numElements0 + 1) / 2)) ? v21_ahead1 : v20_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> const glb_m0_s0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[static_cast<size_t>(elementIndices[(batchIdStack0_0 ? 2 * v20_batchId0 + 0 : (batchIdStack0_0 ? 2 * v20_batchId0 + 0 : batchIdStack0_1 ? 2 * v20_batchId0 + 1 : 2 * v20_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
        tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> const glb_m0_s1 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[static_cast<size_t>(elementIndices[(batchIdStack0_1 ? 2 * v20_batchId0 + 1 : (batchIdStack0_0 ? 2 * v20_batchId0 + 0 : batchIdStack0_1 ? 2 * v20_batchId0 + 1 : 2 * v20_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
        float r1[16]{};
        // ir1 = +(glb_m1 * r0)
        // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
        float ir1[16]{};
        float v49_data = r0[0];
        float v50_data = r0[1];
        float v51_data = r0[2];
        float v52_data = r0[3];
        float v53_data = r0[4];
        float v54_data = r0[5];
        float v55_data = r0[6];
        float v56_data = r0[7];
        float v57_data = r0[8];
        float v58_data = r0[9];
        float v59_data = r0[10];
        float v60_data = r0[11];
        float v61_data = r0[12];
        float v62_data = r0[13];
        float v63_data = r0[14];
        float v64_data = r0[15];
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v346_i0 = 0; v346_i0 < 1; ++v346_i0) {
          int32_t v348_lead = v34_lead + (v346_i0 * 16);
          #pragma unroll
          for (int32_t v349_i1 = 0; v349_i1 < 8; ++v349_i1) {
            int32_t v351_a = v348_lead + (v349_i1 * 16);
            int32_t v352_a = v346_i0 + v349_i1;
            #pragma unroll
            for (int32_t v353_i2 = 0; v353_i2 < 2; ++v353_i2) {
              tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2_se0 = v353_i2 == 0 ? wrap_glb_m2_s0 : wrap_glb_m2_s1;
              float v355_data = __builtin_nontemporal_load(&glb_m2_se0[v351_a]);
              r0[(v352_a + (v353_i2 * 8))] = v355_data;
            }
          }
        }
        tensorforge::transpose16x16b32(v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data);
        tensorforge::VectorT<float, 16> v65_acc{};
        tensorforge::VectorT<float, 4> v73_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v71_vlead];
        float v74_el = v73_data[0];
        tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v74_el, v65_acc, 0, 0, 4);
        float v76_el = v73_data[1];
        tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v76_el, v75_acc, 0, 0, 4);
        float v78_el = v73_data[2];
        tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v78_el, v77_acc, 0, 0, 4);
        float v80_el = v73_data[3];
        tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v80_el, v79_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v74_el, v81_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v76_el, v83_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v78_el, v85_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v80_el, v87_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_el, v89_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v76_el, v91_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v78_el, v93_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v80_el, v95_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v74_el, v97_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v101_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v76_el, v99_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v103_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v78_el, v101_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v105_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v80_el, v103_acc, 0, 0, 7);
        float v106_el = v105_acc[0];
        float v108_el = v105_acc[4];
        float v109_sw = tensorforge::swap<32>(v108_el);
        float v111_el = v105_acc[8];
        float v114_el = v105_acc[12];
        float v115_sw = tensorforge::swap<32>(v114_el);
        ir1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v115_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v109_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v106_el, v106_el))))))));
        float v118_el = v105_acc[1];
        float v120_el = v105_acc[5];
        float v121_sw = tensorforge::swap<32>(v120_el);
        float v123_el = v105_acc[9];
        float v126_el = v105_acc[13];
        float v127_sw = tensorforge::swap<32>(v126_el);
        ir1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v121_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v118_el, v118_el))))))));
        float v130_el = v105_acc[2];
        float v132_el = v105_acc[6];
        float v133_sw = tensorforge::swap<32>(v132_el);
        float v135_el = v105_acc[10];
        float v138_el = v105_acc[14];
        float v139_sw = tensorforge::swap<32>(v138_el);
        ir1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v130_el, v130_el))))))));
        float v142_el = v105_acc[3];
        float v144_el = v105_acc[7];
        float v145_sw = tensorforge::swap<32>(v144_el);
        float v147_el = v105_acc[11];
        float v150_el = v105_acc[15];
        float v151_sw = tensorforge::swap<32>(v150_el);
        ir1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v142_el, v142_el))))))));
        float v155_sw = tensorforge::swap<32>(v106_el);
        float v160_sw = tensorforge::swap<32>(v111_el);
        ir1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v114_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v160_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v108_el, (tensorforge::dppUpdate<228, 1, 15, false>(v155_sw, v155_sw))))))));
        float v167_sw = tensorforge::swap<32>(v118_el);
        float v172_sw = tensorforge::swap<32>(v123_el);
        ir1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v172_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v120_el, (tensorforge::dppUpdate<228, 1, 15, false>(v167_sw, v167_sw))))))));
        float v179_sw = tensorforge::swap<32>(v130_el);
        float v184_sw = tensorforge::swap<32>(v135_el);
        ir1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v184_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v132_el, (tensorforge::dppUpdate<228, 1, 15, false>(v179_sw, v179_sw))))))));
        float v191_sw = tensorforge::swap<32>(v142_el);
        float v196_sw = tensorforge::swap<32>(v147_el);
        ir1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v196_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v144_el, (tensorforge::dppUpdate<228, 1, 15, false>(v191_sw, v191_sw))))))));
        float v203_sw = tensorforge::swap<64>(v106_el);
        ir1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v115_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v111_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v109_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v203_sw, v203_sw))))))));
        float v215_sw = tensorforge::swap<64>(v118_el);
        ir1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v123_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v121_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v215_sw, v215_sw))))))));
        float v227_sw = tensorforge::swap<64>(v130_el);
        ir1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v135_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v227_sw, v227_sw))))))));
        float v239_sw = tensorforge::swap<64>(v142_el);
        ir1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v147_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v239_sw, v239_sw))))))));
        float v252_sw = tensorforge::swap<64>(v155_sw);
        ir1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v114_el, (tensorforge::dppUpdate<228, 4, 15, false>(v160_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v108_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v252_sw, v252_sw))))))));
        float v264_sw = tensorforge::swap<64>(v167_sw);
        ir1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v126_el, (tensorforge::dppUpdate<228, 4, 15, false>(v172_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v120_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v264_sw, v264_sw))))))));
        float v276_sw = tensorforge::swap<64>(v179_sw);
        ir1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v138_el, (tensorforge::dppUpdate<228, 4, 15, false>(v184_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v276_sw, v276_sw))))))));
        float v288_sw = tensorforge::swap<64>(v191_sw);
        ir1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v150_el, (tensorforge::dppUpdate<228, 4, 15, false>(v196_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v288_sw, v288_sw))))))));
        // r1 = ir1
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v305_i0 = 0; v305_i0 < 1; ++v305_i0) {
          #pragma unroll
          for (int32_t v306_i1 = 0; v306_i1 < 8; ++v306_i1) {
            #pragma unroll
            for (int32_t v307_i2 = 0; v307_i2 < 2; ++v307_i2) {
              float v311_data = ir1[(v306_i1 + (v307_i2 * 8))];
              if ((v307_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> const glb_m0_se0 = v307_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                glb_m0_se0[((v34_lead + (v305_i0 * 16)) + (v306_i1 * 16))] = v311_data;
              }
            }
          }
        }
      }
    }
  }
}

