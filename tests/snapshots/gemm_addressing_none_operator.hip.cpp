// === base name ===
kernel_78623ac8bbe857fa

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_78623ac8bbe857fa = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_78623ac8bbe857fa(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_78623ac8bbe857fa(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_78623ac8bbe857fa(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_78623ac8bbe857fa, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_78623ac8bbe857fa, block.x * block.y * block.z, 0));
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
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_78623ac8bbe857fa(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_78623ac8bbe857fa(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_78623ac8bbe857fa), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_78623ac8bbe857fa, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_78623ac8bbe857fa(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} none ordered
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":true,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 4;
      size_t v14_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v30_lead = threadIdx.x % 16;
      int32_t v62_vlead = (v30_lead + ((threadIdx.y % 4) * 16)) * 4;
      float r0[16]{};
      size_t v300_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdActive0 = v300_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v300_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v300_row : v14_batchIdGroup0First) * 256 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v308_i0 = 0; v308_i0 < 1; ++v308_i0) {
        int32_t v310_lead = v30_lead + (v308_i0 * 16);
        #pragma unroll
        for (int32_t v311_i1 = 0; v311_i1 < 16; ++v311_i1) {
          float v314_data = __builtin_nontemporal_load(&peel_glb_m2[(v310_lead + (v311_i1 * 16))]);
          r0[(v308_i0 + v311_i1)] = v314_data;
        }
      }
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v297_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v299_groupNext = (v297_groupNextAhead < numElements0) ? v297_groupNextAhead : v15_batchIdGroup0;
        size_t v304_row = v299_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdActive0 = v304_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v304_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v304_row : v299_groupNext) * 256 + 0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 256 + 0 + m0_extraOffset];
        float r1[16]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 16)] [(0, 16)]
        float v40_data = r0[0];
        float v41_data = r0[1];
        float v42_data = r0[2];
        float v43_data = r0[3];
        float v44_data = r0[4];
        float v45_data = r0[5];
        float v46_data = r0[6];
        float v47_data = r0[7];
        float v48_data = r0[8];
        float v49_data = r0[9];
        float v50_data = r0[10];
        float v51_data = r0[11];
        float v52_data = r0[12];
        float v53_data = r0[13];
        float v54_data = r0[14];
        float v55_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v316_i0 = 0; v316_i0 < 1; ++v316_i0) {
          int32_t v318_lead = v30_lead + (v316_i0 * 16);
          #pragma unroll
          for (int32_t v319_i1 = 0; v319_i1 < 16; ++v319_i1) {
            float v322_data = __builtin_nontemporal_load(&wrap_glb_m2[(v318_lead + (v319_i1 * 16))]);
            r0[(v316_i0 + v319_i1)] = v322_data;
          }
        }
        tensorforge::transpose16x16b32(v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data);
        tensorforge::VectorT<float, 16> v56_acc{};
        tensorforge::VectorT<float, 4> v64_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v62_vlead];
        float v65_el = v64_data[0];
        tensorforge::VectorT<float, 16> v66_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v65_el, v56_acc, 0, 0, 4);
        float v67_el = v64_data[1];
        tensorforge::VectorT<float, 16> v68_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v67_el, v66_acc, 0, 0, 4);
        float v69_el = v64_data[2];
        tensorforge::VectorT<float, 16> v70_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v69_el, v68_acc, 0, 0, 4);
        float v71_el = v64_data[3];
        tensorforge::VectorT<float, 16> v72_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v71_el, v70_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v65_el, v72_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v67_el, v74_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v69_el, v76_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v71_el, v78_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_el, v80_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v67_el, v82_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v69_el, v84_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v71_el, v86_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v65_el, v88_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v67_el, v90_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v69_el, v92_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v71_el, v94_acc, 0, 0, 7);
        float v97_el = v96_acc[0];
        float v99_el = v96_acc[4];
        float v100_sw = tensorforge::swap<32>(v99_el);
        float v102_el = v96_acc[8];
        float v105_el = v96_acc[12];
        float v106_sw = tensorforge::swap<32>(v105_el);
        r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v106_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v102_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v100_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v97_el, v97_el))))))));
        float v109_el = v96_acc[1];
        float v111_el = v96_acc[5];
        float v112_sw = tensorforge::swap<32>(v111_el);
        float v114_el = v96_acc[9];
        float v117_el = v96_acc[13];
        float v118_sw = tensorforge::swap<32>(v117_el);
        r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v118_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v114_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v112_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v109_el, v109_el))))))));
        float v121_el = v96_acc[2];
        float v123_el = v96_acc[6];
        float v124_sw = tensorforge::swap<32>(v123_el);
        float v126_el = v96_acc[10];
        float v129_el = v96_acc[14];
        float v130_sw = tensorforge::swap<32>(v129_el);
        r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v121_el, v121_el))))))));
        float v133_el = v96_acc[3];
        float v135_el = v96_acc[7];
        float v136_sw = tensorforge::swap<32>(v135_el);
        float v138_el = v96_acc[11];
        float v141_el = v96_acc[15];
        float v142_sw = tensorforge::swap<32>(v141_el);
        r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v142_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v133_el, v133_el))))))));
        float v146_sw = tensorforge::swap<32>(v97_el);
        float v151_sw = tensorforge::swap<32>(v102_el);
        r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v99_el, (tensorforge::dppUpdate<228, 1, 15, false>(v146_sw, v146_sw))))))));
        float v158_sw = tensorforge::swap<32>(v109_el);
        float v163_sw = tensorforge::swap<32>(v114_el);
        r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v111_el, (tensorforge::dppUpdate<228, 1, 15, false>(v158_sw, v158_sw))))))));
        float v170_sw = tensorforge::swap<32>(v121_el);
        float v175_sw = tensorforge::swap<32>(v126_el);
        r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v175_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v123_el, (tensorforge::dppUpdate<228, 1, 15, false>(v170_sw, v170_sw))))))));
        float v182_sw = tensorforge::swap<32>(v133_el);
        float v187_sw = tensorforge::swap<32>(v138_el);
        r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v187_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v135_el, (tensorforge::dppUpdate<228, 1, 15, false>(v182_sw, v182_sw))))))));
        float v194_sw = tensorforge::swap<64>(v97_el);
        r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v106_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v102_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v100_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v194_sw, v194_sw))))))));
        float v206_sw = tensorforge::swap<64>(v109_el);
        r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v118_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v114_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v112_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v206_sw, v206_sw))))))));
        float v218_sw = tensorforge::swap<64>(v121_el);
        r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v126_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v218_sw, v218_sw))))))));
        float v230_sw = tensorforge::swap<64>(v133_el);
        r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v142_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v138_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v230_sw, v230_sw))))))));
        float v243_sw = tensorforge::swap<64>(v146_sw);
        r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v105_el, (tensorforge::dppUpdate<228, 4, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v99_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v243_sw, v243_sw))))))));
        float v255_sw = tensorforge::swap<64>(v158_sw);
        r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v117_el, (tensorforge::dppUpdate<228, 4, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v255_sw, v255_sw))))))));
        float v267_sw = tensorforge::swap<64>(v170_sw);
        r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v129_el, (tensorforge::dppUpdate<228, 4, 15, false>(v175_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v267_sw, v267_sw))))))));
        float v279_sw = tensorforge::swap<64>(v182_sw);
        r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v141_el, (tensorforge::dppUpdate<228, 4, 15, false>(v187_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v279_sw, v279_sw))))))));
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v289_i0 = 0; v289_i0 < 1; ++v289_i0) {
          #pragma unroll
          for (int32_t v290_i1 = 0; v290_i1 < 16; ++v290_i1) {
            float v292_data = r1[(v289_i0 + v290_i1)];
            if (batchIdActive0) {
              glb_m0[((v30_lead + (v289_i0 * 16)) + (v290_i1 * 16))] = v292_data;
            }
          }
        }
      }
    }
  }
}

