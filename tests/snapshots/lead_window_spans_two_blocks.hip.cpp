// === base name ===
kernel_e5edeb7d46391680

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e5edeb7d46391680 = {{32, 8, 1}, 32, 64, 1, 8, 256, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e5edeb7d46391680(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e5edeb7d46391680(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e5edeb7d46391680(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e5edeb7d46391680, block.x * block.y * block.z, 64 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (64 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e5edeb7d46391680, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (64 * sizeof(float)));
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
  config.sharedMemBytes = 64 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e5edeb7d46391680(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e5edeb7d46391680(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e5edeb7d46391680), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_e5edeb7d46391680, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1Arg, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e5edeb7d46391680(const float ** m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, float ** m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (64 active) x 8 per block = block 32x8x1, 256 B shared, occupancy grid
    // operands:
    //   m0 64×13(64×13) {0..64}×{0..13} pointer_based
    //   m1 6(6) {0..6} none
    //   m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
    // operations:
    //   t0[i,j,l] = m0[i,j] × m1[l]
    //   m2[i,j,l]@{20..35}×{12..13}×{0..6} += t0[i,j,l]@{20..35}×{12..13}×{0..6}
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":64}],"shared_bytes":256,"shared_elements":64,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"none","alias":"v","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"pointer_based","alias":"D","bbox":[[0,0,0],[64,13,6]],"name":"m2","ordered":false,"parts":1,"shape":[64,13,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0,0],[64,13,6]],"is_tmp":true,"name":"t0","offset":[0,0,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"none","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0,1],[0]],"target":[[0,1],[2]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0,0],[15,1,6]],"is_tmp":false,"name":"m2","offset":[20,12,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0,0],[15,1,6]],"is_tmp":true,"name":"t0","offset":[20,12,0],"shape":[64,13,6]}],"permute":[[0,1,2]],"target":[[0,1,2]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 64];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 6) {
        float v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0]));
      __syncthreads();
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v7_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v7_batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 2; ++v21_i0) {
            int32_t v24_lead = v20_lead + (v21_i0 * 32);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 13; ++v22_i1) {
              float v27_data = __builtin_nontemporal_load(&glb_m0[(v24_lead + (v22_i1 * 64))]);
              r0[(v21_i0 + (v22_i1 * 2))] = v27_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v31_data = glb_m1[0];
          float v32_data = glb_m1[0];
          float v33_data = glb_m1[0];
          float v34_data = glb_m1[0];
          float v35_data = glb_m1[0];
          float v36_data = glb_m1[0];
          float v37_data = glb_m1[0];
          float v38_data = glb_m1[0];
          float v39_data = glb_m1[0];
          float v40_data = glb_m1[0];
          float v41_data = glb_m1[0];
          float v42_data = glb_m1[0];
          float v43_data = glb_m1[0];
          float v44_data = glb_m1[1];
          float v45_data = glb_m1[1];
          float v46_data = glb_m1[1];
          tensorforge::transpose16x16b32(v31_data, v32_data, v33_data, v34_data, v35_data, v36_data, v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data);
          tensorforge::VectorT<float, 16> v47_acc{};
          float v48_data = r0[0];
          tensorforge::VectorT<float, 16> v50_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v31_data, v48_data, v47_acc, 1, 0, 0);
          float v51_el = v50_acc[0];
          float v53_el = v50_acc[4];
          float v54_sw = tensorforge::swap<32>(v53_el);
          float v56_el = v50_acc[8];
          float v59_el = v50_acc[12];
          float v60_sw = tensorforge::swap<32>(v59_el);
          r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v60_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v56_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v54_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v51_el, v51_el))))))));
          float v63_el = v50_acc[1];
          float v65_el = v50_acc[5];
          float v66_sw = tensorforge::swap<32>(v65_el);
          float v68_el = v50_acc[9];
          float v71_el = v50_acc[13];
          float v72_sw = tensorforge::swap<32>(v71_el);
          r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v72_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v68_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v66_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v63_el, v63_el))))))));
          float v75_el = v50_acc[2];
          float v77_el = v50_acc[6];
          float v78_sw = tensorforge::swap<32>(v77_el);
          float v80_el = v50_acc[10];
          float v83_el = v50_acc[14];
          float v84_sw = tensorforge::swap<32>(v83_el);
          r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v84_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v80_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v78_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v75_el, v75_el))))))));
          float v87_el = v50_acc[3];
          float v89_el = v50_acc[7];
          float v90_sw = tensorforge::swap<32>(v89_el);
          float v92_el = v50_acc[11];
          float v95_el = v50_acc[15];
          float v96_sw = tensorforge::swap<32>(v95_el);
          r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v96_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v92_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v90_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v87_el, v87_el))))))));
          float v100_sw = tensorforge::swap<32>(v51_el);
          float v105_sw = tensorforge::swap<32>(v56_el);
          r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v59_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v105_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v53_el, (tensorforge::dppUpdate<228, 1, 15, false>(v100_sw, v100_sw))))))));
          float v112_sw = tensorforge::swap<32>(v63_el);
          float v117_sw = tensorforge::swap<32>(v68_el);
          r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v71_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v117_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v65_el, (tensorforge::dppUpdate<228, 1, 15, false>(v112_sw, v112_sw))))))));
          float v124_sw = tensorforge::swap<32>(v75_el);
          float v129_sw = tensorforge::swap<32>(v80_el);
          r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v83_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v129_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v77_el, (tensorforge::dppUpdate<228, 1, 15, false>(v124_sw, v124_sw))))))));
          float v136_sw = tensorforge::swap<32>(v87_el);
          float v141_sw = tensorforge::swap<32>(v92_el);
          r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v95_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v89_el, (tensorforge::dppUpdate<228, 1, 15, false>(v136_sw, v136_sw))))))));
          float v148_sw = tensorforge::swap<64>(v51_el);
          r1[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v60_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v56_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v54_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v148_sw, v148_sw))))))));
          float v160_sw = tensorforge::swap<64>(v63_el);
          r1[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v72_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v68_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v66_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v160_sw, v160_sw))))))));
          float v172_sw = tensorforge::swap<64>(v75_el);
          r1[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v84_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v80_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v78_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v172_sw, v172_sw))))))));
          float v184_sw = tensorforge::swap<64>(v87_el);
          r1[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v96_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v92_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v90_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v184_sw, v184_sw))))))));
          float v197_sw = tensorforge::swap<64>(v100_sw);
          r1[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v59_el, (tensorforge::dppUpdate<228, 4, 15, false>(v105_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v53_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v197_sw, v197_sw))))))));
          float v209_sw = tensorforge::swap<64>(v112_sw);
          r1[26] = (tensorforge::dppUpdate<228, 8, 15, false>(v71_el, (tensorforge::dppUpdate<228, 4, 15, false>(v117_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v65_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v209_sw, v209_sw))))))));
          float v221_sw = tensorforge::swap<64>(v124_sw);
          r1[28] = (tensorforge::dppUpdate<228, 8, 15, false>(v83_el, (tensorforge::dppUpdate<228, 4, 15, false>(v129_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v77_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v221_sw, v221_sw))))))));
          float v233_sw = tensorforge::swap<64>(v136_sw);
          r1[30] = (tensorforge::dppUpdate<228, 8, 15, false>(v95_el, (tensorforge::dppUpdate<228, 4, 15, false>(v141_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v89_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
          tensorforge::VectorT<float, 16> v243_acc{};
          float v244_data = r0[1];
          tensorforge::VectorT<float, 16> v246_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v31_data, v244_data, v243_acc, 1, 0, 0);
          float v247_el = v246_acc[0];
          float v249_el = v246_acc[4];
          float v250_sw = tensorforge::swap<32>(v249_el);
          float v252_el = v246_acc[8];
          float v255_el = v246_acc[12];
          float v256_sw = tensorforge::swap<32>(v255_el);
          r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v256_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v252_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v250_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v247_el, v247_el))))))));
          float v259_el = v246_acc[1];
          float v261_el = v246_acc[5];
          float v262_sw = tensorforge::swap<32>(v261_el);
          float v264_el = v246_acc[9];
          float v267_el = v246_acc[13];
          float v268_sw = tensorforge::swap<32>(v267_el);
          r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v268_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v264_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v262_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v259_el, v259_el))))))));
          float v271_el = v246_acc[2];
          float v273_el = v246_acc[6];
          float v274_sw = tensorforge::swap<32>(v273_el);
          float v276_el = v246_acc[10];
          float v279_el = v246_acc[14];
          float v280_sw = tensorforge::swap<32>(v279_el);
          r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v280_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v276_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v274_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v271_el, v271_el))))))));
          float v283_el = v246_acc[3];
          float v285_el = v246_acc[7];
          float v286_sw = tensorforge::swap<32>(v285_el);
          float v288_el = v246_acc[11];
          float v291_el = v246_acc[15];
          float v292_sw = tensorforge::swap<32>(v291_el);
          r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v292_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v288_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v286_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v283_el, v283_el))))))));
          float v296_sw = tensorforge::swap<32>(v247_el);
          float v301_sw = tensorforge::swap<32>(v252_el);
          r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v255_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v301_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v249_el, (tensorforge::dppUpdate<228, 1, 15, false>(v296_sw, v296_sw))))))));
          float v308_sw = tensorforge::swap<32>(v259_el);
          float v313_sw = tensorforge::swap<32>(v264_el);
          r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v267_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v313_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v261_el, (tensorforge::dppUpdate<228, 1, 15, false>(v308_sw, v308_sw))))))));
          float v320_sw = tensorforge::swap<32>(v271_el);
          float v325_sw = tensorforge::swap<32>(v276_el);
          r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v279_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v325_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v273_el, (tensorforge::dppUpdate<228, 1, 15, false>(v320_sw, v320_sw))))))));
          float v332_sw = tensorforge::swap<32>(v283_el);
          float v337_sw = tensorforge::swap<32>(v288_el);
          r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v291_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v337_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v285_el, (tensorforge::dppUpdate<228, 1, 15, false>(v332_sw, v332_sw))))))));
          float v344_sw = tensorforge::swap<64>(v247_el);
          r1[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v256_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v252_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v250_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v344_sw, v344_sw))))))));
          float v356_sw = tensorforge::swap<64>(v259_el);
          r1[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v268_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v264_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v262_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v356_sw, v356_sw))))))));
          float v368_sw = tensorforge::swap<64>(v271_el);
          r1[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v280_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v276_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v274_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v368_sw, v368_sw))))))));
          float v380_sw = tensorforge::swap<64>(v283_el);
          r1[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v292_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v288_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v286_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v380_sw, v380_sw))))))));
          float v393_sw = tensorforge::swap<64>(v296_sw);
          r1[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v255_el, (tensorforge::dppUpdate<228, 4, 15, false>(v301_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v249_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v393_sw, v393_sw))))))));
          float v405_sw = tensorforge::swap<64>(v308_sw);
          r1[27] = (tensorforge::dppUpdate<228, 8, 15, false>(v267_el, (tensorforge::dppUpdate<228, 4, 15, false>(v313_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v261_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v405_sw, v405_sw))))))));
          float v417_sw = tensorforge::swap<64>(v320_sw);
          r1[29] = (tensorforge::dppUpdate<228, 8, 15, false>(v279_el, (tensorforge::dppUpdate<228, 4, 15, false>(v325_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v273_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v417_sw, v417_sw))))))));
          float v429_sw = tensorforge::swap<64>(v332_sw);
          r1[31] = (tensorforge::dppUpdate<228, 8, 15, false>(v291_el, (tensorforge::dppUpdate<228, 4, 15, false>(v337_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v285_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v429_sw, v429_sw))))))));
          float v439_data = glb_m1[1];
          float v440_data = glb_m1[1];
          float v441_data = glb_m1[1];
          float v442_data = glb_m1[1];
          float v443_data = glb_m1[1];
          float v444_data = glb_m1[1];
          float v445_data = glb_m1[1];
          float v446_data = glb_m1[1];
          float v447_data = glb_m1[1];
          float v448_data = glb_m1[1];
          float v449_data = glb_m1[2];
          float v450_data = glb_m1[2];
          float v451_data = glb_m1[2];
          float v452_data = glb_m1[2];
          float v453_data = glb_m1[2];
          float v454_data = glb_m1[2];
          tensorforge::transpose16x16b32(v439_data, v440_data, v441_data, v442_data, v443_data, v444_data, v445_data, v446_data, v447_data, v448_data, v449_data, v450_data, v451_data, v452_data, v453_data, v454_data);
          tensorforge::VectorT<float, 16> v455_acc{};
          tensorforge::VectorT<float, 16> v458_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v439_data, v48_data, v455_acc, 1, 0, 0);
          float v459_el = v458_acc[0];
          float v461_el = v458_acc[4];
          float v462_sw = tensorforge::swap<32>(v461_el);
          float v464_el = v458_acc[8];
          float v467_el = v458_acc[12];
          float v468_sw = tensorforge::swap<32>(v467_el);
          r1[32] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v468_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v464_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v462_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v459_el, v459_el))))))));
          float v471_el = v458_acc[1];
          float v473_el = v458_acc[5];
          float v474_sw = tensorforge::swap<32>(v473_el);
          float v476_el = v458_acc[9];
          float v479_el = v458_acc[13];
          float v480_sw = tensorforge::swap<32>(v479_el);
          r1[34] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v480_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v476_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v474_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v471_el, v471_el))))))));
          float v483_el = v458_acc[2];
          float v485_el = v458_acc[6];
          float v486_sw = tensorforge::swap<32>(v485_el);
          float v488_el = v458_acc[10];
          float v491_el = v458_acc[14];
          float v492_sw = tensorforge::swap<32>(v491_el);
          r1[36] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v492_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v488_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v486_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v483_el, v483_el))))))));
          float v495_el = v458_acc[3];
          float v497_el = v458_acc[7];
          float v498_sw = tensorforge::swap<32>(v497_el);
          float v500_el = v458_acc[11];
          float v503_el = v458_acc[15];
          float v504_sw = tensorforge::swap<32>(v503_el);
          r1[38] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v504_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v500_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v498_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v495_el, v495_el))))))));
          float v508_sw = tensorforge::swap<32>(v459_el);
          float v513_sw = tensorforge::swap<32>(v464_el);
          r1[40] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v467_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v513_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v461_el, (tensorforge::dppUpdate<228, 1, 15, false>(v508_sw, v508_sw))))))));
          float v520_sw = tensorforge::swap<32>(v471_el);
          float v525_sw = tensorforge::swap<32>(v476_el);
          r1[42] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v479_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v525_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v473_el, (tensorforge::dppUpdate<228, 1, 15, false>(v520_sw, v520_sw))))))));
          float v532_sw = tensorforge::swap<32>(v483_el);
          float v537_sw = tensorforge::swap<32>(v488_el);
          r1[44] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v491_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v537_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v485_el, (tensorforge::dppUpdate<228, 1, 15, false>(v532_sw, v532_sw))))))));
          float v544_sw = tensorforge::swap<32>(v495_el);
          float v549_sw = tensorforge::swap<32>(v500_el);
          r1[46] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v503_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v549_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v497_el, (tensorforge::dppUpdate<228, 1, 15, false>(v544_sw, v544_sw))))))));
          float v556_sw = tensorforge::swap<64>(v459_el);
          r1[48] = (tensorforge::dppUpdate<228, 8, 15, false>(v468_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v464_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v462_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v556_sw, v556_sw))))))));
          float v568_sw = tensorforge::swap<64>(v471_el);
          r1[50] = (tensorforge::dppUpdate<228, 8, 15, false>(v480_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v476_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v474_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v568_sw, v568_sw))))))));
          float v580_sw = tensorforge::swap<64>(v483_el);
          r1[52] = (tensorforge::dppUpdate<228, 8, 15, false>(v492_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v488_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v486_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v580_sw, v580_sw))))))));
          float v592_sw = tensorforge::swap<64>(v495_el);
          r1[54] = (tensorforge::dppUpdate<228, 8, 15, false>(v504_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v500_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v498_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v592_sw, v592_sw))))))));
          float v605_sw = tensorforge::swap<64>(v508_sw);
          r1[56] = (tensorforge::dppUpdate<228, 8, 15, false>(v467_el, (tensorforge::dppUpdate<228, 4, 15, false>(v513_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v461_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v605_sw, v605_sw))))))));
          float v617_sw = tensorforge::swap<64>(v520_sw);
          r1[58] = (tensorforge::dppUpdate<228, 8, 15, false>(v479_el, (tensorforge::dppUpdate<228, 4, 15, false>(v525_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v473_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v617_sw, v617_sw))))))));
          float v629_sw = tensorforge::swap<64>(v532_sw);
          r1[60] = (tensorforge::dppUpdate<228, 8, 15, false>(v491_el, (tensorforge::dppUpdate<228, 4, 15, false>(v537_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v485_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v629_sw, v629_sw))))))));
          float v641_sw = tensorforge::swap<64>(v544_sw);
          r1[62] = (tensorforge::dppUpdate<228, 8, 15, false>(v503_el, (tensorforge::dppUpdate<228, 4, 15, false>(v549_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v497_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v641_sw, v641_sw))))))));
          tensorforge::VectorT<float, 16> v651_acc{};
          tensorforge::VectorT<float, 16> v654_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v439_data, v244_data, v651_acc, 1, 0, 0);
          float v655_el = v654_acc[0];
          float v657_el = v654_acc[4];
          float v658_sw = tensorforge::swap<32>(v657_el);
          float v660_el = v654_acc[8];
          float v663_el = v654_acc[12];
          float v664_sw = tensorforge::swap<32>(v663_el);
          r1[33] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v664_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v660_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v658_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v655_el, v655_el))))))));
          float v667_el = v654_acc[1];
          float v669_el = v654_acc[5];
          float v670_sw = tensorforge::swap<32>(v669_el);
          float v672_el = v654_acc[9];
          float v675_el = v654_acc[13];
          float v676_sw = tensorforge::swap<32>(v675_el);
          r1[35] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v676_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v672_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v670_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v667_el, v667_el))))))));
          float v679_el = v654_acc[2];
          float v681_el = v654_acc[6];
          float v682_sw = tensorforge::swap<32>(v681_el);
          float v684_el = v654_acc[10];
          float v687_el = v654_acc[14];
          float v688_sw = tensorforge::swap<32>(v687_el);
          r1[37] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v688_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v684_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v682_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v679_el, v679_el))))))));
          float v691_el = v654_acc[3];
          float v693_el = v654_acc[7];
          float v694_sw = tensorforge::swap<32>(v693_el);
          float v696_el = v654_acc[11];
          float v699_el = v654_acc[15];
          float v700_sw = tensorforge::swap<32>(v699_el);
          r1[39] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v700_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v696_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v694_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v691_el, v691_el))))))));
          float v704_sw = tensorforge::swap<32>(v655_el);
          float v709_sw = tensorforge::swap<32>(v660_el);
          r1[41] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v663_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v709_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v657_el, (tensorforge::dppUpdate<228, 1, 15, false>(v704_sw, v704_sw))))))));
          float v716_sw = tensorforge::swap<32>(v667_el);
          float v721_sw = tensorforge::swap<32>(v672_el);
          r1[43] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v675_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v721_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v669_el, (tensorforge::dppUpdate<228, 1, 15, false>(v716_sw, v716_sw))))))));
          float v728_sw = tensorforge::swap<32>(v679_el);
          float v733_sw = tensorforge::swap<32>(v684_el);
          r1[45] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v687_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v733_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v681_el, (tensorforge::dppUpdate<228, 1, 15, false>(v728_sw, v728_sw))))))));
          float v740_sw = tensorforge::swap<32>(v691_el);
          float v745_sw = tensorforge::swap<32>(v696_el);
          r1[47] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v699_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v745_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v693_el, (tensorforge::dppUpdate<228, 1, 15, false>(v740_sw, v740_sw))))))));
          float v752_sw = tensorforge::swap<64>(v655_el);
          r1[49] = (tensorforge::dppUpdate<228, 8, 15, false>(v664_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v660_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v658_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v752_sw, v752_sw))))))));
          float v764_sw = tensorforge::swap<64>(v667_el);
          r1[51] = (tensorforge::dppUpdate<228, 8, 15, false>(v676_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v672_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v670_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v764_sw, v764_sw))))))));
          float v776_sw = tensorforge::swap<64>(v679_el);
          r1[53] = (tensorforge::dppUpdate<228, 8, 15, false>(v688_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v684_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v682_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v776_sw, v776_sw))))))));
          float v788_sw = tensorforge::swap<64>(v691_el);
          r1[55] = (tensorforge::dppUpdate<228, 8, 15, false>(v700_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v696_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v694_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v788_sw, v788_sw))))))));
          float v801_sw = tensorforge::swap<64>(v704_sw);
          r1[57] = (tensorforge::dppUpdate<228, 8, 15, false>(v663_el, (tensorforge::dppUpdate<228, 4, 15, false>(v709_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v657_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v801_sw, v801_sw))))))));
          float v813_sw = tensorforge::swap<64>(v716_sw);
          r1[59] = (tensorforge::dppUpdate<228, 8, 15, false>(v675_el, (tensorforge::dppUpdate<228, 4, 15, false>(v721_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v669_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v813_sw, v813_sw))))))));
          float v825_sw = tensorforge::swap<64>(v728_sw);
          r1[61] = (tensorforge::dppUpdate<228, 8, 15, false>(v687_el, (tensorforge::dppUpdate<228, 4, 15, false>(v733_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v681_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v825_sw, v825_sw))))))));
          float v837_sw = tensorforge::swap<64>(v740_sw);
          r1[63] = (tensorforge::dppUpdate<228, 8, 15, false>(v699_el, (tensorforge::dppUpdate<228, 4, 15, false>(v745_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v693_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v837_sw, v837_sw))))))));
          float v847_data = glb_m1[2];
          float v848_data = glb_m1[2];
          float v849_data = glb_m1[2];
          float v850_data = glb_m1[2];
          float v851_data = glb_m1[2];
          float v852_data = glb_m1[2];
          float v853_data = glb_m1[2];
          float v854_data = glb_m1[3];
          float v855_data = glb_m1[3];
          float v856_data = glb_m1[3];
          float v857_data = glb_m1[3];
          float v858_data = glb_m1[3];
          float v859_data = glb_m1[3];
          float v860_data = glb_m1[3];
          float v861_data = glb_m1[3];
          float v862_data = glb_m1[3];
          tensorforge::transpose16x16b32(v847_data, v848_data, v849_data, v850_data, v851_data, v852_data, v853_data, v854_data, v855_data, v856_data, v857_data, v858_data, v859_data, v860_data, v861_data, v862_data);
          tensorforge::VectorT<float, 16> v863_acc{};
          tensorforge::VectorT<float, 16> v866_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v847_data, v48_data, v863_acc, 1, 0, 0);
          float v867_el = v866_acc[0];
          float v869_el = v866_acc[4];
          float v870_sw = tensorforge::swap<32>(v869_el);
          float v872_el = v866_acc[8];
          float v875_el = v866_acc[12];
          float v876_sw = tensorforge::swap<32>(v875_el);
          r1[64] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v876_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v872_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v870_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v867_el, v867_el))))))));
          float v879_el = v866_acc[1];
          float v881_el = v866_acc[5];
          float v882_sw = tensorforge::swap<32>(v881_el);
          float v884_el = v866_acc[9];
          float v887_el = v866_acc[13];
          float v888_sw = tensorforge::swap<32>(v887_el);
          r1[66] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v888_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v884_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v882_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v879_el, v879_el))))))));
          float v891_el = v866_acc[2];
          float v893_el = v866_acc[6];
          float v894_sw = tensorforge::swap<32>(v893_el);
          float v896_el = v866_acc[10];
          float v899_el = v866_acc[14];
          float v900_sw = tensorforge::swap<32>(v899_el);
          r1[68] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v900_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v896_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v894_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v891_el, v891_el))))))));
          float v903_el = v866_acc[3];
          float v905_el = v866_acc[7];
          float v906_sw = tensorforge::swap<32>(v905_el);
          float v908_el = v866_acc[11];
          float v911_el = v866_acc[15];
          float v912_sw = tensorforge::swap<32>(v911_el);
          r1[70] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v912_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v908_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v906_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v903_el, v903_el))))))));
          float v916_sw = tensorforge::swap<32>(v867_el);
          float v921_sw = tensorforge::swap<32>(v872_el);
          r1[72] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v875_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v921_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v869_el, (tensorforge::dppUpdate<228, 1, 15, false>(v916_sw, v916_sw))))))));
          float v928_sw = tensorforge::swap<32>(v879_el);
          float v933_sw = tensorforge::swap<32>(v884_el);
          r1[74] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v887_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v933_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v881_el, (tensorforge::dppUpdate<228, 1, 15, false>(v928_sw, v928_sw))))))));
          float v940_sw = tensorforge::swap<32>(v891_el);
          float v945_sw = tensorforge::swap<32>(v896_el);
          r1[76] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v899_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v945_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v893_el, (tensorforge::dppUpdate<228, 1, 15, false>(v940_sw, v940_sw))))))));
          float v952_sw = tensorforge::swap<32>(v903_el);
          float v957_sw = tensorforge::swap<32>(v908_el);
          r1[78] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v911_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v957_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v905_el, (tensorforge::dppUpdate<228, 1, 15, false>(v952_sw, v952_sw))))))));
          float v964_sw = tensorforge::swap<64>(v867_el);
          r1[80] = (tensorforge::dppUpdate<228, 8, 15, false>(v876_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v872_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v870_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v964_sw, v964_sw))))))));
          float v976_sw = tensorforge::swap<64>(v879_el);
          r1[82] = (tensorforge::dppUpdate<228, 8, 15, false>(v888_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v884_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v882_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v976_sw, v976_sw))))))));
          float v988_sw = tensorforge::swap<64>(v891_el);
          r1[84] = (tensorforge::dppUpdate<228, 8, 15, false>(v900_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v896_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v894_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v988_sw, v988_sw))))))));
          float v1000_sw = tensorforge::swap<64>(v903_el);
          r1[86] = (tensorforge::dppUpdate<228, 8, 15, false>(v912_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v908_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v906_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1000_sw, v1000_sw))))))));
          float v1013_sw = tensorforge::swap<64>(v916_sw);
          r1[88] = (tensorforge::dppUpdate<228, 8, 15, false>(v875_el, (tensorforge::dppUpdate<228, 4, 15, false>(v921_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v869_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1013_sw, v1013_sw))))))));
          float v1025_sw = tensorforge::swap<64>(v928_sw);
          r1[90] = (tensorforge::dppUpdate<228, 8, 15, false>(v887_el, (tensorforge::dppUpdate<228, 4, 15, false>(v933_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v881_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1025_sw, v1025_sw))))))));
          float v1037_sw = tensorforge::swap<64>(v940_sw);
          r1[92] = (tensorforge::dppUpdate<228, 8, 15, false>(v899_el, (tensorforge::dppUpdate<228, 4, 15, false>(v945_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v893_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1037_sw, v1037_sw))))))));
          float v1049_sw = tensorforge::swap<64>(v952_sw);
          r1[94] = (tensorforge::dppUpdate<228, 8, 15, false>(v911_el, (tensorforge::dppUpdate<228, 4, 15, false>(v957_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v905_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1049_sw, v1049_sw))))))));
          tensorforge::VectorT<float, 16> v1059_acc{};
          tensorforge::VectorT<float, 16> v1062_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v847_data, v244_data, v1059_acc, 1, 0, 0);
          float v1063_el = v1062_acc[0];
          float v1065_el = v1062_acc[4];
          float v1066_sw = tensorforge::swap<32>(v1065_el);
          float v1068_el = v1062_acc[8];
          float v1071_el = v1062_acc[12];
          float v1072_sw = tensorforge::swap<32>(v1071_el);
          r1[65] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1072_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1068_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1066_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1063_el, v1063_el))))))));
          float v1075_el = v1062_acc[1];
          float v1077_el = v1062_acc[5];
          float v1078_sw = tensorforge::swap<32>(v1077_el);
          float v1080_el = v1062_acc[9];
          float v1083_el = v1062_acc[13];
          float v1084_sw = tensorforge::swap<32>(v1083_el);
          r1[67] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1084_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1080_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1078_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1075_el, v1075_el))))))));
          float v1087_el = v1062_acc[2];
          float v1089_el = v1062_acc[6];
          float v1090_sw = tensorforge::swap<32>(v1089_el);
          float v1092_el = v1062_acc[10];
          float v1095_el = v1062_acc[14];
          float v1096_sw = tensorforge::swap<32>(v1095_el);
          r1[69] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1096_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1092_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1090_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1087_el, v1087_el))))))));
          float v1099_el = v1062_acc[3];
          float v1101_el = v1062_acc[7];
          float v1102_sw = tensorforge::swap<32>(v1101_el);
          float v1104_el = v1062_acc[11];
          float v1107_el = v1062_acc[15];
          float v1108_sw = tensorforge::swap<32>(v1107_el);
          r1[71] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1108_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1104_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1102_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1099_el, v1099_el))))))));
          float v1112_sw = tensorforge::swap<32>(v1063_el);
          float v1117_sw = tensorforge::swap<32>(v1068_el);
          r1[73] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1071_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1117_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1065_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1112_sw, v1112_sw))))))));
          float v1124_sw = tensorforge::swap<32>(v1075_el);
          float v1129_sw = tensorforge::swap<32>(v1080_el);
          r1[75] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1083_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1129_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1077_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1124_sw, v1124_sw))))))));
          float v1136_sw = tensorforge::swap<32>(v1087_el);
          float v1141_sw = tensorforge::swap<32>(v1092_el);
          r1[77] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1095_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1141_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1089_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1136_sw, v1136_sw))))))));
          float v1148_sw = tensorforge::swap<32>(v1099_el);
          float v1153_sw = tensorforge::swap<32>(v1104_el);
          r1[79] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1107_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1153_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1101_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1148_sw, v1148_sw))))))));
          float v1160_sw = tensorforge::swap<64>(v1063_el);
          r1[81] = (tensorforge::dppUpdate<228, 8, 15, false>(v1072_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1068_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1066_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1160_sw, v1160_sw))))))));
          float v1172_sw = tensorforge::swap<64>(v1075_el);
          r1[83] = (tensorforge::dppUpdate<228, 8, 15, false>(v1084_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1080_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1078_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1172_sw, v1172_sw))))))));
          float v1184_sw = tensorforge::swap<64>(v1087_el);
          r1[85] = (tensorforge::dppUpdate<228, 8, 15, false>(v1096_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1092_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1090_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1184_sw, v1184_sw))))))));
          float v1196_sw = tensorforge::swap<64>(v1099_el);
          r1[87] = (tensorforge::dppUpdate<228, 8, 15, false>(v1108_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1104_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1102_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1196_sw, v1196_sw))))))));
          float v1209_sw = tensorforge::swap<64>(v1112_sw);
          r1[89] = (tensorforge::dppUpdate<228, 8, 15, false>(v1071_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1117_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1065_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1209_sw, v1209_sw))))))));
          float v1221_sw = tensorforge::swap<64>(v1124_sw);
          r1[91] = (tensorforge::dppUpdate<228, 8, 15, false>(v1083_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1129_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1077_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1221_sw, v1221_sw))))))));
          float v1233_sw = tensorforge::swap<64>(v1136_sw);
          r1[93] = (tensorforge::dppUpdate<228, 8, 15, false>(v1095_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1141_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1089_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1233_sw, v1233_sw))))))));
          float v1245_sw = tensorforge::swap<64>(v1148_sw);
          r1[95] = (tensorforge::dppUpdate<228, 8, 15, false>(v1107_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1153_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1101_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1245_sw, v1245_sw))))))));
          float v1255_data = glb_m1[3];
          float v1256_data = glb_m1[3];
          float v1257_data = glb_m1[3];
          float v1258_data = glb_m1[3];
          float v1259_data = glb_m1[4];
          float v1260_data = glb_m1[4];
          float v1261_data = glb_m1[4];
          float v1262_data = glb_m1[4];
          float v1263_data = glb_m1[4];
          float v1264_data = glb_m1[4];
          float v1265_data = glb_m1[4];
          float v1266_data = glb_m1[4];
          float v1267_data = glb_m1[4];
          float v1268_data = glb_m1[4];
          float v1269_data = glb_m1[4];
          float v1270_data = glb_m1[4];
          tensorforge::transpose16x16b32(v1255_data, v1256_data, v1257_data, v1258_data, v1259_data, v1260_data, v1261_data, v1262_data, v1263_data, v1264_data, v1265_data, v1266_data, v1267_data, v1268_data, v1269_data, v1270_data);
          tensorforge::VectorT<float, 16> v1271_acc{};
          tensorforge::VectorT<float, 16> v1274_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1255_data, v48_data, v1271_acc, 1, 0, 0);
          float v1275_el = v1274_acc[0];
          float v1277_el = v1274_acc[4];
          float v1278_sw = tensorforge::swap<32>(v1277_el);
          float v1280_el = v1274_acc[8];
          float v1283_el = v1274_acc[12];
          float v1284_sw = tensorforge::swap<32>(v1283_el);
          r1[96] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1284_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1280_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1278_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1275_el, v1275_el))))))));
          float v1287_el = v1274_acc[1];
          float v1289_el = v1274_acc[5];
          float v1290_sw = tensorforge::swap<32>(v1289_el);
          float v1292_el = v1274_acc[9];
          float v1295_el = v1274_acc[13];
          float v1296_sw = tensorforge::swap<32>(v1295_el);
          r1[98] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1296_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1292_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1290_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1287_el, v1287_el))))))));
          float v1299_el = v1274_acc[2];
          float v1301_el = v1274_acc[6];
          float v1302_sw = tensorforge::swap<32>(v1301_el);
          float v1304_el = v1274_acc[10];
          float v1307_el = v1274_acc[14];
          float v1308_sw = tensorforge::swap<32>(v1307_el);
          r1[100] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1308_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1304_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1302_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1299_el, v1299_el))))))));
          float v1311_el = v1274_acc[3];
          float v1313_el = v1274_acc[7];
          float v1314_sw = tensorforge::swap<32>(v1313_el);
          float v1316_el = v1274_acc[11];
          float v1319_el = v1274_acc[15];
          float v1320_sw = tensorforge::swap<32>(v1319_el);
          r1[102] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1320_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1316_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1314_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1311_el, v1311_el))))))));
          float v1324_sw = tensorforge::swap<32>(v1275_el);
          float v1329_sw = tensorforge::swap<32>(v1280_el);
          r1[104] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1283_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1329_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1277_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1324_sw, v1324_sw))))))));
          float v1336_sw = tensorforge::swap<32>(v1287_el);
          float v1341_sw = tensorforge::swap<32>(v1292_el);
          r1[106] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1295_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1341_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1289_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1336_sw, v1336_sw))))))));
          float v1348_sw = tensorforge::swap<32>(v1299_el);
          float v1353_sw = tensorforge::swap<32>(v1304_el);
          r1[108] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1307_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1353_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1301_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1348_sw, v1348_sw))))))));
          float v1360_sw = tensorforge::swap<32>(v1311_el);
          float v1365_sw = tensorforge::swap<32>(v1316_el);
          r1[110] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1319_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1365_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1313_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1360_sw, v1360_sw))))))));
          float v1372_sw = tensorforge::swap<64>(v1275_el);
          r1[112] = (tensorforge::dppUpdate<228, 8, 15, false>(v1284_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1280_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1278_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1372_sw, v1372_sw))))))));
          float v1384_sw = tensorforge::swap<64>(v1287_el);
          r1[114] = (tensorforge::dppUpdate<228, 8, 15, false>(v1296_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1292_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1290_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1384_sw, v1384_sw))))))));
          float v1396_sw = tensorforge::swap<64>(v1299_el);
          r1[116] = (tensorforge::dppUpdate<228, 8, 15, false>(v1308_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1304_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1302_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1396_sw, v1396_sw))))))));
          float v1408_sw = tensorforge::swap<64>(v1311_el);
          r1[118] = (tensorforge::dppUpdate<228, 8, 15, false>(v1320_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1316_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1314_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1408_sw, v1408_sw))))))));
          float v1421_sw = tensorforge::swap<64>(v1324_sw);
          r1[120] = (tensorforge::dppUpdate<228, 8, 15, false>(v1283_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1329_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1277_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1421_sw, v1421_sw))))))));
          float v1433_sw = tensorforge::swap<64>(v1336_sw);
          r1[122] = (tensorforge::dppUpdate<228, 8, 15, false>(v1295_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1341_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1289_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1433_sw, v1433_sw))))))));
          float v1445_sw = tensorforge::swap<64>(v1348_sw);
          r1[124] = (tensorforge::dppUpdate<228, 8, 15, false>(v1307_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1353_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1301_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1445_sw, v1445_sw))))))));
          float v1457_sw = tensorforge::swap<64>(v1360_sw);
          r1[126] = (tensorforge::dppUpdate<228, 8, 15, false>(v1319_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1365_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1313_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1457_sw, v1457_sw))))))));
          tensorforge::VectorT<float, 16> v1467_acc{};
          tensorforge::VectorT<float, 16> v1470_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1255_data, v244_data, v1467_acc, 1, 0, 0);
          float v1471_el = v1470_acc[0];
          float v1473_el = v1470_acc[4];
          float v1474_sw = tensorforge::swap<32>(v1473_el);
          float v1476_el = v1470_acc[8];
          float v1479_el = v1470_acc[12];
          float v1480_sw = tensorforge::swap<32>(v1479_el);
          r1[97] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1480_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1476_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1474_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1471_el, v1471_el))))))));
          float v1483_el = v1470_acc[1];
          float v1485_el = v1470_acc[5];
          float v1486_sw = tensorforge::swap<32>(v1485_el);
          float v1488_el = v1470_acc[9];
          float v1491_el = v1470_acc[13];
          float v1492_sw = tensorforge::swap<32>(v1491_el);
          r1[99] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1492_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1488_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1486_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1483_el, v1483_el))))))));
          float v1495_el = v1470_acc[2];
          float v1497_el = v1470_acc[6];
          float v1498_sw = tensorforge::swap<32>(v1497_el);
          float v1500_el = v1470_acc[10];
          float v1503_el = v1470_acc[14];
          float v1504_sw = tensorforge::swap<32>(v1503_el);
          r1[101] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1504_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1500_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1498_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1495_el, v1495_el))))))));
          float v1507_el = v1470_acc[3];
          float v1509_el = v1470_acc[7];
          float v1510_sw = tensorforge::swap<32>(v1509_el);
          float v1512_el = v1470_acc[11];
          float v1515_el = v1470_acc[15];
          float v1516_sw = tensorforge::swap<32>(v1515_el);
          r1[103] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1516_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1512_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1510_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1507_el, v1507_el))))))));
          float v1520_sw = tensorforge::swap<32>(v1471_el);
          float v1525_sw = tensorforge::swap<32>(v1476_el);
          r1[105] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1479_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1525_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1473_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1520_sw, v1520_sw))))))));
          float v1532_sw = tensorforge::swap<32>(v1483_el);
          float v1537_sw = tensorforge::swap<32>(v1488_el);
          r1[107] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1491_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1537_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1485_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1532_sw, v1532_sw))))))));
          float v1544_sw = tensorforge::swap<32>(v1495_el);
          float v1549_sw = tensorforge::swap<32>(v1500_el);
          r1[109] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1503_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1549_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1497_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1544_sw, v1544_sw))))))));
          float v1556_sw = tensorforge::swap<32>(v1507_el);
          float v1561_sw = tensorforge::swap<32>(v1512_el);
          r1[111] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1515_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1561_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1509_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1556_sw, v1556_sw))))))));
          float v1568_sw = tensorforge::swap<64>(v1471_el);
          r1[113] = (tensorforge::dppUpdate<228, 8, 15, false>(v1480_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1476_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1474_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1568_sw, v1568_sw))))))));
          float v1580_sw = tensorforge::swap<64>(v1483_el);
          r1[115] = (tensorforge::dppUpdate<228, 8, 15, false>(v1492_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1488_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1486_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1580_sw, v1580_sw))))))));
          float v1592_sw = tensorforge::swap<64>(v1495_el);
          r1[117] = (tensorforge::dppUpdate<228, 8, 15, false>(v1504_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1500_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1498_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1592_sw, v1592_sw))))))));
          float v1604_sw = tensorforge::swap<64>(v1507_el);
          r1[119] = (tensorforge::dppUpdate<228, 8, 15, false>(v1516_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1512_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1510_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1604_sw, v1604_sw))))))));
          float v1617_sw = tensorforge::swap<64>(v1520_sw);
          r1[121] = (tensorforge::dppUpdate<228, 8, 15, false>(v1479_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1525_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1473_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1617_sw, v1617_sw))))))));
          float v1629_sw = tensorforge::swap<64>(v1532_sw);
          r1[123] = (tensorforge::dppUpdate<228, 8, 15, false>(v1491_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1537_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1485_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1629_sw, v1629_sw))))))));
          float v1641_sw = tensorforge::swap<64>(v1544_sw);
          r1[125] = (tensorforge::dppUpdate<228, 8, 15, false>(v1503_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1549_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1497_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1641_sw, v1641_sw))))))));
          float v1653_sw = tensorforge::swap<64>(v1556_sw);
          r1[127] = (tensorforge::dppUpdate<228, 8, 15, false>(v1515_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1561_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1509_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1653_sw, v1653_sw))))))));
          float v1663_data = glb_m1[4];
          float v1664_data = glb_m1[5];
          float v1665_data = glb_m1[5];
          float v1666_data = glb_m1[5];
          float v1667_data = glb_m1[5];
          float v1668_data = glb_m1[5];
          float v1669_data = glb_m1[5];
          float v1670_data = glb_m1[5];
          float v1671_data = glb_m1[5];
          float v1672_data = glb_m1[5];
          float v1673_data = glb_m1[5];
          float v1674_data = glb_m1[5];
          float v1675_data = glb_m1[5];
          float v1676_data = glb_m1[5];
          float v1677_pad{};
          float v1678_pad{};
          tensorforge::transpose16x16b32(v1663_data, v1664_data, v1665_data, v1666_data, v1667_data, v1668_data, v1669_data, v1670_data, v1671_data, v1672_data, v1673_data, v1674_data, v1675_data, v1676_data, v1677_pad, v1678_pad);
          tensorforge::VectorT<float, 16> v1679_acc{};
          tensorforge::VectorT<float, 16> v1682_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1663_data, v48_data, v1679_acc, 1, 0, 0);
          float v1683_el = v1682_acc[0];
          float v1685_el = v1682_acc[4];
          float v1686_sw = tensorforge::swap<32>(v1685_el);
          float v1688_el = v1682_acc[8];
          float v1691_el = v1682_acc[12];
          float v1692_sw = tensorforge::swap<32>(v1691_el);
          r1[128] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1692_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1688_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1686_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1683_el, v1683_el))))))));
          float v1695_el = v1682_acc[1];
          float v1697_el = v1682_acc[5];
          float v1698_sw = tensorforge::swap<32>(v1697_el);
          float v1700_el = v1682_acc[9];
          float v1703_el = v1682_acc[13];
          float v1704_sw = tensorforge::swap<32>(v1703_el);
          r1[130] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1704_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1700_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1698_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1695_el, v1695_el))))))));
          float v1707_el = v1682_acc[2];
          float v1709_el = v1682_acc[6];
          float v1710_sw = tensorforge::swap<32>(v1709_el);
          float v1712_el = v1682_acc[10];
          float v1715_el = v1682_acc[14];
          float v1716_sw = tensorforge::swap<32>(v1715_el);
          r1[132] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1716_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1712_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1710_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1707_el, v1707_el))))))));
          float v1719_el = v1682_acc[3];
          float v1721_el = v1682_acc[7];
          float v1722_sw = tensorforge::swap<32>(v1721_el);
          float v1724_el = v1682_acc[11];
          float v1727_el = v1682_acc[15];
          float v1728_sw = tensorforge::swap<32>(v1727_el);
          r1[134] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1728_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1724_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1722_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1719_el, v1719_el))))))));
          float v1732_sw = tensorforge::swap<32>(v1683_el);
          float v1737_sw = tensorforge::swap<32>(v1688_el);
          r1[136] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1691_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1737_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1685_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1732_sw, v1732_sw))))))));
          float v1744_sw = tensorforge::swap<32>(v1695_el);
          float v1749_sw = tensorforge::swap<32>(v1700_el);
          r1[138] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1703_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1749_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1697_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1744_sw, v1744_sw))))))));
          float v1756_sw = tensorforge::swap<32>(v1707_el);
          r1[140] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1715_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1712_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1709_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1756_sw, v1756_sw))))))));
          float v1768_sw = tensorforge::swap<32>(v1719_el);
          r1[142] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1727_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1724_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1721_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1768_sw, v1768_sw))))))));
          float v1780_sw = tensorforge::swap<64>(v1683_el);
          r1[144] = (tensorforge::dppUpdate<228, 8, 15, false>(v1692_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1688_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1686_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1780_sw, v1780_sw))))))));
          float v1792_sw = tensorforge::swap<64>(v1695_el);
          r1[146] = (tensorforge::dppUpdate<228, 8, 15, false>(v1704_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1700_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1698_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1792_sw, v1792_sw))))))));
          float v1804_sw = tensorforge::swap<64>(v1707_el);
          r1[148] = (tensorforge::dppUpdate<228, 8, 15, false>(v1716_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1712_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1710_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1804_sw, v1804_sw))))))));
          float v1816_sw = tensorforge::swap<64>(v1719_el);
          r1[150] = (tensorforge::dppUpdate<228, 8, 15, false>(v1728_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1724_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1722_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1816_sw, v1816_sw))))))));
          float v1829_sw = tensorforge::swap<64>(v1732_sw);
          r1[152] = (tensorforge::dppUpdate<228, 8, 15, false>(v1691_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1737_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1685_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1829_sw, v1829_sw))))))));
          float v1841_sw = tensorforge::swap<64>(v1744_sw);
          r1[154] = (tensorforge::dppUpdate<228, 8, 15, false>(v1703_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1749_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1697_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1841_sw, v1841_sw))))))));
          tensorforge::VectorT<float, 16> v1851_acc{};
          tensorforge::VectorT<float, 16> v1854_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1663_data, v244_data, v1851_acc, 1, 0, 0);
          float v1855_el = v1854_acc[0];
          float v1857_el = v1854_acc[4];
          float v1858_sw = tensorforge::swap<32>(v1857_el);
          float v1860_el = v1854_acc[8];
          float v1863_el = v1854_acc[12];
          float v1864_sw = tensorforge::swap<32>(v1863_el);
          r1[129] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1864_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1860_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1858_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1855_el, v1855_el))))))));
          float v1867_el = v1854_acc[1];
          float v1869_el = v1854_acc[5];
          float v1870_sw = tensorforge::swap<32>(v1869_el);
          float v1872_el = v1854_acc[9];
          float v1875_el = v1854_acc[13];
          float v1876_sw = tensorforge::swap<32>(v1875_el);
          r1[131] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1876_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1872_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1870_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1867_el, v1867_el))))))));
          float v1879_el = v1854_acc[2];
          float v1881_el = v1854_acc[6];
          float v1882_sw = tensorforge::swap<32>(v1881_el);
          float v1884_el = v1854_acc[10];
          float v1887_el = v1854_acc[14];
          float v1888_sw = tensorforge::swap<32>(v1887_el);
          r1[133] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1888_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1884_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1882_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1879_el, v1879_el))))))));
          float v1891_el = v1854_acc[3];
          float v1893_el = v1854_acc[7];
          float v1894_sw = tensorforge::swap<32>(v1893_el);
          float v1896_el = v1854_acc[11];
          float v1899_el = v1854_acc[15];
          float v1900_sw = tensorforge::swap<32>(v1899_el);
          r1[135] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1900_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1896_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1894_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1891_el, v1891_el))))))));
          float v1904_sw = tensorforge::swap<32>(v1855_el);
          float v1909_sw = tensorforge::swap<32>(v1860_el);
          r1[137] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1863_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1909_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1857_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1904_sw, v1904_sw))))))));
          float v1916_sw = tensorforge::swap<32>(v1867_el);
          float v1921_sw = tensorforge::swap<32>(v1872_el);
          r1[139] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1875_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1921_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1869_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1916_sw, v1916_sw))))))));
          float v1928_sw = tensorforge::swap<32>(v1879_el);
          r1[141] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1887_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1884_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1881_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1928_sw, v1928_sw))))))));
          float v1940_sw = tensorforge::swap<32>(v1891_el);
          r1[143] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1899_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1896_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1893_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1940_sw, v1940_sw))))))));
          float v1952_sw = tensorforge::swap<64>(v1855_el);
          r1[145] = (tensorforge::dppUpdate<228, 8, 15, false>(v1864_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1860_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1858_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1952_sw, v1952_sw))))))));
          float v1964_sw = tensorforge::swap<64>(v1867_el);
          r1[147] = (tensorforge::dppUpdate<228, 8, 15, false>(v1876_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1872_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1870_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1964_sw, v1964_sw))))))));
          float v1976_sw = tensorforge::swap<64>(v1879_el);
          r1[149] = (tensorforge::dppUpdate<228, 8, 15, false>(v1888_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1884_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1882_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1976_sw, v1976_sw))))))));
          float v1988_sw = tensorforge::swap<64>(v1891_el);
          r1[151] = (tensorforge::dppUpdate<228, 8, 15, false>(v1900_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1896_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1894_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1988_sw, v1988_sw))))))));
          float v2001_sw = tensorforge::swap<64>(v1904_sw);
          r1[153] = (tensorforge::dppUpdate<228, 8, 15, false>(v1863_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1909_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1857_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v2001_sw, v2001_sw))))))));
          float v2013_sw = tensorforge::swap<64>(v1916_sw);
          r1[155] = (tensorforge::dppUpdate<228, 8, 15, false>(v1875_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1921_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1869_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v2013_sw, v2013_sw))))))));
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          bool v2024_g = v20_lead >= 20;
          if (v2024_g) {
            float v2025_data = r1[24];
            float v2026_data = r2[0];
            r2[0] = (v2026_data + v2025_data);
            float v2028_data = r1[50];
            float v2029_data = r2[2];
            r2[2] = (v2029_data + v2028_data);
            float v2031_data = r1[76];
            float v2032_data = r2[4];
            r2[4] = (v2032_data + v2031_data);
            float v2034_data = r1[102];
            float v2035_data = r2[6];
            r2[6] = (v2035_data + v2034_data);
            float v2037_data = r1[128];
            float v2038_data = r2[8];
            r2[8] = (v2038_data + v2037_data);
            float v2040_data = r1[154];
            float v2041_data = r2[10];
            r2[10] = (v2041_data + v2040_data);
          }
          bool v2043_g = v20_lead < 3;
          if (v2043_g) {
            float v2044_data = r1[25];
            float v2045_data = r2[1];
            r2[1] = (v2045_data + v2044_data);
            float v2047_data = r1[51];
            float v2048_data = r2[3];
            r2[3] = (v2048_data + v2047_data);
            float v2050_data = r1[77];
            float v2051_data = r2[5];
            r2[5] = (v2051_data + v2050_data);
            float v2053_data = r1[103];
            float v2054_data = r2[7];
            r2[7] = (v2054_data + v2053_data);
            float v2056_data = r1[129];
            float v2057_data = r2[9];
            r2[9] = (v2057_data + v2056_data);
            float v2059_data = r1[155];
            float v2060_data = r2[11];
            r2[11] = (v2060_data + v2059_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v2024_g) {
            #pragma unroll
            for (int32_t v2063_i1 = 0; v2063_i1 < 1; ++v2063_i1) {
              int32_t v2065_a = v2063_i1 * 2;
              int32_t v2075_a = v20_lead + ((v2063_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v2064_i2 = 0; v2064_i2 < 6; ++v2064_i2) {
                float v2069_data = r2[(v2065_a + (v2064_i2 * 2))];
                int32_t v2076_a = v2075_a + (v2064_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v2076_a], v2069_data);
              }
            }
          }
          if (v2043_g) {
            int32_t v2086_lead = v20_lead + 32_i32;
            #pragma unroll
            for (int32_t v2078_i1 = 0; v2078_i1 < 1; ++v2078_i1) {
              int32_t v2082_a = 1 + (v2078_i1 * 2);
              int32_t v2090_a = v2086_lead + ((v2078_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v2079_i2 = 0; v2079_i2 < 6; ++v2079_i2) {
                float v2084_data = r2[(v2082_a + (v2079_i2 * 2))];
                int32_t v2091_a = v2090_a + (v2079_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v2091_a], v2084_data);
              }
            }
          }
        }
      }
    }
  }
}

