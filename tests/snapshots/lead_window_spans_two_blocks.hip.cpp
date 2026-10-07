// === base name ===
kernel_d39a8dff5da4c456

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d39a8dff5da4c456 = {{32, 8, 1}, 32, 64, 1, 8, 256, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d39a8dff5da4c456(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d39a8dff5da4c456(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d39a8dff5da4c456(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d39a8dff5da4c456, block.x * block.y * block.z, 64 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (64 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d39a8dff5da4c456, block.x * block.y * block.z, 0));
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
void launcher_kernel_d39a8dff5da4c456(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d39a8dff5da4c456(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d39a8dff5da4c456), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_d39a8dff5da4c456, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1Arg, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d39a8dff5da4c456(const float ** m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, float ** m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":64}],"shared_bytes":256,"shared_elements":64,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"none","alias":"v","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"pointer_based","alias":"D","bbox":[[0,0,0],[64,13,6]],"name":"m2","ordered":false,"parts":1,"shape":[64,13,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0,0],[64,13,6]],"is_tmp":true,"name":"t0","offset":[0,0,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"none","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0,1],[0]],"target":[[0,1],[2]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0,0],[15,1,6]],"is_tmp":false,"name":"m2","offset":[20,12,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0,0],[15,1,6]],"is_tmp":true,"name":"t0","offset":[20,12,0],"shape":[64,13,6]}],"permute":[[0,1,2]],"target":[[0,1,2]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 64];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 6) {
        float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      }
      __syncthreads();
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v10_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v10_batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 2; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 32);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 13; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m0[(v27_lead + (v25_i1 * 64))]);
              r0[(v24_i0 + (v25_i1 * 2))] = v30_data;
            }
          }
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
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
          float v44_data = glb_m1[0];
          float v45_data = glb_m1[0];
          float v46_data = glb_m1[0];
          float v47_data = glb_m1[1];
          float v48_data = glb_m1[1];
          float v49_data = glb_m1[1];
          tensorforge::transpose16x16b32(v34_data, v35_data, v36_data, v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data);
          tensorforge::VectorT<float, 16> v50_acc{};
          float v51_data = r0[0];
          tensorforge::VectorT<float, 16> v53_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v34_data, v51_data, v50_acc, 1, 0, 0);
          float v54_el = v53_acc[0];
          float v56_el = v53_acc[4];
          float v57_sw = tensorforge::swap<32>(v56_el);
          float v59_el = v53_acc[8];
          float v62_el = v53_acc[12];
          float v63_sw = tensorforge::swap<32>(v62_el);
          r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v63_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v59_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v57_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v54_el, v54_el))))))));
          float v66_el = v53_acc[1];
          float v68_el = v53_acc[5];
          float v69_sw = tensorforge::swap<32>(v68_el);
          float v71_el = v53_acc[9];
          float v74_el = v53_acc[13];
          float v75_sw = tensorforge::swap<32>(v74_el);
          r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v75_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v71_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v69_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v66_el, v66_el))))))));
          float v78_el = v53_acc[2];
          float v80_el = v53_acc[6];
          float v81_sw = tensorforge::swap<32>(v80_el);
          float v83_el = v53_acc[10];
          float v86_el = v53_acc[14];
          float v87_sw = tensorforge::swap<32>(v86_el);
          r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v87_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v83_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v81_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v78_el, v78_el))))))));
          float v90_el = v53_acc[3];
          float v92_el = v53_acc[7];
          float v93_sw = tensorforge::swap<32>(v92_el);
          float v95_el = v53_acc[11];
          float v98_el = v53_acc[15];
          float v99_sw = tensorforge::swap<32>(v98_el);
          r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v99_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v95_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v93_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v90_el, v90_el))))))));
          float v103_sw = tensorforge::swap<32>(v54_el);
          float v108_sw = tensorforge::swap<32>(v59_el);
          r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v62_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v108_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v56_el, (tensorforge::dppUpdate<228, 1, 15, false>(v103_sw, v103_sw))))))));
          float v115_sw = tensorforge::swap<32>(v66_el);
          float v120_sw = tensorforge::swap<32>(v71_el);
          r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v74_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v120_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v68_el, (tensorforge::dppUpdate<228, 1, 15, false>(v115_sw, v115_sw))))))));
          float v127_sw = tensorforge::swap<32>(v78_el);
          float v132_sw = tensorforge::swap<32>(v83_el);
          r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v86_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v132_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v80_el, (tensorforge::dppUpdate<228, 1, 15, false>(v127_sw, v127_sw))))))));
          float v139_sw = tensorforge::swap<32>(v90_el);
          float v144_sw = tensorforge::swap<32>(v95_el);
          r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v98_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v144_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v92_el, (tensorforge::dppUpdate<228, 1, 15, false>(v139_sw, v139_sw))))))));
          float v151_sw = tensorforge::swap<64>(v54_el);
          r1[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v63_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v59_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v57_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v151_sw, v151_sw))))))));
          float v163_sw = tensorforge::swap<64>(v66_el);
          r1[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v75_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v71_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v69_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v163_sw, v163_sw))))))));
          float v175_sw = tensorforge::swap<64>(v78_el);
          r1[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v87_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v83_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v81_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v175_sw, v175_sw))))))));
          float v187_sw = tensorforge::swap<64>(v90_el);
          r1[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v99_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v95_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v93_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v187_sw, v187_sw))))))));
          float v200_sw = tensorforge::swap<64>(v103_sw);
          r1[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v62_el, (tensorforge::dppUpdate<228, 4, 15, false>(v108_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v56_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v200_sw, v200_sw))))))));
          float v212_sw = tensorforge::swap<64>(v115_sw);
          r1[26] = (tensorforge::dppUpdate<228, 8, 15, false>(v74_el, (tensorforge::dppUpdate<228, 4, 15, false>(v120_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v68_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v212_sw, v212_sw))))))));
          float v224_sw = tensorforge::swap<64>(v127_sw);
          r1[28] = (tensorforge::dppUpdate<228, 8, 15, false>(v86_el, (tensorforge::dppUpdate<228, 4, 15, false>(v132_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v80_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v224_sw, v224_sw))))))));
          float v236_sw = tensorforge::swap<64>(v139_sw);
          r1[30] = (tensorforge::dppUpdate<228, 8, 15, false>(v98_el, (tensorforge::dppUpdate<228, 4, 15, false>(v144_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v92_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v236_sw, v236_sw))))))));
          tensorforge::VectorT<float, 16> v246_acc{};
          float v247_data = r0[1];
          tensorforge::VectorT<float, 16> v249_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v34_data, v247_data, v246_acc, 1, 0, 0);
          float v250_el = v249_acc[0];
          float v252_el = v249_acc[4];
          float v253_sw = tensorforge::swap<32>(v252_el);
          float v255_el = v249_acc[8];
          float v258_el = v249_acc[12];
          float v259_sw = tensorforge::swap<32>(v258_el);
          r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v259_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v255_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v253_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v250_el, v250_el))))))));
          float v262_el = v249_acc[1];
          float v264_el = v249_acc[5];
          float v265_sw = tensorforge::swap<32>(v264_el);
          float v267_el = v249_acc[9];
          float v270_el = v249_acc[13];
          float v271_sw = tensorforge::swap<32>(v270_el);
          r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v271_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v267_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v265_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v262_el, v262_el))))))));
          float v274_el = v249_acc[2];
          float v276_el = v249_acc[6];
          float v277_sw = tensorforge::swap<32>(v276_el);
          float v279_el = v249_acc[10];
          float v282_el = v249_acc[14];
          float v283_sw = tensorforge::swap<32>(v282_el);
          r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v283_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v279_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v277_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v274_el, v274_el))))))));
          float v286_el = v249_acc[3];
          float v288_el = v249_acc[7];
          float v289_sw = tensorforge::swap<32>(v288_el);
          float v291_el = v249_acc[11];
          float v294_el = v249_acc[15];
          float v295_sw = tensorforge::swap<32>(v294_el);
          r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v295_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v291_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v289_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v286_el, v286_el))))))));
          float v299_sw = tensorforge::swap<32>(v250_el);
          float v304_sw = tensorforge::swap<32>(v255_el);
          r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v258_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v304_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v252_el, (tensorforge::dppUpdate<228, 1, 15, false>(v299_sw, v299_sw))))))));
          float v311_sw = tensorforge::swap<32>(v262_el);
          float v316_sw = tensorforge::swap<32>(v267_el);
          r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v270_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v316_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v264_el, (tensorforge::dppUpdate<228, 1, 15, false>(v311_sw, v311_sw))))))));
          float v323_sw = tensorforge::swap<32>(v274_el);
          float v328_sw = tensorforge::swap<32>(v279_el);
          r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v282_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v328_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v276_el, (tensorforge::dppUpdate<228, 1, 15, false>(v323_sw, v323_sw))))))));
          float v335_sw = tensorforge::swap<32>(v286_el);
          float v340_sw = tensorforge::swap<32>(v291_el);
          r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v294_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v340_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v288_el, (tensorforge::dppUpdate<228, 1, 15, false>(v335_sw, v335_sw))))))));
          float v347_sw = tensorforge::swap<64>(v250_el);
          r1[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v259_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v255_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v253_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v347_sw, v347_sw))))))));
          float v359_sw = tensorforge::swap<64>(v262_el);
          r1[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v271_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v267_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v265_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v359_sw, v359_sw))))))));
          float v371_sw = tensorforge::swap<64>(v274_el);
          r1[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v283_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v279_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v277_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v371_sw, v371_sw))))))));
          float v383_sw = tensorforge::swap<64>(v286_el);
          r1[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v295_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v291_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v289_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v383_sw, v383_sw))))))));
          float v396_sw = tensorforge::swap<64>(v299_sw);
          r1[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v258_el, (tensorforge::dppUpdate<228, 4, 15, false>(v304_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v252_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v396_sw, v396_sw))))))));
          float v408_sw = tensorforge::swap<64>(v311_sw);
          r1[27] = (tensorforge::dppUpdate<228, 8, 15, false>(v270_el, (tensorforge::dppUpdate<228, 4, 15, false>(v316_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v264_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v408_sw, v408_sw))))))));
          float v420_sw = tensorforge::swap<64>(v323_sw);
          r1[29] = (tensorforge::dppUpdate<228, 8, 15, false>(v282_el, (tensorforge::dppUpdate<228, 4, 15, false>(v328_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v276_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v420_sw, v420_sw))))))));
          float v432_sw = tensorforge::swap<64>(v335_sw);
          r1[31] = (tensorforge::dppUpdate<228, 8, 15, false>(v294_el, (tensorforge::dppUpdate<228, 4, 15, false>(v340_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v288_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v432_sw, v432_sw))))))));
          float v442_data = glb_m1[1];
          float v443_data = glb_m1[1];
          float v444_data = glb_m1[1];
          float v445_data = glb_m1[1];
          float v446_data = glb_m1[1];
          float v447_data = glb_m1[1];
          float v448_data = glb_m1[1];
          float v449_data = glb_m1[1];
          float v450_data = glb_m1[1];
          float v451_data = glb_m1[1];
          float v452_data = glb_m1[2];
          float v453_data = glb_m1[2];
          float v454_data = glb_m1[2];
          float v455_data = glb_m1[2];
          float v456_data = glb_m1[2];
          float v457_data = glb_m1[2];
          tensorforge::transpose16x16b32(v442_data, v443_data, v444_data, v445_data, v446_data, v447_data, v448_data, v449_data, v450_data, v451_data, v452_data, v453_data, v454_data, v455_data, v456_data, v457_data);
          tensorforge::VectorT<float, 16> v458_acc{};
          tensorforge::VectorT<float, 16> v461_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v442_data, v51_data, v458_acc, 1, 0, 0);
          float v462_el = v461_acc[0];
          float v464_el = v461_acc[4];
          float v465_sw = tensorforge::swap<32>(v464_el);
          float v467_el = v461_acc[8];
          float v470_el = v461_acc[12];
          float v471_sw = tensorforge::swap<32>(v470_el);
          r1[32] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v471_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v467_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v465_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v462_el, v462_el))))))));
          float v474_el = v461_acc[1];
          float v476_el = v461_acc[5];
          float v477_sw = tensorforge::swap<32>(v476_el);
          float v479_el = v461_acc[9];
          float v482_el = v461_acc[13];
          float v483_sw = tensorforge::swap<32>(v482_el);
          r1[34] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v483_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v479_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v477_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v474_el, v474_el))))))));
          float v486_el = v461_acc[2];
          float v488_el = v461_acc[6];
          float v489_sw = tensorforge::swap<32>(v488_el);
          float v491_el = v461_acc[10];
          float v494_el = v461_acc[14];
          float v495_sw = tensorforge::swap<32>(v494_el);
          r1[36] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v495_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v491_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v489_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v486_el, v486_el))))))));
          float v498_el = v461_acc[3];
          float v500_el = v461_acc[7];
          float v501_sw = tensorforge::swap<32>(v500_el);
          float v503_el = v461_acc[11];
          float v506_el = v461_acc[15];
          float v507_sw = tensorforge::swap<32>(v506_el);
          r1[38] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v507_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v503_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v501_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v498_el, v498_el))))))));
          float v511_sw = tensorforge::swap<32>(v462_el);
          float v516_sw = tensorforge::swap<32>(v467_el);
          r1[40] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v470_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v516_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v464_el, (tensorforge::dppUpdate<228, 1, 15, false>(v511_sw, v511_sw))))))));
          float v523_sw = tensorforge::swap<32>(v474_el);
          float v528_sw = tensorforge::swap<32>(v479_el);
          r1[42] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v482_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v528_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v476_el, (tensorforge::dppUpdate<228, 1, 15, false>(v523_sw, v523_sw))))))));
          float v535_sw = tensorforge::swap<32>(v486_el);
          float v540_sw = tensorforge::swap<32>(v491_el);
          r1[44] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v494_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v540_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v488_el, (tensorforge::dppUpdate<228, 1, 15, false>(v535_sw, v535_sw))))))));
          float v547_sw = tensorforge::swap<32>(v498_el);
          float v552_sw = tensorforge::swap<32>(v503_el);
          r1[46] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v506_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v552_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v500_el, (tensorforge::dppUpdate<228, 1, 15, false>(v547_sw, v547_sw))))))));
          float v559_sw = tensorforge::swap<64>(v462_el);
          r1[48] = (tensorforge::dppUpdate<228, 8, 15, false>(v471_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v467_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v465_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v559_sw, v559_sw))))))));
          float v571_sw = tensorforge::swap<64>(v474_el);
          r1[50] = (tensorforge::dppUpdate<228, 8, 15, false>(v483_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v479_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v477_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v571_sw, v571_sw))))))));
          float v583_sw = tensorforge::swap<64>(v486_el);
          r1[52] = (tensorforge::dppUpdate<228, 8, 15, false>(v495_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v491_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v489_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v583_sw, v583_sw))))))));
          float v595_sw = tensorforge::swap<64>(v498_el);
          r1[54] = (tensorforge::dppUpdate<228, 8, 15, false>(v507_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v503_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v501_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v595_sw, v595_sw))))))));
          float v608_sw = tensorforge::swap<64>(v511_sw);
          r1[56] = (tensorforge::dppUpdate<228, 8, 15, false>(v470_el, (tensorforge::dppUpdate<228, 4, 15, false>(v516_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v464_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v608_sw, v608_sw))))))));
          float v620_sw = tensorforge::swap<64>(v523_sw);
          r1[58] = (tensorforge::dppUpdate<228, 8, 15, false>(v482_el, (tensorforge::dppUpdate<228, 4, 15, false>(v528_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v476_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v620_sw, v620_sw))))))));
          float v632_sw = tensorforge::swap<64>(v535_sw);
          r1[60] = (tensorforge::dppUpdate<228, 8, 15, false>(v494_el, (tensorforge::dppUpdate<228, 4, 15, false>(v540_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v488_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v632_sw, v632_sw))))))));
          float v644_sw = tensorforge::swap<64>(v547_sw);
          r1[62] = (tensorforge::dppUpdate<228, 8, 15, false>(v506_el, (tensorforge::dppUpdate<228, 4, 15, false>(v552_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v500_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v644_sw, v644_sw))))))));
          tensorforge::VectorT<float, 16> v654_acc{};
          tensorforge::VectorT<float, 16> v657_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v442_data, v247_data, v654_acc, 1, 0, 0);
          float v658_el = v657_acc[0];
          float v660_el = v657_acc[4];
          float v661_sw = tensorforge::swap<32>(v660_el);
          float v663_el = v657_acc[8];
          float v666_el = v657_acc[12];
          float v667_sw = tensorforge::swap<32>(v666_el);
          r1[33] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v667_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v663_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v661_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v658_el, v658_el))))))));
          float v670_el = v657_acc[1];
          float v672_el = v657_acc[5];
          float v673_sw = tensorforge::swap<32>(v672_el);
          float v675_el = v657_acc[9];
          float v678_el = v657_acc[13];
          float v679_sw = tensorforge::swap<32>(v678_el);
          r1[35] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v679_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v675_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v673_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v670_el, v670_el))))))));
          float v682_el = v657_acc[2];
          float v684_el = v657_acc[6];
          float v685_sw = tensorforge::swap<32>(v684_el);
          float v687_el = v657_acc[10];
          float v690_el = v657_acc[14];
          float v691_sw = tensorforge::swap<32>(v690_el);
          r1[37] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v691_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v687_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v685_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v682_el, v682_el))))))));
          float v694_el = v657_acc[3];
          float v696_el = v657_acc[7];
          float v697_sw = tensorforge::swap<32>(v696_el);
          float v699_el = v657_acc[11];
          float v702_el = v657_acc[15];
          float v703_sw = tensorforge::swap<32>(v702_el);
          r1[39] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v703_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v699_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v697_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v694_el, v694_el))))))));
          float v707_sw = tensorforge::swap<32>(v658_el);
          float v712_sw = tensorforge::swap<32>(v663_el);
          r1[41] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v666_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v712_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v660_el, (tensorforge::dppUpdate<228, 1, 15, false>(v707_sw, v707_sw))))))));
          float v719_sw = tensorforge::swap<32>(v670_el);
          float v724_sw = tensorforge::swap<32>(v675_el);
          r1[43] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v678_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v724_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v672_el, (tensorforge::dppUpdate<228, 1, 15, false>(v719_sw, v719_sw))))))));
          float v731_sw = tensorforge::swap<32>(v682_el);
          float v736_sw = tensorforge::swap<32>(v687_el);
          r1[45] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v690_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v736_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v684_el, (tensorforge::dppUpdate<228, 1, 15, false>(v731_sw, v731_sw))))))));
          float v743_sw = tensorforge::swap<32>(v694_el);
          float v748_sw = tensorforge::swap<32>(v699_el);
          r1[47] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v702_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v748_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v696_el, (tensorforge::dppUpdate<228, 1, 15, false>(v743_sw, v743_sw))))))));
          float v755_sw = tensorforge::swap<64>(v658_el);
          r1[49] = (tensorforge::dppUpdate<228, 8, 15, false>(v667_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v663_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v661_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v755_sw, v755_sw))))))));
          float v767_sw = tensorforge::swap<64>(v670_el);
          r1[51] = (tensorforge::dppUpdate<228, 8, 15, false>(v679_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v675_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v673_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v767_sw, v767_sw))))))));
          float v779_sw = tensorforge::swap<64>(v682_el);
          r1[53] = (tensorforge::dppUpdate<228, 8, 15, false>(v691_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v687_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v685_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v779_sw, v779_sw))))))));
          float v791_sw = tensorforge::swap<64>(v694_el);
          r1[55] = (tensorforge::dppUpdate<228, 8, 15, false>(v703_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v699_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v697_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v791_sw, v791_sw))))))));
          float v804_sw = tensorforge::swap<64>(v707_sw);
          r1[57] = (tensorforge::dppUpdate<228, 8, 15, false>(v666_el, (tensorforge::dppUpdate<228, 4, 15, false>(v712_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v660_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v804_sw, v804_sw))))))));
          float v816_sw = tensorforge::swap<64>(v719_sw);
          r1[59] = (tensorforge::dppUpdate<228, 8, 15, false>(v678_el, (tensorforge::dppUpdate<228, 4, 15, false>(v724_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v672_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v816_sw, v816_sw))))))));
          float v828_sw = tensorforge::swap<64>(v731_sw);
          r1[61] = (tensorforge::dppUpdate<228, 8, 15, false>(v690_el, (tensorforge::dppUpdate<228, 4, 15, false>(v736_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v684_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v828_sw, v828_sw))))))));
          float v840_sw = tensorforge::swap<64>(v743_sw);
          r1[63] = (tensorforge::dppUpdate<228, 8, 15, false>(v702_el, (tensorforge::dppUpdate<228, 4, 15, false>(v748_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v696_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v840_sw, v840_sw))))))));
          float v850_data = glb_m1[2];
          float v851_data = glb_m1[2];
          float v852_data = glb_m1[2];
          float v853_data = glb_m1[2];
          float v854_data = glb_m1[2];
          float v855_data = glb_m1[2];
          float v856_data = glb_m1[2];
          float v857_data = glb_m1[3];
          float v858_data = glb_m1[3];
          float v859_data = glb_m1[3];
          float v860_data = glb_m1[3];
          float v861_data = glb_m1[3];
          float v862_data = glb_m1[3];
          float v863_data = glb_m1[3];
          float v864_data = glb_m1[3];
          float v865_data = glb_m1[3];
          tensorforge::transpose16x16b32(v850_data, v851_data, v852_data, v853_data, v854_data, v855_data, v856_data, v857_data, v858_data, v859_data, v860_data, v861_data, v862_data, v863_data, v864_data, v865_data);
          tensorforge::VectorT<float, 16> v866_acc{};
          tensorforge::VectorT<float, 16> v869_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v850_data, v51_data, v866_acc, 1, 0, 0);
          float v870_el = v869_acc[0];
          float v872_el = v869_acc[4];
          float v873_sw = tensorforge::swap<32>(v872_el);
          float v875_el = v869_acc[8];
          float v878_el = v869_acc[12];
          float v879_sw = tensorforge::swap<32>(v878_el);
          r1[64] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v879_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v875_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v873_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v870_el, v870_el))))))));
          float v882_el = v869_acc[1];
          float v884_el = v869_acc[5];
          float v885_sw = tensorforge::swap<32>(v884_el);
          float v887_el = v869_acc[9];
          float v890_el = v869_acc[13];
          float v891_sw = tensorforge::swap<32>(v890_el);
          r1[66] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v891_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v887_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v885_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v882_el, v882_el))))))));
          float v894_el = v869_acc[2];
          float v896_el = v869_acc[6];
          float v897_sw = tensorforge::swap<32>(v896_el);
          float v899_el = v869_acc[10];
          float v902_el = v869_acc[14];
          float v903_sw = tensorforge::swap<32>(v902_el);
          r1[68] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v903_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v899_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v897_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v894_el, v894_el))))))));
          float v906_el = v869_acc[3];
          float v908_el = v869_acc[7];
          float v909_sw = tensorforge::swap<32>(v908_el);
          float v911_el = v869_acc[11];
          float v914_el = v869_acc[15];
          float v915_sw = tensorforge::swap<32>(v914_el);
          r1[70] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v915_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v911_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v909_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v906_el, v906_el))))))));
          float v919_sw = tensorforge::swap<32>(v870_el);
          float v924_sw = tensorforge::swap<32>(v875_el);
          r1[72] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v878_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v924_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v872_el, (tensorforge::dppUpdate<228, 1, 15, false>(v919_sw, v919_sw))))))));
          float v931_sw = tensorforge::swap<32>(v882_el);
          float v936_sw = tensorforge::swap<32>(v887_el);
          r1[74] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v890_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v936_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v884_el, (tensorforge::dppUpdate<228, 1, 15, false>(v931_sw, v931_sw))))))));
          float v943_sw = tensorforge::swap<32>(v894_el);
          float v948_sw = tensorforge::swap<32>(v899_el);
          r1[76] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v902_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v948_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v896_el, (tensorforge::dppUpdate<228, 1, 15, false>(v943_sw, v943_sw))))))));
          float v955_sw = tensorforge::swap<32>(v906_el);
          float v960_sw = tensorforge::swap<32>(v911_el);
          r1[78] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v914_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v960_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v908_el, (tensorforge::dppUpdate<228, 1, 15, false>(v955_sw, v955_sw))))))));
          float v967_sw = tensorforge::swap<64>(v870_el);
          r1[80] = (tensorforge::dppUpdate<228, 8, 15, false>(v879_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v875_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v873_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v967_sw, v967_sw))))))));
          float v979_sw = tensorforge::swap<64>(v882_el);
          r1[82] = (tensorforge::dppUpdate<228, 8, 15, false>(v891_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v887_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v885_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v979_sw, v979_sw))))))));
          float v991_sw = tensorforge::swap<64>(v894_el);
          r1[84] = (tensorforge::dppUpdate<228, 8, 15, false>(v903_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v899_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v897_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v991_sw, v991_sw))))))));
          float v1003_sw = tensorforge::swap<64>(v906_el);
          r1[86] = (tensorforge::dppUpdate<228, 8, 15, false>(v915_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v911_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v909_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1003_sw, v1003_sw))))))));
          float v1016_sw = tensorforge::swap<64>(v919_sw);
          r1[88] = (tensorforge::dppUpdate<228, 8, 15, false>(v878_el, (tensorforge::dppUpdate<228, 4, 15, false>(v924_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v872_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1016_sw, v1016_sw))))))));
          float v1028_sw = tensorforge::swap<64>(v931_sw);
          r1[90] = (tensorforge::dppUpdate<228, 8, 15, false>(v890_el, (tensorforge::dppUpdate<228, 4, 15, false>(v936_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v884_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1028_sw, v1028_sw))))))));
          float v1040_sw = tensorforge::swap<64>(v943_sw);
          r1[92] = (tensorforge::dppUpdate<228, 8, 15, false>(v902_el, (tensorforge::dppUpdate<228, 4, 15, false>(v948_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v896_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1040_sw, v1040_sw))))))));
          float v1052_sw = tensorforge::swap<64>(v955_sw);
          r1[94] = (tensorforge::dppUpdate<228, 8, 15, false>(v914_el, (tensorforge::dppUpdate<228, 4, 15, false>(v960_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v908_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1052_sw, v1052_sw))))))));
          tensorforge::VectorT<float, 16> v1062_acc{};
          tensorforge::VectorT<float, 16> v1065_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v850_data, v247_data, v1062_acc, 1, 0, 0);
          float v1066_el = v1065_acc[0];
          float v1068_el = v1065_acc[4];
          float v1069_sw = tensorforge::swap<32>(v1068_el);
          float v1071_el = v1065_acc[8];
          float v1074_el = v1065_acc[12];
          float v1075_sw = tensorforge::swap<32>(v1074_el);
          r1[65] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1075_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1071_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1069_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1066_el, v1066_el))))))));
          float v1078_el = v1065_acc[1];
          float v1080_el = v1065_acc[5];
          float v1081_sw = tensorforge::swap<32>(v1080_el);
          float v1083_el = v1065_acc[9];
          float v1086_el = v1065_acc[13];
          float v1087_sw = tensorforge::swap<32>(v1086_el);
          r1[67] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1087_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1083_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1081_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1078_el, v1078_el))))))));
          float v1090_el = v1065_acc[2];
          float v1092_el = v1065_acc[6];
          float v1093_sw = tensorforge::swap<32>(v1092_el);
          float v1095_el = v1065_acc[10];
          float v1098_el = v1065_acc[14];
          float v1099_sw = tensorforge::swap<32>(v1098_el);
          r1[69] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1099_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1095_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1093_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1090_el, v1090_el))))))));
          float v1102_el = v1065_acc[3];
          float v1104_el = v1065_acc[7];
          float v1105_sw = tensorforge::swap<32>(v1104_el);
          float v1107_el = v1065_acc[11];
          float v1110_el = v1065_acc[15];
          float v1111_sw = tensorforge::swap<32>(v1110_el);
          r1[71] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1111_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1107_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1105_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1102_el, v1102_el))))))));
          float v1115_sw = tensorforge::swap<32>(v1066_el);
          float v1120_sw = tensorforge::swap<32>(v1071_el);
          r1[73] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1074_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1120_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1068_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1115_sw, v1115_sw))))))));
          float v1127_sw = tensorforge::swap<32>(v1078_el);
          float v1132_sw = tensorforge::swap<32>(v1083_el);
          r1[75] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1086_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1132_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1080_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1127_sw, v1127_sw))))))));
          float v1139_sw = tensorforge::swap<32>(v1090_el);
          float v1144_sw = tensorforge::swap<32>(v1095_el);
          r1[77] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1098_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1144_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1092_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1139_sw, v1139_sw))))))));
          float v1151_sw = tensorforge::swap<32>(v1102_el);
          float v1156_sw = tensorforge::swap<32>(v1107_el);
          r1[79] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1110_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1156_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1104_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1151_sw, v1151_sw))))))));
          float v1163_sw = tensorforge::swap<64>(v1066_el);
          r1[81] = (tensorforge::dppUpdate<228, 8, 15, false>(v1075_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1071_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1069_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1163_sw, v1163_sw))))))));
          float v1175_sw = tensorforge::swap<64>(v1078_el);
          r1[83] = (tensorforge::dppUpdate<228, 8, 15, false>(v1087_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1083_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1081_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1175_sw, v1175_sw))))))));
          float v1187_sw = tensorforge::swap<64>(v1090_el);
          r1[85] = (tensorforge::dppUpdate<228, 8, 15, false>(v1099_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1095_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1093_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1187_sw, v1187_sw))))))));
          float v1199_sw = tensorforge::swap<64>(v1102_el);
          r1[87] = (tensorforge::dppUpdate<228, 8, 15, false>(v1111_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1107_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1105_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1199_sw, v1199_sw))))))));
          float v1212_sw = tensorforge::swap<64>(v1115_sw);
          r1[89] = (tensorforge::dppUpdate<228, 8, 15, false>(v1074_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1120_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1068_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1212_sw, v1212_sw))))))));
          float v1224_sw = tensorforge::swap<64>(v1127_sw);
          r1[91] = (tensorforge::dppUpdate<228, 8, 15, false>(v1086_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1132_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1080_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1224_sw, v1224_sw))))))));
          float v1236_sw = tensorforge::swap<64>(v1139_sw);
          r1[93] = (tensorforge::dppUpdate<228, 8, 15, false>(v1098_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1144_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1092_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1236_sw, v1236_sw))))))));
          float v1248_sw = tensorforge::swap<64>(v1151_sw);
          r1[95] = (tensorforge::dppUpdate<228, 8, 15, false>(v1110_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1156_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1104_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1248_sw, v1248_sw))))))));
          float v1258_data = glb_m1[3];
          float v1259_data = glb_m1[3];
          float v1260_data = glb_m1[3];
          float v1261_data = glb_m1[3];
          float v1262_data = glb_m1[4];
          float v1263_data = glb_m1[4];
          float v1264_data = glb_m1[4];
          float v1265_data = glb_m1[4];
          float v1266_data = glb_m1[4];
          float v1267_data = glb_m1[4];
          float v1268_data = glb_m1[4];
          float v1269_data = glb_m1[4];
          float v1270_data = glb_m1[4];
          float v1271_data = glb_m1[4];
          float v1272_data = glb_m1[4];
          float v1273_data = glb_m1[4];
          tensorforge::transpose16x16b32(v1258_data, v1259_data, v1260_data, v1261_data, v1262_data, v1263_data, v1264_data, v1265_data, v1266_data, v1267_data, v1268_data, v1269_data, v1270_data, v1271_data, v1272_data, v1273_data);
          tensorforge::VectorT<float, 16> v1274_acc{};
          tensorforge::VectorT<float, 16> v1277_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1258_data, v51_data, v1274_acc, 1, 0, 0);
          float v1278_el = v1277_acc[0];
          float v1280_el = v1277_acc[4];
          float v1281_sw = tensorforge::swap<32>(v1280_el);
          float v1283_el = v1277_acc[8];
          float v1286_el = v1277_acc[12];
          float v1287_sw = tensorforge::swap<32>(v1286_el);
          r1[96] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1287_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1283_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1281_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1278_el, v1278_el))))))));
          float v1290_el = v1277_acc[1];
          float v1292_el = v1277_acc[5];
          float v1293_sw = tensorforge::swap<32>(v1292_el);
          float v1295_el = v1277_acc[9];
          float v1298_el = v1277_acc[13];
          float v1299_sw = tensorforge::swap<32>(v1298_el);
          r1[98] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1299_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1295_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1293_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1290_el, v1290_el))))))));
          float v1302_el = v1277_acc[2];
          float v1304_el = v1277_acc[6];
          float v1305_sw = tensorforge::swap<32>(v1304_el);
          float v1307_el = v1277_acc[10];
          float v1310_el = v1277_acc[14];
          float v1311_sw = tensorforge::swap<32>(v1310_el);
          r1[100] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1311_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1307_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1305_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1302_el, v1302_el))))))));
          float v1314_el = v1277_acc[3];
          float v1316_el = v1277_acc[7];
          float v1317_sw = tensorforge::swap<32>(v1316_el);
          float v1319_el = v1277_acc[11];
          float v1322_el = v1277_acc[15];
          float v1323_sw = tensorforge::swap<32>(v1322_el);
          r1[102] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1323_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1319_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1317_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1314_el, v1314_el))))))));
          float v1327_sw = tensorforge::swap<32>(v1278_el);
          float v1332_sw = tensorforge::swap<32>(v1283_el);
          r1[104] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1286_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1332_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1280_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1327_sw, v1327_sw))))))));
          float v1339_sw = tensorforge::swap<32>(v1290_el);
          float v1344_sw = tensorforge::swap<32>(v1295_el);
          r1[106] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1298_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1344_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1292_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1339_sw, v1339_sw))))))));
          float v1351_sw = tensorforge::swap<32>(v1302_el);
          float v1356_sw = tensorforge::swap<32>(v1307_el);
          r1[108] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1310_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1356_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1304_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1351_sw, v1351_sw))))))));
          float v1363_sw = tensorforge::swap<32>(v1314_el);
          float v1368_sw = tensorforge::swap<32>(v1319_el);
          r1[110] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1322_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1368_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1316_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1363_sw, v1363_sw))))))));
          float v1375_sw = tensorforge::swap<64>(v1278_el);
          r1[112] = (tensorforge::dppUpdate<228, 8, 15, false>(v1287_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1283_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1281_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1375_sw, v1375_sw))))))));
          float v1387_sw = tensorforge::swap<64>(v1290_el);
          r1[114] = (tensorforge::dppUpdate<228, 8, 15, false>(v1299_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1295_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1293_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1387_sw, v1387_sw))))))));
          float v1399_sw = tensorforge::swap<64>(v1302_el);
          r1[116] = (tensorforge::dppUpdate<228, 8, 15, false>(v1311_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1307_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1305_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1399_sw, v1399_sw))))))));
          float v1411_sw = tensorforge::swap<64>(v1314_el);
          r1[118] = (tensorforge::dppUpdate<228, 8, 15, false>(v1323_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1319_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1317_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1411_sw, v1411_sw))))))));
          float v1424_sw = tensorforge::swap<64>(v1327_sw);
          r1[120] = (tensorforge::dppUpdate<228, 8, 15, false>(v1286_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1332_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1280_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1424_sw, v1424_sw))))))));
          float v1436_sw = tensorforge::swap<64>(v1339_sw);
          r1[122] = (tensorforge::dppUpdate<228, 8, 15, false>(v1298_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1344_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1292_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1436_sw, v1436_sw))))))));
          float v1448_sw = tensorforge::swap<64>(v1351_sw);
          r1[124] = (tensorforge::dppUpdate<228, 8, 15, false>(v1310_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1356_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1304_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1448_sw, v1448_sw))))))));
          float v1460_sw = tensorforge::swap<64>(v1363_sw);
          r1[126] = (tensorforge::dppUpdate<228, 8, 15, false>(v1322_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1368_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1316_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1460_sw, v1460_sw))))))));
          tensorforge::VectorT<float, 16> v1470_acc{};
          tensorforge::VectorT<float, 16> v1473_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1258_data, v247_data, v1470_acc, 1, 0, 0);
          float v1474_el = v1473_acc[0];
          float v1476_el = v1473_acc[4];
          float v1477_sw = tensorforge::swap<32>(v1476_el);
          float v1479_el = v1473_acc[8];
          float v1482_el = v1473_acc[12];
          float v1483_sw = tensorforge::swap<32>(v1482_el);
          r1[97] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1483_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1479_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1477_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1474_el, v1474_el))))))));
          float v1486_el = v1473_acc[1];
          float v1488_el = v1473_acc[5];
          float v1489_sw = tensorforge::swap<32>(v1488_el);
          float v1491_el = v1473_acc[9];
          float v1494_el = v1473_acc[13];
          float v1495_sw = tensorforge::swap<32>(v1494_el);
          r1[99] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1495_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1491_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1489_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1486_el, v1486_el))))))));
          float v1498_el = v1473_acc[2];
          float v1500_el = v1473_acc[6];
          float v1501_sw = tensorforge::swap<32>(v1500_el);
          float v1503_el = v1473_acc[10];
          float v1506_el = v1473_acc[14];
          float v1507_sw = tensorforge::swap<32>(v1506_el);
          r1[101] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1507_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1503_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1501_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1498_el, v1498_el))))))));
          float v1510_el = v1473_acc[3];
          float v1512_el = v1473_acc[7];
          float v1513_sw = tensorforge::swap<32>(v1512_el);
          float v1515_el = v1473_acc[11];
          float v1518_el = v1473_acc[15];
          float v1519_sw = tensorforge::swap<32>(v1518_el);
          r1[103] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1519_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1515_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1513_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1510_el, v1510_el))))))));
          float v1523_sw = tensorforge::swap<32>(v1474_el);
          float v1528_sw = tensorforge::swap<32>(v1479_el);
          r1[105] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1482_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1528_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1476_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1523_sw, v1523_sw))))))));
          float v1535_sw = tensorforge::swap<32>(v1486_el);
          float v1540_sw = tensorforge::swap<32>(v1491_el);
          r1[107] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1494_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1540_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1488_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1535_sw, v1535_sw))))))));
          float v1547_sw = tensorforge::swap<32>(v1498_el);
          float v1552_sw = tensorforge::swap<32>(v1503_el);
          r1[109] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1506_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1552_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1500_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1547_sw, v1547_sw))))))));
          float v1559_sw = tensorforge::swap<32>(v1510_el);
          float v1564_sw = tensorforge::swap<32>(v1515_el);
          r1[111] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1518_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1564_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1512_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1559_sw, v1559_sw))))))));
          float v1571_sw = tensorforge::swap<64>(v1474_el);
          r1[113] = (tensorforge::dppUpdate<228, 8, 15, false>(v1483_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1479_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1477_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1571_sw, v1571_sw))))))));
          float v1583_sw = tensorforge::swap<64>(v1486_el);
          r1[115] = (tensorforge::dppUpdate<228, 8, 15, false>(v1495_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1491_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1489_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1583_sw, v1583_sw))))))));
          float v1595_sw = tensorforge::swap<64>(v1498_el);
          r1[117] = (tensorforge::dppUpdate<228, 8, 15, false>(v1507_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1503_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1501_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1595_sw, v1595_sw))))))));
          float v1607_sw = tensorforge::swap<64>(v1510_el);
          r1[119] = (tensorforge::dppUpdate<228, 8, 15, false>(v1519_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1515_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1513_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1607_sw, v1607_sw))))))));
          float v1620_sw = tensorforge::swap<64>(v1523_sw);
          r1[121] = (tensorforge::dppUpdate<228, 8, 15, false>(v1482_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1528_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1476_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1620_sw, v1620_sw))))))));
          float v1632_sw = tensorforge::swap<64>(v1535_sw);
          r1[123] = (tensorforge::dppUpdate<228, 8, 15, false>(v1494_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1540_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1488_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1632_sw, v1632_sw))))))));
          float v1644_sw = tensorforge::swap<64>(v1547_sw);
          r1[125] = (tensorforge::dppUpdate<228, 8, 15, false>(v1506_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1552_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1500_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1644_sw, v1644_sw))))))));
          float v1656_sw = tensorforge::swap<64>(v1559_sw);
          r1[127] = (tensorforge::dppUpdate<228, 8, 15, false>(v1518_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1564_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1512_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1656_sw, v1656_sw))))))));
          float v1666_data = glb_m1[4];
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
          float v1677_data = glb_m1[5];
          float v1678_data = glb_m1[5];
          float v1679_data = glb_m1[5];
          float v1680_pad{};
          float v1681_pad{};
          tensorforge::transpose16x16b32(v1666_data, v1667_data, v1668_data, v1669_data, v1670_data, v1671_data, v1672_data, v1673_data, v1674_data, v1675_data, v1676_data, v1677_data, v1678_data, v1679_data, v1680_pad, v1681_pad);
          tensorforge::VectorT<float, 16> v1682_acc{};
          tensorforge::VectorT<float, 16> v1685_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1666_data, v51_data, v1682_acc, 1, 0, 0);
          float v1686_el = v1685_acc[0];
          float v1688_el = v1685_acc[4];
          float v1689_sw = tensorforge::swap<32>(v1688_el);
          float v1691_el = v1685_acc[8];
          float v1694_el = v1685_acc[12];
          float v1695_sw = tensorforge::swap<32>(v1694_el);
          r1[128] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1695_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1691_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1689_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1686_el, v1686_el))))))));
          float v1698_el = v1685_acc[1];
          float v1700_el = v1685_acc[5];
          float v1701_sw = tensorforge::swap<32>(v1700_el);
          float v1703_el = v1685_acc[9];
          float v1706_el = v1685_acc[13];
          float v1707_sw = tensorforge::swap<32>(v1706_el);
          r1[130] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1707_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1703_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1701_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1698_el, v1698_el))))))));
          float v1710_el = v1685_acc[2];
          float v1712_el = v1685_acc[6];
          float v1713_sw = tensorforge::swap<32>(v1712_el);
          float v1715_el = v1685_acc[10];
          float v1718_el = v1685_acc[14];
          float v1719_sw = tensorforge::swap<32>(v1718_el);
          r1[132] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1719_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1715_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1713_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1710_el, v1710_el))))))));
          float v1722_el = v1685_acc[3];
          float v1724_el = v1685_acc[7];
          float v1725_sw = tensorforge::swap<32>(v1724_el);
          float v1727_el = v1685_acc[11];
          float v1730_el = v1685_acc[15];
          float v1731_sw = tensorforge::swap<32>(v1730_el);
          r1[134] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1731_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1727_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1725_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1722_el, v1722_el))))))));
          float v1735_sw = tensorforge::swap<32>(v1686_el);
          float v1740_sw = tensorforge::swap<32>(v1691_el);
          r1[136] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1694_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1740_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1688_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1735_sw, v1735_sw))))))));
          float v1747_sw = tensorforge::swap<32>(v1698_el);
          float v1752_sw = tensorforge::swap<32>(v1703_el);
          r1[138] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1706_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1752_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1700_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1747_sw, v1747_sw))))))));
          float v1759_sw = tensorforge::swap<32>(v1710_el);
          r1[140] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1718_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1715_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1712_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1759_sw, v1759_sw))))))));
          float v1771_sw = tensorforge::swap<32>(v1722_el);
          r1[142] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1730_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1727_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1724_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1771_sw, v1771_sw))))))));
          float v1783_sw = tensorforge::swap<64>(v1686_el);
          r1[144] = (tensorforge::dppUpdate<228, 8, 15, false>(v1695_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1691_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1689_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1783_sw, v1783_sw))))))));
          float v1795_sw = tensorforge::swap<64>(v1698_el);
          r1[146] = (tensorforge::dppUpdate<228, 8, 15, false>(v1707_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1703_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1701_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1795_sw, v1795_sw))))))));
          float v1807_sw = tensorforge::swap<64>(v1710_el);
          r1[148] = (tensorforge::dppUpdate<228, 8, 15, false>(v1719_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1715_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1713_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1807_sw, v1807_sw))))))));
          float v1819_sw = tensorforge::swap<64>(v1722_el);
          r1[150] = (tensorforge::dppUpdate<228, 8, 15, false>(v1731_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1727_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1725_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1819_sw, v1819_sw))))))));
          float v1832_sw = tensorforge::swap<64>(v1735_sw);
          r1[152] = (tensorforge::dppUpdate<228, 8, 15, false>(v1694_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1740_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1688_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1832_sw, v1832_sw))))))));
          float v1844_sw = tensorforge::swap<64>(v1747_sw);
          r1[154] = (tensorforge::dppUpdate<228, 8, 15, false>(v1706_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1752_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1700_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1844_sw, v1844_sw))))))));
          tensorforge::VectorT<float, 16> v1854_acc{};
          tensorforge::VectorT<float, 16> v1857_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1666_data, v247_data, v1854_acc, 1, 0, 0);
          float v1858_el = v1857_acc[0];
          float v1860_el = v1857_acc[4];
          float v1861_sw = tensorforge::swap<32>(v1860_el);
          float v1863_el = v1857_acc[8];
          float v1866_el = v1857_acc[12];
          float v1867_sw = tensorforge::swap<32>(v1866_el);
          r1[129] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1867_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1863_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1861_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1858_el, v1858_el))))))));
          float v1870_el = v1857_acc[1];
          float v1872_el = v1857_acc[5];
          float v1873_sw = tensorforge::swap<32>(v1872_el);
          float v1875_el = v1857_acc[9];
          float v1878_el = v1857_acc[13];
          float v1879_sw = tensorforge::swap<32>(v1878_el);
          r1[131] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1879_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1875_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1873_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1870_el, v1870_el))))))));
          float v1882_el = v1857_acc[2];
          float v1884_el = v1857_acc[6];
          float v1885_sw = tensorforge::swap<32>(v1884_el);
          float v1887_el = v1857_acc[10];
          float v1890_el = v1857_acc[14];
          float v1891_sw = tensorforge::swap<32>(v1890_el);
          r1[133] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1891_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1887_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1885_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1882_el, v1882_el))))))));
          float v1894_el = v1857_acc[3];
          float v1896_el = v1857_acc[7];
          float v1897_sw = tensorforge::swap<32>(v1896_el);
          float v1899_el = v1857_acc[11];
          float v1902_el = v1857_acc[15];
          float v1903_sw = tensorforge::swap<32>(v1902_el);
          r1[135] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1903_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1899_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1897_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1894_el, v1894_el))))))));
          float v1907_sw = tensorforge::swap<32>(v1858_el);
          float v1912_sw = tensorforge::swap<32>(v1863_el);
          r1[137] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1866_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1912_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1860_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1907_sw, v1907_sw))))))));
          float v1919_sw = tensorforge::swap<32>(v1870_el);
          float v1924_sw = tensorforge::swap<32>(v1875_el);
          r1[139] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1878_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1924_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1872_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1919_sw, v1919_sw))))))));
          float v1931_sw = tensorforge::swap<32>(v1882_el);
          r1[141] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1890_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1887_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1884_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1931_sw, v1931_sw))))))));
          float v1943_sw = tensorforge::swap<32>(v1894_el);
          r1[143] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1902_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1899_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1896_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1943_sw, v1943_sw))))))));
          float v1955_sw = tensorforge::swap<64>(v1858_el);
          r1[145] = (tensorforge::dppUpdate<228, 8, 15, false>(v1867_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1863_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1861_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1955_sw, v1955_sw))))))));
          float v1967_sw = tensorforge::swap<64>(v1870_el);
          r1[147] = (tensorforge::dppUpdate<228, 8, 15, false>(v1879_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1875_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1873_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1967_sw, v1967_sw))))))));
          float v1979_sw = tensorforge::swap<64>(v1882_el);
          r1[149] = (tensorforge::dppUpdate<228, 8, 15, false>(v1891_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1887_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1885_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1979_sw, v1979_sw))))))));
          float v1991_sw = tensorforge::swap<64>(v1894_el);
          r1[151] = (tensorforge::dppUpdate<228, 8, 15, false>(v1903_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1899_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1897_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1991_sw, v1991_sw))))))));
          float v2004_sw = tensorforge::swap<64>(v1907_sw);
          r1[153] = (tensorforge::dppUpdate<228, 8, 15, false>(v1866_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1912_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1860_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v2004_sw, v2004_sw))))))));
          float v2016_sw = tensorforge::swap<64>(v1919_sw);
          r1[155] = (tensorforge::dppUpdate<228, 8, 15, false>(v1878_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1924_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1872_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v2016_sw, v2016_sw))))))));
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          bool v2027_g = v23_lead >= 20;
          if (v2027_g) {
            float v2028_data = r1[24];
            float v2029_data = r2[0];
            r2[0] = (v2029_data + v2028_data);
            float v2031_data = r1[50];
            float v2032_data = r2[2];
            r2[2] = (v2032_data + v2031_data);
            float v2034_data = r1[76];
            float v2035_data = r2[4];
            r2[4] = (v2035_data + v2034_data);
            float v2037_data = r1[102];
            float v2038_data = r2[6];
            r2[6] = (v2038_data + v2037_data);
            float v2040_data = r1[128];
            float v2041_data = r2[8];
            r2[8] = (v2041_data + v2040_data);
            float v2043_data = r1[154];
            float v2044_data = r2[10];
            r2[10] = (v2044_data + v2043_data);
          }
          bool v2046_g = v23_lead < 3;
          if (v2046_g) {
            float v2047_data = r1[25];
            float v2048_data = r2[1];
            r2[1] = (v2048_data + v2047_data);
            float v2050_data = r1[51];
            float v2051_data = r2[3];
            r2[3] = (v2051_data + v2050_data);
            float v2053_data = r1[77];
            float v2054_data = r2[5];
            r2[5] = (v2054_data + v2053_data);
            float v2056_data = r1[103];
            float v2057_data = r2[7];
            r2[7] = (v2057_data + v2056_data);
            float v2059_data = r1[129];
            float v2060_data = r2[9];
            r2[9] = (v2060_data + v2059_data);
            float v2062_data = r1[155];
            float v2063_data = r2[11];
            r2[11] = (v2063_data + v2062_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v2027_g) {
            #pragma unroll
            for (int32_t v2066_i1 = 0; v2066_i1 < 1; ++v2066_i1) {
              int32_t v2068_a = v2066_i1 * 2;
              int32_t v2078_a = v23_lead + ((v2066_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v2067_i2 = 0; v2067_i2 < 6; ++v2067_i2) {
                float v2072_data = r2[(v2068_a + (v2067_i2 * 2))];
                int32_t v2079_a = v2078_a + (v2067_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v2079_a], v2072_data);
              }
            }
          }
          if (v2046_g) {
            int32_t v2089_lead = v23_lead + 32_i32;
            #pragma unroll
            for (int32_t v2081_i1 = 0; v2081_i1 < 1; ++v2081_i1) {
              int32_t v2085_a = 1 + (v2081_i1 * 2);
              int32_t v2093_a = v2089_lead + ((v2081_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v2082_i2 = 0; v2082_i2 < 6; ++v2082_i2) {
                float v2087_data = r2[(v2085_a + (v2082_i2 * 2))];
                int32_t v2094_a = v2093_a + (v2082_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v2094_a], v2087_data);
              }
            }
          }
        }
      }
    }
  }
}

