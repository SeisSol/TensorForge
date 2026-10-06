// === base name ===
kernel_e5657c6226e3d519

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e5657c6226e3d519 = {{32, 8, 1}, 32, 64, 1, 8, 256, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e5657c6226e3d519(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e5657c6226e3d519(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e5657c6226e3d519(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e5657c6226e3d519, block.x * block.y * block.z, 64 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (64 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e5657c6226e3d519, block.x * block.y * block.z, 0));
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
void launcher_kernel_e5657c6226e3d519(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e5657c6226e3d519(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e5657c6226e3d519), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_e5657c6226e3d519, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1Arg, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e5657c6226e3d519(const float ** m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, float ** m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 6) {
        float v12_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      }
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0]));
      __syncthreads();
      for (size_t v13_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v13_batchId0 < numElements0; v13_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v14_ahead1 = v13_batchId0 + (gridDim.x * blockDim.y);
        size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v13_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v13_batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v27_i0 = 0; v27_i0 < 2; ++v27_i0) {
            int32_t v30_lead = v26_lead + (v27_i0 * 32);
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 13; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m0[(v30_lead + (v28_i1 * 64))]);
              r0[(v27_i0 + (v28_i1 * 2))] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
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
          float v47_data = glb_m1[0];
          float v48_data = glb_m1[0];
          float v49_data = glb_m1[0];
          float v50_data = glb_m1[1];
          float v51_data = glb_m1[1];
          float v52_data = glb_m1[1];
          tensorforge::transpose16x16b32(v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data);
          tensorforge::VectorT<float, 16> v53_acc{};
          float v54_data = r0[0];
          tensorforge::VectorT<float, 16> v56_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v37_data, v54_data, v53_acc, 1, 0, 0);
          float v57_el = v56_acc[0];
          float v59_el = v56_acc[4];
          float v60_sw = tensorforge::swap<32>(v59_el);
          float v62_el = v56_acc[8];
          float v65_el = v56_acc[12];
          float v66_sw = tensorforge::swap<32>(v65_el);
          r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v66_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v62_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v60_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v57_el, v57_el))))))));
          float v69_el = v56_acc[1];
          float v71_el = v56_acc[5];
          float v72_sw = tensorforge::swap<32>(v71_el);
          float v74_el = v56_acc[9];
          float v77_el = v56_acc[13];
          float v78_sw = tensorforge::swap<32>(v77_el);
          r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v78_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v74_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v72_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v69_el, v69_el))))))));
          float v81_el = v56_acc[2];
          float v83_el = v56_acc[6];
          float v84_sw = tensorforge::swap<32>(v83_el);
          float v86_el = v56_acc[10];
          float v89_el = v56_acc[14];
          float v90_sw = tensorforge::swap<32>(v89_el);
          r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v90_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v86_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v84_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v81_el, v81_el))))))));
          float v93_el = v56_acc[3];
          float v95_el = v56_acc[7];
          float v96_sw = tensorforge::swap<32>(v95_el);
          float v98_el = v56_acc[11];
          float v101_el = v56_acc[15];
          float v102_sw = tensorforge::swap<32>(v101_el);
          r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v102_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v98_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v96_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v93_el, v93_el))))))));
          float v106_sw = tensorforge::swap<32>(v57_el);
          float v111_sw = tensorforge::swap<32>(v62_el);
          r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v65_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v111_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v59_el, (tensorforge::dppUpdate<228, 1, 15, false>(v106_sw, v106_sw))))))));
          float v118_sw = tensorforge::swap<32>(v69_el);
          float v123_sw = tensorforge::swap<32>(v74_el);
          r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v77_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v123_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v71_el, (tensorforge::dppUpdate<228, 1, 15, false>(v118_sw, v118_sw))))))));
          float v130_sw = tensorforge::swap<32>(v81_el);
          float v135_sw = tensorforge::swap<32>(v86_el);
          r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v89_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v135_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v83_el, (tensorforge::dppUpdate<228, 1, 15, false>(v130_sw, v130_sw))))))));
          float v142_sw = tensorforge::swap<32>(v93_el);
          float v147_sw = tensorforge::swap<32>(v98_el);
          r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v101_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v147_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v95_el, (tensorforge::dppUpdate<228, 1, 15, false>(v142_sw, v142_sw))))))));
          float v154_sw = tensorforge::swap<64>(v57_el);
          r1[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v66_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v62_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v60_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v154_sw, v154_sw))))))));
          float v166_sw = tensorforge::swap<64>(v69_el);
          r1[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v78_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v74_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v72_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v166_sw, v166_sw))))))));
          float v178_sw = tensorforge::swap<64>(v81_el);
          r1[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v90_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v86_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v84_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v178_sw, v178_sw))))))));
          float v190_sw = tensorforge::swap<64>(v93_el);
          r1[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v102_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v98_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v96_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v190_sw, v190_sw))))))));
          float v203_sw = tensorforge::swap<64>(v106_sw);
          r1[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v65_el, (tensorforge::dppUpdate<228, 4, 15, false>(v111_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v59_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v203_sw, v203_sw))))))));
          float v215_sw = tensorforge::swap<64>(v118_sw);
          r1[26] = (tensorforge::dppUpdate<228, 8, 15, false>(v77_el, (tensorforge::dppUpdate<228, 4, 15, false>(v123_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v71_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v215_sw, v215_sw))))))));
          float v227_sw = tensorforge::swap<64>(v130_sw);
          r1[28] = (tensorforge::dppUpdate<228, 8, 15, false>(v89_el, (tensorforge::dppUpdate<228, 4, 15, false>(v135_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v83_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v227_sw, v227_sw))))))));
          float v239_sw = tensorforge::swap<64>(v142_sw);
          r1[30] = (tensorforge::dppUpdate<228, 8, 15, false>(v101_el, (tensorforge::dppUpdate<228, 4, 15, false>(v147_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v95_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v239_sw, v239_sw))))))));
          tensorforge::VectorT<float, 16> v249_acc{};
          float v250_data = r0[1];
          tensorforge::VectorT<float, 16> v252_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v37_data, v250_data, v249_acc, 1, 0, 0);
          float v253_el = v252_acc[0];
          float v255_el = v252_acc[4];
          float v256_sw = tensorforge::swap<32>(v255_el);
          float v258_el = v252_acc[8];
          float v261_el = v252_acc[12];
          float v262_sw = tensorforge::swap<32>(v261_el);
          r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v262_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v258_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v256_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v253_el, v253_el))))))));
          float v265_el = v252_acc[1];
          float v267_el = v252_acc[5];
          float v268_sw = tensorforge::swap<32>(v267_el);
          float v270_el = v252_acc[9];
          float v273_el = v252_acc[13];
          float v274_sw = tensorforge::swap<32>(v273_el);
          r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v274_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v270_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v268_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v265_el, v265_el))))))));
          float v277_el = v252_acc[2];
          float v279_el = v252_acc[6];
          float v280_sw = tensorforge::swap<32>(v279_el);
          float v282_el = v252_acc[10];
          float v285_el = v252_acc[14];
          float v286_sw = tensorforge::swap<32>(v285_el);
          r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v286_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v282_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v280_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v277_el, v277_el))))))));
          float v289_el = v252_acc[3];
          float v291_el = v252_acc[7];
          float v292_sw = tensorforge::swap<32>(v291_el);
          float v294_el = v252_acc[11];
          float v297_el = v252_acc[15];
          float v298_sw = tensorforge::swap<32>(v297_el);
          r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v298_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v294_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v292_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v289_el, v289_el))))))));
          float v302_sw = tensorforge::swap<32>(v253_el);
          float v307_sw = tensorforge::swap<32>(v258_el);
          r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v261_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v307_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v255_el, (tensorforge::dppUpdate<228, 1, 15, false>(v302_sw, v302_sw))))))));
          float v314_sw = tensorforge::swap<32>(v265_el);
          float v319_sw = tensorforge::swap<32>(v270_el);
          r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v273_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v319_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v267_el, (tensorforge::dppUpdate<228, 1, 15, false>(v314_sw, v314_sw))))))));
          float v326_sw = tensorforge::swap<32>(v277_el);
          float v331_sw = tensorforge::swap<32>(v282_el);
          r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v285_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v331_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v279_el, (tensorforge::dppUpdate<228, 1, 15, false>(v326_sw, v326_sw))))))));
          float v338_sw = tensorforge::swap<32>(v289_el);
          float v343_sw = tensorforge::swap<32>(v294_el);
          r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v297_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v343_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v291_el, (tensorforge::dppUpdate<228, 1, 15, false>(v338_sw, v338_sw))))))));
          float v350_sw = tensorforge::swap<64>(v253_el);
          r1[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v262_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v258_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v256_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v350_sw, v350_sw))))))));
          float v362_sw = tensorforge::swap<64>(v265_el);
          r1[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v274_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v270_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v268_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v362_sw, v362_sw))))))));
          float v374_sw = tensorforge::swap<64>(v277_el);
          r1[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v286_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v282_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v280_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v374_sw, v374_sw))))))));
          float v386_sw = tensorforge::swap<64>(v289_el);
          r1[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v298_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v294_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v292_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v386_sw, v386_sw))))))));
          float v399_sw = tensorforge::swap<64>(v302_sw);
          r1[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v261_el, (tensorforge::dppUpdate<228, 4, 15, false>(v307_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v255_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v399_sw, v399_sw))))))));
          float v411_sw = tensorforge::swap<64>(v314_sw);
          r1[27] = (tensorforge::dppUpdate<228, 8, 15, false>(v273_el, (tensorforge::dppUpdate<228, 4, 15, false>(v319_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v267_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v411_sw, v411_sw))))))));
          float v423_sw = tensorforge::swap<64>(v326_sw);
          r1[29] = (tensorforge::dppUpdate<228, 8, 15, false>(v285_el, (tensorforge::dppUpdate<228, 4, 15, false>(v331_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v279_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v423_sw, v423_sw))))))));
          float v435_sw = tensorforge::swap<64>(v338_sw);
          r1[31] = (tensorforge::dppUpdate<228, 8, 15, false>(v297_el, (tensorforge::dppUpdate<228, 4, 15, false>(v343_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v291_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v435_sw, v435_sw))))))));
          float v445_data = glb_m1[1];
          float v446_data = glb_m1[1];
          float v447_data = glb_m1[1];
          float v448_data = glb_m1[1];
          float v449_data = glb_m1[1];
          float v450_data = glb_m1[1];
          float v451_data = glb_m1[1];
          float v452_data = glb_m1[1];
          float v453_data = glb_m1[1];
          float v454_data = glb_m1[1];
          float v455_data = glb_m1[2];
          float v456_data = glb_m1[2];
          float v457_data = glb_m1[2];
          float v458_data = glb_m1[2];
          float v459_data = glb_m1[2];
          float v460_data = glb_m1[2];
          tensorforge::transpose16x16b32(v445_data, v446_data, v447_data, v448_data, v449_data, v450_data, v451_data, v452_data, v453_data, v454_data, v455_data, v456_data, v457_data, v458_data, v459_data, v460_data);
          tensorforge::VectorT<float, 16> v461_acc{};
          tensorforge::VectorT<float, 16> v464_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v445_data, v54_data, v461_acc, 1, 0, 0);
          float v465_el = v464_acc[0];
          float v467_el = v464_acc[4];
          float v468_sw = tensorforge::swap<32>(v467_el);
          float v470_el = v464_acc[8];
          float v473_el = v464_acc[12];
          float v474_sw = tensorforge::swap<32>(v473_el);
          r1[32] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v474_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v470_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v468_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v465_el, v465_el))))))));
          float v477_el = v464_acc[1];
          float v479_el = v464_acc[5];
          float v480_sw = tensorforge::swap<32>(v479_el);
          float v482_el = v464_acc[9];
          float v485_el = v464_acc[13];
          float v486_sw = tensorforge::swap<32>(v485_el);
          r1[34] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v486_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v482_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v480_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v477_el, v477_el))))))));
          float v489_el = v464_acc[2];
          float v491_el = v464_acc[6];
          float v492_sw = tensorforge::swap<32>(v491_el);
          float v494_el = v464_acc[10];
          float v497_el = v464_acc[14];
          float v498_sw = tensorforge::swap<32>(v497_el);
          r1[36] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v498_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v494_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v492_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v489_el, v489_el))))))));
          float v501_el = v464_acc[3];
          float v503_el = v464_acc[7];
          float v504_sw = tensorforge::swap<32>(v503_el);
          float v506_el = v464_acc[11];
          float v509_el = v464_acc[15];
          float v510_sw = tensorforge::swap<32>(v509_el);
          r1[38] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v510_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v506_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v504_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v501_el, v501_el))))))));
          float v514_sw = tensorforge::swap<32>(v465_el);
          float v519_sw = tensorforge::swap<32>(v470_el);
          r1[40] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v473_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v519_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v467_el, (tensorforge::dppUpdate<228, 1, 15, false>(v514_sw, v514_sw))))))));
          float v526_sw = tensorforge::swap<32>(v477_el);
          float v531_sw = tensorforge::swap<32>(v482_el);
          r1[42] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v485_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v531_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v479_el, (tensorforge::dppUpdate<228, 1, 15, false>(v526_sw, v526_sw))))))));
          float v538_sw = tensorforge::swap<32>(v489_el);
          float v543_sw = tensorforge::swap<32>(v494_el);
          r1[44] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v497_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v543_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v491_el, (tensorforge::dppUpdate<228, 1, 15, false>(v538_sw, v538_sw))))))));
          float v550_sw = tensorforge::swap<32>(v501_el);
          float v555_sw = tensorforge::swap<32>(v506_el);
          r1[46] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v509_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v555_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v503_el, (tensorforge::dppUpdate<228, 1, 15, false>(v550_sw, v550_sw))))))));
          float v562_sw = tensorforge::swap<64>(v465_el);
          r1[48] = (tensorforge::dppUpdate<228, 8, 15, false>(v474_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v470_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v468_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v562_sw, v562_sw))))))));
          float v574_sw = tensorforge::swap<64>(v477_el);
          r1[50] = (tensorforge::dppUpdate<228, 8, 15, false>(v486_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v482_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v480_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v574_sw, v574_sw))))))));
          float v586_sw = tensorforge::swap<64>(v489_el);
          r1[52] = (tensorforge::dppUpdate<228, 8, 15, false>(v498_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v494_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v492_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v586_sw, v586_sw))))))));
          float v598_sw = tensorforge::swap<64>(v501_el);
          r1[54] = (tensorforge::dppUpdate<228, 8, 15, false>(v510_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v506_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v504_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v598_sw, v598_sw))))))));
          float v611_sw = tensorforge::swap<64>(v514_sw);
          r1[56] = (tensorforge::dppUpdate<228, 8, 15, false>(v473_el, (tensorforge::dppUpdate<228, 4, 15, false>(v519_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v467_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v611_sw, v611_sw))))))));
          float v623_sw = tensorforge::swap<64>(v526_sw);
          r1[58] = (tensorforge::dppUpdate<228, 8, 15, false>(v485_el, (tensorforge::dppUpdate<228, 4, 15, false>(v531_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v479_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v623_sw, v623_sw))))))));
          float v635_sw = tensorforge::swap<64>(v538_sw);
          r1[60] = (tensorforge::dppUpdate<228, 8, 15, false>(v497_el, (tensorforge::dppUpdate<228, 4, 15, false>(v543_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v491_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v635_sw, v635_sw))))))));
          float v647_sw = tensorforge::swap<64>(v550_sw);
          r1[62] = (tensorforge::dppUpdate<228, 8, 15, false>(v509_el, (tensorforge::dppUpdate<228, 4, 15, false>(v555_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v503_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v647_sw, v647_sw))))))));
          tensorforge::VectorT<float, 16> v657_acc{};
          tensorforge::VectorT<float, 16> v660_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v445_data, v250_data, v657_acc, 1, 0, 0);
          float v661_el = v660_acc[0];
          float v663_el = v660_acc[4];
          float v664_sw = tensorforge::swap<32>(v663_el);
          float v666_el = v660_acc[8];
          float v669_el = v660_acc[12];
          float v670_sw = tensorforge::swap<32>(v669_el);
          r1[33] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v670_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v666_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v664_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v661_el, v661_el))))))));
          float v673_el = v660_acc[1];
          float v675_el = v660_acc[5];
          float v676_sw = tensorforge::swap<32>(v675_el);
          float v678_el = v660_acc[9];
          float v681_el = v660_acc[13];
          float v682_sw = tensorforge::swap<32>(v681_el);
          r1[35] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v682_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v678_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v676_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v673_el, v673_el))))))));
          float v685_el = v660_acc[2];
          float v687_el = v660_acc[6];
          float v688_sw = tensorforge::swap<32>(v687_el);
          float v690_el = v660_acc[10];
          float v693_el = v660_acc[14];
          float v694_sw = tensorforge::swap<32>(v693_el);
          r1[37] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v694_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v690_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v688_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v685_el, v685_el))))))));
          float v697_el = v660_acc[3];
          float v699_el = v660_acc[7];
          float v700_sw = tensorforge::swap<32>(v699_el);
          float v702_el = v660_acc[11];
          float v705_el = v660_acc[15];
          float v706_sw = tensorforge::swap<32>(v705_el);
          r1[39] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v706_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v702_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v700_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v697_el, v697_el))))))));
          float v710_sw = tensorforge::swap<32>(v661_el);
          float v715_sw = tensorforge::swap<32>(v666_el);
          r1[41] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v669_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v715_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v663_el, (tensorforge::dppUpdate<228, 1, 15, false>(v710_sw, v710_sw))))))));
          float v722_sw = tensorforge::swap<32>(v673_el);
          float v727_sw = tensorforge::swap<32>(v678_el);
          r1[43] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v681_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v727_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v675_el, (tensorforge::dppUpdate<228, 1, 15, false>(v722_sw, v722_sw))))))));
          float v734_sw = tensorforge::swap<32>(v685_el);
          float v739_sw = tensorforge::swap<32>(v690_el);
          r1[45] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v693_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v739_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v687_el, (tensorforge::dppUpdate<228, 1, 15, false>(v734_sw, v734_sw))))))));
          float v746_sw = tensorforge::swap<32>(v697_el);
          float v751_sw = tensorforge::swap<32>(v702_el);
          r1[47] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v705_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v751_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v699_el, (tensorforge::dppUpdate<228, 1, 15, false>(v746_sw, v746_sw))))))));
          float v758_sw = tensorforge::swap<64>(v661_el);
          r1[49] = (tensorforge::dppUpdate<228, 8, 15, false>(v670_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v666_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v664_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v758_sw, v758_sw))))))));
          float v770_sw = tensorforge::swap<64>(v673_el);
          r1[51] = (tensorforge::dppUpdate<228, 8, 15, false>(v682_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v678_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v676_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v770_sw, v770_sw))))))));
          float v782_sw = tensorforge::swap<64>(v685_el);
          r1[53] = (tensorforge::dppUpdate<228, 8, 15, false>(v694_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v690_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v688_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v782_sw, v782_sw))))))));
          float v794_sw = tensorforge::swap<64>(v697_el);
          r1[55] = (tensorforge::dppUpdate<228, 8, 15, false>(v706_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v702_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v700_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v794_sw, v794_sw))))))));
          float v807_sw = tensorforge::swap<64>(v710_sw);
          r1[57] = (tensorforge::dppUpdate<228, 8, 15, false>(v669_el, (tensorforge::dppUpdate<228, 4, 15, false>(v715_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v663_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v807_sw, v807_sw))))))));
          float v819_sw = tensorforge::swap<64>(v722_sw);
          r1[59] = (tensorforge::dppUpdate<228, 8, 15, false>(v681_el, (tensorforge::dppUpdate<228, 4, 15, false>(v727_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v675_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v819_sw, v819_sw))))))));
          float v831_sw = tensorforge::swap<64>(v734_sw);
          r1[61] = (tensorforge::dppUpdate<228, 8, 15, false>(v693_el, (tensorforge::dppUpdate<228, 4, 15, false>(v739_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v687_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v831_sw, v831_sw))))))));
          float v843_sw = tensorforge::swap<64>(v746_sw);
          r1[63] = (tensorforge::dppUpdate<228, 8, 15, false>(v705_el, (tensorforge::dppUpdate<228, 4, 15, false>(v751_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v699_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v843_sw, v843_sw))))))));
          float v853_data = glb_m1[2];
          float v854_data = glb_m1[2];
          float v855_data = glb_m1[2];
          float v856_data = glb_m1[2];
          float v857_data = glb_m1[2];
          float v858_data = glb_m1[2];
          float v859_data = glb_m1[2];
          float v860_data = glb_m1[3];
          float v861_data = glb_m1[3];
          float v862_data = glb_m1[3];
          float v863_data = glb_m1[3];
          float v864_data = glb_m1[3];
          float v865_data = glb_m1[3];
          float v866_data = glb_m1[3];
          float v867_data = glb_m1[3];
          float v868_data = glb_m1[3];
          tensorforge::transpose16x16b32(v853_data, v854_data, v855_data, v856_data, v857_data, v858_data, v859_data, v860_data, v861_data, v862_data, v863_data, v864_data, v865_data, v866_data, v867_data, v868_data);
          tensorforge::VectorT<float, 16> v869_acc{};
          tensorforge::VectorT<float, 16> v872_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v853_data, v54_data, v869_acc, 1, 0, 0);
          float v873_el = v872_acc[0];
          float v875_el = v872_acc[4];
          float v876_sw = tensorforge::swap<32>(v875_el);
          float v878_el = v872_acc[8];
          float v881_el = v872_acc[12];
          float v882_sw = tensorforge::swap<32>(v881_el);
          r1[64] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v882_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v878_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v876_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v873_el, v873_el))))))));
          float v885_el = v872_acc[1];
          float v887_el = v872_acc[5];
          float v888_sw = tensorforge::swap<32>(v887_el);
          float v890_el = v872_acc[9];
          float v893_el = v872_acc[13];
          float v894_sw = tensorforge::swap<32>(v893_el);
          r1[66] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v894_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v890_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v888_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v885_el, v885_el))))))));
          float v897_el = v872_acc[2];
          float v899_el = v872_acc[6];
          float v900_sw = tensorforge::swap<32>(v899_el);
          float v902_el = v872_acc[10];
          float v905_el = v872_acc[14];
          float v906_sw = tensorforge::swap<32>(v905_el);
          r1[68] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v906_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v902_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v900_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v897_el, v897_el))))))));
          float v909_el = v872_acc[3];
          float v911_el = v872_acc[7];
          float v912_sw = tensorforge::swap<32>(v911_el);
          float v914_el = v872_acc[11];
          float v917_el = v872_acc[15];
          float v918_sw = tensorforge::swap<32>(v917_el);
          r1[70] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v918_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v914_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v912_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v909_el, v909_el))))))));
          float v922_sw = tensorforge::swap<32>(v873_el);
          float v927_sw = tensorforge::swap<32>(v878_el);
          r1[72] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v881_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v927_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v875_el, (tensorforge::dppUpdate<228, 1, 15, false>(v922_sw, v922_sw))))))));
          float v934_sw = tensorforge::swap<32>(v885_el);
          float v939_sw = tensorforge::swap<32>(v890_el);
          r1[74] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v893_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v939_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v887_el, (tensorforge::dppUpdate<228, 1, 15, false>(v934_sw, v934_sw))))))));
          float v946_sw = tensorforge::swap<32>(v897_el);
          float v951_sw = tensorforge::swap<32>(v902_el);
          r1[76] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v905_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v951_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v899_el, (tensorforge::dppUpdate<228, 1, 15, false>(v946_sw, v946_sw))))))));
          float v958_sw = tensorforge::swap<32>(v909_el);
          float v963_sw = tensorforge::swap<32>(v914_el);
          r1[78] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v917_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v963_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v911_el, (tensorforge::dppUpdate<228, 1, 15, false>(v958_sw, v958_sw))))))));
          float v970_sw = tensorforge::swap<64>(v873_el);
          r1[80] = (tensorforge::dppUpdate<228, 8, 15, false>(v882_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v878_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v876_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v970_sw, v970_sw))))))));
          float v982_sw = tensorforge::swap<64>(v885_el);
          r1[82] = (tensorforge::dppUpdate<228, 8, 15, false>(v894_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v890_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v888_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v982_sw, v982_sw))))))));
          float v994_sw = tensorforge::swap<64>(v897_el);
          r1[84] = (tensorforge::dppUpdate<228, 8, 15, false>(v906_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v902_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v900_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v994_sw, v994_sw))))))));
          float v1006_sw = tensorforge::swap<64>(v909_el);
          r1[86] = (tensorforge::dppUpdate<228, 8, 15, false>(v918_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v914_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v912_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1006_sw, v1006_sw))))))));
          float v1019_sw = tensorforge::swap<64>(v922_sw);
          r1[88] = (tensorforge::dppUpdate<228, 8, 15, false>(v881_el, (tensorforge::dppUpdate<228, 4, 15, false>(v927_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v875_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1019_sw, v1019_sw))))))));
          float v1031_sw = tensorforge::swap<64>(v934_sw);
          r1[90] = (tensorforge::dppUpdate<228, 8, 15, false>(v893_el, (tensorforge::dppUpdate<228, 4, 15, false>(v939_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v887_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1031_sw, v1031_sw))))))));
          float v1043_sw = tensorforge::swap<64>(v946_sw);
          r1[92] = (tensorforge::dppUpdate<228, 8, 15, false>(v905_el, (tensorforge::dppUpdate<228, 4, 15, false>(v951_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v899_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1043_sw, v1043_sw))))))));
          float v1055_sw = tensorforge::swap<64>(v958_sw);
          r1[94] = (tensorforge::dppUpdate<228, 8, 15, false>(v917_el, (tensorforge::dppUpdate<228, 4, 15, false>(v963_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v911_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1055_sw, v1055_sw))))))));
          tensorforge::VectorT<float, 16> v1065_acc{};
          tensorforge::VectorT<float, 16> v1068_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v853_data, v250_data, v1065_acc, 1, 0, 0);
          float v1069_el = v1068_acc[0];
          float v1071_el = v1068_acc[4];
          float v1072_sw = tensorforge::swap<32>(v1071_el);
          float v1074_el = v1068_acc[8];
          float v1077_el = v1068_acc[12];
          float v1078_sw = tensorforge::swap<32>(v1077_el);
          r1[65] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1078_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1074_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1072_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1069_el, v1069_el))))))));
          float v1081_el = v1068_acc[1];
          float v1083_el = v1068_acc[5];
          float v1084_sw = tensorforge::swap<32>(v1083_el);
          float v1086_el = v1068_acc[9];
          float v1089_el = v1068_acc[13];
          float v1090_sw = tensorforge::swap<32>(v1089_el);
          r1[67] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1090_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1086_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1084_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1081_el, v1081_el))))))));
          float v1093_el = v1068_acc[2];
          float v1095_el = v1068_acc[6];
          float v1096_sw = tensorforge::swap<32>(v1095_el);
          float v1098_el = v1068_acc[10];
          float v1101_el = v1068_acc[14];
          float v1102_sw = tensorforge::swap<32>(v1101_el);
          r1[69] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1102_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1098_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1096_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1093_el, v1093_el))))))));
          float v1105_el = v1068_acc[3];
          float v1107_el = v1068_acc[7];
          float v1108_sw = tensorforge::swap<32>(v1107_el);
          float v1110_el = v1068_acc[11];
          float v1113_el = v1068_acc[15];
          float v1114_sw = tensorforge::swap<32>(v1113_el);
          r1[71] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1114_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1110_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1108_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1105_el, v1105_el))))))));
          float v1118_sw = tensorforge::swap<32>(v1069_el);
          float v1123_sw = tensorforge::swap<32>(v1074_el);
          r1[73] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1077_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1123_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1071_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1118_sw, v1118_sw))))))));
          float v1130_sw = tensorforge::swap<32>(v1081_el);
          float v1135_sw = tensorforge::swap<32>(v1086_el);
          r1[75] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1089_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1135_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1083_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1130_sw, v1130_sw))))))));
          float v1142_sw = tensorforge::swap<32>(v1093_el);
          float v1147_sw = tensorforge::swap<32>(v1098_el);
          r1[77] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1101_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1147_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1095_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1142_sw, v1142_sw))))))));
          float v1154_sw = tensorforge::swap<32>(v1105_el);
          float v1159_sw = tensorforge::swap<32>(v1110_el);
          r1[79] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1113_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1159_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1107_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1154_sw, v1154_sw))))))));
          float v1166_sw = tensorforge::swap<64>(v1069_el);
          r1[81] = (tensorforge::dppUpdate<228, 8, 15, false>(v1078_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1074_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1072_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1166_sw, v1166_sw))))))));
          float v1178_sw = tensorforge::swap<64>(v1081_el);
          r1[83] = (tensorforge::dppUpdate<228, 8, 15, false>(v1090_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1086_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1084_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1178_sw, v1178_sw))))))));
          float v1190_sw = tensorforge::swap<64>(v1093_el);
          r1[85] = (tensorforge::dppUpdate<228, 8, 15, false>(v1102_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1098_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1096_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1190_sw, v1190_sw))))))));
          float v1202_sw = tensorforge::swap<64>(v1105_el);
          r1[87] = (tensorforge::dppUpdate<228, 8, 15, false>(v1114_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1110_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1108_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1202_sw, v1202_sw))))))));
          float v1215_sw = tensorforge::swap<64>(v1118_sw);
          r1[89] = (tensorforge::dppUpdate<228, 8, 15, false>(v1077_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1123_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1071_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1215_sw, v1215_sw))))))));
          float v1227_sw = tensorforge::swap<64>(v1130_sw);
          r1[91] = (tensorforge::dppUpdate<228, 8, 15, false>(v1089_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1135_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1083_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1227_sw, v1227_sw))))))));
          float v1239_sw = tensorforge::swap<64>(v1142_sw);
          r1[93] = (tensorforge::dppUpdate<228, 8, 15, false>(v1101_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1147_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1095_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1239_sw, v1239_sw))))))));
          float v1251_sw = tensorforge::swap<64>(v1154_sw);
          r1[95] = (tensorforge::dppUpdate<228, 8, 15, false>(v1113_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1159_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1107_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1251_sw, v1251_sw))))))));
          float v1261_data = glb_m1[3];
          float v1262_data = glb_m1[3];
          float v1263_data = glb_m1[3];
          float v1264_data = glb_m1[3];
          float v1265_data = glb_m1[4];
          float v1266_data = glb_m1[4];
          float v1267_data = glb_m1[4];
          float v1268_data = glb_m1[4];
          float v1269_data = glb_m1[4];
          float v1270_data = glb_m1[4];
          float v1271_data = glb_m1[4];
          float v1272_data = glb_m1[4];
          float v1273_data = glb_m1[4];
          float v1274_data = glb_m1[4];
          float v1275_data = glb_m1[4];
          float v1276_data = glb_m1[4];
          tensorforge::transpose16x16b32(v1261_data, v1262_data, v1263_data, v1264_data, v1265_data, v1266_data, v1267_data, v1268_data, v1269_data, v1270_data, v1271_data, v1272_data, v1273_data, v1274_data, v1275_data, v1276_data);
          tensorforge::VectorT<float, 16> v1277_acc{};
          tensorforge::VectorT<float, 16> v1280_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1261_data, v54_data, v1277_acc, 1, 0, 0);
          float v1281_el = v1280_acc[0];
          float v1283_el = v1280_acc[4];
          float v1284_sw = tensorforge::swap<32>(v1283_el);
          float v1286_el = v1280_acc[8];
          float v1289_el = v1280_acc[12];
          float v1290_sw = tensorforge::swap<32>(v1289_el);
          r1[96] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1290_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1286_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1284_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1281_el, v1281_el))))))));
          float v1293_el = v1280_acc[1];
          float v1295_el = v1280_acc[5];
          float v1296_sw = tensorforge::swap<32>(v1295_el);
          float v1298_el = v1280_acc[9];
          float v1301_el = v1280_acc[13];
          float v1302_sw = tensorforge::swap<32>(v1301_el);
          r1[98] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1302_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1298_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1296_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1293_el, v1293_el))))))));
          float v1305_el = v1280_acc[2];
          float v1307_el = v1280_acc[6];
          float v1308_sw = tensorforge::swap<32>(v1307_el);
          float v1310_el = v1280_acc[10];
          float v1313_el = v1280_acc[14];
          float v1314_sw = tensorforge::swap<32>(v1313_el);
          r1[100] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1314_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1310_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1308_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1305_el, v1305_el))))))));
          float v1317_el = v1280_acc[3];
          float v1319_el = v1280_acc[7];
          float v1320_sw = tensorforge::swap<32>(v1319_el);
          float v1322_el = v1280_acc[11];
          float v1325_el = v1280_acc[15];
          float v1326_sw = tensorforge::swap<32>(v1325_el);
          r1[102] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1326_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1322_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1320_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1317_el, v1317_el))))))));
          float v1330_sw = tensorforge::swap<32>(v1281_el);
          float v1335_sw = tensorforge::swap<32>(v1286_el);
          r1[104] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1289_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1335_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1283_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1330_sw, v1330_sw))))))));
          float v1342_sw = tensorforge::swap<32>(v1293_el);
          float v1347_sw = tensorforge::swap<32>(v1298_el);
          r1[106] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1301_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1347_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1295_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1342_sw, v1342_sw))))))));
          float v1354_sw = tensorforge::swap<32>(v1305_el);
          float v1359_sw = tensorforge::swap<32>(v1310_el);
          r1[108] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1313_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1359_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1307_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1354_sw, v1354_sw))))))));
          float v1366_sw = tensorforge::swap<32>(v1317_el);
          float v1371_sw = tensorforge::swap<32>(v1322_el);
          r1[110] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1325_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1371_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1319_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1366_sw, v1366_sw))))))));
          float v1378_sw = tensorforge::swap<64>(v1281_el);
          r1[112] = (tensorforge::dppUpdate<228, 8, 15, false>(v1290_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1286_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1284_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1378_sw, v1378_sw))))))));
          float v1390_sw = tensorforge::swap<64>(v1293_el);
          r1[114] = (tensorforge::dppUpdate<228, 8, 15, false>(v1302_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1298_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1296_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1390_sw, v1390_sw))))))));
          float v1402_sw = tensorforge::swap<64>(v1305_el);
          r1[116] = (tensorforge::dppUpdate<228, 8, 15, false>(v1314_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1310_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1308_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1402_sw, v1402_sw))))))));
          float v1414_sw = tensorforge::swap<64>(v1317_el);
          r1[118] = (tensorforge::dppUpdate<228, 8, 15, false>(v1326_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1322_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1320_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1414_sw, v1414_sw))))))));
          float v1427_sw = tensorforge::swap<64>(v1330_sw);
          r1[120] = (tensorforge::dppUpdate<228, 8, 15, false>(v1289_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1335_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1283_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1427_sw, v1427_sw))))))));
          float v1439_sw = tensorforge::swap<64>(v1342_sw);
          r1[122] = (tensorforge::dppUpdate<228, 8, 15, false>(v1301_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1347_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1295_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1439_sw, v1439_sw))))))));
          float v1451_sw = tensorforge::swap<64>(v1354_sw);
          r1[124] = (tensorforge::dppUpdate<228, 8, 15, false>(v1313_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1359_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1307_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1451_sw, v1451_sw))))))));
          float v1463_sw = tensorforge::swap<64>(v1366_sw);
          r1[126] = (tensorforge::dppUpdate<228, 8, 15, false>(v1325_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1371_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1319_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1463_sw, v1463_sw))))))));
          tensorforge::VectorT<float, 16> v1473_acc{};
          tensorforge::VectorT<float, 16> v1476_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1261_data, v250_data, v1473_acc, 1, 0, 0);
          float v1477_el = v1476_acc[0];
          float v1479_el = v1476_acc[4];
          float v1480_sw = tensorforge::swap<32>(v1479_el);
          float v1482_el = v1476_acc[8];
          float v1485_el = v1476_acc[12];
          float v1486_sw = tensorforge::swap<32>(v1485_el);
          r1[97] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1486_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1482_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1480_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1477_el, v1477_el))))))));
          float v1489_el = v1476_acc[1];
          float v1491_el = v1476_acc[5];
          float v1492_sw = tensorforge::swap<32>(v1491_el);
          float v1494_el = v1476_acc[9];
          float v1497_el = v1476_acc[13];
          float v1498_sw = tensorforge::swap<32>(v1497_el);
          r1[99] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1498_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1494_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1492_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1489_el, v1489_el))))))));
          float v1501_el = v1476_acc[2];
          float v1503_el = v1476_acc[6];
          float v1504_sw = tensorforge::swap<32>(v1503_el);
          float v1506_el = v1476_acc[10];
          float v1509_el = v1476_acc[14];
          float v1510_sw = tensorforge::swap<32>(v1509_el);
          r1[101] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1510_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1506_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1504_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1501_el, v1501_el))))))));
          float v1513_el = v1476_acc[3];
          float v1515_el = v1476_acc[7];
          float v1516_sw = tensorforge::swap<32>(v1515_el);
          float v1518_el = v1476_acc[11];
          float v1521_el = v1476_acc[15];
          float v1522_sw = tensorforge::swap<32>(v1521_el);
          r1[103] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1522_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1518_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1516_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1513_el, v1513_el))))))));
          float v1526_sw = tensorforge::swap<32>(v1477_el);
          float v1531_sw = tensorforge::swap<32>(v1482_el);
          r1[105] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1485_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1531_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1479_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1526_sw, v1526_sw))))))));
          float v1538_sw = tensorforge::swap<32>(v1489_el);
          float v1543_sw = tensorforge::swap<32>(v1494_el);
          r1[107] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1497_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1543_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1491_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1538_sw, v1538_sw))))))));
          float v1550_sw = tensorforge::swap<32>(v1501_el);
          float v1555_sw = tensorforge::swap<32>(v1506_el);
          r1[109] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1509_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1555_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1503_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1550_sw, v1550_sw))))))));
          float v1562_sw = tensorforge::swap<32>(v1513_el);
          float v1567_sw = tensorforge::swap<32>(v1518_el);
          r1[111] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1521_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1567_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1515_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1562_sw, v1562_sw))))))));
          float v1574_sw = tensorforge::swap<64>(v1477_el);
          r1[113] = (tensorforge::dppUpdate<228, 8, 15, false>(v1486_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1482_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1480_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1574_sw, v1574_sw))))))));
          float v1586_sw = tensorforge::swap<64>(v1489_el);
          r1[115] = (tensorforge::dppUpdate<228, 8, 15, false>(v1498_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1494_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1492_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1586_sw, v1586_sw))))))));
          float v1598_sw = tensorforge::swap<64>(v1501_el);
          r1[117] = (tensorforge::dppUpdate<228, 8, 15, false>(v1510_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1506_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1504_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1598_sw, v1598_sw))))))));
          float v1610_sw = tensorforge::swap<64>(v1513_el);
          r1[119] = (tensorforge::dppUpdate<228, 8, 15, false>(v1522_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1518_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1516_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1610_sw, v1610_sw))))))));
          float v1623_sw = tensorforge::swap<64>(v1526_sw);
          r1[121] = (tensorforge::dppUpdate<228, 8, 15, false>(v1485_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1531_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1479_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1623_sw, v1623_sw))))))));
          float v1635_sw = tensorforge::swap<64>(v1538_sw);
          r1[123] = (tensorforge::dppUpdate<228, 8, 15, false>(v1497_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1543_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1491_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1635_sw, v1635_sw))))))));
          float v1647_sw = tensorforge::swap<64>(v1550_sw);
          r1[125] = (tensorforge::dppUpdate<228, 8, 15, false>(v1509_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1555_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1503_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1647_sw, v1647_sw))))))));
          float v1659_sw = tensorforge::swap<64>(v1562_sw);
          r1[127] = (tensorforge::dppUpdate<228, 8, 15, false>(v1521_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1567_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1515_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1659_sw, v1659_sw))))))));
          float v1669_data = glb_m1[4];
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
          float v1680_data = glb_m1[5];
          float v1681_data = glb_m1[5];
          float v1682_data = glb_m1[5];
          float v1683_pad{};
          float v1684_pad{};
          tensorforge::transpose16x16b32(v1669_data, v1670_data, v1671_data, v1672_data, v1673_data, v1674_data, v1675_data, v1676_data, v1677_data, v1678_data, v1679_data, v1680_data, v1681_data, v1682_data, v1683_pad, v1684_pad);
          tensorforge::VectorT<float, 16> v1685_acc{};
          tensorforge::VectorT<float, 16> v1688_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1669_data, v54_data, v1685_acc, 1, 0, 0);
          float v1689_el = v1688_acc[0];
          float v1691_el = v1688_acc[4];
          float v1692_sw = tensorforge::swap<32>(v1691_el);
          float v1694_el = v1688_acc[8];
          float v1697_el = v1688_acc[12];
          float v1698_sw = tensorforge::swap<32>(v1697_el);
          r1[128] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1698_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1694_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1692_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1689_el, v1689_el))))))));
          float v1701_el = v1688_acc[1];
          float v1703_el = v1688_acc[5];
          float v1704_sw = tensorforge::swap<32>(v1703_el);
          float v1706_el = v1688_acc[9];
          float v1709_el = v1688_acc[13];
          float v1710_sw = tensorforge::swap<32>(v1709_el);
          r1[130] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1710_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1706_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1704_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1701_el, v1701_el))))))));
          float v1713_el = v1688_acc[2];
          float v1715_el = v1688_acc[6];
          float v1716_sw = tensorforge::swap<32>(v1715_el);
          float v1718_el = v1688_acc[10];
          float v1721_el = v1688_acc[14];
          float v1722_sw = tensorforge::swap<32>(v1721_el);
          r1[132] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1722_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1718_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1716_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1713_el, v1713_el))))))));
          float v1725_el = v1688_acc[3];
          float v1727_el = v1688_acc[7];
          float v1728_sw = tensorforge::swap<32>(v1727_el);
          float v1730_el = v1688_acc[11];
          float v1733_el = v1688_acc[15];
          float v1734_sw = tensorforge::swap<32>(v1733_el);
          r1[134] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1734_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1730_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1728_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1725_el, v1725_el))))))));
          float v1738_sw = tensorforge::swap<32>(v1689_el);
          float v1743_sw = tensorforge::swap<32>(v1694_el);
          r1[136] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1697_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1743_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1691_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1738_sw, v1738_sw))))))));
          float v1750_sw = tensorforge::swap<32>(v1701_el);
          float v1755_sw = tensorforge::swap<32>(v1706_el);
          r1[138] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1709_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1755_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1703_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1750_sw, v1750_sw))))))));
          float v1762_sw = tensorforge::swap<32>(v1713_el);
          r1[140] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1721_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1718_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1715_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1762_sw, v1762_sw))))))));
          float v1774_sw = tensorforge::swap<32>(v1725_el);
          r1[142] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1733_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1730_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1727_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1774_sw, v1774_sw))))))));
          float v1786_sw = tensorforge::swap<64>(v1689_el);
          r1[144] = (tensorforge::dppUpdate<228, 8, 15, false>(v1698_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1694_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1692_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1786_sw, v1786_sw))))))));
          float v1798_sw = tensorforge::swap<64>(v1701_el);
          r1[146] = (tensorforge::dppUpdate<228, 8, 15, false>(v1710_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1706_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1704_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1798_sw, v1798_sw))))))));
          float v1810_sw = tensorforge::swap<64>(v1713_el);
          r1[148] = (tensorforge::dppUpdate<228, 8, 15, false>(v1722_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1718_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1716_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1810_sw, v1810_sw))))))));
          float v1822_sw = tensorforge::swap<64>(v1725_el);
          r1[150] = (tensorforge::dppUpdate<228, 8, 15, false>(v1734_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1730_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1728_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1822_sw, v1822_sw))))))));
          float v1835_sw = tensorforge::swap<64>(v1738_sw);
          r1[152] = (tensorforge::dppUpdate<228, 8, 15, false>(v1697_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1743_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1691_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1835_sw, v1835_sw))))))));
          float v1847_sw = tensorforge::swap<64>(v1750_sw);
          r1[154] = (tensorforge::dppUpdate<228, 8, 15, false>(v1709_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1755_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1703_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1847_sw, v1847_sw))))))));
          tensorforge::VectorT<float, 16> v1857_acc{};
          tensorforge::VectorT<float, 16> v1860_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v1669_data, v250_data, v1857_acc, 1, 0, 0);
          float v1861_el = v1860_acc[0];
          float v1863_el = v1860_acc[4];
          float v1864_sw = tensorforge::swap<32>(v1863_el);
          float v1866_el = v1860_acc[8];
          float v1869_el = v1860_acc[12];
          float v1870_sw = tensorforge::swap<32>(v1869_el);
          r1[129] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1870_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1866_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1864_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1861_el, v1861_el))))))));
          float v1873_el = v1860_acc[1];
          float v1875_el = v1860_acc[5];
          float v1876_sw = tensorforge::swap<32>(v1875_el);
          float v1878_el = v1860_acc[9];
          float v1881_el = v1860_acc[13];
          float v1882_sw = tensorforge::swap<32>(v1881_el);
          r1[131] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1882_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1878_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1876_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1873_el, v1873_el))))))));
          float v1885_el = v1860_acc[2];
          float v1887_el = v1860_acc[6];
          float v1888_sw = tensorforge::swap<32>(v1887_el);
          float v1890_el = v1860_acc[10];
          float v1893_el = v1860_acc[14];
          float v1894_sw = tensorforge::swap<32>(v1893_el);
          r1[133] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1894_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1890_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1888_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1885_el, v1885_el))))))));
          float v1897_el = v1860_acc[3];
          float v1899_el = v1860_acc[7];
          float v1900_sw = tensorforge::swap<32>(v1899_el);
          float v1902_el = v1860_acc[11];
          float v1905_el = v1860_acc[15];
          float v1906_sw = tensorforge::swap<32>(v1905_el);
          r1[135] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1906_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1902_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v1900_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v1897_el, v1897_el))))))));
          float v1910_sw = tensorforge::swap<32>(v1861_el);
          float v1915_sw = tensorforge::swap<32>(v1866_el);
          r1[137] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1869_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1915_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1863_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1910_sw, v1910_sw))))))));
          float v1922_sw = tensorforge::swap<32>(v1873_el);
          float v1927_sw = tensorforge::swap<32>(v1878_el);
          r1[139] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1881_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v1927_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v1875_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1922_sw, v1922_sw))))))));
          float v1934_sw = tensorforge::swap<32>(v1885_el);
          r1[141] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1893_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1890_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1887_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1934_sw, v1934_sw))))))));
          float v1946_sw = tensorforge::swap<32>(v1897_el);
          r1[143] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v1905_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v1902_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v1899_el, (tensorforge::dppUpdate<228, 1, 15, false>(v1946_sw, v1946_sw))))))));
          float v1958_sw = tensorforge::swap<64>(v1861_el);
          r1[145] = (tensorforge::dppUpdate<228, 8, 15, false>(v1870_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1866_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1864_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1958_sw, v1958_sw))))))));
          float v1970_sw = tensorforge::swap<64>(v1873_el);
          r1[147] = (tensorforge::dppUpdate<228, 8, 15, false>(v1882_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1878_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1876_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1970_sw, v1970_sw))))))));
          float v1982_sw = tensorforge::swap<64>(v1885_el);
          r1[149] = (tensorforge::dppUpdate<228, 8, 15, false>(v1894_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1890_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1888_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1982_sw, v1982_sw))))))));
          float v1994_sw = tensorforge::swap<64>(v1897_el);
          r1[151] = (tensorforge::dppUpdate<228, 8, 15, false>(v1906_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v1902_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1900_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1994_sw, v1994_sw))))))));
          float v2007_sw = tensorforge::swap<64>(v1910_sw);
          r1[153] = (tensorforge::dppUpdate<228, 8, 15, false>(v1869_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1915_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1863_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v2007_sw, v2007_sw))))))));
          float v2019_sw = tensorforge::swap<64>(v1922_sw);
          r1[155] = (tensorforge::dppUpdate<228, 8, 15, false>(v1881_el, (tensorforge::dppUpdate<228, 4, 15, false>(v1927_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v1875_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v2019_sw, v2019_sw))))))));
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          bool v2030_g = v26_lead >= 20;
          if (v2030_g) {
            float v2031_data = r1[24];
            float v2032_data = r2[0];
            r2[0] = (v2032_data + v2031_data);
            float v2034_data = r1[50];
            float v2035_data = r2[2];
            r2[2] = (v2035_data + v2034_data);
            float v2037_data = r1[76];
            float v2038_data = r2[4];
            r2[4] = (v2038_data + v2037_data);
            float v2040_data = r1[102];
            float v2041_data = r2[6];
            r2[6] = (v2041_data + v2040_data);
            float v2043_data = r1[128];
            float v2044_data = r2[8];
            r2[8] = (v2044_data + v2043_data);
            float v2046_data = r1[154];
            float v2047_data = r2[10];
            r2[10] = (v2047_data + v2046_data);
          }
          bool v2049_g = v26_lead < 3;
          if (v2049_g) {
            float v2050_data = r1[25];
            float v2051_data = r2[1];
            r2[1] = (v2051_data + v2050_data);
            float v2053_data = r1[51];
            float v2054_data = r2[3];
            r2[3] = (v2054_data + v2053_data);
            float v2056_data = r1[77];
            float v2057_data = r2[5];
            r2[5] = (v2057_data + v2056_data);
            float v2059_data = r1[103];
            float v2060_data = r2[7];
            r2[7] = (v2060_data + v2059_data);
            float v2062_data = r1[129];
            float v2063_data = r2[9];
            r2[9] = (v2063_data + v2062_data);
            float v2065_data = r1[155];
            float v2066_data = r2[11];
            r2[11] = (v2066_data + v2065_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v2030_g) {
            #pragma unroll
            for (int32_t v2069_i1 = 0; v2069_i1 < 1; ++v2069_i1) {
              int32_t v2071_a = v2069_i1 * 2;
              int32_t v2081_a = v26_lead + ((v2069_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v2070_i2 = 0; v2070_i2 < 6; ++v2070_i2) {
                float v2075_data = r2[(v2071_a + (v2070_i2 * 2))];
                int32_t v2082_a = v2081_a + (v2070_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v2082_a], v2075_data);
              }
            }
          }
          if (v2049_g) {
            int32_t v2092_lead = v26_lead + 32_i32;
            #pragma unroll
            for (int32_t v2084_i1 = 0; v2084_i1 < 1; ++v2084_i1) {
              int32_t v2088_a = 1 + (v2084_i1 * 2);
              int32_t v2096_a = v2092_lead + ((v2084_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v2085_i2 = 0; v2085_i2 < 6; ++v2085_i2) {
                float v2090_data = r2[(v2088_a + (v2085_i2 * 2))];
                int32_t v2097_a = v2096_a + (v2085_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v2097_a], v2090_data);
              }
            }
          }
        }
      }
    }
  }
}

