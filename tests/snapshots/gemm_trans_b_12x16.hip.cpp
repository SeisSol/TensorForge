// === base name ===
kernel_b98c71a5eb9f3631

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b98c71a5eb9f3631 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b98c71a5eb9f3631(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b98c71a5eb9f3631(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b98c71a5eb9f3631(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b98c71a5eb9f3631, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b98c71a5eb9f3631, block.x * block.y * block.z, 0));
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
void launcher_kernel_b98c71a5eb9f3631(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b98c71a5eb9f3631(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b98c71a5eb9f3631), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b98c71a5eb9f3631, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b98c71a5eb9f3631(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×16(12×16) {0..12}×{0..16} strided
    //   m1 12×20(12×20) {0..12}×{0..20} strided
    //   m2 16×20(16×20) {0..16}×{0..20} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[j,k]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,20]],"name":"m2","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"strided","bbox":[[0,0],[16,20]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,20]}],"permute":[[0,1],[1,0]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 192 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 240 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          bool v19_g = v18_lead < 12;
          if (v19_g) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 20; ++v20_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v18_lead + (v20_i1 * 12))]);
              r0[v20_i1] = v25_data;
            }
          }
          float r1[20]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v31_lead = v18_lead + (v28_i0 * 16);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 20; ++v29_i1) {
              float v34_data = __builtin_nontemporal_load(&glb_m2[(v31_lead + (v29_i1 * 16))]);
              r1[(v28_i0 + v29_i1)] = v34_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v37_data = r1[0];
          float v38_bc = tensorforge::broadcast<16, 1, 0>(v37_data);
          float v40_bc = tensorforge::broadcast<16, 1, 1>(v37_data);
          float v42_bc = tensorforge::broadcast<16, 1, 2>(v37_data);
          float v44_bc = tensorforge::broadcast<16, 1, 3>(v37_data);
          float v46_bc = tensorforge::broadcast<16, 1, 4>(v37_data);
          float v48_bc = tensorforge::broadcast<16, 1, 5>(v37_data);
          float v50_bc = tensorforge::broadcast<16, 1, 6>(v37_data);
          float v52_bc = tensorforge::broadcast<16, 1, 7>(v37_data);
          float v54_bc = tensorforge::broadcast<16, 1, 8>(v37_data);
          float v56_bc = tensorforge::broadcast<16, 1, 9>(v37_data);
          float v58_bc = tensorforge::broadcast<16, 1, 10>(v37_data);
          float v60_bc = tensorforge::broadcast<16, 1, 11>(v37_data);
          float v62_bc = tensorforge::broadcast<16, 1, 12>(v37_data);
          float v64_bc = tensorforge::broadcast<16, 1, 13>(v37_data);
          float v66_bc = tensorforge::broadcast<16, 1, 14>(v37_data);
          float v68_bc = tensorforge::broadcast<16, 1, 15>(v37_data);
          tensorforge::transpose16x16b32(v38_bc, v40_bc, v42_bc, v44_bc, v46_bc, v48_bc, v50_bc, v52_bc, v54_bc, v56_bc, v58_bc, v60_bc, v62_bc, v64_bc, v66_bc, v68_bc);
          float v69_data = r1[1];
          float v70_bc = tensorforge::broadcast<16, 1, 0>(v69_data);
          float v72_bc = tensorforge::broadcast<16, 1, 1>(v69_data);
          float v74_bc = tensorforge::broadcast<16, 1, 2>(v69_data);
          float v76_bc = tensorforge::broadcast<16, 1, 3>(v69_data);
          float v78_bc = tensorforge::broadcast<16, 1, 4>(v69_data);
          float v80_bc = tensorforge::broadcast<16, 1, 5>(v69_data);
          float v82_bc = tensorforge::broadcast<16, 1, 6>(v69_data);
          float v84_bc = tensorforge::broadcast<16, 1, 7>(v69_data);
          float v86_bc = tensorforge::broadcast<16, 1, 8>(v69_data);
          float v88_bc = tensorforge::broadcast<16, 1, 9>(v69_data);
          float v90_bc = tensorforge::broadcast<16, 1, 10>(v69_data);
          float v92_bc = tensorforge::broadcast<16, 1, 11>(v69_data);
          float v94_bc = tensorforge::broadcast<16, 1, 12>(v69_data);
          float v96_bc = tensorforge::broadcast<16, 1, 13>(v69_data);
          float v98_bc = tensorforge::broadcast<16, 1, 14>(v69_data);
          float v100_bc = tensorforge::broadcast<16, 1, 15>(v69_data);
          tensorforge::transpose16x16b32(v70_bc, v72_bc, v74_bc, v76_bc, v78_bc, v80_bc, v82_bc, v84_bc, v86_bc, v88_bc, v90_bc, v92_bc, v94_bc, v96_bc, v98_bc, v100_bc);
          tensorforge::VectorT<float, 16> v101_acc{};
          float v102_data = r0[0];
          float v103_data = r0[1];
          float v104_data = r0[2];
          float v105_data = r0[3];
          float v106_data = r0[4];
          float v107_data = r0[5];
          float v108_data = r0[6];
          float v109_data = r0[7];
          float v110_data = r0[8];
          float v111_data = r0[9];
          float v112_data = r0[10];
          float v113_data = r0[11];
          float v114_data = r0[12];
          float v115_data = r0[13];
          float v116_data = r0[14];
          float v117_data = r0[15];
          tensorforge::VectorT<float, 16> v118_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v38_bc, v102_data, v101_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v119_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_bc, v103_data, v118_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_bc, v104_data, v119_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v121_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_bc, v105_data, v120_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v122_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_bc, v106_data, v121_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_bc, v107_data, v122_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v124_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_bc, v108_data, v123_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v125_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_bc, v109_data, v124_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v126_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_bc, v110_data, v125_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v127_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_bc, v111_data, v126_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v128_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_bc, v112_data, v127_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v129_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_bc, v113_data, v128_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v130_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_bc, v114_data, v129_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v131_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_bc, v115_data, v130_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v132_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_bc, v116_data, v131_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v133_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v68_bc, v117_data, v132_acc, 0, 0, 0);
          float v134_data = r0[16];
          float v135_data = r0[17];
          float v136_data = r0[18];
          float v137_data = r0[19];
          tensorforge::VectorT<float, 16> v139_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v70_bc, v134_data, v133_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v140_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v72_bc, v135_data, v139_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v141_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v74_bc, v136_data, v140_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v142_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v76_bc, v137_data, v141_acc, 0, 0, 0);
          float v143_el = v142_acc[0];
          float v145_el = v142_acc[4];
          float v146_sw = tensorforge::swap<32>(v145_el);
          float v148_el = v142_acc[8];
          float v151_el = v142_acc[12];
          float v152_sw = tensorforge::swap<32>(v151_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v152_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v148_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v146_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v143_el, v143_el))))))));
          float v155_el = v142_acc[1];
          float v157_el = v142_acc[5];
          float v158_sw = tensorforge::swap<32>(v157_el);
          float v160_el = v142_acc[9];
          float v163_el = v142_acc[13];
          float v164_sw = tensorforge::swap<32>(v163_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v164_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v160_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v158_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v155_el, v155_el))))))));
          float v167_el = v142_acc[2];
          float v169_el = v142_acc[6];
          float v170_sw = tensorforge::swap<32>(v169_el);
          float v172_el = v142_acc[10];
          float v175_el = v142_acc[14];
          float v176_sw = tensorforge::swap<32>(v175_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v176_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v172_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v170_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v167_el, v167_el))))))));
          float v179_el = v142_acc[3];
          float v181_el = v142_acc[7];
          float v182_sw = tensorforge::swap<32>(v181_el);
          float v184_el = v142_acc[11];
          float v187_el = v142_acc[15];
          float v188_sw = tensorforge::swap<32>(v187_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v188_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v184_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v182_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v179_el, v179_el))))))));
          float v192_sw = tensorforge::swap<32>(v143_el);
          float v197_sw = tensorforge::swap<32>(v148_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v151_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v197_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v145_el, (tensorforge::dppUpdate<228, 1, 15, false>(v192_sw, v192_sw))))))));
          float v204_sw = tensorforge::swap<32>(v155_el);
          float v209_sw = tensorforge::swap<32>(v160_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v163_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v209_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v157_el, (tensorforge::dppUpdate<228, 1, 15, false>(v204_sw, v204_sw))))))));
          float v216_sw = tensorforge::swap<32>(v167_el);
          float v221_sw = tensorforge::swap<32>(v172_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v175_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v221_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v169_el, (tensorforge::dppUpdate<228, 1, 15, false>(v216_sw, v216_sw))))))));
          float v228_sw = tensorforge::swap<32>(v179_el);
          float v233_sw = tensorforge::swap<32>(v184_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v187_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v233_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v181_el, (tensorforge::dppUpdate<228, 1, 15, false>(v228_sw, v228_sw))))))));
          float v240_sw = tensorforge::swap<64>(v143_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v152_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v148_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v146_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v240_sw, v240_sw))))))));
          float v252_sw = tensorforge::swap<64>(v155_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v164_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v160_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v158_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v252_sw, v252_sw))))))));
          float v264_sw = tensorforge::swap<64>(v167_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v176_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v172_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v170_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v264_sw, v264_sw))))))));
          float v276_sw = tensorforge::swap<64>(v179_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v188_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v184_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v182_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v276_sw, v276_sw))))))));
          float v289_sw = tensorforge::swap<64>(v192_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v151_el, (tensorforge::dppUpdate<228, 4, 15, false>(v197_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v145_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v289_sw, v289_sw))))))));
          float v301_sw = tensorforge::swap<64>(v204_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v163_el, (tensorforge::dppUpdate<228, 4, 15, false>(v209_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v157_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v301_sw, v301_sw))))))));
          float v313_sw = tensorforge::swap<64>(v216_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v175_el, (tensorforge::dppUpdate<228, 4, 15, false>(v221_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v169_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v313_sw, v313_sw))))))));
          float v325_sw = tensorforge::swap<64>(v228_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v187_el, (tensorforge::dppUpdate<228, 4, 15, false>(v233_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v181_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v325_sw, v325_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v19_g) {
            #pragma unroll
            for (int32_t v335_i1 = 0; v335_i1 < 16; ++v335_i1) {
              float v337_data = r2[v335_i1];
              glb_m0[(v18_lead + (v335_i1 * 12))] = v337_data;
            }
          }
        }
      }
    }
  }
}

