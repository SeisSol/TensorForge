// === base name ===
kernel_a85043a99a7ff624

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a85043a99a7ff624 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a85043a99a7ff624(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a85043a99a7ff624(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a85043a99a7ff624(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a85043a99a7ff624, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_a85043a99a7ff624, block.x * block.y * block.z, 0));
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
void launcher_kernel_a85043a99a7ff624(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a85043a99a7ff624(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a85043a99a7ff624), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_a85043a99a7ff624, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a85043a99a7ff624(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×16(12×16) {0..12}×{0..16} strided
    //   m1 12×20(12×20) {0..12}×{0..20} strided
    //   m2 16×20(16×20) {0..16}×{0..20} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[j,k]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,20]],"name":"m2","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"strided","bbox":[[0,0],[16,20]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,20]}],"permute":[[0,1],[1,0]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 192 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 240 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v21_lead = threadIdx.x % 16;
          bool v22_g = v21_lead < 12;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 20; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v23_i1 * 12))]);
              r0[v23_i1] = v28_data;
            }
          }
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          bool v39_g = v21_lead < 4;
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 16; ++v31_i0) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 1; ++v32_i1) {
              int32_t v33_lead = v32_i1 * 16;
              float v37_data = __builtin_nontemporal_load(&glb_m2[(v31_i0 + ((v21_lead + v33_lead) * 16))]);
              r1[(v31_i0 + v33_lead)] = v37_data;
            }
            if (v39_g) {
              float v44_data = __builtin_nontemporal_load(&glb_m2[(v31_i0 + ((v21_lead + 16_i32) * 16))]);
              r1[(v31_i0 + 16)] = v44_data;
            }
          }
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v47_data = r1[0];
          float v48_data = r1[1];
          float v49_data = r1[2];
          float v50_data = r1[3];
          float v51_data = r1[4];
          float v52_data = r1[5];
          float v53_data = r1[6];
          float v54_data = r1[7];
          float v55_data = r1[8];
          float v56_data = r1[9];
          float v57_data = r1[10];
          float v58_data = r1[11];
          float v59_data = r1[12];
          float v60_data = r1[13];
          float v61_data = r1[14];
          float v62_data = r1[15];
          tensorforge::transpose16x16b32(v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data);
          float v63_data = r1[16];
          float v64_data = r1[17];
          float v65_data = r1[18];
          float v66_data = r1[19];
          float v67_data = r1[20];
          float v68_data = r1[21];
          float v69_data = r1[22];
          float v70_data = r1[23];
          float v71_data = r1[24];
          float v72_data = r1[25];
          float v73_data = r1[26];
          float v74_data = r1[27];
          float v75_data = r1[28];
          float v76_data = r1[29];
          float v77_data = r1[30];
          float v78_data = r1[31];
          tensorforge::transpose16x16b32(v63_data, v64_data, v65_data, v66_data, v67_data, v68_data, v69_data, v70_data, v71_data, v72_data, v73_data, v74_data, v75_data, v76_data, v77_data, v78_data);
          tensorforge::VectorT<float, 16> v79_acc{};
          float v80_data = r0[0];
          float v81_data = r0[1];
          float v82_data = r0[2];
          float v83_data = r0[3];
          float v84_data = r0[4];
          float v85_data = r0[5];
          float v86_data = r0[6];
          float v87_data = r0[7];
          float v88_data = r0[8];
          float v89_data = r0[9];
          float v90_data = r0[10];
          float v91_data = r0[11];
          float v92_data = r0[12];
          float v93_data = r0[13];
          float v94_data = r0[14];
          float v95_data = r0[15];
          tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v80_data, v79_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v81_data, v96_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v98_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v82_data, v97_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v83_data, v98_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v100_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v84_data, v99_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v101_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v85_data, v100_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v102_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v86_data, v101_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v103_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v87_data, v102_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v104_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v88_data, v103_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v105_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v89_data, v104_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v106_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v90_data, v105_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v107_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v91_data, v106_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v108_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v92_data, v107_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v109_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v93_data, v108_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v110_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v94_data, v109_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v111_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v95_data, v110_acc, 0, 0, 0);
          float v112_data = r0[16];
          float v113_data = r0[17];
          float v114_data = r0[18];
          float v115_data = r0[19];
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v112_data, v111_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v118_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v113_data, v117_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v119_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v114_data, v118_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v115_data, v119_acc, 0, 0, 0);
          float v121_el = v120_acc[0];
          float v123_el = v120_acc[4];
          float v124_sw = tensorforge::swap<32>(v123_el);
          float v126_el = v120_acc[8];
          float v129_el = v120_acc[12];
          float v130_sw = tensorforge::swap<32>(v129_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v121_el, v121_el))))))));
          float v133_el = v120_acc[1];
          float v135_el = v120_acc[5];
          float v136_sw = tensorforge::swap<32>(v135_el);
          float v138_el = v120_acc[9];
          float v141_el = v120_acc[13];
          float v142_sw = tensorforge::swap<32>(v141_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v142_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v133_el, v133_el))))))));
          float v145_el = v120_acc[2];
          float v147_el = v120_acc[6];
          float v148_sw = tensorforge::swap<32>(v147_el);
          float v150_el = v120_acc[10];
          float v153_el = v120_acc[14];
          float v154_sw = tensorforge::swap<32>(v153_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v154_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v148_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v145_el, v145_el))))))));
          float v157_el = v120_acc[3];
          float v159_el = v120_acc[7];
          float v160_sw = tensorforge::swap<32>(v159_el);
          float v162_el = v120_acc[11];
          float v165_el = v120_acc[15];
          float v166_sw = tensorforge::swap<32>(v165_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v166_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v160_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v157_el, v157_el))))))));
          float v170_sw = tensorforge::swap<32>(v121_el);
          float v175_sw = tensorforge::swap<32>(v126_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v175_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v123_el, (tensorforge::dppUpdate<228, 1, 15, false>(v170_sw, v170_sw))))))));
          float v182_sw = tensorforge::swap<32>(v133_el);
          float v187_sw = tensorforge::swap<32>(v138_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v187_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v135_el, (tensorforge::dppUpdate<228, 1, 15, false>(v182_sw, v182_sw))))))));
          float v194_sw = tensorforge::swap<32>(v145_el);
          float v199_sw = tensorforge::swap<32>(v150_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v153_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v199_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v147_el, (tensorforge::dppUpdate<228, 1, 15, false>(v194_sw, v194_sw))))))));
          float v206_sw = tensorforge::swap<32>(v157_el);
          float v211_sw = tensorforge::swap<32>(v162_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v165_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v211_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v159_el, (tensorforge::dppUpdate<228, 1, 15, false>(v206_sw, v206_sw))))))));
          float v218_sw = tensorforge::swap<64>(v121_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v126_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v218_sw, v218_sw))))))));
          float v230_sw = tensorforge::swap<64>(v133_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v142_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v138_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v230_sw, v230_sw))))))));
          float v242_sw = tensorforge::swap<64>(v145_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v154_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v150_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v148_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v242_sw, v242_sw))))))));
          float v254_sw = tensorforge::swap<64>(v157_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v166_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v162_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v160_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v254_sw, v254_sw))))))));
          float v267_sw = tensorforge::swap<64>(v170_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v129_el, (tensorforge::dppUpdate<228, 4, 15, false>(v175_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v267_sw, v267_sw))))))));
          float v279_sw = tensorforge::swap<64>(v182_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v141_el, (tensorforge::dppUpdate<228, 4, 15, false>(v187_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v279_sw, v279_sw))))))));
          float v291_sw = tensorforge::swap<64>(v194_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v153_el, (tensorforge::dppUpdate<228, 4, 15, false>(v199_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v291_sw, v291_sw))))))));
          float v303_sw = tensorforge::swap<64>(v206_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v165_el, (tensorforge::dppUpdate<228, 4, 15, false>(v211_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v159_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v303_sw, v303_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v22_g) {
            #pragma unroll
            for (int32_t v313_i1 = 0; v313_i1 < 16; ++v313_i1) {
              float v315_data = r2[v313_i1];
              glb_m0[(v21_lead + (v313_i1 * 12))] = v315_data;
            }
          }
        }
      }
    }
  }
}

