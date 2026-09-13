// === base name ===
kernel_2328ef1bf5de7f2c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2328ef1bf5de7f2c = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2328ef1bf5de7f2c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2328ef1bf5de7f2c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2328ef1bf5de7f2c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_2328ef1bf5de7f2c, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_2328ef1bf5de7f2c, block.x * block.y * block.z, 0));
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
void launcher_kernel_2328ef1bf5de7f2c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2328ef1bf5de7f2c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_2328ef1bf5de7f2c), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_2328ef1bf5de7f2c, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_2328ef1bf5de7f2c(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×16(12×16) {0..12}×{0..16} strided
    //   m1 20×12(20×12) {0..20}×{0..12} strided
    //   m2 20×16(20×16) {0..20}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[k,i] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[20,12]],"name":"m1","ordered":false,"parts":1,"shape":[20,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[20,16]],"name":"m2","ordered":false,"parts":1,"shape":[20,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,12]},{"addressing":"strided","bbox":[[0,0],[20,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1\n"}
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
          int32_t v19_lead = threadIdx.x % 16;
          bool v20_g = v19_lead < 12;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 20; ++v16_i0) {
            if (v20_g) {
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v16_i0 + (v19_lead * 20))]);
              r0[v16_i0] = v25_data;
            }
          }
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v30_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v34_lead = v30_lead + (v31_i0 * 16);
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 16; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m2[(v34_lead + (v32_i1 * 20))]);
              r1[(v31_i0 + (v32_i1 * 2))] = v37_data;
            }
          }
          if (v30_lead < 4) {
            int32_t v43_lead = v30_lead + 16_i32;
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 16; ++v41_i1) {
              float v46_data = __builtin_nontemporal_load(&glb_m2[(v43_lead + (v41_i1 * 20))]);
              r1[(1 + (v41_i1 * 2))] = v46_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v50_data = r1[0];
          float v51_data = r1[2];
          float v52_data = r1[4];
          float v53_data = r1[6];
          float v54_data = r1[8];
          float v55_data = r1[10];
          float v56_data = r1[12];
          float v57_data = r1[14];
          float v58_data = r1[16];
          float v59_data = r1[18];
          float v60_data = r1[20];
          float v61_data = r1[22];
          float v62_data = r1[24];
          float v63_data = r1[26];
          float v64_data = r1[28];
          float v65_data = r1[30];
          tensorforge::transpose16x16b32(v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data);
          float v66_data = r1[1];
          float v67_data = r1[3];
          float v68_data = r1[5];
          float v69_data = r1[7];
          float v70_data = r1[9];
          float v71_data = r1[11];
          float v72_data = r1[13];
          float v73_data = r1[15];
          float v74_data = r1[17];
          float v75_data = r1[19];
          float v76_data = r1[21];
          float v77_data = r1[23];
          float v78_data = r1[25];
          float v79_data = r1[27];
          float v80_data = r1[29];
          float v81_data = r1[31];
          tensorforge::transpose16x16b32(v66_data, v67_data, v68_data, v69_data, v70_data, v71_data, v72_data, v73_data, v74_data, v75_data, v76_data, v77_data, v78_data, v79_data, v80_data, v81_data);
          tensorforge::VectorT<float, 16> v82_acc{};
          float v83_data = r0[0];
          float v84_data = r0[1];
          float v85_data = r0[2];
          float v86_data = r0[3];
          float v87_data = r0[4];
          float v88_data = r0[5];
          float v89_data = r0[6];
          float v90_data = r0[7];
          float v91_data = r0[8];
          float v92_data = r0[9];
          float v93_data = r0[10];
          float v94_data = r0[11];
          float v95_data = r0[12];
          float v96_data = r0[13];
          float v97_data = r0[14];
          float v98_data = r0[15];
          tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v83_data, v82_acc, 0, 0, 0);
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
          tensorforge::VectorT<float, 16> v112_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v96_data, v111_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v113_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v97_data, v112_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v114_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v98_data, v113_acc, 0, 0, 0);
          float v115_data = r0[16];
          float v116_data = r0[17];
          float v117_data = r0[18];
          float v118_data = r0[19];
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v115_data, v114_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v121_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v67_data, v116_data, v120_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v122_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v68_data, v117_data, v121_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v69_data, v118_data, v122_acc, 0, 0, 0);
          float v124_el = v123_acc[0];
          float v126_el = v123_acc[4];
          float v127_sw = tensorforge::swap<32>(v126_el);
          float v129_el = v123_acc[8];
          float v132_el = v123_acc[12];
          float v133_sw = tensorforge::swap<32>(v132_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v124_el, v124_el))))))));
          float v136_el = v123_acc[1];
          float v138_el = v123_acc[5];
          float v139_sw = tensorforge::swap<32>(v138_el);
          float v141_el = v123_acc[9];
          float v144_el = v123_acc[13];
          float v145_sw = tensorforge::swap<32>(v144_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v136_el, v136_el))))))));
          float v148_el = v123_acc[2];
          float v150_el = v123_acc[6];
          float v151_sw = tensorforge::swap<32>(v150_el);
          float v153_el = v123_acc[10];
          float v156_el = v123_acc[14];
          float v157_sw = tensorforge::swap<32>(v156_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v153_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v148_el, v148_el))))))));
          float v160_el = v123_acc[3];
          float v162_el = v123_acc[7];
          float v163_sw = tensorforge::swap<32>(v162_el);
          float v165_el = v123_acc[11];
          float v168_el = v123_acc[15];
          float v169_sw = tensorforge::swap<32>(v168_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v169_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v165_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v160_el, v160_el))))))));
          float v173_sw = tensorforge::swap<32>(v124_el);
          float v178_sw = tensorforge::swap<32>(v129_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v178_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v126_el, (tensorforge::dppUpdate<228, 1, 15, false>(v173_sw, v173_sw))))))));
          float v185_sw = tensorforge::swap<32>(v136_el);
          float v190_sw = tensorforge::swap<32>(v141_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v190_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v138_el, (tensorforge::dppUpdate<228, 1, 15, false>(v185_sw, v185_sw))))))));
          float v197_sw = tensorforge::swap<32>(v148_el);
          float v202_sw = tensorforge::swap<32>(v153_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v156_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v202_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v150_el, (tensorforge::dppUpdate<228, 1, 15, false>(v197_sw, v197_sw))))))));
          float v209_sw = tensorforge::swap<32>(v160_el);
          float v214_sw = tensorforge::swap<32>(v165_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v168_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v214_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v162_el, (tensorforge::dppUpdate<228, 1, 15, false>(v209_sw, v209_sw))))))));
          float v221_sw = tensorforge::swap<64>(v124_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v129_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v221_sw, v221_sw))))))));
          float v233_sw = tensorforge::swap<64>(v136_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v141_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
          float v245_sw = tensorforge::swap<64>(v148_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v153_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v245_sw, v245_sw))))))));
          float v257_sw = tensorforge::swap<64>(v160_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v169_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v165_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v257_sw, v257_sw))))))));
          float v270_sw = tensorforge::swap<64>(v173_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v132_el, (tensorforge::dppUpdate<228, 4, 15, false>(v178_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v270_sw, v270_sw))))))));
          float v282_sw = tensorforge::swap<64>(v185_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v144_el, (tensorforge::dppUpdate<228, 4, 15, false>(v190_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v282_sw, v282_sw))))))));
          float v294_sw = tensorforge::swap<64>(v197_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v156_el, (tensorforge::dppUpdate<228, 4, 15, false>(v202_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v294_sw, v294_sw))))))));
          float v306_sw = tensorforge::swap<64>(v209_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v168_el, (tensorforge::dppUpdate<228, 4, 15, false>(v214_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v306_sw, v306_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v30_lead < 12) {
            #pragma unroll
            for (int32_t v317_i1 = 0; v317_i1 < 16; ++v317_i1) {
              float v319_data = r2[v317_i1];
              glb_m0[(v30_lead + (v317_i1 * 12))] = v319_data;
            }
          }
        }
      }
    }
  }
}

