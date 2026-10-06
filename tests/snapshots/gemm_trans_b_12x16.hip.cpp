// === base name ===
kernel_b5f91d3afb322204

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b5f91d3afb322204 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b5f91d3afb322204(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b5f91d3afb322204(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b5f91d3afb322204(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b5f91d3afb322204, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b5f91d3afb322204, block.x * block.y * block.z, 0));
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
void launcher_kernel_b5f91d3afb322204(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b5f91d3afb322204(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b5f91d3afb322204), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b5f91d3afb322204, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b5f91d3afb322204(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          bool v36_g = v18_lead < 4;
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 16; ++v28_i0) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 1; ++v29_i1) {
              int32_t v30_lead = v29_i1 * 16;
              float v34_data = __builtin_nontemporal_load(&glb_m2[(v28_i0 + ((v18_lead + v30_lead) * 16))]);
              r1[(v28_i0 + v30_lead)] = v34_data;
            }
            if (v36_g) {
              float v41_data = __builtin_nontemporal_load(&glb_m2[(v28_i0 + ((v18_lead + 16_i32) * 16))]);
              r1[(v28_i0 + 16)] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v44_data = r1[0];
          float v45_data = r1[1];
          float v46_data = r1[2];
          float v47_data = r1[3];
          float v48_data = r1[4];
          float v49_data = r1[5];
          float v50_data = r1[6];
          float v51_data = r1[7];
          float v52_data = r1[8];
          float v53_data = r1[9];
          float v54_data = r1[10];
          float v55_data = r1[11];
          float v56_data = r1[12];
          float v57_data = r1[13];
          float v58_data = r1[14];
          float v59_data = r1[15];
          tensorforge::transpose16x16b32(v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data);
          float v60_data = r1[16];
          float v61_data = r1[17];
          float v62_data = r1[18];
          float v63_data = r1[19];
          float v64_data = r1[20];
          float v65_data = r1[21];
          float v66_data = r1[22];
          float v67_data = r1[23];
          float v68_data = r1[24];
          float v69_data = r1[25];
          float v70_data = r1[26];
          float v71_data = r1[27];
          float v72_data = r1[28];
          float v73_data = r1[29];
          float v74_data = r1[30];
          float v75_data = r1[31];
          tensorforge::transpose16x16b32(v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data, v67_data, v68_data, v69_data, v70_data, v71_data, v72_data, v73_data, v74_data, v75_data);
          tensorforge::VectorT<float, 16> v76_acc{};
          float v77_data = r0[0];
          float v78_data = r0[1];
          float v79_data = r0[2];
          float v80_data = r0[3];
          float v81_data = r0[4];
          float v82_data = r0[5];
          float v83_data = r0[6];
          float v84_data = r0[7];
          float v85_data = r0[8];
          float v86_data = r0[9];
          float v87_data = r0[10];
          float v88_data = r0[11];
          float v89_data = r0[12];
          float v90_data = r0[13];
          float v91_data = r0[14];
          float v92_data = r0[15];
          tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v77_data, v76_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v78_data, v93_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v79_data, v94_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v80_data, v95_acc, 0, 0, 0);
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
          float v109_data = r0[16];
          float v110_data = r0[17];
          float v111_data = r0[18];
          float v112_data = r0[19];
          tensorforge::VectorT<float, 16> v114_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v109_data, v108_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v115_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v110_data, v114_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v116_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v111_data, v115_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v112_data, v116_acc, 0, 0, 0);
          float v118_el = v117_acc[0];
          float v120_el = v117_acc[4];
          float v121_sw = tensorforge::swap<32>(v120_el);
          float v123_el = v117_acc[8];
          float v126_el = v117_acc[12];
          float v127_sw = tensorforge::swap<32>(v126_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v121_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v118_el, v118_el))))))));
          float v130_el = v117_acc[1];
          float v132_el = v117_acc[5];
          float v133_sw = tensorforge::swap<32>(v132_el);
          float v135_el = v117_acc[9];
          float v138_el = v117_acc[13];
          float v139_sw = tensorforge::swap<32>(v138_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v130_el, v130_el))))))));
          float v142_el = v117_acc[2];
          float v144_el = v117_acc[6];
          float v145_sw = tensorforge::swap<32>(v144_el);
          float v147_el = v117_acc[10];
          float v150_el = v117_acc[14];
          float v151_sw = tensorforge::swap<32>(v150_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v142_el, v142_el))))))));
          float v154_el = v117_acc[3];
          float v156_el = v117_acc[7];
          float v157_sw = tensorforge::swap<32>(v156_el);
          float v159_el = v117_acc[11];
          float v162_el = v117_acc[15];
          float v163_sw = tensorforge::swap<32>(v162_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v159_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v154_el, v154_el))))))));
          float v167_sw = tensorforge::swap<32>(v118_el);
          float v172_sw = tensorforge::swap<32>(v123_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v172_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v120_el, (tensorforge::dppUpdate<228, 1, 15, false>(v167_sw, v167_sw))))))));
          float v179_sw = tensorforge::swap<32>(v130_el);
          float v184_sw = tensorforge::swap<32>(v135_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v184_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v132_el, (tensorforge::dppUpdate<228, 1, 15, false>(v179_sw, v179_sw))))))));
          float v191_sw = tensorforge::swap<32>(v142_el);
          float v196_sw = tensorforge::swap<32>(v147_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v196_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v144_el, (tensorforge::dppUpdate<228, 1, 15, false>(v191_sw, v191_sw))))))));
          float v203_sw = tensorforge::swap<32>(v154_el);
          float v208_sw = tensorforge::swap<32>(v159_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v208_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v156_el, (tensorforge::dppUpdate<228, 1, 15, false>(v203_sw, v203_sw))))))));
          float v215_sw = tensorforge::swap<64>(v118_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v123_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v121_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v215_sw, v215_sw))))))));
          float v227_sw = tensorforge::swap<64>(v130_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v135_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v227_sw, v227_sw))))))));
          float v239_sw = tensorforge::swap<64>(v142_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v147_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v239_sw, v239_sw))))))));
          float v251_sw = tensorforge::swap<64>(v154_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v159_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v251_sw, v251_sw))))))));
          float v264_sw = tensorforge::swap<64>(v167_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v126_el, (tensorforge::dppUpdate<228, 4, 15, false>(v172_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v120_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v264_sw, v264_sw))))))));
          float v276_sw = tensorforge::swap<64>(v179_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v138_el, (tensorforge::dppUpdate<228, 4, 15, false>(v184_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v276_sw, v276_sw))))))));
          float v288_sw = tensorforge::swap<64>(v191_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v150_el, (tensorforge::dppUpdate<228, 4, 15, false>(v196_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v288_sw, v288_sw))))))));
          float v300_sw = tensorforge::swap<64>(v203_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v162_el, (tensorforge::dppUpdate<228, 4, 15, false>(v208_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v156_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v300_sw, v300_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v19_g) {
            #pragma unroll
            for (int32_t v310_i1 = 0; v310_i1 < 16; ++v310_i1) {
              float v312_data = r2[v310_i1];
              glb_m0[(v18_lead + (v310_i1 * 12))] = v312_data;
            }
          }
        }
      }
    }
  }
}

