// === base name ===
kernel_6451ee1f10a2d174

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6451ee1f10a2d174 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6451ee1f10a2d174(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6451ee1f10a2d174(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6451ee1f10a2d174(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6451ee1f10a2d174, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6451ee1f10a2d174, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_6451ee1f10a2d174(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6451ee1f10a2d174(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6451ee1f10a2d174), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_6451ee1f10a2d174, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6451ee1f10a2d174(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×12(32×12) {0..32}×{0..12} strided
    //   m2 12×13(12×13) {0..12}×{0..13} strided
    //   m3 32×13(32×13) {0..32}×{0..13} strided
    //   m4 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   t0[i,j] = m0[i,j]
    //   t0[i,j] += m1[i,k] × m2[k,j]
    //   m0[i,j]@{0..32}×{4..5} = t0[i,j]@{0..32}×{4..5}
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,13]],"name":"m2","ordered":false,"parts":1,"shape":[12,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,4],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 416 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 384 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 156 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 416 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 32);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 13; ++v25_i1) {
              float v30_data = glb_m0[(v27_lead + (v25_i1 * 32))];
              r0[(v24_i0 + v25_i1)] = v30_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
            int32_t v76_lead = v23_lead + (v73_i0 * 32);
            #pragma unroll
            for (int32_t v74_i1 = 0; v74_i1 < 12; ++v74_i1) {
              float v79_data = __builtin_nontemporal_load(&glb_m1[(v76_lead + (v74_i1 * 32))]);
              r2[(v73_i0 + v74_i1)] = v79_data;
            }
          }
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v33_data = r0[0];
          float v34_data = r1[0];
          r1[0] = (v34_data + v33_data);
          float v36_data = r0[1];
          float v37_data = r1[1];
          r1[1] = (v37_data + v36_data);
          float v39_data = r0[2];
          float v40_data = r1[2];
          r1[2] = (v40_data + v39_data);
          float v42_data = r0[3];
          float v43_data = r1[3];
          r1[3] = (v43_data + v42_data);
          float v45_data = r0[4];
          float v46_data = r1[4];
          r1[4] = (v46_data + v45_data);
          float v48_data = r0[5];
          float v49_data = r1[5];
          r1[5] = (v49_data + v48_data);
          float v51_data = r0[6];
          float v52_data = r1[6];
          r1[6] = (v52_data + v51_data);
          float v54_data = r0[7];
          float v55_data = r1[7];
          r1[7] = (v55_data + v54_data);
          float v57_data = r0[8];
          float v58_data = r1[8];
          r1[8] = (v58_data + v57_data);
          float v60_data = r0[9];
          float v61_data = r1[9];
          r1[9] = (v61_data + v60_data);
          float v63_data = r0[10];
          float v64_data = r1[10];
          r1[10] = (v64_data + v63_data);
          float v66_data = r0[11];
          float v67_data = r1[11];
          r1[11] = (v67_data + v66_data);
          float v69_data = r0[12];
          float v70_data = r1[12];
          r1[12] = (v70_data + v69_data);
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v83_i1 = 0; v83_i1 < 13; ++v83_i1) {
              float v88_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v83_i1 * 12))]);
              r3[v83_i1] = v88_data;
            }
          }
          float r4[13]{};
          // ir4 = +(r2 * r3)
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v92_data = r3[0];
          float v93_data = r3[1];
          float v94_data = r3[2];
          float v95_data = r3[3];
          float v96_data = r3[4];
          float v97_data = r3[5];
          float v98_data = r3[6];
          float v99_data = r3[7];
          float v100_data = r3[8];
          float v101_data = r3[9];
          float v102_data = r3[10];
          float v103_data = r3[11];
          float v104_data = r3[12];
          float v105_pad{};
          float v106_pad{};
          float v107_pad{};
          tensorforge::transpose16x16b32(v92_data, v93_data, v94_data, v95_data, v96_data, v97_data, v98_data, v99_data, v100_data, v101_data, v102_data, v103_data, v104_data, v105_pad, v106_pad, v107_pad);
          tensorforge::VectorT<float, 16> v108_acc{};
          float v109_data = r2[0];
          float v110_data = r2[1];
          float v111_data = r2[2];
          float v112_data = r2[3];
          float v113_data = r2[4];
          float v114_data = r2[5];
          float v115_data = r2[6];
          float v116_data = r2[7];
          float v117_data = r2[8];
          float v118_data = r2[9];
          float v119_data = r2[10];
          float v120_data = r2[11];
          tensorforge::VectorT<float, 16> v122_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v92_data, v109_data, v108_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v93_data, v110_data, v122_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v124_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v94_data, v111_data, v123_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v125_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v95_data, v112_data, v124_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v126_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v96_data, v113_data, v125_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v127_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v97_data, v114_data, v126_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v128_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v98_data, v115_data, v127_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v129_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v99_data, v116_data, v128_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v130_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v100_data, v117_data, v129_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v131_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v101_data, v118_data, v130_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v132_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v102_data, v119_data, v131_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v133_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v103_data, v120_data, v132_acc, 1, 0, 0);
          float v134_el = v133_acc[0];
          float v136_el = v133_acc[4];
          float v137_sw = tensorforge::swap<32>(v136_el);
          float v139_el = v133_acc[8];
          float v142_el = v133_acc[12];
          float v143_sw = tensorforge::swap<32>(v142_el);
          ir4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v143_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v139_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v137_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v134_el, v134_el))))))));
          float v146_el = v133_acc[1];
          float v148_el = v133_acc[5];
          float v149_sw = tensorforge::swap<32>(v148_el);
          float v151_el = v133_acc[9];
          float v154_el = v133_acc[13];
          float v155_sw = tensorforge::swap<32>(v154_el);
          ir4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v155_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v151_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v149_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v146_el, v146_el))))))));
          float v158_el = v133_acc[2];
          float v160_el = v133_acc[6];
          float v161_sw = tensorforge::swap<32>(v160_el);
          float v163_el = v133_acc[10];
          float v166_el = v133_acc[14];
          float v167_sw = tensorforge::swap<32>(v166_el);
          ir4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v167_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v163_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v161_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v158_el, v158_el))))))));
          float v170_el = v133_acc[3];
          float v172_el = v133_acc[7];
          float v173_sw = tensorforge::swap<32>(v172_el);
          float v175_el = v133_acc[11];
          float v178_el = v133_acc[15];
          float v179_sw = tensorforge::swap<32>(v178_el);
          ir4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v179_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v175_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v173_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v170_el, v170_el))))))));
          float v183_sw = tensorforge::swap<32>(v134_el);
          float v188_sw = tensorforge::swap<32>(v139_el);
          ir4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v142_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v188_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v136_el, (tensorforge::dppUpdate<228, 1, 15, false>(v183_sw, v183_sw))))))));
          float v195_sw = tensorforge::swap<32>(v146_el);
          ir4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v154_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v151_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v148_el, (tensorforge::dppUpdate<228, 1, 15, false>(v195_sw, v195_sw))))))));
          float v207_sw = tensorforge::swap<32>(v158_el);
          ir4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v166_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v163_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v160_el, (tensorforge::dppUpdate<228, 1, 15, false>(v207_sw, v207_sw))))))));
          float v219_sw = tensorforge::swap<32>(v170_el);
          ir4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v178_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v175_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v172_el, (tensorforge::dppUpdate<228, 1, 15, false>(v219_sw, v219_sw))))))));
          float v231_sw = tensorforge::swap<64>(v134_el);
          ir4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v143_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v139_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v137_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v231_sw, v231_sw))))))));
          float v243_sw = tensorforge::swap<64>(v146_el);
          ir4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v155_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v151_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v149_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v243_sw, v243_sw))))))));
          float v255_sw = tensorforge::swap<64>(v158_el);
          ir4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v167_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v163_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v161_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v255_sw, v255_sw))))))));
          float v267_sw = tensorforge::swap<64>(v170_el);
          ir4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v179_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v175_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v173_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v267_sw, v267_sw))))))));
          float v280_sw = tensorforge::swap<64>(v183_sw);
          ir4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v142_el, (tensorforge::dppUpdate<228, 4, 15, false>(v188_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v136_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v280_sw, v280_sw))))))));
          // r4 = ir4 + r1
          #pragma unroll
          for (int32_t v290_n0 = 0; v290_n0 < 1; ++v290_n0) {
            #pragma unroll
            for (int32_t v291_n1 = 0; v291_n1 < 13; ++v291_n1) {
              int32_t v292_a = v290_n0 + v291_n1;
              float v293_data = ir4[v292_a];
              float v294_data = r1[v292_a];
              r4[v292_a] = (v294_data + v293_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v297_data = r4[4];
          float v298_data = r5[0];
          r5[0] = (v298_data + v297_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v300_i0 = 0; v300_i0 < 1; ++v300_i0) {
            int32_t v305_lead = v23_lead + (v300_i0 * 32);
            #pragma unroll
            for (int32_t v301_i1 = 0; v301_i1 < 1; ++v301_i1) {
              float v303_data = r5[(v300_i0 + v301_i1)];
              glb_m0[(v305_lead + ((v301_i1 + 4) * 32))] = v303_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v310_i0 = 0; v310_i0 < 1; ++v310_i0) {
            int32_t v313_lead = v23_lead + (v310_i0 * 32);
            #pragma unroll
            for (int32_t v311_i1 = 0; v311_i1 < 13; ++v311_i1) {
              float v316_data = glb_m0[(v313_lead + (v311_i1 * 32))];
              r6[(v310_i0 + v311_i1)] = v316_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          if (v23_lead < 13) {
            #pragma unroll
            for (int32_t v320_i1 = 0; v320_i1 < 13; ++v320_i1) {
              float v325_data = __builtin_nontemporal_load(&glb_m4[(v23_lead + (v320_i1 * 13))]);
              r7[v320_i1] = v325_data;
            }
          }
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v328_data = r7[0];
          float v329_data = r7[1];
          float v330_data = r7[2];
          float v331_data = r7[3];
          float v332_data = r7[4];
          float v333_data = r7[5];
          float v334_data = r7[6];
          float v335_data = r7[7];
          float v336_data = r7[8];
          float v337_data = r7[9];
          float v338_data = r7[10];
          float v339_data = r7[11];
          float v340_data = r7[12];
          float v341_pad{};
          float v342_pad{};
          float v343_pad{};
          tensorforge::transpose16x16b32(v328_data, v329_data, v330_data, v331_data, v332_data, v333_data, v334_data, v335_data, v336_data, v337_data, v338_data, v339_data, v340_data, v341_pad, v342_pad, v343_pad);
          tensorforge::VectorT<float, 16> v344_acc{};
          float v345_data = r6[0];
          float v346_data = r6[1];
          float v347_data = r6[2];
          float v348_data = r6[3];
          float v349_data = r6[4];
          float v350_data = r6[5];
          float v351_data = r6[6];
          float v352_data = r6[7];
          float v353_data = r6[8];
          float v354_data = r6[9];
          float v355_data = r6[10];
          float v356_data = r6[11];
          float v357_data = r6[12];
          tensorforge::VectorT<float, 16> v359_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v345_data, v344_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v360_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v346_data, v359_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v361_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v347_data, v360_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v362_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v348_data, v361_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v363_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v362_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v363_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v351_data, v364_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v352_data, v365_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v367_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v336_data, v353_data, v366_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v337_data, v354_data, v367_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v369_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v355_data, v368_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v370_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v339_data, v356_data, v369_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v371_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v340_data, v357_data, v370_acc, 1, 0, 0);
          float v372_el = v371_acc[0];
          float v374_el = v371_acc[4];
          float v375_sw = tensorforge::swap<32>(v374_el);
          float v377_el = v371_acc[8];
          float v380_el = v371_acc[12];
          float v381_sw = tensorforge::swap<32>(v380_el);
          r8[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v381_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v377_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v375_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v372_el, v372_el))))))));
          float v384_el = v371_acc[1];
          float v386_el = v371_acc[5];
          float v387_sw = tensorforge::swap<32>(v386_el);
          float v389_el = v371_acc[9];
          float v392_el = v371_acc[13];
          float v393_sw = tensorforge::swap<32>(v392_el);
          r8[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v393_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v389_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v387_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v384_el, v384_el))))))));
          float v396_el = v371_acc[2];
          float v398_el = v371_acc[6];
          float v399_sw = tensorforge::swap<32>(v398_el);
          float v401_el = v371_acc[10];
          float v404_el = v371_acc[14];
          float v405_sw = tensorforge::swap<32>(v404_el);
          r8[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v405_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v401_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v399_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v396_el, v396_el))))))));
          float v408_el = v371_acc[3];
          float v410_el = v371_acc[7];
          float v411_sw = tensorforge::swap<32>(v410_el);
          float v413_el = v371_acc[11];
          float v416_el = v371_acc[15];
          float v417_sw = tensorforge::swap<32>(v416_el);
          r8[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v417_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v413_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v411_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v408_el, v408_el))))))));
          float v421_sw = tensorforge::swap<32>(v372_el);
          float v426_sw = tensorforge::swap<32>(v377_el);
          r8[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v380_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v426_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v374_el, (tensorforge::dppUpdate<228, 1, 15, false>(v421_sw, v421_sw))))))));
          float v433_sw = tensorforge::swap<32>(v384_el);
          r8[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v392_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v389_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v386_el, (tensorforge::dppUpdate<228, 1, 15, false>(v433_sw, v433_sw))))))));
          float v445_sw = tensorforge::swap<32>(v396_el);
          r8[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v404_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v401_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v398_el, (tensorforge::dppUpdate<228, 1, 15, false>(v445_sw, v445_sw))))))));
          float v457_sw = tensorforge::swap<32>(v408_el);
          r8[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v416_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v413_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v410_el, (tensorforge::dppUpdate<228, 1, 15, false>(v457_sw, v457_sw))))))));
          float v469_sw = tensorforge::swap<64>(v372_el);
          r8[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v381_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v377_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v375_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v469_sw, v469_sw))))))));
          float v481_sw = tensorforge::swap<64>(v384_el);
          r8[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v393_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v389_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v387_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v481_sw, v481_sw))))))));
          float v493_sw = tensorforge::swap<64>(v396_el);
          r8[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v405_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v401_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v399_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v493_sw, v493_sw))))))));
          float v505_sw = tensorforge::swap<64>(v408_el);
          r8[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v417_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v413_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v411_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v505_sw, v505_sw))))))));
          float v518_sw = tensorforge::swap<64>(v421_sw);
          r8[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v380_el, (tensorforge::dppUpdate<228, 4, 15, false>(v426_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v374_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v518_sw, v518_sw))))))));
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v528_i0 = 0; v528_i0 < 1; ++v528_i0) {
            int32_t v533_lead = v23_lead + (v528_i0 * 32);
            #pragma unroll
            for (int32_t v529_i1 = 0; v529_i1 < 13; ++v529_i1) {
              float v531_data = r8[(v528_i0 + v529_i1)];
              glb_m3[(v533_lead + (v529_i1 * 32))] = v531_data;
            }
          }
        }
      }
    }
  }
}

