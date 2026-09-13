// === base name ===
kernel_da801b35dc997cb3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_da801b35dc997cb3 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_da801b35dc997cb3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_da801b35dc997cb3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_da801b35dc997cb3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_da801b35dc997cb3, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_da801b35dc997cb3, block.x * block.y * block.z, 0));
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
void launcher_kernel_da801b35dc997cb3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_da801b35dc997cb3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_da801b35dc997cb3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_da801b35dc997cb3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_da801b35dc997cb3(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,13]],"name":"m2","ordered":false,"parts":1,"shape":[12,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,4],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 416 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 384 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 156 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 416 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v17_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v21_lead = v17_lead + (v18_i0 * 32);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 13; ++v19_i1) {
              float v24_data = glb_m0[(v21_lead + (v19_i1 * 32))];
              r0[(v18_i0 + v19_i1)] = v24_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
            int32_t v30_lead = v17_lead + (v27_i0 * 32);
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m1[(v30_lead + (v28_i1 * 32))]);
              r2[(v27_i0 + v28_i1)] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v36_data = r0[0];
          float v37_data = r1[0];
          r1[0] = (v37_data + v36_data);
          float v39_data = r0[1];
          float v40_data = r1[1];
          r1[1] = (v40_data + v39_data);
          float v42_data = r0[2];
          float v43_data = r1[2];
          r1[2] = (v43_data + v42_data);
          float v45_data = r0[3];
          float v46_data = r1[3];
          r1[3] = (v46_data + v45_data);
          float v48_data = r0[4];
          float v49_data = r1[4];
          r1[4] = (v49_data + v48_data);
          float v51_data = r0[5];
          float v52_data = r1[5];
          r1[5] = (v52_data + v51_data);
          float v54_data = r0[6];
          float v55_data = r1[6];
          r1[6] = (v55_data + v54_data);
          float v57_data = r0[7];
          float v58_data = r1[7];
          r1[7] = (v58_data + v57_data);
          float v60_data = r0[8];
          float v61_data = r1[8];
          r1[8] = (v61_data + v60_data);
          float v63_data = r0[9];
          float v64_data = r1[9];
          r1[9] = (v64_data + v63_data);
          float v66_data = r0[10];
          float v67_data = r1[10];
          r1[10] = (v67_data + v66_data);
          float v69_data = r0[11];
          float v70_data = r1[11];
          r1[11] = (v70_data + v69_data);
          float v72_data = r0[12];
          float v73_data = r1[12];
          r1[12] = (v73_data + v72_data);
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v77_i1 = 0; v77_i1 < 13; ++v77_i1) {
              float v82_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v77_i1 * 12))]);
              r3[v77_i1] = v82_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v86_data = r3[0];
          float v87_data = r3[1];
          float v88_data = r3[2];
          float v89_data = r3[3];
          float v90_data = r3[4];
          float v91_data = r3[5];
          float v92_data = r3[6];
          float v93_data = r3[7];
          float v94_data = r3[8];
          float v95_data = r3[9];
          float v96_data = r3[10];
          float v97_data = r3[11];
          float v98_data = r3[12];
          float v99_pad{};
          float v100_pad{};
          float v101_pad{};
          tensorforge::transpose16x16b32(v86_data, v87_data, v88_data, v89_data, v90_data, v91_data, v92_data, v93_data, v94_data, v95_data, v96_data, v97_data, v98_data, v99_pad, v100_pad, v101_pad);
          tensorforge::VectorT<float, 16> v102_acc{};
          float v103_data = r2[0];
          float v104_data = r2[1];
          float v105_data = r2[2];
          float v106_data = r2[3];
          float v107_data = r2[4];
          float v108_data = r2[5];
          float v109_data = r2[6];
          float v110_data = r2[7];
          float v111_data = r2[8];
          float v112_data = r2[9];
          float v113_data = r2[10];
          float v114_data = r2[11];
          tensorforge::VectorT<float, 16> v116_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v86_data, v103_data, v102_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v87_data, v104_data, v116_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v118_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v88_data, v105_data, v117_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v119_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v89_data, v106_data, v118_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v90_data, v107_data, v119_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v121_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v91_data, v108_data, v120_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v122_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v92_data, v109_data, v121_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v93_data, v110_data, v122_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v124_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v94_data, v111_data, v123_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v125_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v95_data, v112_data, v124_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v126_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v96_data, v113_data, v125_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v127_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v97_data, v114_data, v126_acc, 1, 0, 0);
          float v128_el = v127_acc[0];
          float v130_el = v127_acc[4];
          float v131_sw = tensorforge::swap<32>(v130_el);
          float v133_el = v127_acc[8];
          float v136_el = v127_acc[12];
          float v137_sw = tensorforge::swap<32>(v136_el);
          ir4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v137_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v133_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v131_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v128_el, v128_el))))))));
          float v140_el = v127_acc[1];
          float v142_el = v127_acc[5];
          float v143_sw = tensorforge::swap<32>(v142_el);
          float v145_el = v127_acc[9];
          float v148_el = v127_acc[13];
          float v149_sw = tensorforge::swap<32>(v148_el);
          ir4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v149_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v145_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v143_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v140_el, v140_el))))))));
          float v152_el = v127_acc[2];
          float v154_el = v127_acc[6];
          float v155_sw = tensorforge::swap<32>(v154_el);
          float v157_el = v127_acc[10];
          float v160_el = v127_acc[14];
          float v161_sw = tensorforge::swap<32>(v160_el);
          ir4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v161_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v157_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v155_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v152_el, v152_el))))))));
          float v164_el = v127_acc[3];
          float v166_el = v127_acc[7];
          float v167_sw = tensorforge::swap<32>(v166_el);
          float v169_el = v127_acc[11];
          float v172_el = v127_acc[15];
          float v173_sw = tensorforge::swap<32>(v172_el);
          ir4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v173_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v169_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v167_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v164_el, v164_el))))))));
          float v177_sw = tensorforge::swap<32>(v128_el);
          float v182_sw = tensorforge::swap<32>(v133_el);
          ir4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v136_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v182_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v130_el, (tensorforge::dppUpdate<228, 1, 15, false>(v177_sw, v177_sw))))))));
          float v189_sw = tensorforge::swap<32>(v140_el);
          ir4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v148_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v145_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v142_el, (tensorforge::dppUpdate<228, 1, 15, false>(v189_sw, v189_sw))))))));
          float v201_sw = tensorforge::swap<32>(v152_el);
          ir4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v160_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v157_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v154_el, (tensorforge::dppUpdate<228, 1, 15, false>(v201_sw, v201_sw))))))));
          float v213_sw = tensorforge::swap<32>(v164_el);
          ir4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v172_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v169_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v166_el, (tensorforge::dppUpdate<228, 1, 15, false>(v213_sw, v213_sw))))))));
          float v225_sw = tensorforge::swap<64>(v128_el);
          ir4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v137_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v133_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v131_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v225_sw, v225_sw))))))));
          float v237_sw = tensorforge::swap<64>(v140_el);
          ir4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v149_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v145_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v143_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v237_sw, v237_sw))))))));
          float v249_sw = tensorforge::swap<64>(v152_el);
          ir4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v161_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v157_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v155_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v249_sw, v249_sw))))))));
          float v261_sw = tensorforge::swap<64>(v164_el);
          ir4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v173_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v169_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v167_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v261_sw, v261_sw))))))));
          float v274_sw = tensorforge::swap<64>(v177_sw);
          ir4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v136_el, (tensorforge::dppUpdate<228, 4, 15, false>(v182_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v274_sw, v274_sw))))))));
          #pragma unroll
          for (int32_t v284_n0 = 0; v284_n0 < 1; ++v284_n0) {
            #pragma unroll
            for (int32_t v285_n1 = 0; v285_n1 < 13; ++v285_n1) {
              int32_t v286_a = v284_n0 + v285_n1;
              float v287_data = ir4[v286_a];
              float v288_data = r1[v286_a];
              r4[v286_a] = (v288_data + v287_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v291_data = r4[4];
          float v292_data = r5[0];
          r5[0] = (v292_data + v291_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v294_i0 = 0; v294_i0 < 1; ++v294_i0) {
            int32_t v299_lead = v17_lead + (v294_i0 * 32);
            #pragma unroll
            for (int32_t v295_i1 = 0; v295_i1 < 1; ++v295_i1) {
              float v297_data = r5[(v294_i0 + v295_i1)];
              glb_m0[(v299_lead + ((v295_i1 + 4) * 32))] = v297_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v304_i0 = 0; v304_i0 < 1; ++v304_i0) {
            int32_t v307_lead = v17_lead + (v304_i0 * 32);
            #pragma unroll
            for (int32_t v305_i1 = 0; v305_i1 < 13; ++v305_i1) {
              float v310_data = glb_m0[(v307_lead + (v305_i1 * 32))];
              r6[(v304_i0 + v305_i1)] = v310_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          if (v17_lead < 13) {
            #pragma unroll
            for (int32_t v314_i1 = 0; v314_i1 < 13; ++v314_i1) {
              float v319_data = __builtin_nontemporal_load(&glb_m4[(v17_lead + (v314_i1 * 13))]);
              r7[v314_i1] = v319_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v322_data = r7[0];
          float v323_data = r7[1];
          float v324_data = r7[2];
          float v325_data = r7[3];
          float v326_data = r7[4];
          float v327_data = r7[5];
          float v328_data = r7[6];
          float v329_data = r7[7];
          float v330_data = r7[8];
          float v331_data = r7[9];
          float v332_data = r7[10];
          float v333_data = r7[11];
          float v334_data = r7[12];
          float v335_pad{};
          float v336_pad{};
          float v337_pad{};
          tensorforge::transpose16x16b32(v322_data, v323_data, v324_data, v325_data, v326_data, v327_data, v328_data, v329_data, v330_data, v331_data, v332_data, v333_data, v334_data, v335_pad, v336_pad, v337_pad);
          tensorforge::VectorT<float, 16> v338_acc{};
          float v339_data = r6[0];
          float v340_data = r6[1];
          float v341_data = r6[2];
          float v342_data = r6[3];
          float v343_data = r6[4];
          float v344_data = r6[5];
          float v345_data = r6[6];
          float v346_data = r6[7];
          float v347_data = r6[8];
          float v348_data = r6[9];
          float v349_data = r6[10];
          float v350_data = r6[11];
          float v351_data = r6[12];
          tensorforge::VectorT<float, 16> v353_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v322_data, v339_data, v338_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v354_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v323_data, v340_data, v353_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v355_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v324_data, v341_data, v354_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v356_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v325_data, v342_data, v355_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v357_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v326_data, v343_data, v356_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v358_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v327_data, v344_data, v357_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v359_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v345_data, v358_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v360_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v346_data, v359_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v361_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v347_data, v360_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v362_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v348_data, v361_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v363_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v362_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v363_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v351_data, v364_acc, 1, 0, 0);
          float v366_el = v365_acc[0];
          float v368_el = v365_acc[4];
          float v369_sw = tensorforge::swap<32>(v368_el);
          float v371_el = v365_acc[8];
          float v374_el = v365_acc[12];
          float v375_sw = tensorforge::swap<32>(v374_el);
          r8[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v375_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v371_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v369_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v366_el, v366_el))))))));
          float v378_el = v365_acc[1];
          float v380_el = v365_acc[5];
          float v381_sw = tensorforge::swap<32>(v380_el);
          float v383_el = v365_acc[9];
          float v386_el = v365_acc[13];
          float v387_sw = tensorforge::swap<32>(v386_el);
          r8[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v387_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v383_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v381_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v378_el, v378_el))))))));
          float v390_el = v365_acc[2];
          float v392_el = v365_acc[6];
          float v393_sw = tensorforge::swap<32>(v392_el);
          float v395_el = v365_acc[10];
          float v398_el = v365_acc[14];
          float v399_sw = tensorforge::swap<32>(v398_el);
          r8[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v399_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v395_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v393_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v390_el, v390_el))))))));
          float v402_el = v365_acc[3];
          float v404_el = v365_acc[7];
          float v405_sw = tensorforge::swap<32>(v404_el);
          float v407_el = v365_acc[11];
          float v410_el = v365_acc[15];
          float v411_sw = tensorforge::swap<32>(v410_el);
          r8[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v411_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v407_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v405_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v402_el, v402_el))))))));
          float v415_sw = tensorforge::swap<32>(v366_el);
          float v420_sw = tensorforge::swap<32>(v371_el);
          r8[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v374_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v420_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v368_el, (tensorforge::dppUpdate<228, 1, 15, false>(v415_sw, v415_sw))))))));
          float v427_sw = tensorforge::swap<32>(v378_el);
          r8[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v386_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v383_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v380_el, (tensorforge::dppUpdate<228, 1, 15, false>(v427_sw, v427_sw))))))));
          float v439_sw = tensorforge::swap<32>(v390_el);
          r8[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v398_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v395_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v392_el, (tensorforge::dppUpdate<228, 1, 15, false>(v439_sw, v439_sw))))))));
          float v451_sw = tensorforge::swap<32>(v402_el);
          r8[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v410_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v407_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v404_el, (tensorforge::dppUpdate<228, 1, 15, false>(v451_sw, v451_sw))))))));
          float v463_sw = tensorforge::swap<64>(v366_el);
          r8[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v375_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v371_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v369_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v463_sw, v463_sw))))))));
          float v475_sw = tensorforge::swap<64>(v378_el);
          r8[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v387_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v383_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v381_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v475_sw, v475_sw))))))));
          float v487_sw = tensorforge::swap<64>(v390_el);
          r8[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v399_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v395_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v393_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v487_sw, v487_sw))))))));
          float v499_sw = tensorforge::swap<64>(v402_el);
          r8[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v411_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v407_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v405_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v499_sw, v499_sw))))))));
          float v512_sw = tensorforge::swap<64>(v415_sw);
          r8[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v374_el, (tensorforge::dppUpdate<228, 4, 15, false>(v420_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v368_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v512_sw, v512_sw))))))));
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v522_i0 = 0; v522_i0 < 1; ++v522_i0) {
            int32_t v527_lead = v17_lead + (v522_i0 * 32);
            #pragma unroll
            for (int32_t v523_i1 = 0; v523_i1 < 13; ++v523_i1) {
              float v525_data = r8[(v522_i0 + v523_i1)];
              glb_m3[(v527_lead + (v523_i1 * 32))] = v525_data;
            }
          }
        }
      }
    }
  }
}

