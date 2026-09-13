// === base name ===
kernel_c1de77df608801f3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c1de77df608801f3 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c1de77df608801f3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c1de77df608801f3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c1de77df608801f3(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c1de77df608801f3, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_c1de77df608801f3, block.x * block.y * block.z, 0));
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
void launcher_kernel_c1de77df608801f3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c1de77df608801f3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c1de77df608801f3), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_c1de77df608801f3, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c1de77df608801f3(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×13(32×13) {0..32}×{0..13} strided
    //   m2 13×13(13×13) {0..13}×{0..13} strided
    //   m3 32×13(32×13) {0..32}×{0..13} strided
    //   m4 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   m0[i,j]@{0..32}×{8..9} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{8..9}
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,0],[13,1]],"is_tmp":false,"name":"m2","offset":[0,8],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 416 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 169 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 416 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v21_lead = v17_lead + (v18_i0 * 32);
            #pragma unroll
            for (int32_t v19_i1 = 10; v19_i1 < 13; ++v19_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v19_i1 * 32))]);
              r0[(v18_i0 + (v19_i1 - 10))] = v24_data;
            }
          }
          float r1[1]{};
          // r1 = load{g>r}(glb_m2);
          bool v29_g = v17_lead < 13;
          if ((v17_lead >= 10) && v29_g) {
            #pragma unroll
            for (int32_t v31_i1 = 8; v31_i1 < 9; ++v31_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v31_i1 * 13))]);
              r1[(v31_i1 - 8)] = v36_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v40_data = r0[0];
          float v41_data = r0[1];
          float v42_data = r0[2];
          float v43_acc{};
          float v44_data = r1[0];
          float v45_bc = tensorforge::broadcast<32, 16, 0>(v44_data);
          tensorforge::fmacdpp16<10>(v43_acc, v45_bc, v40_data);
          tensorforge::fmacdpp16<11>(v43_acc, v45_bc, v41_data);
          tensorforge::fmacdpp16<12>(v43_acc, v45_bc, v42_data);
          r2[0] = v43_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v46_i0 = 0; v46_i0 < 1; ++v46_i0) {
            int32_t v51_lead = v17_lead + (v46_i0 * 32);
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 1; ++v47_i1) {
              float v49_data = r2[(v46_i0 + v47_i1)];
              glb_m0[(v51_lead + ((v47_i1 + 8) * 32))] = v49_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v56_i0 = 0; v56_i0 < 1; ++v56_i0) {
            int32_t v59_lead = v17_lead + (v56_i0 * 32);
            #pragma unroll
            for (int32_t v57_i1 = 0; v57_i1 < 13; ++v57_i1) {
              float v62_data = glb_m0[(v59_lead + (v57_i1 * 32))];
              r3[(v56_i0 + v57_i1)] = v62_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          if (v29_g) {
            #pragma unroll
            for (int32_t v65_i1 = 0; v65_i1 < 13; ++v65_i1) {
              float v70_data = __builtin_nontemporal_load(&glb_m4[(v17_lead + (v65_i1 * 13))]);
              r4[v65_i1] = v70_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v73_data = r4[0];
          float v74_data = r4[1];
          float v75_data = r4[2];
          float v76_data = r4[3];
          float v77_data = r4[4];
          float v78_data = r4[5];
          float v79_data = r4[6];
          float v80_data = r4[7];
          float v81_data = r4[8];
          float v82_data = r4[9];
          float v83_data = r4[10];
          float v84_data = r4[11];
          float v85_data = r4[12];
          float v86_pad{};
          float v87_pad{};
          float v88_pad{};
          tensorforge::transpose16x16b32(v73_data, v74_data, v75_data, v76_data, v77_data, v78_data, v79_data, v80_data, v81_data, v82_data, v83_data, v84_data, v85_data, v86_pad, v87_pad, v88_pad);
          tensorforge::VectorT<float, 16> v89_acc{};
          float v90_data = r3[0];
          float v91_data = r3[1];
          float v92_data = r3[2];
          float v93_data = r3[3];
          float v94_data = r3[4];
          float v95_data = r3[5];
          float v96_data = r3[6];
          float v97_data = r3[7];
          float v98_data = r3[8];
          float v99_data = r3[9];
          float v100_data = r3[10];
          float v101_data = r3[11];
          float v102_data = r3[12];
          tensorforge::VectorT<float, 16> v104_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v73_data, v90_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v105_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v74_data, v91_data, v104_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v106_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v75_data, v92_data, v105_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v107_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v76_data, v93_data, v106_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v108_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v77_data, v94_data, v107_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v109_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v78_data, v95_data, v108_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v110_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v79_data, v96_data, v109_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v111_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v80_data, v97_data, v110_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v112_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v81_data, v98_data, v111_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v113_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v82_data, v99_data, v112_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v114_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v83_data, v100_data, v113_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v115_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v84_data, v101_data, v114_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v116_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v85_data, v102_data, v115_acc, 1, 0, 0);
          float v117_el = v116_acc[0];
          float v119_el = v116_acc[4];
          float v120_sw = tensorforge::swap<32>(v119_el);
          float v122_el = v116_acc[8];
          float v125_el = v116_acc[12];
          float v126_sw = tensorforge::swap<32>(v125_el);
          r5[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v126_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v122_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v120_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v117_el, v117_el))))))));
          float v129_el = v116_acc[1];
          float v131_el = v116_acc[5];
          float v132_sw = tensorforge::swap<32>(v131_el);
          float v134_el = v116_acc[9];
          float v137_el = v116_acc[13];
          float v138_sw = tensorforge::swap<32>(v137_el);
          r5[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v138_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v134_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v132_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v129_el, v129_el))))))));
          float v141_el = v116_acc[2];
          float v143_el = v116_acc[6];
          float v144_sw = tensorforge::swap<32>(v143_el);
          float v146_el = v116_acc[10];
          float v149_el = v116_acc[14];
          float v150_sw = tensorforge::swap<32>(v149_el);
          r5[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v150_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v146_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v144_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v141_el, v141_el))))))));
          float v153_el = v116_acc[3];
          float v155_el = v116_acc[7];
          float v156_sw = tensorforge::swap<32>(v155_el);
          float v158_el = v116_acc[11];
          float v161_el = v116_acc[15];
          float v162_sw = tensorforge::swap<32>(v161_el);
          r5[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v162_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v158_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v156_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v153_el, v153_el))))))));
          float v166_sw = tensorforge::swap<32>(v117_el);
          float v171_sw = tensorforge::swap<32>(v122_el);
          r5[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v125_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v171_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v119_el, (tensorforge::dppUpdate<228, 1, 15, false>(v166_sw, v166_sw))))))));
          float v178_sw = tensorforge::swap<32>(v129_el);
          r5[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v137_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v134_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v131_el, (tensorforge::dppUpdate<228, 1, 15, false>(v178_sw, v178_sw))))))));
          float v190_sw = tensorforge::swap<32>(v141_el);
          r5[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v149_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v146_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v143_el, (tensorforge::dppUpdate<228, 1, 15, false>(v190_sw, v190_sw))))))));
          float v202_sw = tensorforge::swap<32>(v153_el);
          r5[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v161_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v158_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v155_el, (tensorforge::dppUpdate<228, 1, 15, false>(v202_sw, v202_sw))))))));
          float v214_sw = tensorforge::swap<64>(v117_el);
          r5[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v126_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v122_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v120_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v214_sw, v214_sw))))))));
          float v226_sw = tensorforge::swap<64>(v129_el);
          r5[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v138_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v134_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v132_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v226_sw, v226_sw))))))));
          float v238_sw = tensorforge::swap<64>(v141_el);
          r5[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v150_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v146_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v144_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v238_sw, v238_sw))))))));
          float v250_sw = tensorforge::swap<64>(v153_el);
          r5[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v162_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v158_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v156_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v250_sw, v250_sw))))))));
          float v263_sw = tensorforge::swap<64>(v166_sw);
          r5[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v125_el, (tensorforge::dppUpdate<228, 4, 15, false>(v171_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v119_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v263_sw, v263_sw))))))));
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v273_i0 = 0; v273_i0 < 1; ++v273_i0) {
            int32_t v278_lead = v17_lead + (v273_i0 * 32);
            #pragma unroll
            for (int32_t v274_i1 = 0; v274_i1 < 13; ++v274_i1) {
              float v276_data = r5[(v273_i0 + v274_i1)];
              glb_m3[(v278_lead + (v274_i1 * 32))] = v276_data;
            }
          }
        }
      }
    }
  }
}

