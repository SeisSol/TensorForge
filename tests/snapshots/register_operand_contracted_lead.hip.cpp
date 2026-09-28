// === base name ===
kernel_8d3212a5dc220532

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8d3212a5dc220532 = {{32, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8d3212a5dc220532(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8d3212a5dc220532(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8d3212a5dc220532(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8d3212a5dc220532, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_8d3212a5dc220532, block.x * block.y * block.z, 0));
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
void launcher_kernel_8d3212a5dc220532(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8d3212a5dc220532(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8d3212a5dc220532), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_8d3212a5dc220532, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8d3212a5dc220532(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (64 active) x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 64×13(64×13) {0..64}×{0..13} strided
    //   m1 13×13(13×13) {0..13}×{0..13} strided
    //   m2 13×13(13×13) {0..13}×{0..13} strided
    //   m3 64×13(64×13) {0..64}×{0..13} strided
    //   m4 64×56(64×56) {0..64}×{0..56} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = t0[i,k]@{35..54}×{0..13} × m2[k,j]
    //   m3[i,j] = m4[i,k]@{0..64}×{35..54} × t1[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[13,13]],"name":"m1","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[64,13]],"name":"m3","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"strided","alias":"K","bbox":[[0,0],[64,56]],"name":"m4","ordered":false,"parts":1,"shape":[64,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[19,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[19,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[19,13]],"is_tmp":true,"name":"t0","offset":[35,0],"shape":[64,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[64,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,19]],"is_tmp":false,"name":"m4","offset":[0,35],"shape":[64,56]},{"addressing":"strided","bbox":[[0,0],[19,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[19,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 832 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 169 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 169 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 832 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 3584 + 0 + m4_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v17_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 2; ++v18_i0) {
            int32_t v21_lead = v17_lead + (v18_i0 * 32);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 13; ++v19_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m0[(v21_lead + (v19_i1 * 64))]);
              r0[(v18_i0 + (v19_i1 * 2))] = v24_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m1);
          bool v28_g = v17_lead < 13;
          if (v28_g) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 13; ++v29_i1) {
              float v34_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v29_i1 * 13))]);
              r1[v29_i1] = v34_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v28_g) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 13; ++v37_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v37_i1 * 13))]);
              r3[v37_i1] = v42_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[26]{};
          // r2 = +(r0 * r1) + None
          // [(0, 64), (0, 13)] [(0, 13)]
          float v45_data = r1[0];
          float v46_data = r1[1];
          float v47_data = r1[2];
          float v48_data = r1[3];
          float v49_data = r1[4];
          float v50_data = r1[5];
          float v51_data = r1[6];
          float v52_data = r1[7];
          float v53_data = r1[8];
          float v54_data = r1[9];
          float v55_data = r1[10];
          float v56_data = r1[11];
          float v57_data = r1[12];
          float v58_pad{};
          float v59_pad{};
          float v60_pad{};
          tensorforge::transpose16x16b32(v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_pad, v59_pad, v60_pad);
          tensorforge::VectorT<float, 16> v61_acc{};
          float v62_data = r0[0];
          float v63_data = r0[2];
          float v64_data = r0[4];
          float v65_data = r0[6];
          float v66_data = r0[8];
          float v67_data = r0[10];
          float v68_data = r0[12];
          float v69_data = r0[14];
          float v70_data = r0[16];
          float v71_data = r0[18];
          float v72_data = r0[20];
          float v73_data = r0[22];
          float v74_data = r0[24];
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v61_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v87_acc, 1, 0, 0);
          float v89_el = v88_acc[0];
          float v91_el = v88_acc[4];
          float v92_sw = tensorforge::swap<32>(v91_el);
          float v94_el = v88_acc[8];
          float v97_el = v88_acc[12];
          float v98_sw = tensorforge::swap<32>(v97_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v98_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v94_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v92_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v89_el, v89_el))))))));
          float v101_el = v88_acc[1];
          float v103_el = v88_acc[5];
          float v104_sw = tensorforge::swap<32>(v103_el);
          float v106_el = v88_acc[9];
          float v109_el = v88_acc[13];
          float v110_sw = tensorforge::swap<32>(v109_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v110_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v104_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v101_el, v101_el))))))));
          float v113_el = v88_acc[2];
          float v115_el = v88_acc[6];
          float v116_sw = tensorforge::swap<32>(v115_el);
          float v118_el = v88_acc[10];
          float v121_el = v88_acc[14];
          float v122_sw = tensorforge::swap<32>(v121_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v122_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v116_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v113_el, v113_el))))))));
          float v125_el = v88_acc[3];
          float v127_el = v88_acc[7];
          float v128_sw = tensorforge::swap<32>(v127_el);
          float v130_el = v88_acc[11];
          float v133_el = v88_acc[15];
          float v134_sw = tensorforge::swap<32>(v133_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v134_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v128_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v125_el, v125_el))))))));
          float v138_sw = tensorforge::swap<32>(v89_el);
          float v143_sw = tensorforge::swap<32>(v94_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v143_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v91_el, (tensorforge::dppUpdate<228, 1, 15, false>(v138_sw, v138_sw))))))));
          float v150_sw = tensorforge::swap<32>(v101_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v106_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v103_el, (tensorforge::dppUpdate<228, 1, 15, false>(v150_sw, v150_sw))))))));
          float v162_sw = tensorforge::swap<32>(v113_el);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v118_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v115_el, (tensorforge::dppUpdate<228, 1, 15, false>(v162_sw, v162_sw))))))));
          float v174_sw = tensorforge::swap<32>(v125_el);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v130_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v127_el, (tensorforge::dppUpdate<228, 1, 15, false>(v174_sw, v174_sw))))))));
          float v186_sw = tensorforge::swap<64>(v89_el);
          r2[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v98_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v94_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v92_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v186_sw, v186_sw))))))));
          float v198_sw = tensorforge::swap<64>(v101_el);
          r2[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v110_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v106_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v104_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v198_sw, v198_sw))))))));
          float v210_sw = tensorforge::swap<64>(v113_el);
          r2[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v122_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v118_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v116_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v210_sw, v210_sw))))))));
          float v222_sw = tensorforge::swap<64>(v125_el);
          r2[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v134_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v130_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v128_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v222_sw, v222_sw))))))));
          float v235_sw = tensorforge::swap<64>(v138_sw);
          r2[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v97_el, (tensorforge::dppUpdate<228, 4, 15, false>(v143_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v91_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v235_sw, v235_sw))))))));
          tensorforge::VectorT<float, 16> v245_acc{};
          float v246_data = r0[1];
          float v247_data = r0[3];
          float v248_data = r0[5];
          float v249_data = r0[7];
          float v250_data = r0[9];
          float v251_data = r0[11];
          float v252_data = r0[13];
          float v253_data = r0[15];
          float v254_data = r0[17];
          float v255_data = r0[19];
          float v256_data = r0[21];
          float v257_data = r0[23];
          float v258_data = r0[25];
          tensorforge::VectorT<float, 16> v260_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v246_data, v245_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v261_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v247_data, v260_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v262_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v248_data, v261_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v263_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v249_data, v262_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v264_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v250_data, v263_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v265_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v251_data, v264_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v266_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v252_data, v265_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v267_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v253_data, v266_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v268_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v254_data, v267_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v269_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v255_data, v268_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v270_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v256_data, v269_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v271_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v257_data, v270_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v272_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v258_data, v271_acc, 1, 0, 0);
          float v273_el = v272_acc[0];
          float v275_el = v272_acc[4];
          float v276_sw = tensorforge::swap<32>(v275_el);
          float v278_el = v272_acc[8];
          float v281_el = v272_acc[12];
          float v282_sw = tensorforge::swap<32>(v281_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v282_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v278_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v276_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v273_el, v273_el))))))));
          float v285_el = v272_acc[1];
          float v287_el = v272_acc[5];
          float v288_sw = tensorforge::swap<32>(v287_el);
          float v290_el = v272_acc[9];
          float v293_el = v272_acc[13];
          float v294_sw = tensorforge::swap<32>(v293_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v294_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v290_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v288_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v285_el, v285_el))))))));
          float v297_el = v272_acc[2];
          float v299_el = v272_acc[6];
          float v300_sw = tensorforge::swap<32>(v299_el);
          float v302_el = v272_acc[10];
          float v305_el = v272_acc[14];
          float v306_sw = tensorforge::swap<32>(v305_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v306_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v302_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v300_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v297_el, v297_el))))))));
          float v309_el = v272_acc[3];
          float v311_el = v272_acc[7];
          float v312_sw = tensorforge::swap<32>(v311_el);
          float v314_el = v272_acc[11];
          float v317_el = v272_acc[15];
          float v318_sw = tensorforge::swap<32>(v317_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v318_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v314_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v312_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v309_el, v309_el))))))));
          float v322_sw = tensorforge::swap<32>(v273_el);
          float v327_sw = tensorforge::swap<32>(v278_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v281_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v327_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v275_el, (tensorforge::dppUpdate<228, 1, 15, false>(v322_sw, v322_sw))))))));
          float v334_sw = tensorforge::swap<32>(v285_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v293_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v290_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v287_el, (tensorforge::dppUpdate<228, 1, 15, false>(v334_sw, v334_sw))))))));
          float v346_sw = tensorforge::swap<32>(v297_el);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v305_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v302_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v299_el, (tensorforge::dppUpdate<228, 1, 15, false>(v346_sw, v346_sw))))))));
          float v358_sw = tensorforge::swap<32>(v309_el);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v317_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v314_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v311_el, (tensorforge::dppUpdate<228, 1, 15, false>(v358_sw, v358_sw))))))));
          float v370_sw = tensorforge::swap<64>(v273_el);
          r2[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v282_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v278_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v276_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v370_sw, v370_sw))))))));
          float v382_sw = tensorforge::swap<64>(v285_el);
          r2[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v294_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v290_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v288_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v382_sw, v382_sw))))))));
          float v394_sw = tensorforge::swap<64>(v297_el);
          r2[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v306_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v302_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v300_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v394_sw, v394_sw))))))));
          float v406_sw = tensorforge::swap<64>(v309_el);
          r2[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v318_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v314_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v312_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v406_sw, v406_sw))))))));
          float v419_sw = tensorforge::swap<64>(v322_sw);
          r2[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v281_el, (tensorforge::dppUpdate<228, 4, 15, false>(v327_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v275_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v419_sw, v419_sw))))))));
          float r5[38]{};
          // r5 = load{g>r}(glb_m4);
          #pragma unroll
          for (int32_t v430_i0 = 0; v430_i0 < 2; ++v430_i0) {
            int32_t v433_lead = v17_lead + (v430_i0 * 32);
            #pragma unroll
            for (int32_t v431_i1 = 35; v431_i1 < 54; ++v431_i1) {
              float v436_data = __builtin_nontemporal_load(&glb_m4[(v433_lead + (v431_i1 * 64))]);
              r5[(v430_i0 + ((v431_i1 - 35) * 2))] = v436_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + None
          // [(3, 22), (0, 13)] [(0, 13)]
          float v441_data = r3[0];
          float v442_data = r3[1];
          float v443_data = r3[2];
          float v444_data = r3[3];
          float v445_data = r3[4];
          float v446_data = r3[5];
          float v447_data = r3[6];
          float v448_data = r3[7];
          float v449_data = r3[8];
          float v450_data = r3[9];
          float v451_data = r3[10];
          float v452_data = r3[11];
          float v453_data = r3[12];
          float v454_pad{};
          float v455_pad{};
          float v456_pad{};
          tensorforge::transpose16x16b32(v441_data, v442_data, v443_data, v444_data, v445_data, v446_data, v447_data, v448_data, v449_data, v450_data, v451_data, v452_data, v453_data, v454_pad, v455_pad, v456_pad);
          tensorforge::VectorT<float, 16> v457_acc{};
          float v458_data = r2[1];
          float v459_data = r2[3];
          float v460_data = r2[5];
          float v461_data = r2[7];
          float v462_data = r2[9];
          float v463_data = r2[11];
          float v464_data = r2[13];
          float v465_data = r2[15];
          float v466_data = r2[17];
          float v467_data = r2[19];
          float v468_data = r2[21];
          float v469_data = r2[23];
          float v470_data = r2[25];
          tensorforge::VectorT<float, 16> v472_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v441_data, v458_data, v457_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v473_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v442_data, v459_data, v472_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v474_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v443_data, v460_data, v473_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v475_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v444_data, v461_data, v474_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v476_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v445_data, v462_data, v475_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v477_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v446_data, v463_data, v476_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v478_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v447_data, v464_data, v477_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v479_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v448_data, v465_data, v478_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v480_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v449_data, v466_data, v479_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v481_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v450_data, v467_data, v480_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v482_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v451_data, v468_data, v481_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v483_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v452_data, v469_data, v482_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v484_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v453_data, v470_data, v483_acc, 1, 0, 0);
          float v485_el = v484_acc[0];
          float v487_el = v484_acc[4];
          float v488_sw = tensorforge::swap<32>(v487_el);
          float v490_el = v484_acc[8];
          float v493_el = v484_acc[12];
          float v494_sw = tensorforge::swap<32>(v493_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v494_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v490_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v488_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v485_el, v485_el))))))));
          float v497_el = v484_acc[1];
          float v499_el = v484_acc[5];
          float v500_sw = tensorforge::swap<32>(v499_el);
          float v502_el = v484_acc[9];
          float v505_el = v484_acc[13];
          float v506_sw = tensorforge::swap<32>(v505_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v506_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v502_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v500_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v497_el, v497_el))))))));
          float v509_el = v484_acc[2];
          float v511_el = v484_acc[6];
          float v512_sw = tensorforge::swap<32>(v511_el);
          float v514_el = v484_acc[10];
          float v517_el = v484_acc[14];
          float v518_sw = tensorforge::swap<32>(v517_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v518_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v514_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v512_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v509_el, v509_el))))))));
          float v521_el = v484_acc[3];
          float v523_el = v484_acc[7];
          float v524_sw = tensorforge::swap<32>(v523_el);
          float v526_el = v484_acc[11];
          float v529_el = v484_acc[15];
          float v530_sw = tensorforge::swap<32>(v529_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v530_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v526_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v524_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v521_el, v521_el))))))));
          float v534_sw = tensorforge::swap<32>(v485_el);
          float v539_sw = tensorforge::swap<32>(v490_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v493_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v539_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v487_el, (tensorforge::dppUpdate<228, 1, 15, false>(v534_sw, v534_sw))))))));
          float v546_sw = tensorforge::swap<32>(v497_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v505_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v502_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v499_el, (tensorforge::dppUpdate<228, 1, 15, false>(v546_sw, v546_sw))))))));
          float v558_sw = tensorforge::swap<32>(v509_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v517_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v514_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v511_el, (tensorforge::dppUpdate<228, 1, 15, false>(v558_sw, v558_sw))))))));
          float v570_sw = tensorforge::swap<32>(v521_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v529_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v526_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v523_el, (tensorforge::dppUpdate<228, 1, 15, false>(v570_sw, v570_sw))))))));
          float v582_sw = tensorforge::swap<64>(v485_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v494_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v490_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v488_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v582_sw, v582_sw))))))));
          float v594_sw = tensorforge::swap<64>(v497_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v506_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v502_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v500_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v594_sw, v594_sw))))))));
          float v606_sw = tensorforge::swap<64>(v509_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v518_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v514_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v512_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v606_sw, v606_sw))))))));
          float v618_sw = tensorforge::swap<64>(v521_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v530_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v526_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v524_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v618_sw, v618_sw))))))));
          float v631_sw = tensorforge::swap<64>(v534_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v493_el, (tensorforge::dppUpdate<228, 4, 15, false>(v539_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v487_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v631_sw, v631_sw))))))));
          // wait(r5 = load{g>r}(glb_m4););
          float r6[26]{};
          // r6 = +(r5 * r4) + None
          // [(0, 64), (0, 13)] [(3, 22)]
          float v642_data = r4[0];
          float v643_data = r4[1];
          float v644_data = r4[2];
          float v645_data = r4[3];
          float v646_data = r4[4];
          float v647_data = r4[5];
          float v648_data = r4[6];
          float v649_data = r4[7];
          float v650_data = r4[8];
          float v651_data = r4[9];
          float v652_data = r4[10];
          float v653_data = r4[11];
          float v654_data = r4[12];
          float v655_pad{};
          float v656_pad{};
          float v657_pad{};
          tensorforge::transpose16x16b32(v642_data, v643_data, v644_data, v645_data, v646_data, v647_data, v648_data, v649_data, v650_data, v651_data, v652_data, v653_data, v654_data, v655_pad, v656_pad, v657_pad);
          tensorforge::VectorT<float, 16> v658_acc{};
          float v659_data = r5[0];
          float v660_data = r5[2];
          float v661_data = r5[4];
          float v662_data = r5[6];
          float v663_data = r5[8];
          float v664_data = r5[10];
          float v665_data = r5[12];
          float v666_data = r5[14];
          float v667_data = r5[16];
          float v668_data = r5[18];
          float v669_data = r5[20];
          float v670_data = r5[22];
          float v671_data = r5[24];
          tensorforge::VectorT<float, 16> v672_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v645_data, v659_data, v658_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v673_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v646_data, v660_data, v672_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v674_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v647_data, v661_data, v673_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v675_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v648_data, v662_data, v674_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v676_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v649_data, v663_data, v675_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v677_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v650_data, v664_data, v676_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v678_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v651_data, v665_data, v677_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v679_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v652_data, v666_data, v678_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v680_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v653_data, v667_data, v679_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v681_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v654_data, v668_data, v680_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v682_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v655_pad, v669_data, v681_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v683_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v656_pad, v670_data, v682_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v684_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v657_pad, v671_data, v683_acc, 1, 0, 0);
          float v685_data = r5[26];
          float v686_data = r5[28];
          float v687_data = r5[30];
          float v688_data = r5[32];
          float v689_data = r5[34];
          float v690_data = r5[36];
          tensorforge::VectorT<float, 16> v692_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v642_data, v685_data, v684_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v693_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v643_data, v686_data, v692_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v694_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v644_data, v687_data, v693_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v695_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v645_data, v688_data, v694_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v696_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v646_data, v689_data, v695_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v697_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v647_data, v690_data, v696_acc, 1, 1, 0);
          float v698_el = v697_acc[0];
          float v700_el = v697_acc[4];
          float v701_sw = tensorforge::swap<32>(v700_el);
          float v703_el = v697_acc[8];
          float v706_el = v697_acc[12];
          float v707_sw = tensorforge::swap<32>(v706_el);
          r6[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v707_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v703_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v701_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v698_el, v698_el))))))));
          float v710_el = v697_acc[1];
          float v712_el = v697_acc[5];
          float v713_sw = tensorforge::swap<32>(v712_el);
          float v715_el = v697_acc[9];
          float v718_el = v697_acc[13];
          float v719_sw = tensorforge::swap<32>(v718_el);
          r6[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v719_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v715_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v713_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v710_el, v710_el))))))));
          float v722_el = v697_acc[2];
          float v724_el = v697_acc[6];
          float v725_sw = tensorforge::swap<32>(v724_el);
          float v727_el = v697_acc[10];
          float v730_el = v697_acc[14];
          float v731_sw = tensorforge::swap<32>(v730_el);
          r6[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v731_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v727_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v725_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v722_el, v722_el))))))));
          float v734_el = v697_acc[3];
          float v736_el = v697_acc[7];
          float v737_sw = tensorforge::swap<32>(v736_el);
          float v739_el = v697_acc[11];
          float v742_el = v697_acc[15];
          float v743_sw = tensorforge::swap<32>(v742_el);
          r6[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v743_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v739_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v737_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v734_el, v734_el))))))));
          float v747_sw = tensorforge::swap<32>(v698_el);
          float v752_sw = tensorforge::swap<32>(v703_el);
          r6[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v706_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v752_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v700_el, (tensorforge::dppUpdate<228, 1, 15, false>(v747_sw, v747_sw))))))));
          float v759_sw = tensorforge::swap<32>(v710_el);
          r6[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v718_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v715_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v712_el, (tensorforge::dppUpdate<228, 1, 15, false>(v759_sw, v759_sw))))))));
          float v771_sw = tensorforge::swap<32>(v722_el);
          r6[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v730_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v727_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v724_el, (tensorforge::dppUpdate<228, 1, 15, false>(v771_sw, v771_sw))))))));
          float v783_sw = tensorforge::swap<32>(v734_el);
          r6[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v742_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v739_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v736_el, (tensorforge::dppUpdate<228, 1, 15, false>(v783_sw, v783_sw))))))));
          float v795_sw = tensorforge::swap<64>(v698_el);
          r6[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v707_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v703_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v701_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v795_sw, v795_sw))))))));
          float v807_sw = tensorforge::swap<64>(v710_el);
          r6[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v719_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v715_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v713_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v807_sw, v807_sw))))))));
          float v819_sw = tensorforge::swap<64>(v722_el);
          r6[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v731_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v727_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v725_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v819_sw, v819_sw))))))));
          float v831_sw = tensorforge::swap<64>(v734_el);
          r6[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v743_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v739_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v737_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v831_sw, v831_sw))))))));
          float v844_sw = tensorforge::swap<64>(v747_sw);
          r6[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v706_el, (tensorforge::dppUpdate<228, 4, 15, false>(v752_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v700_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v844_sw, v844_sw))))))));
          tensorforge::VectorT<float, 16> v854_acc{};
          float v855_data = r5[1];
          float v856_data = r5[3];
          float v857_data = r5[5];
          float v858_data = r5[7];
          float v859_data = r5[9];
          float v860_data = r5[11];
          float v861_data = r5[13];
          float v862_data = r5[15];
          float v863_data = r5[17];
          float v864_data = r5[19];
          float v865_data = r5[21];
          float v866_data = r5[23];
          float v867_data = r5[25];
          tensorforge::VectorT<float, 16> v868_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v645_data, v855_data, v854_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v869_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v646_data, v856_data, v868_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v870_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v647_data, v857_data, v869_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v871_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v648_data, v858_data, v870_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v872_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v649_data, v859_data, v871_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v873_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v650_data, v860_data, v872_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v874_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v651_data, v861_data, v873_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v875_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v652_data, v862_data, v874_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v876_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v653_data, v863_data, v875_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v877_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v654_data, v864_data, v876_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v878_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v655_pad, v865_data, v877_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v879_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v656_pad, v866_data, v878_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v880_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v657_pad, v867_data, v879_acc, 1, 0, 0);
          float v881_data = r5[27];
          float v882_data = r5[29];
          float v883_data = r5[31];
          float v884_data = r5[33];
          float v885_data = r5[35];
          float v886_data = r5[37];
          tensorforge::VectorT<float, 16> v888_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v642_data, v881_data, v880_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v889_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v643_data, v882_data, v888_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v890_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v644_data, v883_data, v889_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v891_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v645_data, v884_data, v890_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v892_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v646_data, v885_data, v891_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v893_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v647_data, v886_data, v892_acc, 1, 1, 0);
          float v894_el = v893_acc[0];
          float v896_el = v893_acc[4];
          float v897_sw = tensorforge::swap<32>(v896_el);
          float v899_el = v893_acc[8];
          float v902_el = v893_acc[12];
          float v903_sw = tensorforge::swap<32>(v902_el);
          r6[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v903_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v899_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v897_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v894_el, v894_el))))))));
          float v906_el = v893_acc[1];
          float v908_el = v893_acc[5];
          float v909_sw = tensorforge::swap<32>(v908_el);
          float v911_el = v893_acc[9];
          float v914_el = v893_acc[13];
          float v915_sw = tensorforge::swap<32>(v914_el);
          r6[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v915_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v911_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v909_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v906_el, v906_el))))))));
          float v918_el = v893_acc[2];
          float v920_el = v893_acc[6];
          float v921_sw = tensorforge::swap<32>(v920_el);
          float v923_el = v893_acc[10];
          float v926_el = v893_acc[14];
          float v927_sw = tensorforge::swap<32>(v926_el);
          r6[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v927_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v923_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v921_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v918_el, v918_el))))))));
          float v930_el = v893_acc[3];
          float v932_el = v893_acc[7];
          float v933_sw = tensorforge::swap<32>(v932_el);
          float v935_el = v893_acc[11];
          float v938_el = v893_acc[15];
          float v939_sw = tensorforge::swap<32>(v938_el);
          r6[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v939_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v935_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v933_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v930_el, v930_el))))))));
          float v943_sw = tensorforge::swap<32>(v894_el);
          float v948_sw = tensorforge::swap<32>(v899_el);
          r6[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v902_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v948_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v896_el, (tensorforge::dppUpdate<228, 1, 15, false>(v943_sw, v943_sw))))))));
          float v955_sw = tensorforge::swap<32>(v906_el);
          r6[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v914_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v911_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v908_el, (tensorforge::dppUpdate<228, 1, 15, false>(v955_sw, v955_sw))))))));
          float v967_sw = tensorforge::swap<32>(v918_el);
          r6[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v926_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v923_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v920_el, (tensorforge::dppUpdate<228, 1, 15, false>(v967_sw, v967_sw))))))));
          float v979_sw = tensorforge::swap<32>(v930_el);
          r6[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v938_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v935_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v932_el, (tensorforge::dppUpdate<228, 1, 15, false>(v979_sw, v979_sw))))))));
          float v991_sw = tensorforge::swap<64>(v894_el);
          r6[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v903_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v899_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v897_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v991_sw, v991_sw))))))));
          float v1003_sw = tensorforge::swap<64>(v906_el);
          r6[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v915_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v911_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v909_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1003_sw, v1003_sw))))))));
          float v1015_sw = tensorforge::swap<64>(v918_el);
          r6[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v927_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v923_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v921_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1015_sw, v1015_sw))))))));
          float v1027_sw = tensorforge::swap<64>(v930_el);
          r6[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v939_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v935_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v933_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1027_sw, v1027_sw))))))));
          float v1040_sw = tensorforge::swap<64>(v943_sw);
          r6[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v902_el, (tensorforge::dppUpdate<228, 4, 15, false>(v948_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v896_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1040_sw, v1040_sw))))))));
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1050_i0 = 0; v1050_i0 < 2; ++v1050_i0) {
            int32_t v1056_lead = v17_lead + (v1050_i0 * 32);
            #pragma unroll
            for (int32_t v1051_i1 = 0; v1051_i1 < 13; ++v1051_i1) {
              float v1054_data = r6[(v1050_i0 + (v1051_i1 * 2))];
              glb_m3[(v1056_lead + (v1051_i1 * 64))] = v1054_data;
            }
          }
        }
      }
    }
  }
}

