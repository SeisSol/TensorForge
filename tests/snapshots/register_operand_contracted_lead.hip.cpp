// === base name ===
kernel_d89f098c59615c57

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d89f098c59615c57 = {{32, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d89f098c59615c57(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d89f098c59615c57(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d89f098c59615c57(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d89f098c59615c57, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d89f098c59615c57, block.x * block.y * block.z, 0));
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
void launcher_kernel_d89f098c59615c57(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d89f098c59615c57(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d89f098c59615c57), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_d89f098c59615c57, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d89f098c59615c57(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[13,13]],"name":"m1","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[64,13]],"name":"m3","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"strided","alias":"K","bbox":[[0,0],[64,56]],"name":"m4","ordered":false,"parts":1,"shape":[64,56],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[19,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[19,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[19,13]],"is_tmp":true,"name":"t0","offset":[35,0],"shape":[64,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[64,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,19]],"is_tmp":false,"name":"m4","offset":[0,35],"shape":[64,56]},{"addressing":"strided","bbox":[[0,0],[19,13]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[19,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 832 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 169 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 169 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 832 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 3584 + 0 + m4_extraOffset];
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
          float r1[13]{};
          // r1 = load{g>r}(glb_m1);
          bool v34_g = v23_lead < 13;
          if (v34_g) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 13; ++v35_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v35_i1 * 13))]);
              r1[v35_i1] = v40_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v34_g) {
            #pragma unroll
            for (int32_t v428_i1 = 0; v428_i1 < 13; ++v428_i1) {
              float v433_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v428_i1 * 13))]);
              r3[v428_i1] = v433_data;
            }
          }
          float r2[26]{};
          // r2 = +(r0 * r1) + None
          // [(0, 64), (0, 13)] [(0, 13)]
          float v43_data = r1[0];
          float v44_data = r1[1];
          float v45_data = r1[2];
          float v46_data = r1[3];
          float v47_data = r1[4];
          float v48_data = r1[5];
          float v49_data = r1[6];
          float v50_data = r1[7];
          float v51_data = r1[8];
          float v52_data = r1[9];
          float v53_data = r1[10];
          float v54_data = r1[11];
          float v55_data = r1[12];
          float v56_pad{};
          float v57_pad{};
          float v58_pad{};
          tensorforge::transpose16x16b32(v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_pad, v57_pad, v58_pad);
          tensorforge::VectorT<float, 16> v59_acc{};
          float v60_data = r0[0];
          float v61_data = r0[2];
          float v62_data = r0[4];
          float v63_data = r0[6];
          float v64_data = r0[8];
          float v65_data = r0[10];
          float v66_data = r0[12];
          float v67_data = r0[14];
          float v68_data = r0[16];
          float v69_data = r0[18];
          float v70_data = r0[20];
          float v71_data = r0[22];
          float v72_data = r0[24];
          tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v60_data, v59_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v74_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v75_acc, 1, 0, 0);
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
          float v87_el = v86_acc[0];
          float v89_el = v86_acc[4];
          float v90_sw = tensorforge::swap<32>(v89_el);
          float v92_el = v86_acc[8];
          float v95_el = v86_acc[12];
          float v96_sw = tensorforge::swap<32>(v95_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v96_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v92_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v90_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v87_el, v87_el))))))));
          float v99_el = v86_acc[1];
          float v101_el = v86_acc[5];
          float v102_sw = tensorforge::swap<32>(v101_el);
          float v104_el = v86_acc[9];
          float v107_el = v86_acc[13];
          float v108_sw = tensorforge::swap<32>(v107_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v108_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v104_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v102_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v99_el, v99_el))))))));
          float v111_el = v86_acc[2];
          float v113_el = v86_acc[6];
          float v114_sw = tensorforge::swap<32>(v113_el);
          float v116_el = v86_acc[10];
          float v119_el = v86_acc[14];
          float v120_sw = tensorforge::swap<32>(v119_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v120_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v116_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v114_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v111_el, v111_el))))))));
          float v123_el = v86_acc[3];
          float v125_el = v86_acc[7];
          float v126_sw = tensorforge::swap<32>(v125_el);
          float v128_el = v86_acc[11];
          float v131_el = v86_acc[15];
          float v132_sw = tensorforge::swap<32>(v131_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v132_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v128_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v126_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v123_el, v123_el))))))));
          float v136_sw = tensorforge::swap<32>(v87_el);
          float v141_sw = tensorforge::swap<32>(v92_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v95_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v89_el, (tensorforge::dppUpdate<228, 1, 15, false>(v136_sw, v136_sw))))))));
          float v148_sw = tensorforge::swap<32>(v99_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v107_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v104_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v101_el, (tensorforge::dppUpdate<228, 1, 15, false>(v148_sw, v148_sw))))))));
          float v160_sw = tensorforge::swap<32>(v111_el);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v119_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v116_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v113_el, (tensorforge::dppUpdate<228, 1, 15, false>(v160_sw, v160_sw))))))));
          float v172_sw = tensorforge::swap<32>(v123_el);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v131_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v128_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v125_el, (tensorforge::dppUpdate<228, 1, 15, false>(v172_sw, v172_sw))))))));
          float v184_sw = tensorforge::swap<64>(v87_el);
          r2[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v96_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v92_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v90_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v184_sw, v184_sw))))))));
          float v196_sw = tensorforge::swap<64>(v99_el);
          r2[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v108_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v104_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v102_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v196_sw, v196_sw))))))));
          float v208_sw = tensorforge::swap<64>(v111_el);
          r2[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v120_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v116_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v114_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v208_sw, v208_sw))))))));
          float v220_sw = tensorforge::swap<64>(v123_el);
          r2[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v132_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v128_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v126_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v220_sw, v220_sw))))))));
          float v233_sw = tensorforge::swap<64>(v136_sw);
          r2[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v95_el, (tensorforge::dppUpdate<228, 4, 15, false>(v141_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v89_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
          tensorforge::VectorT<float, 16> v243_acc{};
          float v244_data = r0[1];
          float v245_data = r0[3];
          float v246_data = r0[5];
          float v247_data = r0[7];
          float v248_data = r0[9];
          float v249_data = r0[11];
          float v250_data = r0[13];
          float v251_data = r0[15];
          float v252_data = r0[17];
          float v253_data = r0[19];
          float v254_data = r0[21];
          float v255_data = r0[23];
          float v256_data = r0[25];
          tensorforge::VectorT<float, 16> v258_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v244_data, v243_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v259_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v245_data, v258_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v260_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v246_data, v259_acc, 1, 0, 0);
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
          float v271_el = v270_acc[0];
          float v273_el = v270_acc[4];
          float v274_sw = tensorforge::swap<32>(v273_el);
          float v276_el = v270_acc[8];
          float v279_el = v270_acc[12];
          float v280_sw = tensorforge::swap<32>(v279_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v280_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v276_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v274_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v271_el, v271_el))))))));
          float v283_el = v270_acc[1];
          float v285_el = v270_acc[5];
          float v286_sw = tensorforge::swap<32>(v285_el);
          float v288_el = v270_acc[9];
          float v291_el = v270_acc[13];
          float v292_sw = tensorforge::swap<32>(v291_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v292_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v288_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v286_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v283_el, v283_el))))))));
          float v295_el = v270_acc[2];
          float v297_el = v270_acc[6];
          float v298_sw = tensorforge::swap<32>(v297_el);
          float v300_el = v270_acc[10];
          float v303_el = v270_acc[14];
          float v304_sw = tensorforge::swap<32>(v303_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v304_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v300_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v298_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v295_el, v295_el))))))));
          float v307_el = v270_acc[3];
          float v309_el = v270_acc[7];
          float v310_sw = tensorforge::swap<32>(v309_el);
          float v312_el = v270_acc[11];
          float v315_el = v270_acc[15];
          float v316_sw = tensorforge::swap<32>(v315_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v316_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v312_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v310_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v307_el, v307_el))))))));
          float v320_sw = tensorforge::swap<32>(v271_el);
          float v325_sw = tensorforge::swap<32>(v276_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v279_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v325_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v273_el, (tensorforge::dppUpdate<228, 1, 15, false>(v320_sw, v320_sw))))))));
          float v332_sw = tensorforge::swap<32>(v283_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v291_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v288_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v285_el, (tensorforge::dppUpdate<228, 1, 15, false>(v332_sw, v332_sw))))))));
          float v344_sw = tensorforge::swap<32>(v295_el);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v303_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v300_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v297_el, (tensorforge::dppUpdate<228, 1, 15, false>(v344_sw, v344_sw))))))));
          float v356_sw = tensorforge::swap<32>(v307_el);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v315_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v312_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v309_el, (tensorforge::dppUpdate<228, 1, 15, false>(v356_sw, v356_sw))))))));
          float v368_sw = tensorforge::swap<64>(v271_el);
          r2[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v280_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v276_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v274_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v368_sw, v368_sw))))))));
          float v380_sw = tensorforge::swap<64>(v283_el);
          r2[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v292_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v288_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v286_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v380_sw, v380_sw))))))));
          float v392_sw = tensorforge::swap<64>(v295_el);
          r2[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v304_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v300_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v298_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v392_sw, v392_sw))))))));
          float v404_sw = tensorforge::swap<64>(v307_el);
          r2[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v316_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v312_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v310_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v404_sw, v404_sw))))))));
          float v417_sw = tensorforge::swap<64>(v320_sw);
          r2[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v279_el, (tensorforge::dppUpdate<228, 4, 15, false>(v325_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v273_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v417_sw, v417_sw))))))));
          float r5[38]{};
          // r5 = load{g>r}(glb_m4);
          #pragma unroll
          for (int32_t v637_i0 = 0; v637_i0 < 2; ++v637_i0) {
            int32_t v640_lead = v23_lead + (v637_i0 * 32);
            #pragma unroll
            for (int32_t v638_i1 = 35; v638_i1 < 54; ++v638_i1) {
              float v643_data = __builtin_nontemporal_load(&glb_m4[(v640_lead + (v638_i1 * 64))]);
              r5[(v637_i0 + ((v638_i1 - 35) * 2))] = v643_data;
            }
          }
          float r4[13]{};
          // r4 = +(r2 * r3) + None
          // [(3, 22), (0, 13)] [(0, 13)]
          float v436_data = r3[0];
          float v437_data = r3[1];
          float v438_data = r3[2];
          float v439_data = r3[3];
          float v440_data = r3[4];
          float v441_data = r3[5];
          float v442_data = r3[6];
          float v443_data = r3[7];
          float v444_data = r3[8];
          float v445_data = r3[9];
          float v446_data = r3[10];
          float v447_data = r3[11];
          float v448_data = r3[12];
          float v449_pad{};
          float v450_pad{};
          float v451_pad{};
          tensorforge::transpose16x16b32(v436_data, v437_data, v438_data, v439_data, v440_data, v441_data, v442_data, v443_data, v444_data, v445_data, v446_data, v447_data, v448_data, v449_pad, v450_pad, v451_pad);
          tensorforge::VectorT<float, 16> v452_acc{};
          float v453_data = r2[1];
          float v454_data = r2[3];
          float v455_data = r2[5];
          float v456_data = r2[7];
          float v457_data = r2[9];
          float v458_data = r2[11];
          float v459_data = r2[13];
          float v460_data = r2[15];
          float v461_data = r2[17];
          float v462_data = r2[19];
          float v463_data = r2[21];
          float v464_data = r2[23];
          float v465_data = r2[25];
          tensorforge::VectorT<float, 16> v467_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v436_data, v453_data, v452_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v468_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v437_data, v454_data, v467_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v469_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v438_data, v455_data, v468_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v470_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v439_data, v456_data, v469_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v471_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v440_data, v457_data, v470_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v472_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v441_data, v458_data, v471_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v473_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v442_data, v459_data, v472_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v474_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v443_data, v460_data, v473_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v475_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v444_data, v461_data, v474_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v476_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v445_data, v462_data, v475_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v477_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v446_data, v463_data, v476_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v478_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v447_data, v464_data, v477_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v479_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v448_data, v465_data, v478_acc, 1, 0, 0);
          float v480_el = v479_acc[0];
          float v482_el = v479_acc[4];
          float v483_sw = tensorforge::swap<32>(v482_el);
          float v485_el = v479_acc[8];
          float v488_el = v479_acc[12];
          float v489_sw = tensorforge::swap<32>(v488_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v489_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v485_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v483_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v480_el, v480_el))))))));
          float v492_el = v479_acc[1];
          float v494_el = v479_acc[5];
          float v495_sw = tensorforge::swap<32>(v494_el);
          float v497_el = v479_acc[9];
          float v500_el = v479_acc[13];
          float v501_sw = tensorforge::swap<32>(v500_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v501_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v497_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v495_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v492_el, v492_el))))))));
          float v504_el = v479_acc[2];
          float v506_el = v479_acc[6];
          float v507_sw = tensorforge::swap<32>(v506_el);
          float v509_el = v479_acc[10];
          float v512_el = v479_acc[14];
          float v513_sw = tensorforge::swap<32>(v512_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v513_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v509_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v507_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v504_el, v504_el))))))));
          float v516_el = v479_acc[3];
          float v518_el = v479_acc[7];
          float v519_sw = tensorforge::swap<32>(v518_el);
          float v521_el = v479_acc[11];
          float v524_el = v479_acc[15];
          float v525_sw = tensorforge::swap<32>(v524_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v525_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v521_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v519_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v516_el, v516_el))))))));
          float v529_sw = tensorforge::swap<32>(v480_el);
          float v534_sw = tensorforge::swap<32>(v485_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v488_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v534_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v482_el, (tensorforge::dppUpdate<228, 1, 15, false>(v529_sw, v529_sw))))))));
          float v541_sw = tensorforge::swap<32>(v492_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v500_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v497_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v494_el, (tensorforge::dppUpdate<228, 1, 15, false>(v541_sw, v541_sw))))))));
          float v553_sw = tensorforge::swap<32>(v504_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v512_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v509_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v506_el, (tensorforge::dppUpdate<228, 1, 15, false>(v553_sw, v553_sw))))))));
          float v565_sw = tensorforge::swap<32>(v516_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v524_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v521_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v518_el, (tensorforge::dppUpdate<228, 1, 15, false>(v565_sw, v565_sw))))))));
          float v577_sw = tensorforge::swap<64>(v480_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v489_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v485_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v483_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v577_sw, v577_sw))))))));
          float v589_sw = tensorforge::swap<64>(v492_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v501_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v497_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v495_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v589_sw, v589_sw))))))));
          float v601_sw = tensorforge::swap<64>(v504_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v513_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v509_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v507_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v601_sw, v601_sw))))))));
          float v613_sw = tensorforge::swap<64>(v516_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v525_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v521_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v519_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v613_sw, v613_sw))))))));
          float v626_sw = tensorforge::swap<64>(v529_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v488_el, (tensorforge::dppUpdate<228, 4, 15, false>(v534_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v482_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v626_sw, v626_sw))))))));
          float r6[26]{};
          // r6 = +(r5 * r4) + None
          // [(0, 64), (0, 13)] [(3, 22)]
          float v648_data = r4[0];
          float v649_data = r4[1];
          float v650_data = r4[2];
          float v651_data = r4[3];
          float v652_data = r4[4];
          float v653_data = r4[5];
          float v654_data = r4[6];
          float v655_data = r4[7];
          float v656_data = r4[8];
          float v657_data = r4[9];
          float v658_data = r4[10];
          float v659_data = r4[11];
          float v660_data = r4[12];
          float v661_pad{};
          float v662_pad{};
          float v663_pad{};
          tensorforge::transpose16x16b32(v648_data, v649_data, v650_data, v651_data, v652_data, v653_data, v654_data, v655_data, v656_data, v657_data, v658_data, v659_data, v660_data, v661_pad, v662_pad, v663_pad);
          tensorforge::VectorT<float, 16> v664_acc{};
          float v665_data = r5[0];
          float v666_data = r5[2];
          float v667_data = r5[4];
          float v668_data = r5[6];
          float v669_data = r5[8];
          float v670_data = r5[10];
          float v671_data = r5[12];
          float v672_data = r5[14];
          float v673_data = r5[16];
          float v674_data = r5[18];
          float v675_data = r5[20];
          float v676_data = r5[22];
          float v677_data = r5[24];
          tensorforge::VectorT<float, 16> v678_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v651_data, v665_data, v664_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v679_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v652_data, v666_data, v678_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v680_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v653_data, v667_data, v679_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v681_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v654_data, v668_data, v680_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v682_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v655_data, v669_data, v681_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v683_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v656_data, v670_data, v682_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v684_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v657_data, v671_data, v683_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v685_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v658_data, v672_data, v684_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v686_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v659_data, v673_data, v685_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v687_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v660_data, v674_data, v686_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v688_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v661_pad, v675_data, v687_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v689_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v662_pad, v676_data, v688_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v690_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v663_pad, v677_data, v689_acc, 1, 0, 0);
          float v691_data = r5[26];
          float v692_data = r5[28];
          float v693_data = r5[30];
          float v694_data = r5[32];
          float v695_data = r5[34];
          float v696_data = r5[36];
          tensorforge::VectorT<float, 16> v698_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v648_data, v691_data, v690_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v699_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v649_data, v692_data, v698_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v700_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v650_data, v693_data, v699_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v701_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v651_data, v694_data, v700_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v702_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v652_data, v695_data, v701_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v703_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v653_data, v696_data, v702_acc, 1, 1, 0);
          float v704_el = v703_acc[0];
          float v706_el = v703_acc[4];
          float v707_sw = tensorforge::swap<32>(v706_el);
          float v709_el = v703_acc[8];
          float v712_el = v703_acc[12];
          float v713_sw = tensorforge::swap<32>(v712_el);
          r6[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v713_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v709_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v707_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v704_el, v704_el))))))));
          float v716_el = v703_acc[1];
          float v718_el = v703_acc[5];
          float v719_sw = tensorforge::swap<32>(v718_el);
          float v721_el = v703_acc[9];
          float v724_el = v703_acc[13];
          float v725_sw = tensorforge::swap<32>(v724_el);
          r6[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v725_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v721_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v719_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v716_el, v716_el))))))));
          float v728_el = v703_acc[2];
          float v730_el = v703_acc[6];
          float v731_sw = tensorforge::swap<32>(v730_el);
          float v733_el = v703_acc[10];
          float v736_el = v703_acc[14];
          float v737_sw = tensorforge::swap<32>(v736_el);
          r6[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v737_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v733_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v731_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v728_el, v728_el))))))));
          float v740_el = v703_acc[3];
          float v742_el = v703_acc[7];
          float v743_sw = tensorforge::swap<32>(v742_el);
          float v745_el = v703_acc[11];
          float v748_el = v703_acc[15];
          float v749_sw = tensorforge::swap<32>(v748_el);
          r6[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v749_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v745_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v743_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v740_el, v740_el))))))));
          float v753_sw = tensorforge::swap<32>(v704_el);
          float v758_sw = tensorforge::swap<32>(v709_el);
          r6[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v712_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v758_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v706_el, (tensorforge::dppUpdate<228, 1, 15, false>(v753_sw, v753_sw))))))));
          float v765_sw = tensorforge::swap<32>(v716_el);
          r6[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v724_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v721_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v718_el, (tensorforge::dppUpdate<228, 1, 15, false>(v765_sw, v765_sw))))))));
          float v777_sw = tensorforge::swap<32>(v728_el);
          r6[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v736_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v733_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v730_el, (tensorforge::dppUpdate<228, 1, 15, false>(v777_sw, v777_sw))))))));
          float v789_sw = tensorforge::swap<32>(v740_el);
          r6[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v748_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v745_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v742_el, (tensorforge::dppUpdate<228, 1, 15, false>(v789_sw, v789_sw))))))));
          float v801_sw = tensorforge::swap<64>(v704_el);
          r6[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v713_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v709_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v707_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v801_sw, v801_sw))))))));
          float v813_sw = tensorforge::swap<64>(v716_el);
          r6[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v725_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v721_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v719_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v813_sw, v813_sw))))))));
          float v825_sw = tensorforge::swap<64>(v728_el);
          r6[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v737_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v733_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v731_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v825_sw, v825_sw))))))));
          float v837_sw = tensorforge::swap<64>(v740_el);
          r6[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v749_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v745_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v743_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v837_sw, v837_sw))))))));
          float v850_sw = tensorforge::swap<64>(v753_sw);
          r6[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v712_el, (tensorforge::dppUpdate<228, 4, 15, false>(v758_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v706_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v850_sw, v850_sw))))))));
          tensorforge::VectorT<float, 16> v860_acc{};
          float v861_data = r5[1];
          float v862_data = r5[3];
          float v863_data = r5[5];
          float v864_data = r5[7];
          float v865_data = r5[9];
          float v866_data = r5[11];
          float v867_data = r5[13];
          float v868_data = r5[15];
          float v869_data = r5[17];
          float v870_data = r5[19];
          float v871_data = r5[21];
          float v872_data = r5[23];
          float v873_data = r5[25];
          tensorforge::VectorT<float, 16> v874_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v651_data, v861_data, v860_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v875_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v652_data, v862_data, v874_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v876_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v653_data, v863_data, v875_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v877_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v654_data, v864_data, v876_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v878_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v655_data, v865_data, v877_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v879_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v656_data, v866_data, v878_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v880_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v657_data, v867_data, v879_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v881_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v658_data, v868_data, v880_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v882_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v659_data, v869_data, v881_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v883_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v660_data, v870_data, v882_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v884_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v661_pad, v871_data, v883_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v885_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v662_pad, v872_data, v884_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v886_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v663_pad, v873_data, v885_acc, 1, 0, 0);
          float v887_data = r5[27];
          float v888_data = r5[29];
          float v889_data = r5[31];
          float v890_data = r5[33];
          float v891_data = r5[35];
          float v892_data = r5[37];
          tensorforge::VectorT<float, 16> v894_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v648_data, v887_data, v886_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v895_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v649_data, v888_data, v894_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v896_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v650_data, v889_data, v895_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v897_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v651_data, v890_data, v896_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v898_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v652_data, v891_data, v897_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v899_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v653_data, v892_data, v898_acc, 1, 1, 0);
          float v900_el = v899_acc[0];
          float v902_el = v899_acc[4];
          float v903_sw = tensorforge::swap<32>(v902_el);
          float v905_el = v899_acc[8];
          float v908_el = v899_acc[12];
          float v909_sw = tensorforge::swap<32>(v908_el);
          r6[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v909_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v905_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v903_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v900_el, v900_el))))))));
          float v912_el = v899_acc[1];
          float v914_el = v899_acc[5];
          float v915_sw = tensorforge::swap<32>(v914_el);
          float v917_el = v899_acc[9];
          float v920_el = v899_acc[13];
          float v921_sw = tensorforge::swap<32>(v920_el);
          r6[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v921_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v917_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v915_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v912_el, v912_el))))))));
          float v924_el = v899_acc[2];
          float v926_el = v899_acc[6];
          float v927_sw = tensorforge::swap<32>(v926_el);
          float v929_el = v899_acc[10];
          float v932_el = v899_acc[14];
          float v933_sw = tensorforge::swap<32>(v932_el);
          r6[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v933_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v929_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v927_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v924_el, v924_el))))))));
          float v936_el = v899_acc[3];
          float v938_el = v899_acc[7];
          float v939_sw = tensorforge::swap<32>(v938_el);
          float v941_el = v899_acc[11];
          float v944_el = v899_acc[15];
          float v945_sw = tensorforge::swap<32>(v944_el);
          r6[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v945_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v941_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v939_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v936_el, v936_el))))))));
          float v949_sw = tensorforge::swap<32>(v900_el);
          float v954_sw = tensorforge::swap<32>(v905_el);
          r6[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v908_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v954_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v902_el, (tensorforge::dppUpdate<228, 1, 15, false>(v949_sw, v949_sw))))))));
          float v961_sw = tensorforge::swap<32>(v912_el);
          r6[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v920_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v917_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v914_el, (tensorforge::dppUpdate<228, 1, 15, false>(v961_sw, v961_sw))))))));
          float v973_sw = tensorforge::swap<32>(v924_el);
          r6[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v932_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v929_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v926_el, (tensorforge::dppUpdate<228, 1, 15, false>(v973_sw, v973_sw))))))));
          float v985_sw = tensorforge::swap<32>(v936_el);
          r6[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v944_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v941_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v938_el, (tensorforge::dppUpdate<228, 1, 15, false>(v985_sw, v985_sw))))))));
          float v997_sw = tensorforge::swap<64>(v900_el);
          r6[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v909_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v905_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v903_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v997_sw, v997_sw))))))));
          float v1009_sw = tensorforge::swap<64>(v912_el);
          r6[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v921_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v917_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v915_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1009_sw, v1009_sw))))))));
          float v1021_sw = tensorforge::swap<64>(v924_el);
          r6[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v933_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v929_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v927_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1021_sw, v1021_sw))))))));
          float v1033_sw = tensorforge::swap<64>(v936_el);
          r6[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v945_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v941_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v939_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v1033_sw, v1033_sw))))))));
          float v1046_sw = tensorforge::swap<64>(v949_sw);
          r6[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v908_el, (tensorforge::dppUpdate<228, 4, 15, false>(v954_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v902_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1046_sw, v1046_sw))))))));
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1056_i0 = 0; v1056_i0 < 2; ++v1056_i0) {
            int32_t v1062_lead = v23_lead + (v1056_i0 * 32);
            #pragma unroll
            for (int32_t v1057_i1 = 0; v1057_i1 < 13; ++v1057_i1) {
              float v1060_data = r6[(v1056_i0 + (v1057_i1 * 2))];
              glb_m3[(v1062_lead + (v1057_i1 * 64))] = v1060_data;
            }
          }
        }
      }
    }
  }
}

