// === base name ===
kernel_a930783d4ed990eb

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a930783d4ed990eb = {{32, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a930783d4ed990eb(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a930783d4ed990eb(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a930783d4ed990eb(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a930783d4ed990eb, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_a930783d4ed990eb, block.x * block.y * block.z, 0));
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
void launcher_kernel_a930783d4ed990eb(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a930783d4ed990eb(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a930783d4ed990eb), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_a930783d4ed990eb, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a930783d4ed990eb(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m0););
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v34_g) {
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 13; ++v43_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v43_i1 * 13))]);
              r3[v43_i1] = v48_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[26]{};
          // r2 = +(r0 * r1) + None
          // [(0, 64), (0, 13)] [(0, 13)]
          float v51_data = r1[0];
          float v52_data = r1[1];
          float v53_data = r1[2];
          float v54_data = r1[3];
          float v55_data = r1[4];
          float v56_data = r1[5];
          float v57_data = r1[6];
          float v58_data = r1[7];
          float v59_data = r1[8];
          float v60_data = r1[9];
          float v61_data = r1[10];
          float v62_data = r1[11];
          float v63_data = r1[12];
          float v64_pad{};
          float v65_pad{};
          float v66_pad{};
          tensorforge::transpose16x16b32(v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_pad, v65_pad, v66_pad);
          tensorforge::VectorT<float, 16> v67_acc{};
          float v68_data = r0[0];
          float v69_data = r0[2];
          float v70_data = r0[4];
          float v71_data = r0[6];
          float v72_data = r0[8];
          float v73_data = r0[10];
          float v74_data = r0[12];
          float v75_data = r0[14];
          float v76_data = r0[16];
          float v77_data = r0[18];
          float v78_data = r0[20];
          float v79_data = r0[22];
          float v80_data = r0[24];
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v67_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v90_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v91_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v92_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v93_acc, 1, 0, 0);
          float v95_el = v94_acc[0];
          float v97_el = v94_acc[4];
          float v98_sw = tensorforge::swap<32>(v97_el);
          float v100_el = v94_acc[8];
          float v103_el = v94_acc[12];
          float v104_sw = tensorforge::swap<32>(v103_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v104_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v98_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v95_el, v95_el))))))));
          float v107_el = v94_acc[1];
          float v109_el = v94_acc[5];
          float v110_sw = tensorforge::swap<32>(v109_el);
          float v112_el = v94_acc[9];
          float v115_el = v94_acc[13];
          float v116_sw = tensorforge::swap<32>(v115_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v116_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v110_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v107_el, v107_el))))))));
          float v119_el = v94_acc[2];
          float v121_el = v94_acc[6];
          float v122_sw = tensorforge::swap<32>(v121_el);
          float v124_el = v94_acc[10];
          float v127_el = v94_acc[14];
          float v128_sw = tensorforge::swap<32>(v127_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v128_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v122_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v119_el, v119_el))))))));
          float v131_el = v94_acc[3];
          float v133_el = v94_acc[7];
          float v134_sw = tensorforge::swap<32>(v133_el);
          float v136_el = v94_acc[11];
          float v139_el = v94_acc[15];
          float v140_sw = tensorforge::swap<32>(v139_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v140_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v136_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v134_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v131_el, v131_el))))))));
          float v144_sw = tensorforge::swap<32>(v95_el);
          float v149_sw = tensorforge::swap<32>(v100_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v149_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v97_el, (tensorforge::dppUpdate<228, 1, 15, false>(v144_sw, v144_sw))))))));
          float v156_sw = tensorforge::swap<32>(v107_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v112_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v109_el, (tensorforge::dppUpdate<228, 1, 15, false>(v156_sw, v156_sw))))))));
          float v168_sw = tensorforge::swap<32>(v119_el);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v124_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v121_el, (tensorforge::dppUpdate<228, 1, 15, false>(v168_sw, v168_sw))))))));
          float v180_sw = tensorforge::swap<32>(v131_el);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v139_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v136_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v133_el, (tensorforge::dppUpdate<228, 1, 15, false>(v180_sw, v180_sw))))))));
          float v192_sw = tensorforge::swap<64>(v95_el);
          r2[16] = (tensorforge::dppUpdate<228, 8, 15, false>(v104_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v100_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v98_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v192_sw, v192_sw))))))));
          float v204_sw = tensorforge::swap<64>(v107_el);
          r2[18] = (tensorforge::dppUpdate<228, 8, 15, false>(v116_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v112_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v110_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v204_sw, v204_sw))))))));
          float v216_sw = tensorforge::swap<64>(v119_el);
          r2[20] = (tensorforge::dppUpdate<228, 8, 15, false>(v128_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v124_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v122_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v216_sw, v216_sw))))))));
          float v228_sw = tensorforge::swap<64>(v131_el);
          r2[22] = (tensorforge::dppUpdate<228, 8, 15, false>(v140_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v136_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v134_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v228_sw, v228_sw))))))));
          float v241_sw = tensorforge::swap<64>(v144_sw);
          r2[24] = (tensorforge::dppUpdate<228, 8, 15, false>(v103_el, (tensorforge::dppUpdate<228, 4, 15, false>(v149_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v241_sw, v241_sw))))))));
          tensorforge::VectorT<float, 16> v251_acc{};
          float v252_data = r0[1];
          float v253_data = r0[3];
          float v254_data = r0[5];
          float v255_data = r0[7];
          float v256_data = r0[9];
          float v257_data = r0[11];
          float v258_data = r0[13];
          float v259_data = r0[15];
          float v260_data = r0[17];
          float v261_data = r0[19];
          float v262_data = r0[21];
          float v263_data = r0[23];
          float v264_data = r0[25];
          tensorforge::VectorT<float, 16> v266_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v252_data, v251_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v267_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v253_data, v266_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v268_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v254_data, v267_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v269_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v255_data, v268_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v270_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v256_data, v269_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v271_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v257_data, v270_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v272_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v258_data, v271_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v273_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v259_data, v272_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v274_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v260_data, v273_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v275_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v261_data, v274_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v276_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v262_data, v275_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v277_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v263_data, v276_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v278_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v264_data, v277_acc, 1, 0, 0);
          float v279_el = v278_acc[0];
          float v281_el = v278_acc[4];
          float v282_sw = tensorforge::swap<32>(v281_el);
          float v284_el = v278_acc[8];
          float v287_el = v278_acc[12];
          float v288_sw = tensorforge::swap<32>(v287_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v288_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v284_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v282_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v279_el, v279_el))))))));
          float v291_el = v278_acc[1];
          float v293_el = v278_acc[5];
          float v294_sw = tensorforge::swap<32>(v293_el);
          float v296_el = v278_acc[9];
          float v299_el = v278_acc[13];
          float v300_sw = tensorforge::swap<32>(v299_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v300_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v296_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v294_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v291_el, v291_el))))))));
          float v303_el = v278_acc[2];
          float v305_el = v278_acc[6];
          float v306_sw = tensorforge::swap<32>(v305_el);
          float v308_el = v278_acc[10];
          float v311_el = v278_acc[14];
          float v312_sw = tensorforge::swap<32>(v311_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v312_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v308_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v306_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v303_el, v303_el))))))));
          float v315_el = v278_acc[3];
          float v317_el = v278_acc[7];
          float v318_sw = tensorforge::swap<32>(v317_el);
          float v320_el = v278_acc[11];
          float v323_el = v278_acc[15];
          float v324_sw = tensorforge::swap<32>(v323_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v324_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v320_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v318_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v315_el, v315_el))))))));
          float v328_sw = tensorforge::swap<32>(v279_el);
          float v333_sw = tensorforge::swap<32>(v284_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v287_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v333_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v281_el, (tensorforge::dppUpdate<228, 1, 15, false>(v328_sw, v328_sw))))))));
          float v340_sw = tensorforge::swap<32>(v291_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v299_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v296_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v293_el, (tensorforge::dppUpdate<228, 1, 15, false>(v340_sw, v340_sw))))))));
          float v352_sw = tensorforge::swap<32>(v303_el);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v311_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v308_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v305_el, (tensorforge::dppUpdate<228, 1, 15, false>(v352_sw, v352_sw))))))));
          float v364_sw = tensorforge::swap<32>(v315_el);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v323_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v320_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v317_el, (tensorforge::dppUpdate<228, 1, 15, false>(v364_sw, v364_sw))))))));
          float v376_sw = tensorforge::swap<64>(v279_el);
          r2[17] = (tensorforge::dppUpdate<228, 8, 15, false>(v288_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v284_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v282_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v376_sw, v376_sw))))))));
          float v388_sw = tensorforge::swap<64>(v291_el);
          r2[19] = (tensorforge::dppUpdate<228, 8, 15, false>(v300_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v296_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v294_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v388_sw, v388_sw))))))));
          float v400_sw = tensorforge::swap<64>(v303_el);
          r2[21] = (tensorforge::dppUpdate<228, 8, 15, false>(v312_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v308_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v306_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v400_sw, v400_sw))))))));
          float v412_sw = tensorforge::swap<64>(v315_el);
          r2[23] = (tensorforge::dppUpdate<228, 8, 15, false>(v324_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v320_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v318_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v412_sw, v412_sw))))))));
          float v425_sw = tensorforge::swap<64>(v328_sw);
          r2[25] = (tensorforge::dppUpdate<228, 8, 15, false>(v287_el, (tensorforge::dppUpdate<228, 4, 15, false>(v333_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v281_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v425_sw, v425_sw))))))));
          float r5[38]{};
          // r5 = load{g>r}(glb_m4);
          #pragma unroll
          for (int32_t v436_i0 = 0; v436_i0 < 2; ++v436_i0) {
            int32_t v439_lead = v23_lead + (v436_i0 * 32);
            #pragma unroll
            for (int32_t v437_i1 = 35; v437_i1 < 54; ++v437_i1) {
              float v442_data = __builtin_nontemporal_load(&glb_m4[(v439_lead + (v437_i1 * 64))]);
              r5[(v436_i0 + ((v437_i1 - 35) * 2))] = v442_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + None
          // [(3, 22), (0, 13)] [(0, 13)]
          float v447_data = r3[0];
          float v448_data = r3[1];
          float v449_data = r3[2];
          float v450_data = r3[3];
          float v451_data = r3[4];
          float v452_data = r3[5];
          float v453_data = r3[6];
          float v454_data = r3[7];
          float v455_data = r3[8];
          float v456_data = r3[9];
          float v457_data = r3[10];
          float v458_data = r3[11];
          float v459_data = r3[12];
          float v460_pad{};
          float v461_pad{};
          float v462_pad{};
          tensorforge::transpose16x16b32(v447_data, v448_data, v449_data, v450_data, v451_data, v452_data, v453_data, v454_data, v455_data, v456_data, v457_data, v458_data, v459_data, v460_pad, v461_pad, v462_pad);
          tensorforge::VectorT<float, 16> v463_acc{};
          float v464_data = r2[1];
          float v465_data = r2[3];
          float v466_data = r2[5];
          float v467_data = r2[7];
          float v468_data = r2[9];
          float v469_data = r2[11];
          float v470_data = r2[13];
          float v471_data = r2[15];
          float v472_data = r2[17];
          float v473_data = r2[19];
          float v474_data = r2[21];
          float v475_data = r2[23];
          float v476_data = r2[25];
          tensorforge::VectorT<float, 16> v478_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v447_data, v464_data, v463_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v479_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v448_data, v465_data, v478_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v480_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v449_data, v466_data, v479_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v481_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v450_data, v467_data, v480_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v482_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v451_data, v468_data, v481_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v483_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v452_data, v469_data, v482_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v484_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v453_data, v470_data, v483_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v485_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v454_data, v471_data, v484_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v486_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v455_data, v472_data, v485_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v487_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v456_data, v473_data, v486_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v488_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v457_data, v474_data, v487_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v489_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v458_data, v475_data, v488_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v490_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v459_data, v476_data, v489_acc, 1, 0, 0);
          float v491_el = v490_acc[0];
          float v493_el = v490_acc[4];
          float v494_sw = tensorforge::swap<32>(v493_el);
          float v496_el = v490_acc[8];
          float v499_el = v490_acc[12];
          float v500_sw = tensorforge::swap<32>(v499_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v500_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v496_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v494_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v491_el, v491_el))))))));
          float v503_el = v490_acc[1];
          float v505_el = v490_acc[5];
          float v506_sw = tensorforge::swap<32>(v505_el);
          float v508_el = v490_acc[9];
          float v511_el = v490_acc[13];
          float v512_sw = tensorforge::swap<32>(v511_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v512_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v508_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v506_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v503_el, v503_el))))))));
          float v515_el = v490_acc[2];
          float v517_el = v490_acc[6];
          float v518_sw = tensorforge::swap<32>(v517_el);
          float v520_el = v490_acc[10];
          float v523_el = v490_acc[14];
          float v524_sw = tensorforge::swap<32>(v523_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v524_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v520_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v518_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v515_el, v515_el))))))));
          float v527_el = v490_acc[3];
          float v529_el = v490_acc[7];
          float v530_sw = tensorforge::swap<32>(v529_el);
          float v532_el = v490_acc[11];
          float v535_el = v490_acc[15];
          float v536_sw = tensorforge::swap<32>(v535_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v536_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v532_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v530_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v527_el, v527_el))))))));
          float v540_sw = tensorforge::swap<32>(v491_el);
          float v545_sw = tensorforge::swap<32>(v496_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v499_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v545_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v493_el, (tensorforge::dppUpdate<228, 1, 15, false>(v540_sw, v540_sw))))))));
          float v552_sw = tensorforge::swap<32>(v503_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v511_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v508_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v505_el, (tensorforge::dppUpdate<228, 1, 15, false>(v552_sw, v552_sw))))))));
          float v564_sw = tensorforge::swap<32>(v515_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v523_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v520_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v517_el, (tensorforge::dppUpdate<228, 1, 15, false>(v564_sw, v564_sw))))))));
          float v576_sw = tensorforge::swap<32>(v527_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v535_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>((tensorforge::swap<32>(v532_el)))), (tensorforge::dppUpdate<228, 2, 15, false>(v529_el, (tensorforge::dppUpdate<228, 1, 15, false>(v576_sw, v576_sw))))))));
          float v588_sw = tensorforge::swap<64>(v491_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v500_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v496_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v494_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v588_sw, v588_sw))))))));
          float v600_sw = tensorforge::swap<64>(v503_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v512_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v508_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v506_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v600_sw, v600_sw))))))));
          float v612_sw = tensorforge::swap<64>(v515_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v524_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v520_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v518_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v612_sw, v612_sw))))))));
          float v624_sw = tensorforge::swap<64>(v527_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v536_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v532_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v530_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v624_sw, v624_sw))))))));
          float v637_sw = tensorforge::swap<64>(v540_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v499_el, (tensorforge::dppUpdate<228, 4, 15, false>(v545_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v493_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v637_sw, v637_sw))))))));
          // wait(r5 = load{g>r}(glb_m4););
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

