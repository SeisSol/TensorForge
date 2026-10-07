// === base name ===
kernel_1f441f363a31109d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1f441f363a31109d = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1f441f363a31109d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1f441f363a31109d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1f441f363a31109d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1f441f363a31109d, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_1f441f363a31109d, block.x * block.y * block.z, 0));
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
void launcher_kernel_1f441f363a31109d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1f441f363a31109d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_1f441f363a31109d), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m6;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_1f441f363a31109d, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_1f441f363a31109d(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 32×16(32×16) {0..32}×{0..16} strided
    //   m1 32×12(32×12) {0..32}×{0..12} strided
    //   m2 12×16(12×16) {0..12}×{0..16} strided
    //   m3 32×12(32×12) {0..32}×{0..12} strided
    //   m4 12×8(12×8) {0..12}×{0..8} strided
    //   m5 32×12(32×12) {0..32}×{0..12} strided
    //   m6 12×8(12×8) {0..12}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j]@{0..32}×{0..8} += m3[i,k] × m4[k,j]
    //   m0[i,j]@{0..32}×{8..16} += m5[i,k] × m6[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,16]],"name":"m0","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,16]],"name":"m2","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[32,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[32,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 512 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 384 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 192 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 384 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 96 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v7_batchId0 * 384 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v7_batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v27_i1 * 32))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          bool v35_g = v25_lead < 12;
          if (v35_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 16; ++v36_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m2[(v25_lead + (v36_i1 * 12))]);
              r1[v36_i1] = v41_data;
            }
          }
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v287_i0 = 0; v287_i0 < 1; ++v287_i0) {
            int32_t v290_lead = v25_lead + (v287_i0 * 32);
            #pragma unroll
            for (int32_t v288_i1 = 0; v288_i1 < 12; ++v288_i1) {
              float v293_data = __builtin_nontemporal_load(&glb_m3[(v290_lead + (v288_i1 * 32))]);
              r3[(v287_i0 + v288_i1)] = v293_data;
            }
          }
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
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
          tensorforge::VectorT<float, 16> v60_acc{};
          float v61_data = r0[0];
          float v62_data = r0[1];
          float v63_data = r0[2];
          float v64_data = r0[3];
          float v65_data = r0[4];
          float v66_data = r0[5];
          float v67_data = r0[6];
          float v68_data = r0[7];
          float v69_data = r0[8];
          float v70_data = r0[9];
          float v71_data = r0[10];
          float v72_data = r0[11];
          tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v60_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v74_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v75_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v84_acc, 1, 0, 0);
          float v86_el = v85_acc[0];
          float v88_el = v85_acc[4];
          float v89_sw = tensorforge::swap<32>(v88_el);
          float v91_el = v85_acc[8];
          float v94_el = v85_acc[12];
          float v95_sw = tensorforge::swap<32>(v94_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v95_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v91_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v89_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v86_el, v86_el))))))));
          float v98_el = v85_acc[1];
          float v100_el = v85_acc[5];
          float v101_sw = tensorforge::swap<32>(v100_el);
          float v103_el = v85_acc[9];
          float v106_el = v85_acc[13];
          float v107_sw = tensorforge::swap<32>(v106_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v107_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v101_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v98_el, v98_el))))))));
          float v110_el = v85_acc[2];
          float v112_el = v85_acc[6];
          float v113_sw = tensorforge::swap<32>(v112_el);
          float v115_el = v85_acc[10];
          float v118_el = v85_acc[14];
          float v119_sw = tensorforge::swap<32>(v118_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v119_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v113_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v110_el, v110_el))))))));
          float v122_el = v85_acc[3];
          float v124_el = v85_acc[7];
          float v125_sw = tensorforge::swap<32>(v124_el);
          float v127_el = v85_acc[11];
          float v130_el = v85_acc[15];
          float v131_sw = tensorforge::swap<32>(v130_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v131_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v125_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v122_el, v122_el))))))));
          float v135_sw = tensorforge::swap<32>(v86_el);
          float v140_sw = tensorforge::swap<32>(v91_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v94_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v140_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v88_el, (tensorforge::dppUpdate<228, 1, 15, false>(v135_sw, v135_sw))))))));
          float v147_sw = tensorforge::swap<32>(v98_el);
          float v152_sw = tensorforge::swap<32>(v103_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v152_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v100_el, (tensorforge::dppUpdate<228, 1, 15, false>(v147_sw, v147_sw))))))));
          float v159_sw = tensorforge::swap<32>(v110_el);
          float v164_sw = tensorforge::swap<32>(v115_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v164_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v112_el, (tensorforge::dppUpdate<228, 1, 15, false>(v159_sw, v159_sw))))))));
          float v171_sw = tensorforge::swap<32>(v122_el);
          float v176_sw = tensorforge::swap<32>(v127_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v176_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_el, (tensorforge::dppUpdate<228, 1, 15, false>(v171_sw, v171_sw))))))));
          float v183_sw = tensorforge::swap<64>(v86_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v95_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v91_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v89_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v183_sw, v183_sw))))))));
          float v195_sw = tensorforge::swap<64>(v98_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v107_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v103_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v101_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v195_sw, v195_sw))))))));
          float v207_sw = tensorforge::swap<64>(v110_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v119_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v115_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v113_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v207_sw, v207_sw))))))));
          float v219_sw = tensorforge::swap<64>(v122_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v131_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v127_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v125_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v219_sw, v219_sw))))))));
          float v232_sw = tensorforge::swap<64>(v135_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v94_el, (tensorforge::dppUpdate<228, 4, 15, false>(v140_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v88_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v232_sw, v232_sw))))))));
          float v244_sw = tensorforge::swap<64>(v147_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v106_el, (tensorforge::dppUpdate<228, 4, 15, false>(v152_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v244_sw, v244_sw))))))));
          float v256_sw = tensorforge::swap<64>(v159_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v118_el, (tensorforge::dppUpdate<228, 4, 15, false>(v164_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v256_sw, v256_sw))))))));
          float v268_sw = tensorforge::swap<64>(v171_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_el, (tensorforge::dppUpdate<228, 4, 15, false>(v176_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v268_sw, v268_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v278_i0 = 0; v278_i0 < 1; ++v278_i0) {
            int32_t v283_lead = v25_lead + (v278_i0 * 32);
            #pragma unroll
            for (int32_t v279_i1 = 0; v279_i1 < 16; ++v279_i1) {
              float v281_data = r2[(v278_i0 + v279_i1)];
              glb_m0[(v283_lead + (v279_i1 * 32))] = v281_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v35_g) {
            #pragma unroll
            for (int32_t v296_i1 = 0; v296_i1 < 8; ++v296_i1) {
              float v301_data = __builtin_nontemporal_load(&glb_m4[(v25_lead + (v296_i1 * 12))]);
              r4[v296_i1] = v301_data;
            }
          }
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v387_i0 = 0; v387_i0 < 1; ++v387_i0) {
            int32_t v390_lead = v25_lead + (v387_i0 * 32);
            #pragma unroll
            for (int32_t v388_i1 = 0; v388_i1 < 12; ++v388_i1) {
              float v393_data = __builtin_nontemporal_load(&glb_m5[(v390_lead + (v388_i1 * 32))]);
              r6[(v387_i0 + v388_i1)] = v393_data;
            }
          }
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v304_data = r4[0];
          float v305_data = r4[1];
          float v306_data = r4[2];
          float v307_data = r4[3];
          float v308_tp{};
          float v309_tp{};
          float v310_tp{};
          float v311_tp{};
          tensorforge::transpose4x4b32(v308_tp, v309_tp, v310_tp, v311_tp, v304_data, v305_data, v306_data, v307_data);
          tensorforge::VectorT<float, 4> v312_acc{};
          float v313_data = r3[0];
          float v314_data = r3[1];
          float v315_data = r3[2];
          float v316_data = r3[3];
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v313_data, v312_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v314_data, v317_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v315_data, v318_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v316_data, v319_acc, 3, 0, 0);
          float v321_data = r3[4];
          float v322_data = r3[5];
          float v323_data = r3[6];
          float v324_data = r3[7];
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v321_data, v320_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v322_data, v325_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v323_data, v326_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v324_data, v327_acc, 3, 1, 0);
          float v329_data = r3[8];
          float v330_data = r3[9];
          float v331_data = r3[10];
          float v332_data = r3[11];
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v308_tp, v329_data, v328_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v309_tp, v330_data, v333_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v310_tp, v331_data, v334_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v332_data, v335_acc, 3, 2, 0);
          r5[0] = (v336_acc[0]);
          r5[1] = (v336_acc[1]);
          r5[2] = (v336_acc[2]);
          r5[3] = (v336_acc[3]);
          float v341_data = r4[4];
          float v342_data = r4[5];
          float v343_data = r4[6];
          float v344_data = r4[7];
          float v345_tp{};
          float v346_tp{};
          float v347_tp{};
          float v348_tp{};
          tensorforge::transpose4x4b32(v345_tp, v346_tp, v347_tp, v348_tp, v341_data, v342_data, v343_data, v344_data);
          tensorforge::VectorT<float, 4> v349_acc{};
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v313_data, v349_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v314_data, v354_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v315_data, v355_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v316_data, v356_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v321_data, v357_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v322_data, v362_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v323_data, v363_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v324_data, v364_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v345_tp, v329_data, v365_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v346_tp, v330_data, v370_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v347_tp, v331_data, v371_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v332_data, v372_acc, 3, 2, 0);
          r5[4] = (v373_acc[0]);
          r5[5] = (v373_acc[1]);
          r5[6] = (v373_acc[2]);
          r5[7] = (v373_acc[3]);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v378_i0 = 0; v378_i0 < 1; ++v378_i0) {
            int32_t v383_lead = v25_lead + (v378_i0 * 32);
            #pragma unroll
            for (int32_t v379_i1 = 0; v379_i1 < 8; ++v379_i1) {
              float v381_data = r5[(v378_i0 + v379_i1)];
              int32_t v385_a = v383_lead + (v379_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v385_a], v381_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v35_g) {
            #pragma unroll
            for (int32_t v396_i1 = 0; v396_i1 < 8; ++v396_i1) {
              float v401_data = __builtin_nontemporal_load(&glb_m6[(v25_lead + (v396_i1 * 12))]);
              r7[v396_i1] = v401_data;
            }
          }
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v404_data = r7[0];
          float v405_data = r7[1];
          float v406_data = r7[2];
          float v407_data = r7[3];
          float v408_tp{};
          float v409_tp{};
          float v410_tp{};
          float v411_tp{};
          tensorforge::transpose4x4b32(v408_tp, v409_tp, v410_tp, v411_tp, v404_data, v405_data, v406_data, v407_data);
          tensorforge::VectorT<float, 4> v412_acc{};
          float v413_data = r6[0];
          float v414_data = r6[1];
          float v415_data = r6[2];
          float v416_data = r6[3];
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v413_data, v412_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v414_data, v417_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v415_data, v418_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v416_data, v419_acc, 3, 0, 0);
          float v421_data = r6[4];
          float v422_data = r6[5];
          float v423_data = r6[6];
          float v424_data = r6[7];
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v421_data, v420_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v422_data, v425_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v423_data, v426_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v424_data, v427_acc, 3, 1, 0);
          float v429_data = r6[8];
          float v430_data = r6[9];
          float v431_data = r6[10];
          float v432_data = r6[11];
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v408_tp, v429_data, v428_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v409_tp, v430_data, v433_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v410_tp, v431_data, v434_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v411_tp, v432_data, v435_acc, 3, 2, 0);
          r8[0] = (v436_acc[0]);
          r8[1] = (v436_acc[1]);
          r8[2] = (v436_acc[2]);
          r8[3] = (v436_acc[3]);
          float v441_data = r7[4];
          float v442_data = r7[5];
          float v443_data = r7[6];
          float v444_data = r7[7];
          float v445_tp{};
          float v446_tp{};
          float v447_tp{};
          float v448_tp{};
          tensorforge::transpose4x4b32(v445_tp, v446_tp, v447_tp, v448_tp, v441_data, v442_data, v443_data, v444_data);
          tensorforge::VectorT<float, 4> v449_acc{};
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v413_data, v449_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v414_data, v454_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v415_data, v455_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v448_tp, v416_data, v456_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v421_data, v457_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v422_data, v462_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v423_data, v463_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v448_tp, v424_data, v464_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v445_tp, v429_data, v465_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v446_tp, v430_data, v470_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v447_tp, v431_data, v471_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v448_tp, v432_data, v472_acc, 3, 2, 0);
          r8[4] = (v473_acc[0]);
          r8[5] = (v473_acc[1]);
          r8[6] = (v473_acc[2]);
          r8[7] = (v473_acc[3]);
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v478_i0 = 0; v478_i0 < 1; ++v478_i0) {
            int32_t v483_lead = v25_lead + (v478_i0 * 32);
            #pragma unroll
            for (int32_t v479_i1 = 0; v479_i1 < 8; ++v479_i1) {
              float v481_data = r8[(v478_i0 + v479_i1)];
              int32_t v486_a = v483_lead + ((v479_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v486_a], v481_data);
            }
          }
        }
      }
    }
  }
}

