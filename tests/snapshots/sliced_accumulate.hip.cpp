// === base name ===
kernel_1ce9eaae15af55d8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1ce9eaae15af55d8 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1ce9eaae15af55d8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1ce9eaae15af55d8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1ce9eaae15af55d8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1ce9eaae15af55d8, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_1ce9eaae15af55d8, block.x * block.y * block.z, 0));
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
void launcher_kernel_1ce9eaae15af55d8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1ce9eaae15af55d8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_1ce9eaae15af55d8), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_1ce9eaae15af55d8, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_1ce9eaae15af55d8(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,16]],"name":"m0","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,16]],"name":"m2","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[32,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[32,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 512 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 384 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 192 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 384 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 96 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v1_batchId0 * 384 + 0 + m5_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m6 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m6[v1_batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v23_lead = v19_lead + (v20_i0 * 32);
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v21_i1 * 32))]);
              r0[(v20_i0 + v21_i1)] = v26_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          bool v29_g = v19_lead < 12;
          if (v29_g) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 16; ++v30_i1) {
              float v35_data = __builtin_nontemporal_load(&glb_m2[(v19_lead + (v30_i1 * 12))]);
              r1[v30_i1] = v35_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v38_i0 = 0; v38_i0 < 1; ++v38_i0) {
            int32_t v41_lead = v19_lead + (v38_i0 * 32);
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m3[(v41_lead + (v39_i1 * 32))]);
              r3[(v38_i0 + v39_i1)] = v44_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
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
          tensorforge::VectorT<float, 16> v63_acc{};
          float v64_data = r0[0];
          float v65_data = r0[1];
          float v66_data = r0[2];
          float v67_data = r0[3];
          float v68_data = r0[4];
          float v69_data = r0[5];
          float v70_data = r0[6];
          float v71_data = r0[7];
          float v72_data = r0[8];
          float v73_data = r0[9];
          float v74_data = r0[10];
          float v75_data = r0[11];
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v63_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v87_acc, 1, 0, 0);
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
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v110_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v104_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v101_el, v101_el))))))));
          float v113_el = v88_acc[2];
          float v115_el = v88_acc[6];
          float v116_sw = tensorforge::swap<32>(v115_el);
          float v118_el = v88_acc[10];
          float v121_el = v88_acc[14];
          float v122_sw = tensorforge::swap<32>(v121_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v122_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v116_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v113_el, v113_el))))))));
          float v125_el = v88_acc[3];
          float v127_el = v88_acc[7];
          float v128_sw = tensorforge::swap<32>(v127_el);
          float v130_el = v88_acc[11];
          float v133_el = v88_acc[15];
          float v134_sw = tensorforge::swap<32>(v133_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v134_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v128_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v125_el, v125_el))))))));
          float v138_sw = tensorforge::swap<32>(v89_el);
          float v143_sw = tensorforge::swap<32>(v94_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v143_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v91_el, (tensorforge::dppUpdate<228, 1, 15, false>(v138_sw, v138_sw))))))));
          float v150_sw = tensorforge::swap<32>(v101_el);
          float v155_sw = tensorforge::swap<32>(v106_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v155_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v103_el, (tensorforge::dppUpdate<228, 1, 15, false>(v150_sw, v150_sw))))))));
          float v162_sw = tensorforge::swap<32>(v113_el);
          float v167_sw = tensorforge::swap<32>(v118_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v167_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v115_el, (tensorforge::dppUpdate<228, 1, 15, false>(v162_sw, v162_sw))))))));
          float v174_sw = tensorforge::swap<32>(v125_el);
          float v179_sw = tensorforge::swap<32>(v130_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v179_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v127_el, (tensorforge::dppUpdate<228, 1, 15, false>(v174_sw, v174_sw))))))));
          float v186_sw = tensorforge::swap<64>(v89_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v98_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v94_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v92_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v186_sw, v186_sw))))))));
          float v198_sw = tensorforge::swap<64>(v101_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v110_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v106_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v104_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v198_sw, v198_sw))))))));
          float v210_sw = tensorforge::swap<64>(v113_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v122_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v118_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v116_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v210_sw, v210_sw))))))));
          float v222_sw = tensorforge::swap<64>(v125_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v134_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v130_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v128_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v222_sw, v222_sw))))))));
          float v235_sw = tensorforge::swap<64>(v138_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v97_el, (tensorforge::dppUpdate<228, 4, 15, false>(v143_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v91_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v235_sw, v235_sw))))))));
          float v247_sw = tensorforge::swap<64>(v150_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v109_el, (tensorforge::dppUpdate<228, 4, 15, false>(v155_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v247_sw, v247_sw))))))));
          float v259_sw = tensorforge::swap<64>(v162_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v121_el, (tensorforge::dppUpdate<228, 4, 15, false>(v167_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v259_sw, v259_sw))))))));
          float v271_sw = tensorforge::swap<64>(v174_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v133_el, (tensorforge::dppUpdate<228, 4, 15, false>(v179_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v271_sw, v271_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v281_i0 = 0; v281_i0 < 1; ++v281_i0) {
            int32_t v286_lead = v19_lead + (v281_i0 * 32);
            #pragma unroll
            for (int32_t v282_i1 = 0; v282_i1 < 16; ++v282_i1) {
              float v284_data = r2[(v281_i0 + v282_i1)];
              glb_m0[(v286_lead + (v282_i1 * 32))] = v284_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v29_g) {
            #pragma unroll
            for (int32_t v290_i1 = 0; v290_i1 < 8; ++v290_i1) {
              float v295_data = __builtin_nontemporal_load(&glb_m4[(v19_lead + (v290_i1 * 12))]);
              r4[v290_i1] = v295_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v298_i0 = 0; v298_i0 < 1; ++v298_i0) {
            int32_t v301_lead = v19_lead + (v298_i0 * 32);
            #pragma unroll
            for (int32_t v299_i1 = 0; v299_i1 < 12; ++v299_i1) {
              float v304_data = __builtin_nontemporal_load(&glb_m5[(v301_lead + (v299_i1 * 32))]);
              r6[(v298_i0 + v299_i1)] = v304_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v307_data = r4[0];
          float v308_data = r4[1];
          float v309_data = r4[2];
          float v310_data = r4[3];
          float v311_tp{};
          float v312_tp{};
          float v313_tp{};
          float v314_tp{};
          tensorforge::transpose4x4b32(v311_tp, v312_tp, v313_tp, v314_tp, v307_data, v308_data, v309_data, v310_data);
          tensorforge::VectorT<float, 4> v315_acc{};
          float v316_data = r3[0];
          float v317_data = r3[1];
          float v318_data = r3[2];
          float v319_data = r3[3];
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v316_data, v315_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v317_data, v320_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v318_data, v321_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v319_data, v322_acc, 3, 0, 0);
          float v324_data = r3[4];
          float v325_data = r3[5];
          float v326_data = r3[6];
          float v327_data = r3[7];
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v324_data, v323_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v325_data, v328_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v326_data, v329_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v327_data, v330_acc, 3, 1, 0);
          float v332_data = r3[8];
          float v333_data = r3[9];
          float v334_data = r3[10];
          float v335_data = r3[11];
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v311_tp, v332_data, v331_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v333_data, v336_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v334_data, v337_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v335_data, v338_acc, 3, 2, 0);
          r5[0] = (v339_acc[0]);
          r5[1] = (v339_acc[1]);
          r5[2] = (v339_acc[2]);
          r5[3] = (v339_acc[3]);
          float v344_data = r4[4];
          float v345_data = r4[5];
          float v346_data = r4[6];
          float v347_data = r4[7];
          float v348_tp{};
          float v349_tp{};
          float v350_tp{};
          float v351_tp{};
          tensorforge::transpose4x4b32(v348_tp, v349_tp, v350_tp, v351_tp, v344_data, v345_data, v346_data, v347_data);
          tensorforge::VectorT<float, 4> v352_acc{};
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v316_data, v352_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v317_data, v357_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v318_data, v358_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v319_data, v359_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v324_data, v360_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v325_data, v365_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v326_data, v366_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v327_data, v367_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v348_tp, v332_data, v368_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v349_tp, v333_data, v373_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v350_tp, v334_data, v374_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v351_tp, v335_data, v375_acc, 3, 2, 0);
          r5[4] = (v376_acc[0]);
          r5[5] = (v376_acc[1]);
          r5[6] = (v376_acc[2]);
          r5[7] = (v376_acc[3]);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v381_i0 = 0; v381_i0 < 1; ++v381_i0) {
            int32_t v386_lead = v19_lead + (v381_i0 * 32);
            #pragma unroll
            for (int32_t v382_i1 = 0; v382_i1 < 8; ++v382_i1) {
              float v384_data = r5[(v381_i0 + v382_i1)];
              int32_t v388_a = v386_lead + (v382_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v388_a], v384_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v29_g) {
            #pragma unroll
            for (int32_t v390_i1 = 0; v390_i1 < 8; ++v390_i1) {
              float v395_data = __builtin_nontemporal_load(&glb_m6[(v19_lead + (v390_i1 * 12))]);
              r7[v390_i1] = v395_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v398_data = r7[0];
          float v399_data = r7[1];
          float v400_data = r7[2];
          float v401_data = r7[3];
          float v402_tp{};
          float v403_tp{};
          float v404_tp{};
          float v405_tp{};
          tensorforge::transpose4x4b32(v402_tp, v403_tp, v404_tp, v405_tp, v398_data, v399_data, v400_data, v401_data);
          tensorforge::VectorT<float, 4> v406_acc{};
          float v407_data = r6[0];
          float v408_data = r6[1];
          float v409_data = r6[2];
          float v410_data = r6[3];
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v407_data, v406_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v408_data, v411_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v409_data, v412_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v410_data, v413_acc, 3, 0, 0);
          float v415_data = r6[4];
          float v416_data = r6[5];
          float v417_data = r6[6];
          float v418_data = r6[7];
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v415_data, v414_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v416_data, v419_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v417_data, v420_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v418_data, v421_acc, 3, 1, 0);
          float v423_data = r6[8];
          float v424_data = r6[9];
          float v425_data = r6[10];
          float v426_data = r6[11];
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v402_tp, v423_data, v422_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v403_tp, v424_data, v427_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v404_tp, v425_data, v428_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v405_tp, v426_data, v429_acc, 3, 2, 0);
          r8[0] = (v430_acc[0]);
          r8[1] = (v430_acc[1]);
          r8[2] = (v430_acc[2]);
          r8[3] = (v430_acc[3]);
          float v435_data = r7[4];
          float v436_data = r7[5];
          float v437_data = r7[6];
          float v438_data = r7[7];
          float v439_tp{};
          float v440_tp{};
          float v441_tp{};
          float v442_tp{};
          tensorforge::transpose4x4b32(v439_tp, v440_tp, v441_tp, v442_tp, v435_data, v436_data, v437_data, v438_data);
          tensorforge::VectorT<float, 4> v443_acc{};
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v407_data, v443_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v408_data, v448_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v409_data, v449_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v410_data, v450_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v415_data, v451_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v416_data, v456_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v417_data, v457_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v418_data, v458_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v439_tp, v423_data, v459_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v440_tp, v424_data, v464_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v441_tp, v425_data, v465_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v442_tp, v426_data, v466_acc, 3, 2, 0);
          r8[4] = (v467_acc[0]);
          r8[5] = (v467_acc[1]);
          r8[6] = (v467_acc[2]);
          r8[7] = (v467_acc[3]);
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v472_i0 = 0; v472_i0 < 1; ++v472_i0) {
            int32_t v477_lead = v19_lead + (v472_i0 * 32);
            #pragma unroll
            for (int32_t v473_i1 = 0; v473_i1 < 8; ++v473_i1) {
              float v475_data = r8[(v472_i0 + v473_i1)];
              int32_t v480_a = v477_lead + ((v473_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v480_a], v475_data);
            }
          }
        }
      }
    }
  }
}

