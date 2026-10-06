// === base name ===
kernel_7e5ff106a5a4a3c1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7e5ff106a5a4a3c1 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7e5ff106a5a4a3c1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7e5ff106a5a4a3c1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7e5ff106a5a4a3c1(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7e5ff106a5a4a3c1, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_7e5ff106a5a4a3c1, block.x * block.y * block.z, 0));
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
void launcher_kernel_7e5ff106a5a4a3c1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7e5ff106a5a4a3c1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_7e5ff106a5a4a3c1), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_7e5ff106a5a4a3c1, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, m6Arg, m6_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7e5ff106a5a4a3c1(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m6, size_t m6_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v44_i0 = 0; v44_i0 < 1; ++v44_i0) {
            int32_t v47_lead = v25_lead + (v44_i0 * 32);
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              float v50_data = __builtin_nontemporal_load(&glb_m3[(v47_lead + (v45_i1 * 32))]);
              r3[(v44_i0 + v45_i1)] = v50_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          float v53_data = r1[0];
          float v54_data = r1[1];
          float v55_data = r1[2];
          float v56_data = r1[3];
          float v57_data = r1[4];
          float v58_data = r1[5];
          float v59_data = r1[6];
          float v60_data = r1[7];
          float v61_data = r1[8];
          float v62_data = r1[9];
          float v63_data = r1[10];
          float v64_data = r1[11];
          float v65_data = r1[12];
          float v66_data = r1[13];
          float v67_data = r1[14];
          float v68_data = r1[15];
          tensorforge::transpose16x16b32(v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data, v67_data, v68_data);
          tensorforge::VectorT<float, 16> v69_acc{};
          float v70_data = r0[0];
          float v71_data = r0[1];
          float v72_data = r0[2];
          float v73_data = r0[3];
          float v74_data = r0[4];
          float v75_data = r0[5];
          float v76_data = r0[6];
          float v77_data = r0[7];
          float v78_data = r0[8];
          float v79_data = r0[9];
          float v80_data = r0[10];
          float v81_data = r0[11];
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v69_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v90_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v91_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v92_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v81_data, v93_acc, 1, 0, 0);
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
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v116_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v110_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v107_el, v107_el))))))));
          float v119_el = v94_acc[2];
          float v121_el = v94_acc[6];
          float v122_sw = tensorforge::swap<32>(v121_el);
          float v124_el = v94_acc[10];
          float v127_el = v94_acc[14];
          float v128_sw = tensorforge::swap<32>(v127_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v128_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v122_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v119_el, v119_el))))))));
          float v131_el = v94_acc[3];
          float v133_el = v94_acc[7];
          float v134_sw = tensorforge::swap<32>(v133_el);
          float v136_el = v94_acc[11];
          float v139_el = v94_acc[15];
          float v140_sw = tensorforge::swap<32>(v139_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v140_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v136_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v134_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v131_el, v131_el))))))));
          float v144_sw = tensorforge::swap<32>(v95_el);
          float v149_sw = tensorforge::swap<32>(v100_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v149_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v97_el, (tensorforge::dppUpdate<228, 1, 15, false>(v144_sw, v144_sw))))))));
          float v156_sw = tensorforge::swap<32>(v107_el);
          float v161_sw = tensorforge::swap<32>(v112_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v161_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v109_el, (tensorforge::dppUpdate<228, 1, 15, false>(v156_sw, v156_sw))))))));
          float v168_sw = tensorforge::swap<32>(v119_el);
          float v173_sw = tensorforge::swap<32>(v124_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v173_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v121_el, (tensorforge::dppUpdate<228, 1, 15, false>(v168_sw, v168_sw))))))));
          float v180_sw = tensorforge::swap<32>(v131_el);
          float v185_sw = tensorforge::swap<32>(v136_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v139_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v185_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v133_el, (tensorforge::dppUpdate<228, 1, 15, false>(v180_sw, v180_sw))))))));
          float v192_sw = tensorforge::swap<64>(v95_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v104_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v100_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v98_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v192_sw, v192_sw))))))));
          float v204_sw = tensorforge::swap<64>(v107_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v116_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v112_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v110_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v204_sw, v204_sw))))))));
          float v216_sw = tensorforge::swap<64>(v119_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v128_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v124_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v122_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v216_sw, v216_sw))))))));
          float v228_sw = tensorforge::swap<64>(v131_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v140_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v136_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v134_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v228_sw, v228_sw))))))));
          float v241_sw = tensorforge::swap<64>(v144_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v103_el, (tensorforge::dppUpdate<228, 4, 15, false>(v149_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v241_sw, v241_sw))))))));
          float v253_sw = tensorforge::swap<64>(v156_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v115_el, (tensorforge::dppUpdate<228, 4, 15, false>(v161_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v253_sw, v253_sw))))))));
          float v265_sw = tensorforge::swap<64>(v168_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v127_el, (tensorforge::dppUpdate<228, 4, 15, false>(v173_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v265_sw, v265_sw))))))));
          float v277_sw = tensorforge::swap<64>(v180_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v139_el, (tensorforge::dppUpdate<228, 4, 15, false>(v185_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v133_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v277_sw, v277_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v287_i0 = 0; v287_i0 < 1; ++v287_i0) {
            int32_t v292_lead = v25_lead + (v287_i0 * 32);
            #pragma unroll
            for (int32_t v288_i1 = 0; v288_i1 < 16; ++v288_i1) {
              float v290_data = r2[(v287_i0 + v288_i1)];
              glb_m0[(v292_lead + (v288_i1 * 32))] = v290_data;
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
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v304_i0 = 0; v304_i0 < 1; ++v304_i0) {
            int32_t v307_lead = v25_lead + (v304_i0 * 32);
            #pragma unroll
            for (int32_t v305_i1 = 0; v305_i1 < 12; ++v305_i1) {
              float v310_data = __builtin_nontemporal_load(&glb_m5[(v307_lead + (v305_i1 * 32))]);
              r6[(v304_i0 + v305_i1)] = v310_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v313_data = r4[0];
          float v314_data = r4[1];
          float v315_data = r4[2];
          float v316_data = r4[3];
          float v317_tp{};
          float v318_tp{};
          float v319_tp{};
          float v320_tp{};
          tensorforge::transpose4x4b32(v317_tp, v318_tp, v319_tp, v320_tp, v313_data, v314_data, v315_data, v316_data);
          tensorforge::VectorT<float, 4> v321_acc{};
          float v322_data = r3[0];
          float v323_data = r3[1];
          float v324_data = r3[2];
          float v325_data = r3[3];
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v322_data, v321_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v323_data, v326_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v324_data, v327_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v329_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v325_data, v328_acc, 3, 0, 0);
          float v330_data = r3[4];
          float v331_data = r3[5];
          float v332_data = r3[6];
          float v333_data = r3[7];
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v330_data, v329_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v331_data, v334_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v332_data, v335_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v333_data, v336_acc, 3, 1, 0);
          float v338_data = r3[8];
          float v339_data = r3[9];
          float v340_data = r3[10];
          float v341_data = r3[11];
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v338_data, v337_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v339_data, v342_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v340_data, v343_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v341_data, v344_acc, 3, 2, 0);
          r5[0] = (v345_acc[0]);
          r5[1] = (v345_acc[1]);
          r5[2] = (v345_acc[2]);
          r5[3] = (v345_acc[3]);
          float v350_data = r4[4];
          float v351_data = r4[5];
          float v352_data = r4[6];
          float v353_data = r4[7];
          float v354_tp{};
          float v355_tp{};
          float v356_tp{};
          float v357_tp{};
          tensorforge::transpose4x4b32(v354_tp, v355_tp, v356_tp, v357_tp, v350_data, v351_data, v352_data, v353_data);
          tensorforge::VectorT<float, 4> v358_acc{};
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v322_data, v358_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v323_data, v363_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v324_data, v364_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v325_data, v365_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v330_data, v366_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v331_data, v371_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v332_data, v372_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v333_data, v373_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v338_data, v374_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v339_data, v379_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v340_data, v380_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v357_tp, v341_data, v381_acc, 3, 2, 0);
          r5[4] = (v382_acc[0]);
          r5[5] = (v382_acc[1]);
          r5[6] = (v382_acc[2]);
          r5[7] = (v382_acc[3]);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v387_i0 = 0; v387_i0 < 1; ++v387_i0) {
            int32_t v392_lead = v25_lead + (v387_i0 * 32);
            #pragma unroll
            for (int32_t v388_i1 = 0; v388_i1 < 8; ++v388_i1) {
              float v390_data = r5[(v387_i0 + v388_i1)];
              int32_t v394_a = v392_lead + (v388_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v394_a], v390_data);
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
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
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

