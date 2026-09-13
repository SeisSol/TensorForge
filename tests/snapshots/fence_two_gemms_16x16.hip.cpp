// === base name ===
kernel_d73e075c965d9e96

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d73e075c965d9e96 = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 2};
tensorforge::LaunchConfig launch_config_kernel_d73e075c965d9e96(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_d73e075c965d9e96(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d73e075c965d9e96(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d73e075c965d9e96, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d73e075c965d9e96, block.x * block.y * block.z, 0));
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
void launcher_kernel_d73e075c965d9e96(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d73e075c965d9e96(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d73e075c965d9e96), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags1;
  hipLaunchKernelGGL(kernel_kernel_d73e075c965d9e96, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d73e075c965d9e96(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} strided
    //   m5 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   fence
    //   m3[i,j] = m4[i,k] × m5[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0},{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"fence"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 256 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v1_batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 32;
          bool v19_g = v18_lead < 16;
          if (v19_g) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v18_lead + (v20_i1 * 16))]);
              r0[v20_i1] = v25_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v19_g) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 16; ++v28_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m2[(v18_lead + (v28_i1 * 16))]);
              r1[v28_i1] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v36_data = r1[0];
          float v37_data = r1[1];
          float v38_data = r1[2];
          float v39_data = r1[3];
          float v40_data = r1[4];
          float v41_data = r1[5];
          float v42_data = r1[6];
          float v43_data = r1[7];
          float v44_data = r1[8];
          float v45_data = r1[9];
          float v46_data = r1[10];
          float v47_data = r1[11];
          float v48_data = r1[12];
          float v49_data = r1[13];
          float v50_data = r1[14];
          float v51_data = r1[15];
          tensorforge::transpose16x16b32(v36_data, v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data);
          tensorforge::VectorT<float, 16> v52_acc{};
          float v53_data = r0[0];
          float v54_data = r0[1];
          float v55_data = r0[2];
          float v56_data = r0[3];
          float v57_data = r0[4];
          float v58_data = r0[5];
          float v59_data = r0[6];
          float v60_data = r0[7];
          float v61_data = r0[8];
          float v62_data = r0[9];
          float v63_data = r0[10];
          float v64_data = r0[11];
          float v65_data = r0[12];
          float v66_data = r0[13];
          float v67_data = r0[14];
          float v68_data = r0[15];
          tensorforge::VectorT<float, 16> v69_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v36_data, v53_data, v52_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v70_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v37_data, v54_data, v69_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v71_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v38_data, v55_data, v70_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v72_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v39_data, v56_data, v71_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v73_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v57_data, v72_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v58_data, v73_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v59_data, v74_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v60_data, v75_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 1, 0, 0);
          float v85_el = v84_acc[0];
          float v87_el = v84_acc[4];
          float v88_sw = tensorforge::swap<32>(v87_el);
          float v90_el = v84_acc[8];
          float v93_el = v84_acc[12];
          float v94_sw = tensorforge::swap<32>(v93_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v94_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v90_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v88_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v85_el, v85_el))))))));
          float v97_el = v84_acc[1];
          float v99_el = v84_acc[5];
          float v100_sw = tensorforge::swap<32>(v99_el);
          float v102_el = v84_acc[9];
          float v105_el = v84_acc[13];
          float v106_sw = tensorforge::swap<32>(v105_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v106_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v102_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v100_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v97_el, v97_el))))))));
          float v109_el = v84_acc[2];
          float v111_el = v84_acc[6];
          float v112_sw = tensorforge::swap<32>(v111_el);
          float v114_el = v84_acc[10];
          float v117_el = v84_acc[14];
          float v118_sw = tensorforge::swap<32>(v117_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v118_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v114_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v112_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v109_el, v109_el))))))));
          float v121_el = v84_acc[3];
          float v123_el = v84_acc[7];
          float v124_sw = tensorforge::swap<32>(v123_el);
          float v126_el = v84_acc[11];
          float v129_el = v84_acc[15];
          float v130_sw = tensorforge::swap<32>(v129_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v121_el, v121_el))))))));
          float v134_sw = tensorforge::swap<32>(v85_el);
          float v139_sw = tensorforge::swap<32>(v90_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v93_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v87_el, (tensorforge::dppUpdate<228, 1, 15, false>(v134_sw, v134_sw))))))));
          float v146_sw = tensorforge::swap<32>(v97_el);
          float v151_sw = tensorforge::swap<32>(v102_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v99_el, (tensorforge::dppUpdate<228, 1, 15, false>(v146_sw, v146_sw))))))));
          float v158_sw = tensorforge::swap<32>(v109_el);
          float v163_sw = tensorforge::swap<32>(v114_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v111_el, (tensorforge::dppUpdate<228, 1, 15, false>(v158_sw, v158_sw))))))));
          float v170_sw = tensorforge::swap<32>(v121_el);
          float v175_sw = tensorforge::swap<32>(v126_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v175_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v123_el, (tensorforge::dppUpdate<228, 1, 15, false>(v170_sw, v170_sw))))))));
          float v182_sw = tensorforge::swap<64>(v85_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v94_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v90_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v88_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v182_sw, v182_sw))))))));
          float v194_sw = tensorforge::swap<64>(v97_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v106_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v102_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v100_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v194_sw, v194_sw))))))));
          float v206_sw = tensorforge::swap<64>(v109_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v118_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v114_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v112_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v206_sw, v206_sw))))))));
          float v218_sw = tensorforge::swap<64>(v121_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v126_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v218_sw, v218_sw))))))));
          float v231_sw = tensorforge::swap<64>(v134_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v93_el, (tensorforge::dppUpdate<228, 4, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v87_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v231_sw, v231_sw))))))));
          float v243_sw = tensorforge::swap<64>(v146_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v105_el, (tensorforge::dppUpdate<228, 4, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v99_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v243_sw, v243_sw))))))));
          float v255_sw = tensorforge::swap<64>(v158_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v117_el, (tensorforge::dppUpdate<228, 4, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v255_sw, v255_sw))))))));
          float v267_sw = tensorforge::swap<64>(v170_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v129_el, (tensorforge::dppUpdate<228, 4, 15, false>(v175_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v267_sw, v267_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v19_g) {
            #pragma unroll
            for (int32_t v277_i1 = 0; v277_i1 < 16; ++v277_i1) {
              float v279_data = r2[v277_i1];
              glb_m0[(v18_lead + (v277_i1 * 16))] = v279_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y);
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t v285_batchId0 = ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y); v285_batchId0 < numElements1; v285_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v286_ahead1 = v285_batchId0 + (gridDim.x * blockDim.y);
        size_t v288_batchId1 = (v286_ahead1 < numElements1) ? v286_ahead1 : v285_batchId0;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[v285_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v285_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v285_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v285_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v285_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v285_batchId0 * 256 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v285_batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m4);
          int32_t v302_lead = threadIdx.x % 32;
          bool v303_g = v302_lead < 16;
          if (v303_g) {
            #pragma unroll
            for (int32_t v304_i1 = 0; v304_i1 < 16; ++v304_i1) {
              float v309_data = __builtin_nontemporal_load(&glb_m4[(v302_lead + (v304_i1 * 16))]);
              r0[v304_i1] = v309_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m5);
          if (v303_g) {
            #pragma unroll
            for (int32_t v312_i1 = 0; v312_i1 < 16; ++v312_i1) {
              float v317_data = __builtin_nontemporal_load(&glb_m5[(v302_lead + (v312_i1 * 16))]);
              r1[v312_i1] = v317_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m4););
          // wait(r1 = load{g>r}(glb_m5););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v320_data = r1[0];
          float v321_data = r1[1];
          float v322_data = r1[2];
          float v323_data = r1[3];
          float v324_data = r1[4];
          float v325_data = r1[5];
          float v326_data = r1[6];
          float v327_data = r1[7];
          float v328_data = r1[8];
          float v329_data = r1[9];
          float v330_data = r1[10];
          float v331_data = r1[11];
          float v332_data = r1[12];
          float v333_data = r1[13];
          float v334_data = r1[14];
          float v335_data = r1[15];
          tensorforge::transpose16x16b32(v320_data, v321_data, v322_data, v323_data, v324_data, v325_data, v326_data, v327_data, v328_data, v329_data, v330_data, v331_data, v332_data, v333_data, v334_data, v335_data);
          tensorforge::VectorT<float, 16> v336_acc{};
          float v337_data = r0[0];
          float v338_data = r0[1];
          float v339_data = r0[2];
          float v340_data = r0[3];
          float v341_data = r0[4];
          float v342_data = r0[5];
          float v343_data = r0[6];
          float v344_data = r0[7];
          float v345_data = r0[8];
          float v346_data = r0[9];
          float v347_data = r0[10];
          float v348_data = r0[11];
          float v349_data = r0[12];
          float v350_data = r0[13];
          float v351_data = r0[14];
          float v352_data = r0[15];
          tensorforge::VectorT<float, 16> v353_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v320_data, v337_data, v336_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v354_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v321_data, v338_data, v353_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v355_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v322_data, v339_data, v354_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v356_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v323_data, v340_data, v355_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v357_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v324_data, v341_data, v356_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v358_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v325_data, v342_data, v357_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v359_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v326_data, v343_data, v358_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v360_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v327_data, v344_data, v359_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v361_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v345_data, v360_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v362_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v346_data, v361_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v363_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v347_data, v362_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v348_data, v363_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v364_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v365_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v367_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v351_data, v366_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v352_data, v367_acc, 1, 0, 0);
          float v369_el = v368_acc[0];
          float v371_el = v368_acc[4];
          float v372_sw = tensorforge::swap<32>(v371_el);
          float v374_el = v368_acc[8];
          float v377_el = v368_acc[12];
          float v378_sw = tensorforge::swap<32>(v377_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v378_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v374_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v372_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v369_el, v369_el))))))));
          float v381_el = v368_acc[1];
          float v383_el = v368_acc[5];
          float v384_sw = tensorforge::swap<32>(v383_el);
          float v386_el = v368_acc[9];
          float v389_el = v368_acc[13];
          float v390_sw = tensorforge::swap<32>(v389_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v390_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v386_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v384_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v381_el, v381_el))))))));
          float v393_el = v368_acc[2];
          float v395_el = v368_acc[6];
          float v396_sw = tensorforge::swap<32>(v395_el);
          float v398_el = v368_acc[10];
          float v401_el = v368_acc[14];
          float v402_sw = tensorforge::swap<32>(v401_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v402_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v398_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v396_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v393_el, v393_el))))))));
          float v405_el = v368_acc[3];
          float v407_el = v368_acc[7];
          float v408_sw = tensorforge::swap<32>(v407_el);
          float v410_el = v368_acc[11];
          float v413_el = v368_acc[15];
          float v414_sw = tensorforge::swap<32>(v413_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v414_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v410_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v408_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v405_el, v405_el))))))));
          float v418_sw = tensorforge::swap<32>(v369_el);
          float v423_sw = tensorforge::swap<32>(v374_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v377_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v423_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v371_el, (tensorforge::dppUpdate<228, 1, 15, false>(v418_sw, v418_sw))))))));
          float v430_sw = tensorforge::swap<32>(v381_el);
          float v435_sw = tensorforge::swap<32>(v386_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v389_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v435_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v383_el, (tensorforge::dppUpdate<228, 1, 15, false>(v430_sw, v430_sw))))))));
          float v442_sw = tensorforge::swap<32>(v393_el);
          float v447_sw = tensorforge::swap<32>(v398_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v401_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v447_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v395_el, (tensorforge::dppUpdate<228, 1, 15, false>(v442_sw, v442_sw))))))));
          float v454_sw = tensorforge::swap<32>(v405_el);
          float v459_sw = tensorforge::swap<32>(v410_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v413_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v459_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v407_el, (tensorforge::dppUpdate<228, 1, 15, false>(v454_sw, v454_sw))))))));
          float v466_sw = tensorforge::swap<64>(v369_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v378_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v374_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v372_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v466_sw, v466_sw))))))));
          float v478_sw = tensorforge::swap<64>(v381_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v390_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v386_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v384_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v478_sw, v478_sw))))))));
          float v490_sw = tensorforge::swap<64>(v393_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v402_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v398_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v396_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v490_sw, v490_sw))))))));
          float v502_sw = tensorforge::swap<64>(v405_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v414_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v410_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v408_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v502_sw, v502_sw))))))));
          float v515_sw = tensorforge::swap<64>(v418_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v377_el, (tensorforge::dppUpdate<228, 4, 15, false>(v423_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v371_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v515_sw, v515_sw))))))));
          float v527_sw = tensorforge::swap<64>(v430_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v389_el, (tensorforge::dppUpdate<228, 4, 15, false>(v435_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v383_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v527_sw, v527_sw))))))));
          float v539_sw = tensorforge::swap<64>(v442_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v401_el, (tensorforge::dppUpdate<228, 4, 15, false>(v447_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v395_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v539_sw, v539_sw))))))));
          float v551_sw = tensorforge::swap<64>(v454_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v413_el, (tensorforge::dppUpdate<228, 4, 15, false>(v459_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v407_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v551_sw, v551_sw))))))));
          // glb_m3 = store{r>g}(r2);
          if (v303_g) {
            #pragma unroll
            for (int32_t v561_i1 = 0; v561_i1 < 16; ++v561_i1) {
              float v563_data = r2[v561_i1];
              glb_m3[(v302_lead + (v561_i1 * 16))] = v563_data;
            }
          }
        }
      }
    }
  }
}

