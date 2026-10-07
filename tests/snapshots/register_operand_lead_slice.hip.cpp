// === base name ===
kernel_95dc02aa574516b8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_95dc02aa574516b8 = {{32, 8, 1}, 32, 32, 1, 8, 16384, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_95dc02aa574516b8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_95dc02aa574516b8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_95dc02aa574516b8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_95dc02aa574516b8, block.x * block.y * block.z, 4096 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (4096 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_95dc02aa574516b8, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (4096 * sizeof(float)));
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
  config.sharedMemBytes = 4096 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_95dc02aa574516b8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_95dc02aa574516b8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_95dc02aa574516b8), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_95dc02aa574516b8, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_95dc02aa574516b8(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 16384 B shared, occupancy grid
    // operands:
    //   m0 32×32(32×32) {0..32}×{0..32} strided
    //   m1 32×16(32×16) {0..32}×{0..16} strided
    //   m2 32×32(32×32) {0..32}×{0..32} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    //   m4 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = m2[i,k] × m1[k,j]
    //   t2[i,j] = t0[k,i] × t1[k,j]
    //   m3[i,j] = t2[i,k]@{8..16}×{0..16} × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":4096}],"shared_bytes":16384,"shared_elements":4096,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[32,32]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[16,8]],"name":"m4","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,32]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,32]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,16]},{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,16]],"is_tmp":true,"name":"t2","offset":[8,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[512 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v8_batchId0 * 1024 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v8_batchId0 * 512 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v8_batchId0 * 1024 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 64 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v8_batchId0 * 128 + 0 + m4_extraOffset];
          float r0[32]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
            int32_t v28_lead = v24_lead + (v25_i0 * 32);
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 32; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m0[(v28_lead + (v26_i1 * 32))]);
              r0[(v25_i0 + v26_i1)] = v31_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
            int32_t v37_lead = v24_lead + (v34_i0 * 32);
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 16; ++v35_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v37_lead + (v35_i1 * 32))]);
              r1[(v34_i0 + v35_i1)] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[32]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v43_i0 = 0; v43_i0 < 1; ++v43_i0) {
            int32_t v46_lead = v24_lead + (v43_i0 * 32);
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 32; ++v44_i1) {
              float v49_data = __builtin_nontemporal_load(&glb_m2[(v46_lead + (v44_i1 * 32))]);
              r3[(v43_i0 + v44_i1)] = v49_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v52_data = r1[0];
          float v53_data = r1[1];
          float v54_data = r1[2];
          float v55_data = r1[3];
          float v56_data = r1[4];
          float v57_data = r1[5];
          float v58_data = r1[6];
          float v59_data = r1[7];
          float v60_data = r1[8];
          float v61_data = r1[9];
          float v62_data = r1[10];
          float v63_data = r1[11];
          float v64_data = r1[12];
          float v65_data = r1[13];
          float v66_data = r1[14];
          float v67_data = r1[15];
          tensorforge::transpose16x16b32(v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data, v67_data);
          tensorforge::VectorT<float, 16> v68_acc{};
          float v69_data = r0[0];
          float v70_data = r0[1];
          float v71_data = r0[2];
          float v72_data = r0[3];
          float v73_data = r0[4];
          float v74_data = r0[5];
          float v75_data = r0[6];
          float v76_data = r0[7];
          float v77_data = r0[8];
          float v78_data = r0[9];
          float v79_data = r0[10];
          float v80_data = r0[11];
          float v81_data = r0[12];
          float v82_data = r0[13];
          float v83_data = r0[14];
          float v84_data = r0[15];
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v68_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v90_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v91_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v92_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v93_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v94_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v95_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v81_data, v96_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v98_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v82_data, v97_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v99_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v83_data, v98_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v100_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v67_data, v84_data, v99_acc, 1, 0, 0);
          float v101_data = r0[16];
          float v102_data = r0[17];
          float v103_data = r0[18];
          float v104_data = r0[19];
          float v105_data = r0[20];
          float v106_data = r0[21];
          float v107_data = r0[22];
          float v108_data = r0[23];
          float v109_data = r0[24];
          float v110_data = r0[25];
          float v111_data = r0[26];
          float v112_data = r0[27];
          float v113_data = r0[28];
          float v114_data = r0[29];
          float v115_data = r0[30];
          float v116_data = r0[31];
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v101_data, v100_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v118_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v102_data, v117_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v119_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v103_data, v118_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v104_data, v119_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v121_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v105_data, v120_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v122_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v106_data, v121_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v107_data, v122_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v124_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v108_data, v123_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v125_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v109_data, v124_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v126_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v110_data, v125_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v127_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v111_data, v126_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v128_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v112_data, v127_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v129_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v113_data, v128_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v130_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v114_data, v129_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v131_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v115_data, v130_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v132_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v67_data, v116_data, v131_acc, 1, 1, 0);
          float v133_el = v132_acc[0];
          float v135_el = v132_acc[4];
          float v136_sw = tensorforge::swap<32>(v135_el);
          float v138_el = v132_acc[8];
          float v141_el = v132_acc[12];
          float v142_sw = tensorforge::swap<32>(v141_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v142_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v133_el, v133_el))))))));
          float v145_el = v132_acc[1];
          float v147_el = v132_acc[5];
          float v148_sw = tensorforge::swap<32>(v147_el);
          float v150_el = v132_acc[9];
          float v153_el = v132_acc[13];
          float v154_sw = tensorforge::swap<32>(v153_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v154_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v148_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v145_el, v145_el))))))));
          float v157_el = v132_acc[2];
          float v159_el = v132_acc[6];
          float v160_sw = tensorforge::swap<32>(v159_el);
          float v162_el = v132_acc[10];
          float v165_el = v132_acc[14];
          float v166_sw = tensorforge::swap<32>(v165_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v166_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v160_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v157_el, v157_el))))))));
          float v169_el = v132_acc[3];
          float v171_el = v132_acc[7];
          float v172_sw = tensorforge::swap<32>(v171_el);
          float v174_el = v132_acc[11];
          float v177_el = v132_acc[15];
          float v178_sw = tensorforge::swap<32>(v177_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v178_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v174_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v172_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v169_el, v169_el))))))));
          float v182_sw = tensorforge::swap<32>(v133_el);
          float v187_sw = tensorforge::swap<32>(v138_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v187_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v135_el, (tensorforge::dppUpdate<228, 1, 15, false>(v182_sw, v182_sw))))))));
          float v194_sw = tensorforge::swap<32>(v145_el);
          float v199_sw = tensorforge::swap<32>(v150_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v153_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v199_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v147_el, (tensorforge::dppUpdate<228, 1, 15, false>(v194_sw, v194_sw))))))));
          float v206_sw = tensorforge::swap<32>(v157_el);
          float v211_sw = tensorforge::swap<32>(v162_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v165_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v211_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v159_el, (tensorforge::dppUpdate<228, 1, 15, false>(v206_sw, v206_sw))))))));
          float v218_sw = tensorforge::swap<32>(v169_el);
          float v223_sw = tensorforge::swap<32>(v174_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v177_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v223_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v171_el, (tensorforge::dppUpdate<228, 1, 15, false>(v218_sw, v218_sw))))))));
          float v230_sw = tensorforge::swap<64>(v133_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v142_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v138_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v230_sw, v230_sw))))))));
          float v242_sw = tensorforge::swap<64>(v145_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v154_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v150_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v148_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v242_sw, v242_sw))))))));
          float v254_sw = tensorforge::swap<64>(v157_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v166_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v162_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v160_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v254_sw, v254_sw))))))));
          float v266_sw = tensorforge::swap<64>(v169_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v178_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v174_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v172_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v266_sw, v266_sw))))))));
          float v279_sw = tensorforge::swap<64>(v182_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v141_el, (tensorforge::dppUpdate<228, 4, 15, false>(v187_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v279_sw, v279_sw))))))));
          float v291_sw = tensorforge::swap<64>(v194_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v153_el, (tensorforge::dppUpdate<228, 4, 15, false>(v199_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v291_sw, v291_sw))))))));
          float v303_sw = tensorforge::swap<64>(v206_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v165_el, (tensorforge::dppUpdate<228, 4, 15, false>(v211_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v159_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v303_sw, v303_sw))))))));
          float v315_sw = tensorforge::swap<64>(v218_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v177_el, (tensorforge::dppUpdate<228, 4, 15, false>(v223_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v171_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v315_sw, v315_sw))))))));
          float r6[8]{};
          // r6 = load{g>r}(glb_m4);
          bool v326_g = v24_lead < 16;
          if (v326_g) {
            #pragma unroll
            for (int32_t v327_i1 = 0; v327_i1 < 8; ++v327_i1) {
              float v332_data = __builtin_nontemporal_load(&glb_m4[(v24_lead + (v327_i1 * 16))]);
              r6[v327_i1] = v332_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[16]{};
          // r4 = +(r3 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v335_data = r1[0];
          float v336_data = r1[1];
          float v337_data = r1[2];
          float v338_data = r1[3];
          float v339_data = r1[4];
          float v340_data = r1[5];
          float v341_data = r1[6];
          float v342_data = r1[7];
          float v343_data = r1[8];
          float v344_data = r1[9];
          float v345_data = r1[10];
          float v346_data = r1[11];
          float v347_data = r1[12];
          float v348_data = r1[13];
          float v349_data = r1[14];
          float v350_data = r1[15];
          tensorforge::transpose16x16b32(v335_data, v336_data, v337_data, v338_data, v339_data, v340_data, v341_data, v342_data, v343_data, v344_data, v345_data, v346_data, v347_data, v348_data, v349_data, v350_data);
          tensorforge::VectorT<float, 16> v351_acc{};
          float v352_data = r3[0];
          float v353_data = r3[1];
          float v354_data = r3[2];
          float v355_data = r3[3];
          float v356_data = r3[4];
          float v357_data = r3[5];
          float v358_data = r3[6];
          float v359_data = r3[7];
          float v360_data = r3[8];
          float v361_data = r3[9];
          float v362_data = r3[10];
          float v363_data = r3[11];
          float v364_data = r3[12];
          float v365_data = r3[13];
          float v366_data = r3[14];
          float v367_data = r3[15];
          tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v352_data, v351_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v369_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v336_data, v353_data, v368_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v370_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v337_data, v354_data, v369_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v371_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v355_data, v370_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v372_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v339_data, v356_data, v371_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v373_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v340_data, v357_data, v372_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v374_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v341_data, v358_data, v373_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v375_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v342_data, v359_data, v374_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v376_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v343_data, v360_data, v375_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v377_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v344_data, v361_data, v376_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v378_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v345_data, v362_data, v377_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v379_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v346_data, v363_data, v378_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v380_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v347_data, v364_data, v379_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v381_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v348_data, v365_data, v380_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v382_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v349_data, v366_data, v381_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v383_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v350_data, v367_data, v382_acc, 1, 0, 0);
          float v384_data = r3[16];
          float v385_data = r3[17];
          float v386_data = r3[18];
          float v387_data = r3[19];
          float v388_data = r3[20];
          float v389_data = r3[21];
          float v390_data = r3[22];
          float v391_data = r3[23];
          float v392_data = r3[24];
          float v393_data = r3[25];
          float v394_data = r3[26];
          float v395_data = r3[27];
          float v396_data = r3[28];
          float v397_data = r3[29];
          float v398_data = r3[30];
          float v399_data = r3[31];
          tensorforge::VectorT<float, 16> v400_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v384_data, v383_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v401_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v336_data, v385_data, v400_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v402_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v337_data, v386_data, v401_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v403_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v387_data, v402_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v404_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v339_data, v388_data, v403_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v405_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v340_data, v389_data, v404_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v406_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v341_data, v390_data, v405_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v407_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v342_data, v391_data, v406_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v408_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v343_data, v392_data, v407_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v409_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v344_data, v393_data, v408_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v410_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v345_data, v394_data, v409_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v411_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v346_data, v395_data, v410_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v412_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v347_data, v396_data, v411_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v413_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v348_data, v397_data, v412_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v414_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v349_data, v398_data, v413_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v415_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v350_data, v399_data, v414_acc, 1, 1, 0);
          float v416_el = v415_acc[0];
          float v418_el = v415_acc[4];
          float v419_sw = tensorforge::swap<32>(v418_el);
          float v421_el = v415_acc[8];
          float v424_el = v415_acc[12];
          float v425_sw = tensorforge::swap<32>(v424_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v425_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v421_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v419_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v416_el, v416_el))))))));
          float v428_el = v415_acc[1];
          float v430_el = v415_acc[5];
          float v431_sw = tensorforge::swap<32>(v430_el);
          float v433_el = v415_acc[9];
          float v436_el = v415_acc[13];
          float v437_sw = tensorforge::swap<32>(v436_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v437_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v433_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v431_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v428_el, v428_el))))))));
          float v440_el = v415_acc[2];
          float v442_el = v415_acc[6];
          float v443_sw = tensorforge::swap<32>(v442_el);
          float v445_el = v415_acc[10];
          float v448_el = v415_acc[14];
          float v449_sw = tensorforge::swap<32>(v448_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v449_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v445_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v443_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v440_el, v440_el))))))));
          float v452_el = v415_acc[3];
          float v454_el = v415_acc[7];
          float v455_sw = tensorforge::swap<32>(v454_el);
          float v457_el = v415_acc[11];
          float v460_el = v415_acc[15];
          float v461_sw = tensorforge::swap<32>(v460_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v461_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v457_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v455_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v452_el, v452_el))))))));
          float v465_sw = tensorforge::swap<32>(v416_el);
          float v470_sw = tensorforge::swap<32>(v421_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v424_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v470_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v418_el, (tensorforge::dppUpdate<228, 1, 15, false>(v465_sw, v465_sw))))))));
          float v477_sw = tensorforge::swap<32>(v428_el);
          float v482_sw = tensorforge::swap<32>(v433_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v436_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v482_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v430_el, (tensorforge::dppUpdate<228, 1, 15, false>(v477_sw, v477_sw))))))));
          float v489_sw = tensorforge::swap<32>(v440_el);
          float v494_sw = tensorforge::swap<32>(v445_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v448_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v494_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v442_el, (tensorforge::dppUpdate<228, 1, 15, false>(v489_sw, v489_sw))))))));
          float v501_sw = tensorforge::swap<32>(v452_el);
          float v506_sw = tensorforge::swap<32>(v457_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v460_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v506_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v454_el, (tensorforge::dppUpdate<228, 1, 15, false>(v501_sw, v501_sw))))))));
          float v513_sw = tensorforge::swap<64>(v416_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v425_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v421_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v419_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v513_sw, v513_sw))))))));
          float v525_sw = tensorforge::swap<64>(v428_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v437_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v433_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v431_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v525_sw, v525_sw))))))));
          float v537_sw = tensorforge::swap<64>(v440_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v449_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v445_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v443_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v537_sw, v537_sw))))))));
          float v549_sw = tensorforge::swap<64>(v452_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v461_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v457_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v455_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v549_sw, v549_sw))))))));
          float v562_sw = tensorforge::swap<64>(v465_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v424_el, (tensorforge::dppUpdate<228, 4, 15, false>(v470_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v418_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v562_sw, v562_sw))))))));
          float v574_sw = tensorforge::swap<64>(v477_sw);
          r4[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v436_el, (tensorforge::dppUpdate<228, 4, 15, false>(v482_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v430_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v574_sw, v574_sw))))))));
          float v586_sw = tensorforge::swap<64>(v489_sw);
          r4[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v448_el, (tensorforge::dppUpdate<228, 4, 15, false>(v494_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v442_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v586_sw, v586_sw))))))));
          float v598_sw = tensorforge::swap<64>(v501_sw);
          r4[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v460_el, (tensorforge::dppUpdate<228, 4, 15, false>(v506_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v454_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v598_sw, v598_sw))))))));
          // s0 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v608_i0 = 0; v608_i0 < 1; ++v608_i0) {
            int32_t v613_lead = v24_lead + (v608_i0 * 32);
            #pragma unroll
            for (int32_t v609_i1 = 0; v609_i1 < 16; ++v609_i1) {
              float v611_data = r2[(v608_i0 + v609_i1)];
              int32_t v615_a = v613_lead + (v609_i1 * 32);
              s0[(v615_a ^ ((v615_a >> 5) & 31))] = v611_data;
            }
          }
          float r5[16]{};
          // r5 = +(s0 * r4) + None
          // [(0, 16), (0, 16)] [(0, 32)]
          float v620_data = r4[0];
          float v621_data = r4[1];
          float v622_data = r4[2];
          float v623_data = r4[3];
          float v624_data = r4[4];
          float v625_data = r4[5];
          float v626_data = r4[6];
          float v627_data = r4[7];
          float v628_data = r4[8];
          float v629_data = r4[9];
          float v630_data = r4[10];
          float v631_data = r4[11];
          float v632_data = r4[12];
          float v633_data = r4[13];
          float v634_data = r4[14];
          float v635_data = r4[15];
          tensorforge::transpose16x16b32(v620_data, v621_data, v622_data, v623_data, v624_data, v625_data, v626_data, v627_data, v628_data, v629_data, v630_data, v631_data, v632_data, v633_data, v634_data, v635_data);
          tensorforge::VectorT<float, 16> v636_acc{};
          int32_t v639_a = v24_lead * 32;
          int32_t v643_sw = v639_a ^ ((v639_a >> 5) & 31);
          float v644_data = s0[v643_sw];
          int32_t v645_a = 1 + v639_a;
          float v649_data = s0[(v645_a ^ ((v645_a >> 5) & 31))];
          int32_t v650_a = 2 + v639_a;
          float v654_data = s0[(v650_a ^ ((v650_a >> 5) & 31))];
          int32_t v655_a = 3 + v639_a;
          float v659_data = s0[(v655_a ^ ((v655_a >> 5) & 31))];
          int32_t v660_a = 4 + v639_a;
          float v664_data = s0[(v660_a ^ ((v660_a >> 5) & 31))];
          int32_t v665_a = 5 + v639_a;
          float v669_data = s0[(v665_a ^ ((v665_a >> 5) & 31))];
          int32_t v670_a = 6 + v639_a;
          float v674_data = s0[(v670_a ^ ((v670_a >> 5) & 31))];
          int32_t v675_a = 7 + v639_a;
          float v679_data = s0[(v675_a ^ ((v675_a >> 5) & 31))];
          int32_t v680_a = 8 + v639_a;
          float v684_data = s0[(v680_a ^ ((v680_a >> 5) & 31))];
          int32_t v685_a = 9 + v639_a;
          float v689_data = s0[(v685_a ^ ((v685_a >> 5) & 31))];
          int32_t v690_a = 10 + v639_a;
          float v694_data = s0[(v690_a ^ ((v690_a >> 5) & 31))];
          int32_t v695_a = 11 + v639_a;
          float v699_data = s0[(v695_a ^ ((v695_a >> 5) & 31))];
          int32_t v700_a = 12 + v639_a;
          float v704_data = s0[(v700_a ^ ((v700_a >> 5) & 31))];
          int32_t v705_a = 13 + v639_a;
          float v709_data = s0[(v705_a ^ ((v705_a >> 5) & 31))];
          int32_t v710_a = 14 + v639_a;
          float v714_data = s0[(v710_a ^ ((v710_a >> 5) & 31))];
          int32_t v715_a = 15 + v639_a;
          float v719_data = s0[(v715_a ^ ((v715_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v720_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v620_data, v644_data, v636_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v721_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v621_data, v649_data, v720_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v722_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v622_data, v654_data, v721_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v723_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v659_data, v722_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v724_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v664_data, v723_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v725_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v669_data, v724_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v726_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v674_data, v725_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v727_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v627_data, v679_data, v726_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v728_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v628_data, v684_data, v727_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v729_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v629_data, v689_data, v728_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v730_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v630_data, v694_data, v729_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v731_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v631_data, v699_data, v730_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v732_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v632_data, v704_data, v731_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v733_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v633_data, v709_data, v732_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v734_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v634_data, v714_data, v733_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v735_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v635_data, v719_data, v734_acc, 1, 0, 0);
          int32_t v736_a = 16 + v639_a;
          float v740_data = s0[(v736_a ^ ((v736_a >> 5) & 31))];
          int32_t v741_a = 17 + v639_a;
          float v745_data = s0[(v741_a ^ ((v741_a >> 5) & 31))];
          int32_t v746_a = 18 + v639_a;
          float v750_data = s0[(v746_a ^ ((v746_a >> 5) & 31))];
          int32_t v751_a = 19 + v639_a;
          float v755_data = s0[(v751_a ^ ((v751_a >> 5) & 31))];
          int32_t v756_a = 20 + v639_a;
          float v760_data = s0[(v756_a ^ ((v756_a >> 5) & 31))];
          int32_t v761_a = 21 + v639_a;
          float v765_data = s0[(v761_a ^ ((v761_a >> 5) & 31))];
          int32_t v766_a = 22 + v639_a;
          float v770_data = s0[(v766_a ^ ((v766_a >> 5) & 31))];
          int32_t v771_a = 23 + v639_a;
          float v775_data = s0[(v771_a ^ ((v771_a >> 5) & 31))];
          int32_t v776_a = 24 + v639_a;
          float v780_data = s0[(v776_a ^ ((v776_a >> 5) & 31))];
          int32_t v781_a = 25 + v639_a;
          float v785_data = s0[(v781_a ^ ((v781_a >> 5) & 31))];
          int32_t v786_a = 26 + v639_a;
          float v790_data = s0[(v786_a ^ ((v786_a >> 5) & 31))];
          int32_t v791_a = 27 + v639_a;
          float v795_data = s0[(v791_a ^ ((v791_a >> 5) & 31))];
          int32_t v796_a = 28 + v639_a;
          float v800_data = s0[(v796_a ^ ((v796_a >> 5) & 31))];
          int32_t v801_a = 29 + v639_a;
          float v805_data = s0[(v801_a ^ ((v801_a >> 5) & 31))];
          int32_t v806_a = 30 + v639_a;
          float v810_data = s0[(v806_a ^ ((v806_a >> 5) & 31))];
          int32_t v811_a = 31 + v639_a;
          float v815_data = s0[(v811_a ^ ((v811_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v816_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v620_data, v740_data, v735_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v817_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v621_data, v745_data, v816_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v818_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v622_data, v750_data, v817_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v819_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v755_data, v818_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v820_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v760_data, v819_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v821_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v765_data, v820_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v822_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v770_data, v821_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v823_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v627_data, v775_data, v822_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v824_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v628_data, v780_data, v823_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v825_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v629_data, v785_data, v824_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v826_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v630_data, v790_data, v825_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v827_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v631_data, v795_data, v826_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v828_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v632_data, v800_data, v827_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v829_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v633_data, v805_data, v828_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v830_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v634_data, v810_data, v829_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v831_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v635_data, v815_data, v830_acc, 1, 1, 0);
          float v832_el = v831_acc[0];
          float v834_el = v831_acc[4];
          float v835_sw = tensorforge::swap<32>(v834_el);
          float v837_el = v831_acc[8];
          float v840_el = v831_acc[12];
          float v841_sw = tensorforge::swap<32>(v840_el);
          r5[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v841_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v837_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v835_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v832_el, v832_el))))))));
          float v844_el = v831_acc[1];
          float v846_el = v831_acc[5];
          float v847_sw = tensorforge::swap<32>(v846_el);
          float v849_el = v831_acc[9];
          float v852_el = v831_acc[13];
          float v853_sw = tensorforge::swap<32>(v852_el);
          r5[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v853_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v849_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v847_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v844_el, v844_el))))))));
          float v856_el = v831_acc[2];
          float v858_el = v831_acc[6];
          float v859_sw = tensorforge::swap<32>(v858_el);
          float v861_el = v831_acc[10];
          float v864_el = v831_acc[14];
          float v865_sw = tensorforge::swap<32>(v864_el);
          r5[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v865_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v861_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v859_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v856_el, v856_el))))))));
          float v868_el = v831_acc[3];
          float v870_el = v831_acc[7];
          float v871_sw = tensorforge::swap<32>(v870_el);
          float v873_el = v831_acc[11];
          float v876_el = v831_acc[15];
          float v877_sw = tensorforge::swap<32>(v876_el);
          r5[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v877_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v873_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v871_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v868_el, v868_el))))))));
          float v881_sw = tensorforge::swap<32>(v832_el);
          float v886_sw = tensorforge::swap<32>(v837_el);
          r5[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v840_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v886_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v834_el, (tensorforge::dppUpdate<228, 1, 15, false>(v881_sw, v881_sw))))))));
          float v893_sw = tensorforge::swap<32>(v844_el);
          float v898_sw = tensorforge::swap<32>(v849_el);
          r5[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v852_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v898_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v846_el, (tensorforge::dppUpdate<228, 1, 15, false>(v893_sw, v893_sw))))))));
          float v905_sw = tensorforge::swap<32>(v856_el);
          float v910_sw = tensorforge::swap<32>(v861_el);
          r5[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v864_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v910_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v858_el, (tensorforge::dppUpdate<228, 1, 15, false>(v905_sw, v905_sw))))))));
          float v917_sw = tensorforge::swap<32>(v868_el);
          float v922_sw = tensorforge::swap<32>(v873_el);
          r5[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v876_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v922_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v870_el, (tensorforge::dppUpdate<228, 1, 15, false>(v917_sw, v917_sw))))))));
          float v929_sw = tensorforge::swap<64>(v832_el);
          r5[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v841_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v837_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v835_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v929_sw, v929_sw))))))));
          float v941_sw = tensorforge::swap<64>(v844_el);
          r5[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v853_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v849_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v847_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v941_sw, v941_sw))))))));
          float v953_sw = tensorforge::swap<64>(v856_el);
          r5[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v865_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v861_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v859_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v953_sw, v953_sw))))))));
          float v965_sw = tensorforge::swap<64>(v868_el);
          r5[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v877_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v873_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v871_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v965_sw, v965_sw))))))));
          float v978_sw = tensorforge::swap<64>(v881_sw);
          r5[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v840_el, (tensorforge::dppUpdate<228, 4, 15, false>(v886_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v834_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v978_sw, v978_sw))))))));
          float v990_sw = tensorforge::swap<64>(v893_sw);
          r5[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v852_el, (tensorforge::dppUpdate<228, 4, 15, false>(v898_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v846_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v990_sw, v990_sw))))))));
          float v1002_sw = tensorforge::swap<64>(v905_sw);
          r5[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v864_el, (tensorforge::dppUpdate<228, 4, 15, false>(v910_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v858_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1002_sw, v1002_sw))))))));
          float v1014_sw = tensorforge::swap<64>(v917_sw);
          r5[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v876_el, (tensorforge::dppUpdate<228, 4, 15, false>(v922_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v870_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1014_sw, v1014_sw))))))));
          // wait(r6 = load{g>r}(glb_m4););
          float r7[8]{};
          // r7 = +(r5 * r6) + None
          // [(8, 16), (0, 8)] [(0, 16)]
          float v1025_data = r6[0];
          float v1026_data = r6[1];
          float v1027_data = r6[2];
          float v1028_data = r6[3];
          float v1029_tp{};
          float v1030_tp{};
          float v1031_tp{};
          float v1032_tp{};
          tensorforge::transpose4x4b32(v1029_tp, v1030_tp, v1031_tp, v1032_tp, v1025_data, v1026_data, v1027_data, v1028_data);
          tensorforge::VectorT<float, 4> v1033_acc{};
          float v1034_data = r5[0];
          float v1035_data = r5[1];
          float v1036_data = r5[2];
          float v1037_data = r5[3];
          tensorforge::VectorT<float, 4> v1038_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1034_data, v1033_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1039_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1035_data, v1038_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1040_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1036_data, v1039_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1041_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1037_data, v1040_acc, 3, 0, 0);
          float v1042_data = r5[4];
          float v1043_data = r5[5];
          float v1044_data = r5[6];
          float v1045_data = r5[7];
          tensorforge::VectorT<float, 4> v1046_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1042_data, v1041_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1047_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1043_data, v1046_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1048_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1044_data, v1047_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1049_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1045_data, v1048_acc, 3, 1, 0);
          float v1050_data = r5[8];
          float v1051_data = r5[9];
          float v1052_data = r5[10];
          float v1053_data = r5[11];
          tensorforge::VectorT<float, 4> v1054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1050_data, v1049_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1055_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1051_data, v1054_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1056_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1052_data, v1055_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1053_data, v1056_acc, 3, 2, 0);
          float v1058_data = r5[12];
          float v1059_data = r5[13];
          float v1060_data = r5[14];
          float v1061_data = r5[15];
          tensorforge::VectorT<float, 4> v1062_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1058_data, v1057_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1063_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1030_tp, v1059_data, v1062_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1064_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1031_tp, v1060_data, v1063_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1065_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1061_data, v1064_acc, 3, 3, 0);
          r7[0] = (v1065_acc[0]);
          r7[1] = (v1065_acc[1]);
          r7[2] = (v1065_acc[2]);
          r7[3] = (v1065_acc[3]);
          float v1070_data = r6[4];
          float v1071_data = r6[5];
          float v1072_data = r6[6];
          float v1073_data = r6[7];
          float v1074_tp{};
          float v1075_tp{};
          float v1076_tp{};
          float v1077_tp{};
          tensorforge::transpose4x4b32(v1074_tp, v1075_tp, v1076_tp, v1077_tp, v1070_data, v1071_data, v1072_data, v1073_data);
          tensorforge::VectorT<float, 4> v1078_acc{};
          tensorforge::VectorT<float, 4> v1083_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1034_data, v1078_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1084_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1075_tp, v1035_data, v1083_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1085_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1076_tp, v1036_data, v1084_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1086_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1037_data, v1085_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1091_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1042_data, v1086_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1092_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1075_tp, v1043_data, v1091_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1093_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1076_tp, v1044_data, v1092_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1094_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1045_data, v1093_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1099_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1050_data, v1094_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1075_tp, v1051_data, v1099_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1076_tp, v1052_data, v1100_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1053_data, v1101_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1058_data, v1102_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1075_tp, v1059_data, v1107_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1076_tp, v1060_data, v1108_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1061_data, v1109_acc, 3, 3, 0);
          r7[4] = (v1110_acc[0]);
          r7[5] = (v1110_acc[1]);
          r7[6] = (v1110_acc[2]);
          r7[7] = (v1110_acc[3]);
          // glb_m3 = store{r>g}(r7);
          if ((v24_lead >= 8) && v326_g) {
            int32_t v1122_off = v24_lead + -8;
            #pragma unroll
            for (int32_t v1117_i1 = 0; v1117_i1 < 8; ++v1117_i1) {
              float v1119_data = r7[v1117_i1];
              glb_m3[(v1122_off + (v1117_i1 * 8))] = v1119_data;
            }
          }
        }
      }
    }
  }
}

