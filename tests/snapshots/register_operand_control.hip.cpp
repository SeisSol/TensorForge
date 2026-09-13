// === base name ===
kernel_a77cdae20fc5dbf8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a77cdae20fc5dbf8 = {{32, 8, 1}, 32, 32, 1, 8, 16384, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a77cdae20fc5dbf8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a77cdae20fc5dbf8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a77cdae20fc5dbf8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a77cdae20fc5dbf8, block.x * block.y * block.z, 4096 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (4096 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_a77cdae20fc5dbf8, block.x * block.y * block.z, 0));
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
void launcher_kernel_a77cdae20fc5dbf8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a77cdae20fc5dbf8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a77cdae20fc5dbf8), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_a77cdae20fc5dbf8, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a77cdae20fc5dbf8(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 16384 B shared, occupancy grid
    // operands:
    //   m0 32×32(32×32) {0..32}×{0..32} strided
    //   m1 32×16(32×16) {0..32}×{0..16} strided
    //   m2 32×32(32×32) {0..32}×{0..32} strided
    //   m3 16×8(16×8) {0..16}×{0..8} strided
    //   m4 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = m2[i,k] × m1[k,j]
    //   t2[i,j] = t0[k,i] × t1[k,j]
    //   m3[i,j] = t2[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":4096}],"shared_bytes":16384,"shared_elements":4096,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[32,32]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[16,8]],"name":"m3","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[16,8]],"name":"m4","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,32]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,32]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,16]},{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[512 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[512];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 1024 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v5_batchId0 * 512 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 1024 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v5_batchId0 * 128 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v5_batchId0 * 128 + 0 + m4_extraOffset];
          float r0[32]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v25_lead = v21_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 32; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v25_lead + (v23_i1 * 32))]);
              r0[(v22_i0 + v23_i1)] = v28_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v34_lead = v21_lead + (v31_i0 * 32);
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 16; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v34_lead + (v32_i1 * 32))]);
              r1[(v31_i0 + v32_i1)] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[32]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
            int32_t v43_lead = v21_lead + (v40_i0 * 32);
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 32; ++v41_i1) {
              float v46_data = __builtin_nontemporal_load(&glb_m2[(v43_lead + (v41_i1 * 32))]);
              r3[(v40_i0 + v41_i1)] = v46_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v49_data = r1[0];
          float v50_data = r1[1];
          float v51_data = r1[2];
          float v52_data = r1[3];
          float v53_data = r1[4];
          float v54_data = r1[5];
          float v55_data = r1[6];
          float v56_data = r1[7];
          float v57_data = r1[8];
          float v58_data = r1[9];
          float v59_data = r1[10];
          float v60_data = r1[11];
          float v61_data = r1[12];
          float v62_data = r1[13];
          float v63_data = r1[14];
          float v64_data = r1[15];
          tensorforge::transpose16x16b32(v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data);
          tensorforge::VectorT<float, 16> v65_acc{};
          float v66_data = r0[0];
          float v67_data = r0[1];
          float v68_data = r0[2];
          float v69_data = r0[3];
          float v70_data = r0[4];
          float v71_data = r0[5];
          float v72_data = r0[6];
          float v73_data = r0[7];
          float v74_data = r0[8];
          float v75_data = r0[9];
          float v76_data = r0[10];
          float v77_data = r0[11];
          float v78_data = r0[12];
          float v79_data = r0[13];
          float v80_data = r0[14];
          float v81_data = r0[15];
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v65_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 1, 0, 0);
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
          float v98_data = r0[16];
          float v99_data = r0[17];
          float v100_data = r0[18];
          float v101_data = r0[19];
          float v102_data = r0[20];
          float v103_data = r0[21];
          float v104_data = r0[22];
          float v105_data = r0[23];
          float v106_data = r0[24];
          float v107_data = r0[25];
          float v108_data = r0[26];
          float v109_data = r0[27];
          float v110_data = r0[28];
          float v111_data = r0[29];
          float v112_data = r0[30];
          float v113_data = r0[31];
          tensorforge::VectorT<float, 16> v114_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v98_data, v97_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v115_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v99_data, v114_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v116_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v100_data, v115_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v101_data, v116_acc, 1, 1, 0);
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
          float v130_el = v129_acc[0];
          float v132_el = v129_acc[4];
          float v133_sw = tensorforge::swap<32>(v132_el);
          float v135_el = v129_acc[8];
          float v138_el = v129_acc[12];
          float v139_sw = tensorforge::swap<32>(v138_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v130_el, v130_el))))))));
          float v142_el = v129_acc[1];
          float v144_el = v129_acc[5];
          float v145_sw = tensorforge::swap<32>(v144_el);
          float v147_el = v129_acc[9];
          float v150_el = v129_acc[13];
          float v151_sw = tensorforge::swap<32>(v150_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v142_el, v142_el))))))));
          float v154_el = v129_acc[2];
          float v156_el = v129_acc[6];
          float v157_sw = tensorforge::swap<32>(v156_el);
          float v159_el = v129_acc[10];
          float v162_el = v129_acc[14];
          float v163_sw = tensorforge::swap<32>(v162_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v159_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v154_el, v154_el))))))));
          float v166_el = v129_acc[3];
          float v168_el = v129_acc[7];
          float v169_sw = tensorforge::swap<32>(v168_el);
          float v171_el = v129_acc[11];
          float v174_el = v129_acc[15];
          float v175_sw = tensorforge::swap<32>(v174_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v175_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v171_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v169_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v166_el, v166_el))))))));
          float v179_sw = tensorforge::swap<32>(v130_el);
          float v184_sw = tensorforge::swap<32>(v135_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v184_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v132_el, (tensorforge::dppUpdate<228, 1, 15, false>(v179_sw, v179_sw))))))));
          float v191_sw = tensorforge::swap<32>(v142_el);
          float v196_sw = tensorforge::swap<32>(v147_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v196_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v144_el, (tensorforge::dppUpdate<228, 1, 15, false>(v191_sw, v191_sw))))))));
          float v203_sw = tensorforge::swap<32>(v154_el);
          float v208_sw = tensorforge::swap<32>(v159_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v208_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v156_el, (tensorforge::dppUpdate<228, 1, 15, false>(v203_sw, v203_sw))))))));
          float v215_sw = tensorforge::swap<32>(v166_el);
          float v220_sw = tensorforge::swap<32>(v171_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v174_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v220_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v168_el, (tensorforge::dppUpdate<228, 1, 15, false>(v215_sw, v215_sw))))))));
          float v227_sw = tensorforge::swap<64>(v130_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v135_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v227_sw, v227_sw))))))));
          float v239_sw = tensorforge::swap<64>(v142_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v147_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v239_sw, v239_sw))))))));
          float v251_sw = tensorforge::swap<64>(v154_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v159_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v251_sw, v251_sw))))))));
          float v263_sw = tensorforge::swap<64>(v166_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v175_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v171_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v169_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v263_sw, v263_sw))))))));
          float v276_sw = tensorforge::swap<64>(v179_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v138_el, (tensorforge::dppUpdate<228, 4, 15, false>(v184_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v276_sw, v276_sw))))))));
          float v288_sw = tensorforge::swap<64>(v191_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v150_el, (tensorforge::dppUpdate<228, 4, 15, false>(v196_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v288_sw, v288_sw))))))));
          float v300_sw = tensorforge::swap<64>(v203_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v162_el, (tensorforge::dppUpdate<228, 4, 15, false>(v208_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v156_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v300_sw, v300_sw))))))));
          float v312_sw = tensorforge::swap<64>(v215_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v174_el, (tensorforge::dppUpdate<228, 4, 15, false>(v220_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v168_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v312_sw, v312_sw))))))));
          float r6[8]{};
          // r6 = load{g>r}(glb_m4);
          bool v323_g = v21_lead < 16;
          if (v323_g) {
            #pragma unroll
            for (int32_t v324_i1 = 0; v324_i1 < 8; ++v324_i1) {
              float v329_data = __builtin_nontemporal_load(&glb_m4[(v21_lead + (v324_i1 * 16))]);
              r6[v324_i1] = v329_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[16]{};
          // r4 = +(r3 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v332_data = r1[0];
          float v333_data = r1[1];
          float v334_data = r1[2];
          float v335_data = r1[3];
          float v336_data = r1[4];
          float v337_data = r1[5];
          float v338_data = r1[6];
          float v339_data = r1[7];
          float v340_data = r1[8];
          float v341_data = r1[9];
          float v342_data = r1[10];
          float v343_data = r1[11];
          float v344_data = r1[12];
          float v345_data = r1[13];
          float v346_data = r1[14];
          float v347_data = r1[15];
          tensorforge::transpose16x16b32(v332_data, v333_data, v334_data, v335_data, v336_data, v337_data, v338_data, v339_data, v340_data, v341_data, v342_data, v343_data, v344_data, v345_data, v346_data, v347_data);
          tensorforge::VectorT<float, 16> v348_acc{};
          float v349_data = r3[0];
          float v350_data = r3[1];
          float v351_data = r3[2];
          float v352_data = r3[3];
          float v353_data = r3[4];
          float v354_data = r3[5];
          float v355_data = r3[6];
          float v356_data = r3[7];
          float v357_data = r3[8];
          float v358_data = r3[9];
          float v359_data = r3[10];
          float v360_data = r3[11];
          float v361_data = r3[12];
          float v362_data = r3[13];
          float v363_data = r3[14];
          float v364_data = r3[15];
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v348_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v365_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v367_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v351_data, v366_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v352_data, v367_acc, 1, 0, 0);
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
          float v381_data = r3[16];
          float v382_data = r3[17];
          float v383_data = r3[18];
          float v384_data = r3[19];
          float v385_data = r3[20];
          float v386_data = r3[21];
          float v387_data = r3[22];
          float v388_data = r3[23];
          float v389_data = r3[24];
          float v390_data = r3[25];
          float v391_data = r3[26];
          float v392_data = r3[27];
          float v393_data = r3[28];
          float v394_data = r3[29];
          float v395_data = r3[30];
          float v396_data = r3[31];
          tensorforge::VectorT<float, 16> v397_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v381_data, v380_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v398_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v382_data, v397_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v399_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v383_data, v398_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v400_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v384_data, v399_acc, 1, 1, 0);
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
          float v413_el = v412_acc[0];
          float v415_el = v412_acc[4];
          float v416_sw = tensorforge::swap<32>(v415_el);
          float v418_el = v412_acc[8];
          float v421_el = v412_acc[12];
          float v422_sw = tensorforge::swap<32>(v421_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v422_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v418_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v416_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v413_el, v413_el))))))));
          float v425_el = v412_acc[1];
          float v427_el = v412_acc[5];
          float v428_sw = tensorforge::swap<32>(v427_el);
          float v430_el = v412_acc[9];
          float v433_el = v412_acc[13];
          float v434_sw = tensorforge::swap<32>(v433_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v434_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v430_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v428_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v425_el, v425_el))))))));
          float v437_el = v412_acc[2];
          float v439_el = v412_acc[6];
          float v440_sw = tensorforge::swap<32>(v439_el);
          float v442_el = v412_acc[10];
          float v445_el = v412_acc[14];
          float v446_sw = tensorforge::swap<32>(v445_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v446_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v442_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v440_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v437_el, v437_el))))))));
          float v449_el = v412_acc[3];
          float v451_el = v412_acc[7];
          float v452_sw = tensorforge::swap<32>(v451_el);
          float v454_el = v412_acc[11];
          float v457_el = v412_acc[15];
          float v458_sw = tensorforge::swap<32>(v457_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v458_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v454_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v452_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v449_el, v449_el))))))));
          float v462_sw = tensorforge::swap<32>(v413_el);
          float v467_sw = tensorforge::swap<32>(v418_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v421_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v467_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v415_el, (tensorforge::dppUpdate<228, 1, 15, false>(v462_sw, v462_sw))))))));
          float v474_sw = tensorforge::swap<32>(v425_el);
          float v479_sw = tensorforge::swap<32>(v430_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v433_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v479_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v427_el, (tensorforge::dppUpdate<228, 1, 15, false>(v474_sw, v474_sw))))))));
          float v486_sw = tensorforge::swap<32>(v437_el);
          float v491_sw = tensorforge::swap<32>(v442_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v445_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v491_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v439_el, (tensorforge::dppUpdate<228, 1, 15, false>(v486_sw, v486_sw))))))));
          float v498_sw = tensorforge::swap<32>(v449_el);
          float v503_sw = tensorforge::swap<32>(v454_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v457_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v503_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v451_el, (tensorforge::dppUpdate<228, 1, 15, false>(v498_sw, v498_sw))))))));
          float v510_sw = tensorforge::swap<64>(v413_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v422_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v418_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v416_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v510_sw, v510_sw))))))));
          float v522_sw = tensorforge::swap<64>(v425_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v434_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v430_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v428_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v522_sw, v522_sw))))))));
          float v534_sw = tensorforge::swap<64>(v437_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v446_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v442_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v440_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v534_sw, v534_sw))))))));
          float v546_sw = tensorforge::swap<64>(v449_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v458_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v454_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v452_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v546_sw, v546_sw))))))));
          float v559_sw = tensorforge::swap<64>(v462_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v421_el, (tensorforge::dppUpdate<228, 4, 15, false>(v467_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v415_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v559_sw, v559_sw))))))));
          float v571_sw = tensorforge::swap<64>(v474_sw);
          r4[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v433_el, (tensorforge::dppUpdate<228, 4, 15, false>(v479_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v427_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v571_sw, v571_sw))))))));
          float v583_sw = tensorforge::swap<64>(v486_sw);
          r4[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v445_el, (tensorforge::dppUpdate<228, 4, 15, false>(v491_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v439_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v583_sw, v583_sw))))))));
          float v595_sw = tensorforge::swap<64>(v498_sw);
          r4[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v457_el, (tensorforge::dppUpdate<228, 4, 15, false>(v503_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v451_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v595_sw, v595_sw))))))));
          // s0 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v605_i0 = 0; v605_i0 < 1; ++v605_i0) {
            int32_t v610_lead = v21_lead + (v605_i0 * 32);
            #pragma unroll
            for (int32_t v606_i1 = 0; v606_i1 < 16; ++v606_i1) {
              float v608_data = r2[(v605_i0 + v606_i1)];
              int32_t v612_a = v610_lead + (v606_i1 * 32);
              s0[(v612_a ^ ((v612_a >> 5) & 31))] = v608_data;
            }
          }
          float r5[16]{};
          // r5 = +(s0 * r4) + None
          // [(0, 16), (0, 16)] [(0, 32)]
          float v617_data = r4[0];
          float v618_data = r4[1];
          float v619_data = r4[2];
          float v620_data = r4[3];
          float v621_data = r4[4];
          float v622_data = r4[5];
          float v623_data = r4[6];
          float v624_data = r4[7];
          float v625_data = r4[8];
          float v626_data = r4[9];
          float v627_data = r4[10];
          float v628_data = r4[11];
          float v629_data = r4[12];
          float v630_data = r4[13];
          float v631_data = r4[14];
          float v632_data = r4[15];
          tensorforge::transpose16x16b32(v617_data, v618_data, v619_data, v620_data, v621_data, v622_data, v623_data, v624_data, v625_data, v626_data, v627_data, v628_data, v629_data, v630_data, v631_data, v632_data);
          tensorforge::VectorT<float, 16> v633_acc{};
          int32_t v636_a = v21_lead * 32;
          float v641_data = s0[(v636_a ^ ((v636_a >> 5) & 31))];
          int32_t v642_a = 1 + v636_a;
          float v646_data = s0[(v642_a ^ ((v642_a >> 5) & 31))];
          int32_t v647_a = 2 + v636_a;
          float v651_data = s0[(v647_a ^ ((v647_a >> 5) & 31))];
          int32_t v652_a = 3 + v636_a;
          float v656_data = s0[(v652_a ^ ((v652_a >> 5) & 31))];
          int32_t v657_a = 4 + v636_a;
          float v661_data = s0[(v657_a ^ ((v657_a >> 5) & 31))];
          int32_t v662_a = 5 + v636_a;
          float v666_data = s0[(v662_a ^ ((v662_a >> 5) & 31))];
          int32_t v667_a = 6 + v636_a;
          float v671_data = s0[(v667_a ^ ((v667_a >> 5) & 31))];
          int32_t v672_a = 7 + v636_a;
          float v676_data = s0[(v672_a ^ ((v672_a >> 5) & 31))];
          int32_t v677_a = 8 + v636_a;
          float v681_data = s0[(v677_a ^ ((v677_a >> 5) & 31))];
          int32_t v682_a = 9 + v636_a;
          float v686_data = s0[(v682_a ^ ((v682_a >> 5) & 31))];
          int32_t v687_a = 10 + v636_a;
          float v691_data = s0[(v687_a ^ ((v687_a >> 5) & 31))];
          int32_t v692_a = 11 + v636_a;
          float v696_data = s0[(v692_a ^ ((v692_a >> 5) & 31))];
          int32_t v697_a = 12 + v636_a;
          float v701_data = s0[(v697_a ^ ((v697_a >> 5) & 31))];
          int32_t v702_a = 13 + v636_a;
          float v706_data = s0[(v702_a ^ ((v702_a >> 5) & 31))];
          int32_t v707_a = 14 + v636_a;
          float v711_data = s0[(v707_a ^ ((v707_a >> 5) & 31))];
          int32_t v712_a = 15 + v636_a;
          float v716_data = s0[(v712_a ^ ((v712_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v717_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v617_data, v641_data, v633_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v718_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v618_data, v646_data, v717_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v719_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v619_data, v651_data, v718_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v720_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v620_data, v656_data, v719_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v721_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v621_data, v661_data, v720_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v722_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v622_data, v666_data, v721_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v723_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v671_data, v722_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v724_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v676_data, v723_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v725_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v681_data, v724_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v726_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v686_data, v725_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v727_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v627_data, v691_data, v726_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v728_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v628_data, v696_data, v727_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v729_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v629_data, v701_data, v728_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v730_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v630_data, v706_data, v729_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v731_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v631_data, v711_data, v730_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v732_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v632_data, v716_data, v731_acc, 1, 0, 0);
          int32_t v733_a = 16 + v636_a;
          float v737_data = s0[(v733_a ^ ((v733_a >> 5) & 31))];
          int32_t v738_a = 17 + v636_a;
          float v742_data = s0[(v738_a ^ ((v738_a >> 5) & 31))];
          int32_t v743_a = 18 + v636_a;
          float v747_data = s0[(v743_a ^ ((v743_a >> 5) & 31))];
          int32_t v748_a = 19 + v636_a;
          float v752_data = s0[(v748_a ^ ((v748_a >> 5) & 31))];
          int32_t v753_a = 20 + v636_a;
          float v757_data = s0[(v753_a ^ ((v753_a >> 5) & 31))];
          int32_t v758_a = 21 + v636_a;
          float v762_data = s0[(v758_a ^ ((v758_a >> 5) & 31))];
          int32_t v763_a = 22 + v636_a;
          float v767_data = s0[(v763_a ^ ((v763_a >> 5) & 31))];
          int32_t v768_a = 23 + v636_a;
          float v772_data = s0[(v768_a ^ ((v768_a >> 5) & 31))];
          int32_t v773_a = 24 + v636_a;
          float v777_data = s0[(v773_a ^ ((v773_a >> 5) & 31))];
          int32_t v778_a = 25 + v636_a;
          float v782_data = s0[(v778_a ^ ((v778_a >> 5) & 31))];
          int32_t v783_a = 26 + v636_a;
          float v787_data = s0[(v783_a ^ ((v783_a >> 5) & 31))];
          int32_t v788_a = 27 + v636_a;
          float v792_data = s0[(v788_a ^ ((v788_a >> 5) & 31))];
          int32_t v793_a = 28 + v636_a;
          float v797_data = s0[(v793_a ^ ((v793_a >> 5) & 31))];
          int32_t v798_a = 29 + v636_a;
          float v802_data = s0[(v798_a ^ ((v798_a >> 5) & 31))];
          int32_t v803_a = 30 + v636_a;
          float v807_data = s0[(v803_a ^ ((v803_a >> 5) & 31))];
          int32_t v808_a = 31 + v636_a;
          float v812_data = s0[(v808_a ^ ((v808_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v813_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v617_data, v737_data, v732_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v814_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v618_data, v742_data, v813_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v815_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v619_data, v747_data, v814_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v816_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v620_data, v752_data, v815_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v817_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v621_data, v757_data, v816_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v818_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v622_data, v762_data, v817_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v819_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v767_data, v818_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v820_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v772_data, v819_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v821_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v777_data, v820_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v822_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v782_data, v821_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v823_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v627_data, v787_data, v822_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v824_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v628_data, v792_data, v823_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v825_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v629_data, v797_data, v824_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v826_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v630_data, v802_data, v825_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v827_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v631_data, v807_data, v826_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v828_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v632_data, v812_data, v827_acc, 1, 1, 0);
          float v829_el = v828_acc[0];
          float v831_el = v828_acc[4];
          float v832_sw = tensorforge::swap<32>(v831_el);
          float v834_el = v828_acc[8];
          float v837_el = v828_acc[12];
          float v838_sw = tensorforge::swap<32>(v837_el);
          r5[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v838_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v834_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v832_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v829_el, v829_el))))))));
          float v841_el = v828_acc[1];
          float v843_el = v828_acc[5];
          float v844_sw = tensorforge::swap<32>(v843_el);
          float v846_el = v828_acc[9];
          float v849_el = v828_acc[13];
          float v850_sw = tensorforge::swap<32>(v849_el);
          r5[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v850_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v846_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v844_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v841_el, v841_el))))))));
          float v853_el = v828_acc[2];
          float v855_el = v828_acc[6];
          float v856_sw = tensorforge::swap<32>(v855_el);
          float v858_el = v828_acc[10];
          float v861_el = v828_acc[14];
          float v862_sw = tensorforge::swap<32>(v861_el);
          r5[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v862_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v858_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v856_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v853_el, v853_el))))))));
          float v865_el = v828_acc[3];
          float v867_el = v828_acc[7];
          float v868_sw = tensorforge::swap<32>(v867_el);
          float v870_el = v828_acc[11];
          float v873_el = v828_acc[15];
          float v874_sw = tensorforge::swap<32>(v873_el);
          r5[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v874_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v870_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v868_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v865_el, v865_el))))))));
          float v878_sw = tensorforge::swap<32>(v829_el);
          float v883_sw = tensorforge::swap<32>(v834_el);
          r5[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v837_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v883_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v831_el, (tensorforge::dppUpdate<228, 1, 15, false>(v878_sw, v878_sw))))))));
          float v890_sw = tensorforge::swap<32>(v841_el);
          float v895_sw = tensorforge::swap<32>(v846_el);
          r5[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v849_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v895_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v843_el, (tensorforge::dppUpdate<228, 1, 15, false>(v890_sw, v890_sw))))))));
          float v902_sw = tensorforge::swap<32>(v853_el);
          float v907_sw = tensorforge::swap<32>(v858_el);
          r5[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v861_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v907_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v855_el, (tensorforge::dppUpdate<228, 1, 15, false>(v902_sw, v902_sw))))))));
          float v914_sw = tensorforge::swap<32>(v865_el);
          float v919_sw = tensorforge::swap<32>(v870_el);
          r5[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v873_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v919_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v867_el, (tensorforge::dppUpdate<228, 1, 15, false>(v914_sw, v914_sw))))))));
          float v926_sw = tensorforge::swap<64>(v829_el);
          r5[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v838_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v834_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v832_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v926_sw, v926_sw))))))));
          float v938_sw = tensorforge::swap<64>(v841_el);
          r5[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v850_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v846_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v844_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v938_sw, v938_sw))))))));
          float v950_sw = tensorforge::swap<64>(v853_el);
          r5[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v862_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v858_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v856_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v950_sw, v950_sw))))))));
          float v962_sw = tensorforge::swap<64>(v865_el);
          r5[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v874_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v870_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v868_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v962_sw, v962_sw))))))));
          float v975_sw = tensorforge::swap<64>(v878_sw);
          r5[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v837_el, (tensorforge::dppUpdate<228, 4, 15, false>(v883_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v831_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v975_sw, v975_sw))))))));
          float v987_sw = tensorforge::swap<64>(v890_sw);
          r5[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v849_el, (tensorforge::dppUpdate<228, 4, 15, false>(v895_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v843_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v987_sw, v987_sw))))))));
          float v999_sw = tensorforge::swap<64>(v902_sw);
          r5[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v861_el, (tensorforge::dppUpdate<228, 4, 15, false>(v907_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v855_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v999_sw, v999_sw))))))));
          float v1011_sw = tensorforge::swap<64>(v914_sw);
          r5[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v873_el, (tensorforge::dppUpdate<228, 4, 15, false>(v919_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v867_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1011_sw, v1011_sw))))))));
          // wait(r6 = load{g>r}(glb_m4););
          float r7[8]{};
          // r7 = +(r5 * r6) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float v1022_data = r6[0];
          float v1023_data = r6[1];
          float v1024_data = r6[2];
          float v1025_data = r6[3];
          float v1026_tp{};
          float v1027_tp{};
          float v1028_tp{};
          float v1029_tp{};
          tensorforge::transpose4x4b32(v1026_tp, v1027_tp, v1028_tp, v1029_tp, v1022_data, v1023_data, v1024_data, v1025_data);
          tensorforge::VectorT<float, 4> v1030_acc{};
          float v1031_data = r5[0];
          float v1032_data = r5[1];
          float v1033_data = r5[2];
          float v1034_data = r5[3];
          tensorforge::VectorT<float, 4> v1035_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1031_data, v1030_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1036_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1032_data, v1035_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1037_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1033_data, v1036_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1038_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1034_data, v1037_acc, 3, 0, 0);
          float v1039_data = r5[4];
          float v1040_data = r5[5];
          float v1041_data = r5[6];
          float v1042_data = r5[7];
          tensorforge::VectorT<float, 4> v1043_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1039_data, v1038_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1044_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1040_data, v1043_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1045_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1041_data, v1044_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1046_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1042_data, v1045_acc, 3, 1, 0);
          float v1047_data = r5[8];
          float v1048_data = r5[9];
          float v1049_data = r5[10];
          float v1050_data = r5[11];
          tensorforge::VectorT<float, 4> v1051_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1047_data, v1046_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1052_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1048_data, v1051_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1053_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1049_data, v1052_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1050_data, v1053_acc, 3, 2, 0);
          float v1055_data = r5[12];
          float v1056_data = r5[13];
          float v1057_data = r5[14];
          float v1058_data = r5[15];
          tensorforge::VectorT<float, 4> v1059_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1026_tp, v1055_data, v1054_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1060_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1027_tp, v1056_data, v1059_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1061_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1028_tp, v1057_data, v1060_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1062_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1029_tp, v1058_data, v1061_acc, 3, 3, 0);
          r7[0] = (v1062_acc[0]);
          r7[1] = (v1062_acc[1]);
          r7[2] = (v1062_acc[2]);
          r7[3] = (v1062_acc[3]);
          float v1067_data = r6[4];
          float v1068_data = r6[5];
          float v1069_data = r6[6];
          float v1070_data = r6[7];
          float v1071_tp{};
          float v1072_tp{};
          float v1073_tp{};
          float v1074_tp{};
          tensorforge::transpose4x4b32(v1071_tp, v1072_tp, v1073_tp, v1074_tp, v1067_data, v1068_data, v1069_data, v1070_data);
          tensorforge::VectorT<float, 4> v1075_acc{};
          tensorforge::VectorT<float, 4> v1080_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1071_tp, v1031_data, v1075_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1081_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1072_tp, v1032_data, v1080_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1082_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1073_tp, v1033_data, v1081_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1083_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1034_data, v1082_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1088_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1071_tp, v1039_data, v1083_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1089_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1072_tp, v1040_data, v1088_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1090_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1073_tp, v1041_data, v1089_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1091_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1042_data, v1090_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1096_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1071_tp, v1047_data, v1091_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1097_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1072_tp, v1048_data, v1096_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1098_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1073_tp, v1049_data, v1097_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1099_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1050_data, v1098_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1071_tp, v1055_data, v1099_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1072_tp, v1056_data, v1104_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1073_tp, v1057_data, v1105_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1074_tp, v1058_data, v1106_acc, 3, 3, 0);
          r7[4] = (v1107_acc[0]);
          r7[5] = (v1107_acc[1]);
          r7[6] = (v1107_acc[2]);
          r7[7] = (v1107_acc[3]);
          // glb_m3 = store{r>g}(r7);
          if (v323_g) {
            #pragma unroll
            for (int32_t v1112_i1 = 0; v1112_i1 < 8; ++v1112_i1) {
              float v1114_data = r7[v1112_i1];
              glb_m3[(v21_lead + (v1112_i1 * 16))] = v1114_data;
            }
          }
        }
      }
    }
  }
}

