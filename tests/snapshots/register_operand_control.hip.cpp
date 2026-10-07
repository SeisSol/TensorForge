// === base name ===
kernel_f5cccd7fd834c319

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f5cccd7fd834c319 = {{32, 8, 1}, 32, 32, 1, 8, 16384, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f5cccd7fd834c319(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f5cccd7fd834c319(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f5cccd7fd834c319(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f5cccd7fd834c319, block.x * block.y * block.z, 4096 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (4096 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_f5cccd7fd834c319, block.x * block.y * block.z, 0));
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
void launcher_kernel_f5cccd7fd834c319(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f5cccd7fd834c319(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_f5cccd7fd834c319), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_f5cccd7fd834c319, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f5cccd7fd834c319(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
    //   m3 16×8(16×8) {0..16}×{0..8} strided
    //   m4 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t1[i,j] = m2[i,k] × m1[k,j]
    //   t2[i,j] = t0[k,i] × t1[k,j]
    //   m3[i,j] = t2[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":4096}],"shared_bytes":16384,"shared_elements":4096,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[32,32]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[16,8]],"name":"m3","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[16,8]],"name":"m4","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,32]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,32]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,16]},{"addressing":"pointer_based","bbox":[[0,0],[32,16]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v8_batchId0 * 128 + 0 + m3_extraOffset];
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
          float r3[32]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v317_i0 = 0; v317_i0 < 1; ++v317_i0) {
            int32_t v320_lead = v24_lead + (v317_i0 * 32);
            #pragma unroll
            for (int32_t v318_i1 = 0; v318_i1 < 32; ++v318_i1) {
              float v323_data = __builtin_nontemporal_load(&glb_m2[(v320_lead + (v318_i1 * 32))]);
              r3[(v317_i0 + v318_i1)] = v323_data;
            }
          }
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
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
          float v56_data = r1[13];
          float v57_data = r1[14];
          float v58_data = r1[15];
          tensorforge::transpose16x16b32(v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 16> v59_acc{};
          float v60_data = r0[0];
          float v61_data = r0[1];
          float v62_data = r0[2];
          float v63_data = r0[3];
          float v64_data = r0[4];
          float v65_data = r0[5];
          float v66_data = r0[6];
          float v67_data = r0[7];
          float v68_data = r0[8];
          float v69_data = r0[9];
          float v70_data = r0[10];
          float v71_data = r0[11];
          float v72_data = r0[12];
          float v73_data = r0[13];
          float v74_data = r0[14];
          float v75_data = r0[15];
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v60_data, v59_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v90_acc, 1, 0, 0);
          float v92_data = r0[16];
          float v93_data = r0[17];
          float v94_data = r0[18];
          float v95_data = r0[19];
          float v96_data = r0[20];
          float v97_data = r0[21];
          float v98_data = r0[22];
          float v99_data = r0[23];
          float v100_data = r0[24];
          float v101_data = r0[25];
          float v102_data = r0[26];
          float v103_data = r0[27];
          float v104_data = r0[28];
          float v105_data = r0[29];
          float v106_data = r0[30];
          float v107_data = r0[31];
          tensorforge::VectorT<float, 16> v108_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v92_data, v91_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v109_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v93_data, v108_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v110_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v94_data, v109_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v111_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v95_data, v110_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v112_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v96_data, v111_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v113_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v97_data, v112_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v114_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v98_data, v113_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v115_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v99_data, v114_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v116_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v100_data, v115_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v101_data, v116_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v118_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v102_data, v117_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v119_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v103_data, v118_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v104_data, v119_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v121_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v105_data, v120_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v122_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v106_data, v121_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v107_data, v122_acc, 1, 1, 0);
          float v124_el = v123_acc[0];
          float v126_el = v123_acc[4];
          float v127_sw = tensorforge::swap<32>(v126_el);
          float v129_el = v123_acc[8];
          float v132_el = v123_acc[12];
          float v133_sw = tensorforge::swap<32>(v132_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v127_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v124_el, v124_el))))))));
          float v136_el = v123_acc[1];
          float v138_el = v123_acc[5];
          float v139_sw = tensorforge::swap<32>(v138_el);
          float v141_el = v123_acc[9];
          float v144_el = v123_acc[13];
          float v145_sw = tensorforge::swap<32>(v144_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v136_el, v136_el))))))));
          float v148_el = v123_acc[2];
          float v150_el = v123_acc[6];
          float v151_sw = tensorforge::swap<32>(v150_el);
          float v153_el = v123_acc[10];
          float v156_el = v123_acc[14];
          float v157_sw = tensorforge::swap<32>(v156_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v153_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v148_el, v148_el))))))));
          float v160_el = v123_acc[3];
          float v162_el = v123_acc[7];
          float v163_sw = tensorforge::swap<32>(v162_el);
          float v165_el = v123_acc[11];
          float v168_el = v123_acc[15];
          float v169_sw = tensorforge::swap<32>(v168_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v169_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v165_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v160_el, v160_el))))))));
          float v173_sw = tensorforge::swap<32>(v124_el);
          float v178_sw = tensorforge::swap<32>(v129_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v178_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v126_el, (tensorforge::dppUpdate<228, 1, 15, false>(v173_sw, v173_sw))))))));
          float v185_sw = tensorforge::swap<32>(v136_el);
          float v190_sw = tensorforge::swap<32>(v141_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v190_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v138_el, (tensorforge::dppUpdate<228, 1, 15, false>(v185_sw, v185_sw))))))));
          float v197_sw = tensorforge::swap<32>(v148_el);
          float v202_sw = tensorforge::swap<32>(v153_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v156_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v202_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v150_el, (tensorforge::dppUpdate<228, 1, 15, false>(v197_sw, v197_sw))))))));
          float v209_sw = tensorforge::swap<32>(v160_el);
          float v214_sw = tensorforge::swap<32>(v165_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v168_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v214_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v162_el, (tensorforge::dppUpdate<228, 1, 15, false>(v209_sw, v209_sw))))))));
          float v221_sw = tensorforge::swap<64>(v124_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v133_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v129_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v127_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v221_sw, v221_sw))))))));
          float v233_sw = tensorforge::swap<64>(v136_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v141_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
          float v245_sw = tensorforge::swap<64>(v148_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v153_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v245_sw, v245_sw))))))));
          float v257_sw = tensorforge::swap<64>(v160_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v169_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v165_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v257_sw, v257_sw))))))));
          float v270_sw = tensorforge::swap<64>(v173_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v132_el, (tensorforge::dppUpdate<228, 4, 15, false>(v178_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v126_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v270_sw, v270_sw))))))));
          float v282_sw = tensorforge::swap<64>(v185_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v144_el, (tensorforge::dppUpdate<228, 4, 15, false>(v190_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v282_sw, v282_sw))))))));
          float v294_sw = tensorforge::swap<64>(v197_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v156_el, (tensorforge::dppUpdate<228, 4, 15, false>(v202_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v294_sw, v294_sw))))))));
          float v306_sw = tensorforge::swap<64>(v209_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v168_el, (tensorforge::dppUpdate<228, 4, 15, false>(v214_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v306_sw, v306_sw))))))));
          float r6[8]{};
          // r6 = load{g>r}(glb_m4);
          bool v1016_g = v24_lead < 16;
          if (v1016_g) {
            #pragma unroll
            for (int32_t v1017_i1 = 0; v1017_i1 < 8; ++v1017_i1) {
              float v1022_data = __builtin_nontemporal_load(&glb_m4[(v24_lead + (v1017_i1 * 16))]);
              r6[v1017_i1] = v1022_data;
            }
          }
          float r4[16]{};
          // r4 = +(r3 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v326_data = r1[0];
          float v327_data = r1[1];
          float v328_data = r1[2];
          float v329_data = r1[3];
          float v330_data = r1[4];
          float v331_data = r1[5];
          float v332_data = r1[6];
          float v333_data = r1[7];
          float v334_data = r1[8];
          float v335_data = r1[9];
          float v336_data = r1[10];
          float v337_data = r1[11];
          float v338_data = r1[12];
          float v339_data = r1[13];
          float v340_data = r1[14];
          float v341_data = r1[15];
          tensorforge::transpose16x16b32(v326_data, v327_data, v328_data, v329_data, v330_data, v331_data, v332_data, v333_data, v334_data, v335_data, v336_data, v337_data, v338_data, v339_data, v340_data, v341_data);
          tensorforge::VectorT<float, 16> v342_acc{};
          float v343_data = r3[0];
          float v344_data = r3[1];
          float v345_data = r3[2];
          float v346_data = r3[3];
          float v347_data = r3[4];
          float v348_data = r3[5];
          float v349_data = r3[6];
          float v350_data = r3[7];
          float v351_data = r3[8];
          float v352_data = r3[9];
          float v353_data = r3[10];
          float v354_data = r3[11];
          float v355_data = r3[12];
          float v356_data = r3[13];
          float v357_data = r3[14];
          float v358_data = r3[15];
          tensorforge::VectorT<float, 16> v359_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v326_data, v343_data, v342_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v360_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v327_data, v344_data, v359_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v361_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v345_data, v360_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v362_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v346_data, v361_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v363_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v347_data, v362_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v348_data, v363_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v364_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v365_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v367_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v351_data, v366_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v352_data, v367_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v369_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v336_data, v353_data, v368_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v370_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v337_data, v354_data, v369_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v371_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v355_data, v370_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v372_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v339_data, v356_data, v371_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v373_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v340_data, v357_data, v372_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v374_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v341_data, v358_data, v373_acc, 1, 0, 0);
          float v375_data = r3[16];
          float v376_data = r3[17];
          float v377_data = r3[18];
          float v378_data = r3[19];
          float v379_data = r3[20];
          float v380_data = r3[21];
          float v381_data = r3[22];
          float v382_data = r3[23];
          float v383_data = r3[24];
          float v384_data = r3[25];
          float v385_data = r3[26];
          float v386_data = r3[27];
          float v387_data = r3[28];
          float v388_data = r3[29];
          float v389_data = r3[30];
          float v390_data = r3[31];
          tensorforge::VectorT<float, 16> v391_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v326_data, v375_data, v374_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v392_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v327_data, v376_data, v391_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v393_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v377_data, v392_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v394_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v378_data, v393_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v395_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v379_data, v394_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v396_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v380_data, v395_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v397_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v381_data, v396_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v398_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v382_data, v397_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v399_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v383_data, v398_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v400_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v384_data, v399_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v401_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v336_data, v385_data, v400_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v402_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v337_data, v386_data, v401_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v403_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v387_data, v402_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v404_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v339_data, v388_data, v403_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v405_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v340_data, v389_data, v404_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v406_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v341_data, v390_data, v405_acc, 1, 1, 0);
          float v407_el = v406_acc[0];
          float v409_el = v406_acc[4];
          float v410_sw = tensorforge::swap<32>(v409_el);
          float v412_el = v406_acc[8];
          float v415_el = v406_acc[12];
          float v416_sw = tensorforge::swap<32>(v415_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v416_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v412_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v410_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v407_el, v407_el))))))));
          float v419_el = v406_acc[1];
          float v421_el = v406_acc[5];
          float v422_sw = tensorforge::swap<32>(v421_el);
          float v424_el = v406_acc[9];
          float v427_el = v406_acc[13];
          float v428_sw = tensorforge::swap<32>(v427_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v428_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v424_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v422_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v419_el, v419_el))))))));
          float v431_el = v406_acc[2];
          float v433_el = v406_acc[6];
          float v434_sw = tensorforge::swap<32>(v433_el);
          float v436_el = v406_acc[10];
          float v439_el = v406_acc[14];
          float v440_sw = tensorforge::swap<32>(v439_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v440_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v436_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v434_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v431_el, v431_el))))))));
          float v443_el = v406_acc[3];
          float v445_el = v406_acc[7];
          float v446_sw = tensorforge::swap<32>(v445_el);
          float v448_el = v406_acc[11];
          float v451_el = v406_acc[15];
          float v452_sw = tensorforge::swap<32>(v451_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v452_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v448_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v446_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v443_el, v443_el))))))));
          float v456_sw = tensorforge::swap<32>(v407_el);
          float v461_sw = tensorforge::swap<32>(v412_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v415_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v461_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v409_el, (tensorforge::dppUpdate<228, 1, 15, false>(v456_sw, v456_sw))))))));
          float v468_sw = tensorforge::swap<32>(v419_el);
          float v473_sw = tensorforge::swap<32>(v424_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v427_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v473_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v421_el, (tensorforge::dppUpdate<228, 1, 15, false>(v468_sw, v468_sw))))))));
          float v480_sw = tensorforge::swap<32>(v431_el);
          float v485_sw = tensorforge::swap<32>(v436_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v439_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v485_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v433_el, (tensorforge::dppUpdate<228, 1, 15, false>(v480_sw, v480_sw))))))));
          float v492_sw = tensorforge::swap<32>(v443_el);
          float v497_sw = tensorforge::swap<32>(v448_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v451_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v497_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v445_el, (tensorforge::dppUpdate<228, 1, 15, false>(v492_sw, v492_sw))))))));
          float v504_sw = tensorforge::swap<64>(v407_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v416_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v412_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v410_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v504_sw, v504_sw))))))));
          float v516_sw = tensorforge::swap<64>(v419_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v428_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v424_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v422_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v516_sw, v516_sw))))))));
          float v528_sw = tensorforge::swap<64>(v431_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v440_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v436_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v434_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v528_sw, v528_sw))))))));
          float v540_sw = tensorforge::swap<64>(v443_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v452_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v448_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v446_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v540_sw, v540_sw))))))));
          float v553_sw = tensorforge::swap<64>(v456_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v415_el, (tensorforge::dppUpdate<228, 4, 15, false>(v461_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v409_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v553_sw, v553_sw))))))));
          float v565_sw = tensorforge::swap<64>(v468_sw);
          r4[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v427_el, (tensorforge::dppUpdate<228, 4, 15, false>(v473_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v421_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v565_sw, v565_sw))))))));
          float v577_sw = tensorforge::swap<64>(v480_sw);
          r4[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v439_el, (tensorforge::dppUpdate<228, 4, 15, false>(v485_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v433_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v577_sw, v577_sw))))))));
          float v589_sw = tensorforge::swap<64>(v492_sw);
          r4[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v451_el, (tensorforge::dppUpdate<228, 4, 15, false>(v497_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v445_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v589_sw, v589_sw))))))));
          // s0 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v599_i0 = 0; v599_i0 < 1; ++v599_i0) {
            int32_t v604_lead = v24_lead + (v599_i0 * 32);
            #pragma unroll
            for (int32_t v600_i1 = 0; v600_i1 < 16; ++v600_i1) {
              float v602_data = r2[(v599_i0 + v600_i1)];
              int32_t v606_a = v604_lead + (v600_i1 * 32);
              s0[(v606_a ^ ((v606_a >> 5) & 31))] = v602_data;
            }
          }
          float r5[16]{};
          // r5 = +(s0 * r4) + None
          // [(0, 16), (0, 16)] [(0, 32)]
          float v611_data = r4[0];
          float v612_data = r4[1];
          float v613_data = r4[2];
          float v614_data = r4[3];
          float v615_data = r4[4];
          float v616_data = r4[5];
          float v617_data = r4[6];
          float v618_data = r4[7];
          float v619_data = r4[8];
          float v620_data = r4[9];
          float v621_data = r4[10];
          float v622_data = r4[11];
          float v623_data = r4[12];
          float v624_data = r4[13];
          float v625_data = r4[14];
          float v626_data = r4[15];
          tensorforge::transpose16x16b32(v611_data, v612_data, v613_data, v614_data, v615_data, v616_data, v617_data, v618_data, v619_data, v620_data, v621_data, v622_data, v623_data, v624_data, v625_data, v626_data);
          tensorforge::VectorT<float, 16> v627_acc{};
          int32_t v630_a = v24_lead * 32;
          int32_t v634_sw = v630_a ^ ((v630_a >> 5) & 31);
          float v635_data = s0[v634_sw];
          int32_t v636_a = 1 + v630_a;
          float v640_data = s0[(v636_a ^ ((v636_a >> 5) & 31))];
          int32_t v641_a = 2 + v630_a;
          float v645_data = s0[(v641_a ^ ((v641_a >> 5) & 31))];
          int32_t v646_a = 3 + v630_a;
          float v650_data = s0[(v646_a ^ ((v646_a >> 5) & 31))];
          int32_t v651_a = 4 + v630_a;
          float v655_data = s0[(v651_a ^ ((v651_a >> 5) & 31))];
          int32_t v656_a = 5 + v630_a;
          float v660_data = s0[(v656_a ^ ((v656_a >> 5) & 31))];
          int32_t v661_a = 6 + v630_a;
          float v665_data = s0[(v661_a ^ ((v661_a >> 5) & 31))];
          int32_t v666_a = 7 + v630_a;
          float v670_data = s0[(v666_a ^ ((v666_a >> 5) & 31))];
          int32_t v671_a = 8 + v630_a;
          float v675_data = s0[(v671_a ^ ((v671_a >> 5) & 31))];
          int32_t v676_a = 9 + v630_a;
          float v680_data = s0[(v676_a ^ ((v676_a >> 5) & 31))];
          int32_t v681_a = 10 + v630_a;
          float v685_data = s0[(v681_a ^ ((v681_a >> 5) & 31))];
          int32_t v686_a = 11 + v630_a;
          float v690_data = s0[(v686_a ^ ((v686_a >> 5) & 31))];
          int32_t v691_a = 12 + v630_a;
          float v695_data = s0[(v691_a ^ ((v691_a >> 5) & 31))];
          int32_t v696_a = 13 + v630_a;
          float v700_data = s0[(v696_a ^ ((v696_a >> 5) & 31))];
          int32_t v701_a = 14 + v630_a;
          float v705_data = s0[(v701_a ^ ((v701_a >> 5) & 31))];
          int32_t v706_a = 15 + v630_a;
          float v710_data = s0[(v706_a ^ ((v706_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v711_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v611_data, v635_data, v627_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v712_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v612_data, v640_data, v711_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v713_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v613_data, v645_data, v712_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v714_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v614_data, v650_data, v713_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v715_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v615_data, v655_data, v714_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v716_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v616_data, v660_data, v715_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v717_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v617_data, v665_data, v716_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v718_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v618_data, v670_data, v717_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v719_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v619_data, v675_data, v718_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v720_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v620_data, v680_data, v719_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v721_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v621_data, v685_data, v720_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v722_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v622_data, v690_data, v721_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v723_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v695_data, v722_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v724_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v700_data, v723_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v725_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v705_data, v724_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v726_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v710_data, v725_acc, 1, 0, 0);
          int32_t v727_a = 16 + v630_a;
          float v731_data = s0[(v727_a ^ ((v727_a >> 5) & 31))];
          int32_t v732_a = 17 + v630_a;
          float v736_data = s0[(v732_a ^ ((v732_a >> 5) & 31))];
          int32_t v737_a = 18 + v630_a;
          float v741_data = s0[(v737_a ^ ((v737_a >> 5) & 31))];
          int32_t v742_a = 19 + v630_a;
          float v746_data = s0[(v742_a ^ ((v742_a >> 5) & 31))];
          int32_t v747_a = 20 + v630_a;
          float v751_data = s0[(v747_a ^ ((v747_a >> 5) & 31))];
          int32_t v752_a = 21 + v630_a;
          float v756_data = s0[(v752_a ^ ((v752_a >> 5) & 31))];
          int32_t v757_a = 22 + v630_a;
          float v761_data = s0[(v757_a ^ ((v757_a >> 5) & 31))];
          int32_t v762_a = 23 + v630_a;
          float v766_data = s0[(v762_a ^ ((v762_a >> 5) & 31))];
          int32_t v767_a = 24 + v630_a;
          float v771_data = s0[(v767_a ^ ((v767_a >> 5) & 31))];
          int32_t v772_a = 25 + v630_a;
          float v776_data = s0[(v772_a ^ ((v772_a >> 5) & 31))];
          int32_t v777_a = 26 + v630_a;
          float v781_data = s0[(v777_a ^ ((v777_a >> 5) & 31))];
          int32_t v782_a = 27 + v630_a;
          float v786_data = s0[(v782_a ^ ((v782_a >> 5) & 31))];
          int32_t v787_a = 28 + v630_a;
          float v791_data = s0[(v787_a ^ ((v787_a >> 5) & 31))];
          int32_t v792_a = 29 + v630_a;
          float v796_data = s0[(v792_a ^ ((v792_a >> 5) & 31))];
          int32_t v797_a = 30 + v630_a;
          float v801_data = s0[(v797_a ^ ((v797_a >> 5) & 31))];
          int32_t v802_a = 31 + v630_a;
          float v806_data = s0[(v802_a ^ ((v802_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v807_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v611_data, v731_data, v726_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v808_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v612_data, v736_data, v807_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v809_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v613_data, v741_data, v808_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v810_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v614_data, v746_data, v809_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v811_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v615_data, v751_data, v810_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v812_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v616_data, v756_data, v811_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v813_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v617_data, v761_data, v812_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v814_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v618_data, v766_data, v813_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v815_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v619_data, v771_data, v814_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v816_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v620_data, v776_data, v815_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v817_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v621_data, v781_data, v816_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v818_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v622_data, v786_data, v817_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v819_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v791_data, v818_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v820_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v796_data, v819_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v821_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v801_data, v820_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v822_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v806_data, v821_acc, 1, 1, 0);
          float v823_el = v822_acc[0];
          float v825_el = v822_acc[4];
          float v826_sw = tensorforge::swap<32>(v825_el);
          float v828_el = v822_acc[8];
          float v831_el = v822_acc[12];
          float v832_sw = tensorforge::swap<32>(v831_el);
          r5[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v832_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v828_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v826_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v823_el, v823_el))))))));
          float v835_el = v822_acc[1];
          float v837_el = v822_acc[5];
          float v838_sw = tensorforge::swap<32>(v837_el);
          float v840_el = v822_acc[9];
          float v843_el = v822_acc[13];
          float v844_sw = tensorforge::swap<32>(v843_el);
          r5[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v844_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v840_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v838_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v835_el, v835_el))))))));
          float v847_el = v822_acc[2];
          float v849_el = v822_acc[6];
          float v850_sw = tensorforge::swap<32>(v849_el);
          float v852_el = v822_acc[10];
          float v855_el = v822_acc[14];
          float v856_sw = tensorforge::swap<32>(v855_el);
          r5[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v856_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v852_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v850_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v847_el, v847_el))))))));
          float v859_el = v822_acc[3];
          float v861_el = v822_acc[7];
          float v862_sw = tensorforge::swap<32>(v861_el);
          float v864_el = v822_acc[11];
          float v867_el = v822_acc[15];
          float v868_sw = tensorforge::swap<32>(v867_el);
          r5[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v868_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v864_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v862_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v859_el, v859_el))))))));
          float v872_sw = tensorforge::swap<32>(v823_el);
          float v877_sw = tensorforge::swap<32>(v828_el);
          r5[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v831_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v877_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v825_el, (tensorforge::dppUpdate<228, 1, 15, false>(v872_sw, v872_sw))))))));
          float v884_sw = tensorforge::swap<32>(v835_el);
          float v889_sw = tensorforge::swap<32>(v840_el);
          r5[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v843_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v889_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v837_el, (tensorforge::dppUpdate<228, 1, 15, false>(v884_sw, v884_sw))))))));
          float v896_sw = tensorforge::swap<32>(v847_el);
          float v901_sw = tensorforge::swap<32>(v852_el);
          r5[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v855_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v901_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v849_el, (tensorforge::dppUpdate<228, 1, 15, false>(v896_sw, v896_sw))))))));
          float v908_sw = tensorforge::swap<32>(v859_el);
          float v913_sw = tensorforge::swap<32>(v864_el);
          r5[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v867_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v913_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v861_el, (tensorforge::dppUpdate<228, 1, 15, false>(v908_sw, v908_sw))))))));
          float v920_sw = tensorforge::swap<64>(v823_el);
          r5[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v832_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v828_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v826_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v920_sw, v920_sw))))))));
          float v932_sw = tensorforge::swap<64>(v835_el);
          r5[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v844_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v840_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v838_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v932_sw, v932_sw))))))));
          float v944_sw = tensorforge::swap<64>(v847_el);
          r5[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v856_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v852_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v850_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v944_sw, v944_sw))))))));
          float v956_sw = tensorforge::swap<64>(v859_el);
          r5[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v868_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v864_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v862_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v956_sw, v956_sw))))))));
          float v969_sw = tensorforge::swap<64>(v872_sw);
          r5[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v831_el, (tensorforge::dppUpdate<228, 4, 15, false>(v877_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v825_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v969_sw, v969_sw))))))));
          float v981_sw = tensorforge::swap<64>(v884_sw);
          r5[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v843_el, (tensorforge::dppUpdate<228, 4, 15, false>(v889_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v837_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v981_sw, v981_sw))))))));
          float v993_sw = tensorforge::swap<64>(v896_sw);
          r5[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v855_el, (tensorforge::dppUpdate<228, 4, 15, false>(v901_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v849_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v993_sw, v993_sw))))))));
          float v1005_sw = tensorforge::swap<64>(v908_sw);
          r5[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v867_el, (tensorforge::dppUpdate<228, 4, 15, false>(v913_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v861_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1005_sw, v1005_sw))))))));
          float r7[8]{};
          // r7 = +(r5 * r6) + None
          // [(0, 16), (0, 8)] [(0, 16)]
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
          if (v1016_g) {
            #pragma unroll
            for (int32_t v1115_i1 = 0; v1115_i1 < 8; ++v1115_i1) {
              float v1117_data = r7[v1115_i1];
              glb_m3[(v24_lead + (v1115_i1 * 16))] = v1117_data;
            }
          }
        }
      }
    }
  }
}

