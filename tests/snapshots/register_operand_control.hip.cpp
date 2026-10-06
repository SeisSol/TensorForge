// === base name ===
kernel_4625c9d17dac8a77

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4625c9d17dac8a77 = {{32, 8, 1}, 32, 32, 1, 8, 16384, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4625c9d17dac8a77(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4625c9d17dac8a77(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4625c9d17dac8a77(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4625c9d17dac8a77, block.x * block.y * block.z, 4096 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (4096 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4625c9d17dac8a77, block.x * block.y * block.z, 0));
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
void launcher_kernel_4625c9d17dac8a77(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4625c9d17dac8a77(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4625c9d17dac8a77), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
  hipLaunchKernelGGL(kernel_kernel_4625c9d17dac8a77, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4625c9d17dac8a77(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float* tempShrMem = &localShrMem0[512];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 1024 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v11_batchId0 * 512 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 1024 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v11_batchId0 * 128 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v11_batchId0 * 128 + 0 + m4_extraOffset];
          float r0[32]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v27_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v31_lead = v27_lead + (v28_i0 * 32);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 32; ++v29_i1) {
              float v34_data = __builtin_nontemporal_load(&glb_m0[(v31_lead + (v29_i1 * 32))]);
              r0[(v28_i0 + v29_i1)] = v34_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v37_i0 = 0; v37_i0 < 1; ++v37_i0) {
            int32_t v40_lead = v27_lead + (v37_i0 * 32);
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 16; ++v38_i1) {
              float v43_data = __builtin_nontemporal_load(&glb_m1[(v40_lead + (v38_i1 * 32))]);
              r1[(v37_i0 + v38_i1)] = v43_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[32]{};
          // r3 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v46_i0 = 0; v46_i0 < 1; ++v46_i0) {
            int32_t v49_lead = v27_lead + (v46_i0 * 32);
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 32; ++v47_i1) {
              float v52_data = __builtin_nontemporal_load(&glb_m2[(v49_lead + (v47_i1 * 32))]);
              r3[(v46_i0 + v47_i1)] = v52_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v55_data = r1[0];
          float v56_data = r1[1];
          float v57_data = r1[2];
          float v58_data = r1[3];
          float v59_data = r1[4];
          float v60_data = r1[5];
          float v61_data = r1[6];
          float v62_data = r1[7];
          float v63_data = r1[8];
          float v64_data = r1[9];
          float v65_data = r1[10];
          float v66_data = r1[11];
          float v67_data = r1[12];
          float v68_data = r1[13];
          float v69_data = r1[14];
          float v70_data = r1[15];
          tensorforge::transpose16x16b32(v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data, v67_data, v68_data, v69_data, v70_data);
          tensorforge::VectorT<float, 16> v71_acc{};
          float v72_data = r0[0];
          float v73_data = r0[1];
          float v74_data = r0[2];
          float v75_data = r0[3];
          float v76_data = r0[4];
          float v77_data = r0[5];
          float v78_data = r0[6];
          float v79_data = r0[7];
          float v80_data = r0[8];
          float v81_data = r0[9];
          float v82_data = r0[10];
          float v83_data = r0[11];
          float v84_data = r0[12];
          float v85_data = r0[13];
          float v86_data = r0[14];
          float v87_data = r0[15];
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v71_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v101_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v68_data, v85_data, v100_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v102_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v69_data, v86_data, v101_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v103_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v70_data, v87_data, v102_acc, 1, 0, 0);
          float v104_data = r0[16];
          float v105_data = r0[17];
          float v106_data = r0[18];
          float v107_data = r0[19];
          float v108_data = r0[20];
          float v109_data = r0[21];
          float v110_data = r0[22];
          float v111_data = r0[23];
          float v112_data = r0[24];
          float v113_data = r0[25];
          float v114_data = r0[26];
          float v115_data = r0[27];
          float v116_data = r0[28];
          float v117_data = r0[29];
          float v118_data = r0[30];
          float v119_data = r0[31];
          tensorforge::VectorT<float, 16> v120_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v104_data, v103_acc, 1, 1, 0);
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
          tensorforge::VectorT<float, 16> v133_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v68_data, v117_data, v132_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v134_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v69_data, v118_data, v133_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v135_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v70_data, v119_data, v134_acc, 1, 1, 0);
          float v136_el = v135_acc[0];
          float v138_el = v135_acc[4];
          float v139_sw = tensorforge::swap<32>(v138_el);
          float v141_el = v135_acc[8];
          float v144_el = v135_acc[12];
          float v145_sw = tensorforge::swap<32>(v144_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v139_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v136_el, v136_el))))))));
          float v148_el = v135_acc[1];
          float v150_el = v135_acc[5];
          float v151_sw = tensorforge::swap<32>(v150_el);
          float v153_el = v135_acc[9];
          float v156_el = v135_acc[13];
          float v157_sw = tensorforge::swap<32>(v156_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v153_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v151_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v148_el, v148_el))))))));
          float v160_el = v135_acc[2];
          float v162_el = v135_acc[6];
          float v163_sw = tensorforge::swap<32>(v162_el);
          float v165_el = v135_acc[10];
          float v168_el = v135_acc[14];
          float v169_sw = tensorforge::swap<32>(v168_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v169_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v165_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v163_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v160_el, v160_el))))))));
          float v172_el = v135_acc[3];
          float v174_el = v135_acc[7];
          float v175_sw = tensorforge::swap<32>(v174_el);
          float v177_el = v135_acc[11];
          float v180_el = v135_acc[15];
          float v181_sw = tensorforge::swap<32>(v180_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v181_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v177_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v175_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v172_el, v172_el))))))));
          float v185_sw = tensorforge::swap<32>(v136_el);
          float v190_sw = tensorforge::swap<32>(v141_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v190_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v138_el, (tensorforge::dppUpdate<228, 1, 15, false>(v185_sw, v185_sw))))))));
          float v197_sw = tensorforge::swap<32>(v148_el);
          float v202_sw = tensorforge::swap<32>(v153_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v156_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v202_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v150_el, (tensorforge::dppUpdate<228, 1, 15, false>(v197_sw, v197_sw))))))));
          float v209_sw = tensorforge::swap<32>(v160_el);
          float v214_sw = tensorforge::swap<32>(v165_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v168_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v214_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v162_el, (tensorforge::dppUpdate<228, 1, 15, false>(v209_sw, v209_sw))))))));
          float v221_sw = tensorforge::swap<32>(v172_el);
          float v226_sw = tensorforge::swap<32>(v177_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v180_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v226_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v174_el, (tensorforge::dppUpdate<228, 1, 15, false>(v221_sw, v221_sw))))))));
          float v233_sw = tensorforge::swap<64>(v136_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v141_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v139_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v233_sw, v233_sw))))))));
          float v245_sw = tensorforge::swap<64>(v148_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v153_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v151_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v245_sw, v245_sw))))))));
          float v257_sw = tensorforge::swap<64>(v160_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v169_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v165_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v163_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v257_sw, v257_sw))))))));
          float v269_sw = tensorforge::swap<64>(v172_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v181_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v177_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v175_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v269_sw, v269_sw))))))));
          float v282_sw = tensorforge::swap<64>(v185_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v144_el, (tensorforge::dppUpdate<228, 4, 15, false>(v190_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v138_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v282_sw, v282_sw))))))));
          float v294_sw = tensorforge::swap<64>(v197_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v156_el, (tensorforge::dppUpdate<228, 4, 15, false>(v202_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v150_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v294_sw, v294_sw))))))));
          float v306_sw = tensorforge::swap<64>(v209_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v168_el, (tensorforge::dppUpdate<228, 4, 15, false>(v214_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v162_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v306_sw, v306_sw))))))));
          float v318_sw = tensorforge::swap<64>(v221_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v180_el, (tensorforge::dppUpdate<228, 4, 15, false>(v226_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v174_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v318_sw, v318_sw))))))));
          float r6[8]{};
          // r6 = load{g>r}(glb_m4);
          bool v329_g = v27_lead < 16;
          if (v329_g) {
            #pragma unroll
            for (int32_t v330_i1 = 0; v330_i1 < 8; ++v330_i1) {
              float v335_data = __builtin_nontemporal_load(&glb_m4[(v27_lead + (v330_i1 * 16))]);
              r6[v330_i1] = v335_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[16]{};
          // r4 = +(r3 * r1) + None
          // [(0, 32), (0, 16)] [(0, 32)]
          float v338_data = r1[0];
          float v339_data = r1[1];
          float v340_data = r1[2];
          float v341_data = r1[3];
          float v342_data = r1[4];
          float v343_data = r1[5];
          float v344_data = r1[6];
          float v345_data = r1[7];
          float v346_data = r1[8];
          float v347_data = r1[9];
          float v348_data = r1[10];
          float v349_data = r1[11];
          float v350_data = r1[12];
          float v351_data = r1[13];
          float v352_data = r1[14];
          float v353_data = r1[15];
          tensorforge::transpose16x16b32(v338_data, v339_data, v340_data, v341_data, v342_data, v343_data, v344_data, v345_data, v346_data, v347_data, v348_data, v349_data, v350_data, v351_data, v352_data, v353_data);
          tensorforge::VectorT<float, 16> v354_acc{};
          float v355_data = r3[0];
          float v356_data = r3[1];
          float v357_data = r3[2];
          float v358_data = r3[3];
          float v359_data = r3[4];
          float v360_data = r3[5];
          float v361_data = r3[6];
          float v362_data = r3[7];
          float v363_data = r3[8];
          float v364_data = r3[9];
          float v365_data = r3[10];
          float v366_data = r3[11];
          float v367_data = r3[12];
          float v368_data = r3[13];
          float v369_data = r3[14];
          float v370_data = r3[15];
          tensorforge::VectorT<float, 16> v371_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v355_data, v354_acc, 1, 0, 0);
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
          tensorforge::VectorT<float, 16> v384_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v351_data, v368_data, v383_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v385_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v352_data, v369_data, v384_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v386_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v353_data, v370_data, v385_acc, 1, 0, 0);
          float v387_data = r3[16];
          float v388_data = r3[17];
          float v389_data = r3[18];
          float v390_data = r3[19];
          float v391_data = r3[20];
          float v392_data = r3[21];
          float v393_data = r3[22];
          float v394_data = r3[23];
          float v395_data = r3[24];
          float v396_data = r3[25];
          float v397_data = r3[26];
          float v398_data = r3[27];
          float v399_data = r3[28];
          float v400_data = r3[29];
          float v401_data = r3[30];
          float v402_data = r3[31];
          tensorforge::VectorT<float, 16> v403_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v387_data, v386_acc, 1, 1, 0);
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
          tensorforge::VectorT<float, 16> v416_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v351_data, v400_data, v415_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v417_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v352_data, v401_data, v416_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v418_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v353_data, v402_data, v417_acc, 1, 1, 0);
          float v419_el = v418_acc[0];
          float v421_el = v418_acc[4];
          float v422_sw = tensorforge::swap<32>(v421_el);
          float v424_el = v418_acc[8];
          float v427_el = v418_acc[12];
          float v428_sw = tensorforge::swap<32>(v427_el);
          r4[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v428_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v424_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v422_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v419_el, v419_el))))))));
          float v431_el = v418_acc[1];
          float v433_el = v418_acc[5];
          float v434_sw = tensorforge::swap<32>(v433_el);
          float v436_el = v418_acc[9];
          float v439_el = v418_acc[13];
          float v440_sw = tensorforge::swap<32>(v439_el);
          r4[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v440_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v436_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v434_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v431_el, v431_el))))))));
          float v443_el = v418_acc[2];
          float v445_el = v418_acc[6];
          float v446_sw = tensorforge::swap<32>(v445_el);
          float v448_el = v418_acc[10];
          float v451_el = v418_acc[14];
          float v452_sw = tensorforge::swap<32>(v451_el);
          r4[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v452_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v448_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v446_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v443_el, v443_el))))))));
          float v455_el = v418_acc[3];
          float v457_el = v418_acc[7];
          float v458_sw = tensorforge::swap<32>(v457_el);
          float v460_el = v418_acc[11];
          float v463_el = v418_acc[15];
          float v464_sw = tensorforge::swap<32>(v463_el);
          r4[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v464_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v460_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v458_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v455_el, v455_el))))))));
          float v468_sw = tensorforge::swap<32>(v419_el);
          float v473_sw = tensorforge::swap<32>(v424_el);
          r4[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v427_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v473_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v421_el, (tensorforge::dppUpdate<228, 1, 15, false>(v468_sw, v468_sw))))))));
          float v480_sw = tensorforge::swap<32>(v431_el);
          float v485_sw = tensorforge::swap<32>(v436_el);
          r4[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v439_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v485_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v433_el, (tensorforge::dppUpdate<228, 1, 15, false>(v480_sw, v480_sw))))))));
          float v492_sw = tensorforge::swap<32>(v443_el);
          float v497_sw = tensorforge::swap<32>(v448_el);
          r4[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v451_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v497_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v445_el, (tensorforge::dppUpdate<228, 1, 15, false>(v492_sw, v492_sw))))))));
          float v504_sw = tensorforge::swap<32>(v455_el);
          float v509_sw = tensorforge::swap<32>(v460_el);
          r4[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v463_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v509_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v457_el, (tensorforge::dppUpdate<228, 1, 15, false>(v504_sw, v504_sw))))))));
          float v516_sw = tensorforge::swap<64>(v419_el);
          r4[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v428_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v424_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v422_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v516_sw, v516_sw))))))));
          float v528_sw = tensorforge::swap<64>(v431_el);
          r4[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v440_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v436_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v434_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v528_sw, v528_sw))))))));
          float v540_sw = tensorforge::swap<64>(v443_el);
          r4[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v452_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v448_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v446_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v540_sw, v540_sw))))))));
          float v552_sw = tensorforge::swap<64>(v455_el);
          r4[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v464_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v460_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v458_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v552_sw, v552_sw))))))));
          float v565_sw = tensorforge::swap<64>(v468_sw);
          r4[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v427_el, (tensorforge::dppUpdate<228, 4, 15, false>(v473_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v421_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v565_sw, v565_sw))))))));
          float v577_sw = tensorforge::swap<64>(v480_sw);
          r4[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v439_el, (tensorforge::dppUpdate<228, 4, 15, false>(v485_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v433_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v577_sw, v577_sw))))))));
          float v589_sw = tensorforge::swap<64>(v492_sw);
          r4[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v451_el, (tensorforge::dppUpdate<228, 4, 15, false>(v497_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v445_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v589_sw, v589_sw))))))));
          float v601_sw = tensorforge::swap<64>(v504_sw);
          r4[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v463_el, (tensorforge::dppUpdate<228, 4, 15, false>(v509_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v457_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v601_sw, v601_sw))))))));
          // s0 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v611_i0 = 0; v611_i0 < 1; ++v611_i0) {
            int32_t v616_lead = v27_lead + (v611_i0 * 32);
            #pragma unroll
            for (int32_t v612_i1 = 0; v612_i1 < 16; ++v612_i1) {
              float v614_data = r2[(v611_i0 + v612_i1)];
              int32_t v618_a = v616_lead + (v612_i1 * 32);
              s0[(v618_a ^ ((v618_a >> 5) & 31))] = v614_data;
            }
          }
          float r5[16]{};
          // r5 = +(s0 * r4) + None
          // [(0, 16), (0, 16)] [(0, 32)]
          float v623_data = r4[0];
          float v624_data = r4[1];
          float v625_data = r4[2];
          float v626_data = r4[3];
          float v627_data = r4[4];
          float v628_data = r4[5];
          float v629_data = r4[6];
          float v630_data = r4[7];
          float v631_data = r4[8];
          float v632_data = r4[9];
          float v633_data = r4[10];
          float v634_data = r4[11];
          float v635_data = r4[12];
          float v636_data = r4[13];
          float v637_data = r4[14];
          float v638_data = r4[15];
          tensorforge::transpose16x16b32(v623_data, v624_data, v625_data, v626_data, v627_data, v628_data, v629_data, v630_data, v631_data, v632_data, v633_data, v634_data, v635_data, v636_data, v637_data, v638_data);
          tensorforge::VectorT<float, 16> v639_acc{};
          int32_t v642_a = v27_lead * 32;
          float v647_data = s0[(v642_a ^ ((v642_a >> 5) & 31))];
          int32_t v648_a = 1 + v642_a;
          float v652_data = s0[(v648_a ^ ((v648_a >> 5) & 31))];
          int32_t v653_a = 2 + v642_a;
          float v657_data = s0[(v653_a ^ ((v653_a >> 5) & 31))];
          int32_t v658_a = 3 + v642_a;
          float v662_data = s0[(v658_a ^ ((v658_a >> 5) & 31))];
          int32_t v663_a = 4 + v642_a;
          float v667_data = s0[(v663_a ^ ((v663_a >> 5) & 31))];
          int32_t v668_a = 5 + v642_a;
          float v672_data = s0[(v668_a ^ ((v668_a >> 5) & 31))];
          int32_t v673_a = 6 + v642_a;
          float v677_data = s0[(v673_a ^ ((v673_a >> 5) & 31))];
          int32_t v678_a = 7 + v642_a;
          float v682_data = s0[(v678_a ^ ((v678_a >> 5) & 31))];
          int32_t v683_a = 8 + v642_a;
          float v687_data = s0[(v683_a ^ ((v683_a >> 5) & 31))];
          int32_t v688_a = 9 + v642_a;
          float v692_data = s0[(v688_a ^ ((v688_a >> 5) & 31))];
          int32_t v693_a = 10 + v642_a;
          float v697_data = s0[(v693_a ^ ((v693_a >> 5) & 31))];
          int32_t v698_a = 11 + v642_a;
          float v702_data = s0[(v698_a ^ ((v698_a >> 5) & 31))];
          int32_t v703_a = 12 + v642_a;
          float v707_data = s0[(v703_a ^ ((v703_a >> 5) & 31))];
          int32_t v708_a = 13 + v642_a;
          float v712_data = s0[(v708_a ^ ((v708_a >> 5) & 31))];
          int32_t v713_a = 14 + v642_a;
          float v717_data = s0[(v713_a ^ ((v713_a >> 5) & 31))];
          int32_t v718_a = 15 + v642_a;
          float v722_data = s0[(v718_a ^ ((v718_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v723_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v647_data, v639_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v724_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v652_data, v723_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v725_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v657_data, v724_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v726_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v662_data, v725_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v727_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v627_data, v667_data, v726_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v728_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v628_data, v672_data, v727_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v729_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v629_data, v677_data, v728_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v730_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v630_data, v682_data, v729_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v731_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v631_data, v687_data, v730_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v732_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v632_data, v692_data, v731_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v733_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v633_data, v697_data, v732_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v734_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v634_data, v702_data, v733_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v735_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v635_data, v707_data, v734_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v736_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v636_data, v712_data, v735_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v737_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v637_data, v717_data, v736_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v738_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v638_data, v722_data, v737_acc, 1, 0, 0);
          int32_t v739_a = 16 + v642_a;
          float v743_data = s0[(v739_a ^ ((v739_a >> 5) & 31))];
          int32_t v744_a = 17 + v642_a;
          float v748_data = s0[(v744_a ^ ((v744_a >> 5) & 31))];
          int32_t v749_a = 18 + v642_a;
          float v753_data = s0[(v749_a ^ ((v749_a >> 5) & 31))];
          int32_t v754_a = 19 + v642_a;
          float v758_data = s0[(v754_a ^ ((v754_a >> 5) & 31))];
          int32_t v759_a = 20 + v642_a;
          float v763_data = s0[(v759_a ^ ((v759_a >> 5) & 31))];
          int32_t v764_a = 21 + v642_a;
          float v768_data = s0[(v764_a ^ ((v764_a >> 5) & 31))];
          int32_t v769_a = 22 + v642_a;
          float v773_data = s0[(v769_a ^ ((v769_a >> 5) & 31))];
          int32_t v774_a = 23 + v642_a;
          float v778_data = s0[(v774_a ^ ((v774_a >> 5) & 31))];
          int32_t v779_a = 24 + v642_a;
          float v783_data = s0[(v779_a ^ ((v779_a >> 5) & 31))];
          int32_t v784_a = 25 + v642_a;
          float v788_data = s0[(v784_a ^ ((v784_a >> 5) & 31))];
          int32_t v789_a = 26 + v642_a;
          float v793_data = s0[(v789_a ^ ((v789_a >> 5) & 31))];
          int32_t v794_a = 27 + v642_a;
          float v798_data = s0[(v794_a ^ ((v794_a >> 5) & 31))];
          int32_t v799_a = 28 + v642_a;
          float v803_data = s0[(v799_a ^ ((v799_a >> 5) & 31))];
          int32_t v804_a = 29 + v642_a;
          float v808_data = s0[(v804_a ^ ((v804_a >> 5) & 31))];
          int32_t v809_a = 30 + v642_a;
          float v813_data = s0[(v809_a ^ ((v809_a >> 5) & 31))];
          int32_t v814_a = 31 + v642_a;
          float v818_data = s0[(v814_a ^ ((v814_a >> 5) & 31))];
          tensorforge::VectorT<float, 16> v819_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v623_data, v743_data, v738_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v820_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v624_data, v748_data, v819_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v821_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v625_data, v753_data, v820_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v822_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v626_data, v758_data, v821_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v823_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v627_data, v763_data, v822_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v824_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v628_data, v768_data, v823_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v825_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v629_data, v773_data, v824_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v826_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v630_data, v778_data, v825_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v827_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v631_data, v783_data, v826_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v828_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v632_data, v788_data, v827_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v829_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v633_data, v793_data, v828_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v830_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v634_data, v798_data, v829_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v831_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v635_data, v803_data, v830_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v832_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v636_data, v808_data, v831_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v833_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v637_data, v813_data, v832_acc, 1, 1, 0);
          tensorforge::VectorT<float, 16> v834_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v638_data, v818_data, v833_acc, 1, 1, 0);
          float v835_el = v834_acc[0];
          float v837_el = v834_acc[4];
          float v838_sw = tensorforge::swap<32>(v837_el);
          float v840_el = v834_acc[8];
          float v843_el = v834_acc[12];
          float v844_sw = tensorforge::swap<32>(v843_el);
          r5[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v844_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v840_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v838_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v835_el, v835_el))))))));
          float v847_el = v834_acc[1];
          float v849_el = v834_acc[5];
          float v850_sw = tensorforge::swap<32>(v849_el);
          float v852_el = v834_acc[9];
          float v855_el = v834_acc[13];
          float v856_sw = tensorforge::swap<32>(v855_el);
          r5[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v856_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v852_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v850_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v847_el, v847_el))))))));
          float v859_el = v834_acc[2];
          float v861_el = v834_acc[6];
          float v862_sw = tensorforge::swap<32>(v861_el);
          float v864_el = v834_acc[10];
          float v867_el = v834_acc[14];
          float v868_sw = tensorforge::swap<32>(v867_el);
          r5[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v868_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v864_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v862_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v859_el, v859_el))))))));
          float v871_el = v834_acc[3];
          float v873_el = v834_acc[7];
          float v874_sw = tensorforge::swap<32>(v873_el);
          float v876_el = v834_acc[11];
          float v879_el = v834_acc[15];
          float v880_sw = tensorforge::swap<32>(v879_el);
          r5[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v880_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v876_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v874_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v871_el, v871_el))))))));
          float v884_sw = tensorforge::swap<32>(v835_el);
          float v889_sw = tensorforge::swap<32>(v840_el);
          r5[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v843_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v889_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v837_el, (tensorforge::dppUpdate<228, 1, 15, false>(v884_sw, v884_sw))))))));
          float v896_sw = tensorforge::swap<32>(v847_el);
          float v901_sw = tensorforge::swap<32>(v852_el);
          r5[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v855_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v901_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v849_el, (tensorforge::dppUpdate<228, 1, 15, false>(v896_sw, v896_sw))))))));
          float v908_sw = tensorforge::swap<32>(v859_el);
          float v913_sw = tensorforge::swap<32>(v864_el);
          r5[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v867_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v913_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v861_el, (tensorforge::dppUpdate<228, 1, 15, false>(v908_sw, v908_sw))))))));
          float v920_sw = tensorforge::swap<32>(v871_el);
          float v925_sw = tensorforge::swap<32>(v876_el);
          r5[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v879_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v925_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v873_el, (tensorforge::dppUpdate<228, 1, 15, false>(v920_sw, v920_sw))))))));
          float v932_sw = tensorforge::swap<64>(v835_el);
          r5[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v844_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v840_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v838_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v932_sw, v932_sw))))))));
          float v944_sw = tensorforge::swap<64>(v847_el);
          r5[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v856_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v852_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v850_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v944_sw, v944_sw))))))));
          float v956_sw = tensorforge::swap<64>(v859_el);
          r5[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v868_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v864_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v862_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v956_sw, v956_sw))))))));
          float v968_sw = tensorforge::swap<64>(v871_el);
          r5[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v880_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v876_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v874_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v968_sw, v968_sw))))))));
          float v981_sw = tensorforge::swap<64>(v884_sw);
          r5[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v843_el, (tensorforge::dppUpdate<228, 4, 15, false>(v889_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v837_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v981_sw, v981_sw))))))));
          float v993_sw = tensorforge::swap<64>(v896_sw);
          r5[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v855_el, (tensorforge::dppUpdate<228, 4, 15, false>(v901_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v849_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v993_sw, v993_sw))))))));
          float v1005_sw = tensorforge::swap<64>(v908_sw);
          r5[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v867_el, (tensorforge::dppUpdate<228, 4, 15, false>(v913_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v861_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1005_sw, v1005_sw))))))));
          float v1017_sw = tensorforge::swap<64>(v920_sw);
          r5[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v879_el, (tensorforge::dppUpdate<228, 4, 15, false>(v925_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v873_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v1017_sw, v1017_sw))))))));
          // wait(r6 = load{g>r}(glb_m4););
          float r7[8]{};
          // r7 = +(r5 * r6) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float v1028_data = r6[0];
          float v1029_data = r6[1];
          float v1030_data = r6[2];
          float v1031_data = r6[3];
          float v1032_tp{};
          float v1033_tp{};
          float v1034_tp{};
          float v1035_tp{};
          tensorforge::transpose4x4b32(v1032_tp, v1033_tp, v1034_tp, v1035_tp, v1028_data, v1029_data, v1030_data, v1031_data);
          tensorforge::VectorT<float, 4> v1036_acc{};
          float v1037_data = r5[0];
          float v1038_data = r5[1];
          float v1039_data = r5[2];
          float v1040_data = r5[3];
          tensorforge::VectorT<float, 4> v1041_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1037_data, v1036_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1042_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1038_data, v1041_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1043_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1034_tp, v1039_data, v1042_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1044_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1035_tp, v1040_data, v1043_acc, 3, 0, 0);
          float v1045_data = r5[4];
          float v1046_data = r5[5];
          float v1047_data = r5[6];
          float v1048_data = r5[7];
          tensorforge::VectorT<float, 4> v1049_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1045_data, v1044_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1050_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1046_data, v1049_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1051_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1034_tp, v1047_data, v1050_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1052_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1035_tp, v1048_data, v1051_acc, 3, 1, 0);
          float v1053_data = r5[8];
          float v1054_data = r5[9];
          float v1055_data = r5[10];
          float v1056_data = r5[11];
          tensorforge::VectorT<float, 4> v1057_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1053_data, v1052_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1058_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1054_data, v1057_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1059_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1034_tp, v1055_data, v1058_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1060_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1035_tp, v1056_data, v1059_acc, 3, 2, 0);
          float v1061_data = r5[12];
          float v1062_data = r5[13];
          float v1063_data = r5[14];
          float v1064_data = r5[15];
          tensorforge::VectorT<float, 4> v1065_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1032_tp, v1061_data, v1060_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1066_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1033_tp, v1062_data, v1065_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1067_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1034_tp, v1063_data, v1066_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1068_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1035_tp, v1064_data, v1067_acc, 3, 3, 0);
          r7[0] = (v1068_acc[0]);
          r7[1] = (v1068_acc[1]);
          r7[2] = (v1068_acc[2]);
          r7[3] = (v1068_acc[3]);
          float v1073_data = r6[4];
          float v1074_data = r6[5];
          float v1075_data = r6[6];
          float v1076_data = r6[7];
          float v1077_tp{};
          float v1078_tp{};
          float v1079_tp{};
          float v1080_tp{};
          tensorforge::transpose4x4b32(v1077_tp, v1078_tp, v1079_tp, v1080_tp, v1073_data, v1074_data, v1075_data, v1076_data);
          tensorforge::VectorT<float, 4> v1081_acc{};
          tensorforge::VectorT<float, 4> v1086_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1037_data, v1081_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1087_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1078_tp, v1038_data, v1086_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1088_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1079_tp, v1039_data, v1087_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1089_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1080_tp, v1040_data, v1088_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1094_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1045_data, v1089_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1095_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1078_tp, v1046_data, v1094_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1096_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1079_tp, v1047_data, v1095_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1097_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1080_tp, v1048_data, v1096_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1053_data, v1097_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1078_tp, v1054_data, v1102_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1079_tp, v1055_data, v1103_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1080_tp, v1056_data, v1104_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1077_tp, v1061_data, v1105_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1078_tp, v1062_data, v1110_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1079_tp, v1063_data, v1111_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1080_tp, v1064_data, v1112_acc, 3, 3, 0);
          r7[4] = (v1113_acc[0]);
          r7[5] = (v1113_acc[1]);
          r7[6] = (v1113_acc[2]);
          r7[7] = (v1113_acc[3]);
          // glb_m3 = store{r>g}(r7);
          if (v329_g) {
            #pragma unroll
            for (int32_t v1118_i1 = 0; v1118_i1 < 8; ++v1118_i1) {
              float v1120_data = r7[v1118_i1];
              glb_m3[(v27_lead + (v1118_i1 * 16))] = v1120_data;
            }
          }
        }
      }
    }
  }
}

