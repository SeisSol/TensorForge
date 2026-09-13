// === base name ===
kernel_976a0371aff6a302

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_976a0371aff6a302 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_976a0371aff6a302(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_976a0371aff6a302(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_976a0371aff6a302(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_976a0371aff6a302, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_976a0371aff6a302, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(float)));
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
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_976a0371aff6a302(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_976a0371aff6a302(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_976a0371aff6a302), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_976a0371aff6a302, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_976a0371aff6a302(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} none
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v5_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v5_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v7_batchIdLane0 = threadIdx.y % 4;
      int32_t v24_lead = threadIdx.x % 16;
      int32_t v55_a = v24_lead + ((threadIdx.y % 4) * 16);
      int32_t v62_a = v55_a + 64;
      int32_t v68_a = v55_a + 128;
      int32_t v74_a = v55_a + 192;
      for (size_t v8_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v8_batchIdGroup0 < numElements0; v8_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v9_row = v8_batchIdGroup0 + v7_batchIdLane0;
        const bool batchIdActive0 = v9_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v9_row]));
        size_t v11_batchId0 = batchIdActive0 ? v9_row : v8_batchIdGroup0;
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v11_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v11_batchId0 * 256 + 0 + m2_extraOffset];
        float r0[16]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
          int32_t v28_lead = v24_lead + (v25_i0 * 16);
          #pragma unroll
          for (int32_t v26_i1 = 0; v26_i1 < 16; ++v26_i1) {
            float v31_data = __builtin_nontemporal_load(&glb_m2[(v28_lead + (v26_i1 * 16))]);
            r0[(v25_i0 + v26_i1)] = v31_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[16]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 16)] [(0, 16)]
        float v34_data = r0[0];
        float v35_data = r0[1];
        float v36_data = r0[2];
        float v37_data = r0[3];
        float v38_data = r0[4];
        float v39_data = r0[5];
        float v40_data = r0[6];
        float v41_data = r0[7];
        float v42_data = r0[8];
        float v43_data = r0[9];
        float v44_data = r0[10];
        float v45_data = r0[11];
        float v46_data = r0[12];
        float v47_data = r0[13];
        float v48_data = r0[14];
        float v49_data = r0[15];
        tensorforge::transpose16x16b32(v34_data, v35_data, v36_data, v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data);
        tensorforge::VectorT<float, 16> v50_acc{};
        float v57_data = glb_m1[v55_a];
        tensorforge::VectorT<float, 16> v58_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v34_data, v57_data, v50_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v59_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v35_data, v57_data, v58_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v60_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v36_data, v57_data, v59_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v61_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v37_data, v57_data, v60_acc, 0, 0, 7);
        float v63_data = glb_m1[v62_a];
        tensorforge::VectorT<float, 16> v64_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v38_data, v63_data, v61_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v65_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v39_data, v63_data, v64_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v66_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v63_data, v65_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v67_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v63_data, v66_acc, 0, 0, 7);
        float v69_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 16> v70_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v69_data, v67_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v71_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v69_data, v70_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v72_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v69_data, v71_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v73_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v69_data, v72_acc, 0, 0, 7);
        float v75_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v75_data, v73_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v75_data, v76_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v75_data, v77_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v75_data, v78_acc, 0, 0, 7);
        float v80_el = v79_acc[0];
        float v82_el = v79_acc[4];
        float v83_sw = tensorforge::swap<32>(v82_el);
        float v85_el = v79_acc[8];
        float v88_el = v79_acc[12];
        float v89_sw = tensorforge::swap<32>(v88_el);
        r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v89_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v85_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v83_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v80_el, v80_el))))))));
        float v92_el = v79_acc[1];
        float v94_el = v79_acc[5];
        float v95_sw = tensorforge::swap<32>(v94_el);
        float v97_el = v79_acc[9];
        float v100_el = v79_acc[13];
        float v101_sw = tensorforge::swap<32>(v100_el);
        r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v101_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v95_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v92_el, v92_el))))))));
        float v104_el = v79_acc[2];
        float v106_el = v79_acc[6];
        float v107_sw = tensorforge::swap<32>(v106_el);
        float v109_el = v79_acc[10];
        float v112_el = v79_acc[14];
        float v113_sw = tensorforge::swap<32>(v112_el);
        r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v113_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v107_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v104_el, v104_el))))))));
        float v116_el = v79_acc[3];
        float v118_el = v79_acc[7];
        float v119_sw = tensorforge::swap<32>(v118_el);
        float v121_el = v79_acc[11];
        float v124_el = v79_acc[15];
        float v125_sw = tensorforge::swap<32>(v124_el);
        r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v125_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v119_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v116_el, v116_el))))))));
        float v129_sw = tensorforge::swap<32>(v80_el);
        float v134_sw = tensorforge::swap<32>(v85_el);
        r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v88_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v134_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v82_el, (tensorforge::dppUpdate<228, 1, 15, false>(v129_sw, v129_sw))))))));
        float v141_sw = tensorforge::swap<32>(v92_el);
        float v146_sw = tensorforge::swap<32>(v97_el);
        r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v146_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v94_el, (tensorforge::dppUpdate<228, 1, 15, false>(v141_sw, v141_sw))))))));
        float v153_sw = tensorforge::swap<32>(v104_el);
        float v158_sw = tensorforge::swap<32>(v109_el);
        r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v158_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v106_el, (tensorforge::dppUpdate<228, 1, 15, false>(v153_sw, v153_sw))))))));
        float v165_sw = tensorforge::swap<32>(v116_el);
        float v170_sw = tensorforge::swap<32>(v121_el);
        r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v170_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v118_el, (tensorforge::dppUpdate<228, 1, 15, false>(v165_sw, v165_sw))))))));
        float v177_sw = tensorforge::swap<64>(v80_el);
        r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v89_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v85_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v83_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v177_sw, v177_sw))))))));
        float v189_sw = tensorforge::swap<64>(v92_el);
        r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v101_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v97_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v95_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v189_sw, v189_sw))))))));
        float v201_sw = tensorforge::swap<64>(v104_el);
        r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v113_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v109_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v107_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v201_sw, v201_sw))))))));
        float v213_sw = tensorforge::swap<64>(v116_el);
        r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v125_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v121_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v119_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v213_sw, v213_sw))))))));
        float v226_sw = tensorforge::swap<64>(v129_sw);
        r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v88_el, (tensorforge::dppUpdate<228, 4, 15, false>(v134_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v82_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v226_sw, v226_sw))))))));
        float v238_sw = tensorforge::swap<64>(v141_sw);
        r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v100_el, (tensorforge::dppUpdate<228, 4, 15, false>(v146_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v94_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v238_sw, v238_sw))))))));
        float v250_sw = tensorforge::swap<64>(v153_sw);
        r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v112_el, (tensorforge::dppUpdate<228, 4, 15, false>(v158_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v250_sw, v250_sw))))))));
        float v262_sw = tensorforge::swap<64>(v165_sw);
        r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v124_el, (tensorforge::dppUpdate<228, 4, 15, false>(v170_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v262_sw, v262_sw))))))));
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v272_i0 = 0; v272_i0 < 1; ++v272_i0) {
          #pragma unroll
          for (int32_t v273_i1 = 0; v273_i1 < 16; ++v273_i1) {
            float v275_data = r1[(v272_i0 + v273_i1)];
            if (batchIdActive0) {
              glb_m0[((v24_lead + (v272_i0 * 16)) + (v273_i1 * 16))] = v275_data;
            }
          }
        }
      }
    }
  }
}

