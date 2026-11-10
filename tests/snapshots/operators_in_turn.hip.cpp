// === base name ===
kernel_b2d5ecdd299ca155

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b2d5ecdd299ca155 = {{16, 16, 1}, 16, 16, 1, 16, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b2d5ecdd299ca155(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b2d5ecdd299ca155(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b2d5ecdd299ca155(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b2d5ecdd299ca155, block.x * block.y * block.z, 768 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (768 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b2d5ecdd299ca155, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (768 * sizeof(float)));
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
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b2d5ecdd299ca155(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b2d5ecdd299ca155(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b2d5ecdd299ca155), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m4;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b2d5ecdd299ca155, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m5Arg, m5_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b2d5ecdd299ca155(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m4, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} none ordered
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} none ordered
    //   m5 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m3[i,j] = m4[i,k] × m0[k,j]
    //   m5[i,j] = m1[i,k] × m3[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"P","bbox":[[0,0],[16,16]],"name":"m1","ordered":true,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"Q","bbox":[[0,0],[16,16]],"name":"m4","ordered":true,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 512];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m4[0];
      float * __restrict__ glb_m4 = &totalShrMem[256];
      // glb_m4 = load{g>s}(ptr_glb_m4[0, 1])
      float v12_ld = ptr_glb_m4[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      glb_m4[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      __syncthreads();
      size_t v14_batchIdLane0 = threadIdx.y % 4;
      size_t v17_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v35_lead = threadIdx.x % 16;
      int32_t v68_vlead = (v35_lead + ((threadIdx.y % 4) * 16)) * 4;
      float r0[16]{};
      size_t v854_row = v17_batchIdGroup0First + v14_batchIdLane0;
      const bool peel_batchIdActive0 = v854_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v854_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v854_row : v17_batchIdGroup0First) * 256 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v862_i0 = 0; v862_i0 < 1; ++v862_i0) {
        int32_t v864_lead = v35_lead + (v862_i0 * 16);
        #pragma unroll
        for (int32_t v865_i1 = 0; v865_i1 < 16; ++v865_i1) {
          float v868_data = __builtin_nontemporal_load(&peel_glb_m2[(v864_lead + (v865_i1 * 16))]);
          r0[(v862_i0 + v865_i1)] = v868_data;
        }
      }
      for (size_t v18_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v18_batchIdGroup0 < numElements0; v18_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v851_groupNextAhead = v18_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v853_groupNext = (v851_groupNextAhead < numElements0) ? v851_groupNextAhead : v18_batchIdGroup0;
        size_t v858_row = v853_groupNext + v14_batchIdLane0;
        const bool wrap_batchIdActive0 = v858_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v858_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v858_row : v853_groupNext) * 256 + 0 + m2_extraOffset];
        size_t v19_row = v18_batchIdGroup0 + v14_batchIdLane0;
        const bool batchIdActive0 = v19_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v19_row]));
        size_t v21_batchId0 = batchIdActive0 ? v19_row : v18_batchIdGroup0;
        size_t v22_ahead1 = v21_batchId0 + (gridDim.x * blockDim.y);
        size_t v24_batchId1 = (v22_ahead1 < numElements0) ? v22_ahead1 : v21_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v21_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v21_batchId0 * 256 + 0 + m3_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m5[v21_batchId0 * 256 + 0 + m5_extraOffset];
        float r1[16]{};
        // ir1 = +(glb_m1 * r0)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir1[16]{};
        float v46_data = r0[0];
        float v47_data = r0[1];
        float v48_data = r0[2];
        float v49_data = r0[3];
        float v50_data = r0[4];
        float v51_data = r0[5];
        float v52_data = r0[6];
        float v53_data = r0[7];
        float v54_data = r0[8];
        float v55_data = r0[9];
        float v56_data = r0[10];
        float v57_data = r0[11];
        float v58_data = r0[12];
        float v59_data = r0[13];
        float v60_data = r0[14];
        float v61_data = r0[15];
        tensorforge::transpose16x16b32(v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data);
        tensorforge::VectorT<float, 16> v62_acc{};
        tensorforge::VectorT<float, 4> v70_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v68_vlead];
        float v71_el = v70_data[0];
        tensorforge::VectorT<float, 16> v72_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v71_el, v62_acc, 0, 0, 4);
        float v73_el = v70_data[1];
        tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v73_el, v72_acc, 0, 0, 4);
        float v75_el = v70_data[2];
        tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v75_el, v74_acc, 0, 0, 4);
        float v77_el = v70_data[3];
        tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v77_el, v76_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v71_el, v78_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v73_el, v80_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v75_el, v82_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v77_el, v84_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_el, v86_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v73_el, v88_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v75_el, v90_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v77_el, v92_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v71_el, v94_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v98_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v73_el, v96_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v100_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v75_el, v98_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v102_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v77_el, v100_acc, 0, 0, 7);
        float v103_el = v102_acc[0];
        float v105_el = v102_acc[4];
        float v106_sw = tensorforge::swap<32>(v105_el);
        float v108_el = v102_acc[8];
        float v111_el = v102_acc[12];
        float v112_sw = tensorforge::swap<32>(v111_el);
        ir1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v112_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v108_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v106_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v103_el, v103_el))))))));
        float v115_el = v102_acc[1];
        float v117_el = v102_acc[5];
        float v118_sw = tensorforge::swap<32>(v117_el);
        float v120_el = v102_acc[9];
        float v123_el = v102_acc[13];
        float v124_sw = tensorforge::swap<32>(v123_el);
        ir1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v124_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v120_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v118_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v115_el, v115_el))))))));
        float v127_el = v102_acc[2];
        float v129_el = v102_acc[6];
        float v130_sw = tensorforge::swap<32>(v129_el);
        float v132_el = v102_acc[10];
        float v135_el = v102_acc[14];
        float v136_sw = tensorforge::swap<32>(v135_el);
        ir1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v127_el, v127_el))))))));
        float v139_el = v102_acc[3];
        float v141_el = v102_acc[7];
        float v142_sw = tensorforge::swap<32>(v141_el);
        float v144_el = v102_acc[11];
        float v147_el = v102_acc[15];
        float v148_sw = tensorforge::swap<32>(v147_el);
        ir1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v148_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v142_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v139_el, v139_el))))))));
        float v152_sw = tensorforge::swap<32>(v103_el);
        float v157_sw = tensorforge::swap<32>(v108_el);
        ir1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v105_el, (tensorforge::dppUpdate<228, 1, 15, false>(v152_sw, v152_sw))))))));
        float v164_sw = tensorforge::swap<32>(v115_el);
        float v169_sw = tensorforge::swap<32>(v120_el);
        ir1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v169_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v117_el, (tensorforge::dppUpdate<228, 1, 15, false>(v164_sw, v164_sw))))))));
        float v176_sw = tensorforge::swap<32>(v127_el);
        float v181_sw = tensorforge::swap<32>(v132_el);
        ir1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v181_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v129_el, (tensorforge::dppUpdate<228, 1, 15, false>(v176_sw, v176_sw))))))));
        float v188_sw = tensorforge::swap<32>(v139_el);
        float v193_sw = tensorforge::swap<32>(v144_el);
        ir1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v193_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v141_el, (tensorforge::dppUpdate<228, 1, 15, false>(v188_sw, v188_sw))))))));
        float v200_sw = tensorforge::swap<64>(v103_el);
        ir1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v112_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v108_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v106_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v200_sw, v200_sw))))))));
        float v212_sw = tensorforge::swap<64>(v115_el);
        ir1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v124_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v120_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v118_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v212_sw, v212_sw))))))));
        float v224_sw = tensorforge::swap<64>(v127_el);
        ir1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v132_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v224_sw, v224_sw))))))));
        float v236_sw = tensorforge::swap<64>(v139_el);
        ir1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v148_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v144_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v142_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v236_sw, v236_sw))))))));
        float v249_sw = tensorforge::swap<64>(v152_sw);
        ir1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v111_el, (tensorforge::dppUpdate<228, 4, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v249_sw, v249_sw))))))));
        float v261_sw = tensorforge::swap<64>(v164_sw);
        ir1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v123_el, (tensorforge::dppUpdate<228, 4, 15, false>(v169_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v261_sw, v261_sw))))))));
        float v273_sw = tensorforge::swap<64>(v176_sw);
        ir1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v135_el, (tensorforge::dppUpdate<228, 4, 15, false>(v181_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v273_sw, v273_sw))))))));
        float v285_sw = tensorforge::swap<64>(v188_sw);
        ir1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v147_el, (tensorforge::dppUpdate<228, 4, 15, false>(v193_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v285_sw, v285_sw))))))));
        // r1 = ir1
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v299_i0 = 0; v299_i0 < 1; ++v299_i0) {
          #pragma unroll
          for (int32_t v300_i1 = 0; v300_i1 < 16; ++v300_i1) {
            float v302_data = ir1[v300_i1];
            if (batchIdActive0) {
              glb_m0[((v35_lead + (v299_i0 * 16)) + (v300_i1 * 16))] = v302_data;
            }
          }
        }
        float r2[16]{};
        // r2 = load{g>r}(glb_m0);
        #pragma unroll
        for (int32_t v308_i0 = 0; v308_i0 < 1; ++v308_i0) {
          #pragma unroll
          for (int32_t v309_i1 = 0; v309_i1 < 16; ++v309_i1) {
            float v314_data = ir1[v309_i1];
            r2[(v308_i0 + v309_i1)] = v314_data;
          }
        }
        float r3[16]{};
        // ir3 = +(glb_m4 * r2)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir3[16]{};
        float v318_data = r2[0];
        float v319_data = r2[1];
        float v320_data = r2[2];
        float v321_data = r2[3];
        float v322_data = r2[4];
        float v323_data = r2[5];
        float v324_data = r2[6];
        float v325_data = r2[7];
        float v326_data = r2[8];
        float v327_data = r2[9];
        float v328_data = r2[10];
        float v329_data = r2[11];
        float v330_data = r2[12];
        float v331_data = r2[13];
        float v332_data = r2[14];
        float v333_data = r2[15];
        tensorforge::transpose16x16b32(v318_data, v319_data, v320_data, v321_data, v322_data, v323_data, v324_data, v325_data, v326_data, v327_data, v328_data, v329_data, v330_data, v331_data, v332_data, v333_data);
        tensorforge::VectorT<float, 16> v334_acc{};
        tensorforge::VectorT<float, 4> v342_data = *(tensorforge::VectorT<float, 4>*)&glb_m4[v68_vlead];
        float v343_el = v342_data[0];
        tensorforge::VectorT<float, 16> v344_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v318_data, v343_el, v334_acc, 0, 0, 4);
        float v345_el = v342_data[1];
        tensorforge::VectorT<float, 16> v346_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v319_data, v345_el, v344_acc, 0, 0, 4);
        float v347_el = v342_data[2];
        tensorforge::VectorT<float, 16> v348_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v320_data, v347_el, v346_acc, 0, 0, 4);
        float v349_el = v342_data[3];
        tensorforge::VectorT<float, 16> v350_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v321_data, v349_el, v348_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v352_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v322_data, v343_el, v350_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v354_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v323_data, v345_el, v352_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v356_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v324_data, v347_el, v354_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v358_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v325_data, v349_el, v356_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v360_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v326_data, v343_el, v358_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v362_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v327_data, v345_el, v360_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v347_el, v362_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v349_el, v364_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v343_el, v366_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v370_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v345_el, v368_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v372_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v347_el, v370_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v374_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v349_el, v372_acc, 0, 0, 7);
        float v375_el = v374_acc[0];
        float v377_el = v374_acc[4];
        float v378_sw = tensorforge::swap<32>(v377_el);
        float v380_el = v374_acc[8];
        float v383_el = v374_acc[12];
        float v384_sw = tensorforge::swap<32>(v383_el);
        ir3[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v384_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v380_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v378_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v375_el, v375_el))))))));
        float v387_el = v374_acc[1];
        float v389_el = v374_acc[5];
        float v390_sw = tensorforge::swap<32>(v389_el);
        float v392_el = v374_acc[9];
        float v395_el = v374_acc[13];
        float v396_sw = tensorforge::swap<32>(v395_el);
        ir3[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v396_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v392_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v390_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v387_el, v387_el))))))));
        float v399_el = v374_acc[2];
        float v401_el = v374_acc[6];
        float v402_sw = tensorforge::swap<32>(v401_el);
        float v404_el = v374_acc[10];
        float v407_el = v374_acc[14];
        float v408_sw = tensorforge::swap<32>(v407_el);
        ir3[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v408_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v404_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v402_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v399_el, v399_el))))))));
        float v411_el = v374_acc[3];
        float v413_el = v374_acc[7];
        float v414_sw = tensorforge::swap<32>(v413_el);
        float v416_el = v374_acc[11];
        float v419_el = v374_acc[15];
        float v420_sw = tensorforge::swap<32>(v419_el);
        ir3[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v420_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v416_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v414_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v411_el, v411_el))))))));
        float v424_sw = tensorforge::swap<32>(v375_el);
        float v429_sw = tensorforge::swap<32>(v380_el);
        ir3[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v383_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v429_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v377_el, (tensorforge::dppUpdate<228, 1, 15, false>(v424_sw, v424_sw))))))));
        float v436_sw = tensorforge::swap<32>(v387_el);
        float v441_sw = tensorforge::swap<32>(v392_el);
        ir3[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v395_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v441_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v389_el, (tensorforge::dppUpdate<228, 1, 15, false>(v436_sw, v436_sw))))))));
        float v448_sw = tensorforge::swap<32>(v399_el);
        float v453_sw = tensorforge::swap<32>(v404_el);
        ir3[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v407_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v453_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v401_el, (tensorforge::dppUpdate<228, 1, 15, false>(v448_sw, v448_sw))))))));
        float v460_sw = tensorforge::swap<32>(v411_el);
        float v465_sw = tensorforge::swap<32>(v416_el);
        ir3[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v419_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v465_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v413_el, (tensorforge::dppUpdate<228, 1, 15, false>(v460_sw, v460_sw))))))));
        float v472_sw = tensorforge::swap<64>(v375_el);
        ir3[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v384_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v380_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v378_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v472_sw, v472_sw))))))));
        float v484_sw = tensorforge::swap<64>(v387_el);
        ir3[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v396_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v392_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v390_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v484_sw, v484_sw))))))));
        float v496_sw = tensorforge::swap<64>(v399_el);
        ir3[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v408_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v404_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v402_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v496_sw, v496_sw))))))));
        float v508_sw = tensorforge::swap<64>(v411_el);
        ir3[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v420_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v416_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v414_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v508_sw, v508_sw))))))));
        float v521_sw = tensorforge::swap<64>(v424_sw);
        ir3[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v383_el, (tensorforge::dppUpdate<228, 4, 15, false>(v429_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v377_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v521_sw, v521_sw))))))));
        float v533_sw = tensorforge::swap<64>(v436_sw);
        ir3[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v395_el, (tensorforge::dppUpdate<228, 4, 15, false>(v441_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v389_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v533_sw, v533_sw))))))));
        float v545_sw = tensorforge::swap<64>(v448_sw);
        ir3[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v407_el, (tensorforge::dppUpdate<228, 4, 15, false>(v453_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v401_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v545_sw, v545_sw))))))));
        float v557_sw = tensorforge::swap<64>(v460_sw);
        ir3[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v419_el, (tensorforge::dppUpdate<228, 4, 15, false>(v465_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v413_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v557_sw, v557_sw))))))));
        // r3 = ir3
        // glb_m3 = store{r>g}(r3);
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v870_i0 = 0; v870_i0 < 1; ++v870_i0) {
          int32_t v872_lead = v35_lead + (v870_i0 * 16);
          #pragma unroll
          for (int32_t v873_i1 = 0; v873_i1 < 16; ++v873_i1) {
            float v876_data = __builtin_nontemporal_load(&wrap_glb_m2[(v872_lead + (v873_i1 * 16))]);
            r0[(v870_i0 + v873_i1)] = v876_data;
          }
        }
        #pragma unroll
        for (int32_t v571_i0 = 0; v571_i0 < 1; ++v571_i0) {
          #pragma unroll
          for (int32_t v572_i1 = 0; v572_i1 < 16; ++v572_i1) {
            float v574_data = ir3[v572_i1];
            if (batchIdActive0) {
              glb_m3[((v35_lead + (v571_i0 * 16)) + (v572_i1 * 16))] = v574_data;
            }
          }
        }
        float r4[16]{};
        // r4 = load{g>r}(glb_m3);
        #pragma unroll
        for (int32_t v580_i0 = 0; v580_i0 < 1; ++v580_i0) {
          #pragma unroll
          for (int32_t v581_i1 = 0; v581_i1 < 16; ++v581_i1) {
            float v586_data = ir3[v581_i1];
            r4[(v580_i0 + v581_i1)] = v586_data;
          }
        }
        float r5[16]{};
        // ir5 = +(glb_m1 * r4)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir5[16]{};
        float v590_data = r4[0];
        float v591_data = r4[1];
        float v592_data = r4[2];
        float v593_data = r4[3];
        float v594_data = r4[4];
        float v595_data = r4[5];
        float v596_data = r4[6];
        float v597_data = r4[7];
        float v598_data = r4[8];
        float v599_data = r4[9];
        float v600_data = r4[10];
        float v601_data = r4[11];
        float v602_data = r4[12];
        float v603_data = r4[13];
        float v604_data = r4[14];
        float v605_data = r4[15];
        tensorforge::transpose16x16b32(v590_data, v591_data, v592_data, v593_data, v594_data, v595_data, v596_data, v597_data, v598_data, v599_data, v600_data, v601_data, v602_data, v603_data, v604_data, v605_data);
        tensorforge::VectorT<float, 16> v606_acc{};
        tensorforge::VectorT<float, 16> v616_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v590_data, v71_el, v606_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v618_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v591_data, v73_el, v616_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v620_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v592_data, v75_el, v618_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v622_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v593_data, v77_el, v620_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v624_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v594_data, v71_el, v622_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v626_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v595_data, v73_el, v624_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v628_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v596_data, v75_el, v626_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v630_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v597_data, v77_el, v628_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v632_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v598_data, v71_el, v630_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v634_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v599_data, v73_el, v632_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v636_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v600_data, v75_el, v634_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v638_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v601_data, v77_el, v636_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v640_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v602_data, v71_el, v638_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v642_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v603_data, v73_el, v640_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v644_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v604_data, v75_el, v642_acc, 0, 0, 7);
        tensorforge::VectorT<float, 16> v646_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v605_data, v77_el, v644_acc, 0, 0, 7);
        float v647_el = v646_acc[0];
        float v649_el = v646_acc[4];
        float v650_sw = tensorforge::swap<32>(v649_el);
        float v652_el = v646_acc[8];
        float v655_el = v646_acc[12];
        float v656_sw = tensorforge::swap<32>(v655_el);
        ir5[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v656_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v652_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v650_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v647_el, v647_el))))))));
        float v659_el = v646_acc[1];
        float v661_el = v646_acc[5];
        float v662_sw = tensorforge::swap<32>(v661_el);
        float v664_el = v646_acc[9];
        float v667_el = v646_acc[13];
        float v668_sw = tensorforge::swap<32>(v667_el);
        ir5[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v668_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v664_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v662_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v659_el, v659_el))))))));
        float v671_el = v646_acc[2];
        float v673_el = v646_acc[6];
        float v674_sw = tensorforge::swap<32>(v673_el);
        float v676_el = v646_acc[10];
        float v679_el = v646_acc[14];
        float v680_sw = tensorforge::swap<32>(v679_el);
        ir5[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v680_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v676_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v674_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v671_el, v671_el))))))));
        float v683_el = v646_acc[3];
        float v685_el = v646_acc[7];
        float v686_sw = tensorforge::swap<32>(v685_el);
        float v688_el = v646_acc[11];
        float v691_el = v646_acc[15];
        float v692_sw = tensorforge::swap<32>(v691_el);
        ir5[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v692_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v688_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v686_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v683_el, v683_el))))))));
        float v696_sw = tensorforge::swap<32>(v647_el);
        float v701_sw = tensorforge::swap<32>(v652_el);
        ir5[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v655_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v701_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v649_el, (tensorforge::dppUpdate<228, 1, 15, false>(v696_sw, v696_sw))))))));
        float v708_sw = tensorforge::swap<32>(v659_el);
        float v713_sw = tensorforge::swap<32>(v664_el);
        ir5[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v667_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v713_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v661_el, (tensorforge::dppUpdate<228, 1, 15, false>(v708_sw, v708_sw))))))));
        float v720_sw = tensorforge::swap<32>(v671_el);
        float v725_sw = tensorforge::swap<32>(v676_el);
        ir5[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v679_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v725_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v673_el, (tensorforge::dppUpdate<228, 1, 15, false>(v720_sw, v720_sw))))))));
        float v732_sw = tensorforge::swap<32>(v683_el);
        float v737_sw = tensorforge::swap<32>(v688_el);
        ir5[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v691_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v737_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v685_el, (tensorforge::dppUpdate<228, 1, 15, false>(v732_sw, v732_sw))))))));
        float v744_sw = tensorforge::swap<64>(v647_el);
        ir5[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v656_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v652_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v650_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v744_sw, v744_sw))))))));
        float v756_sw = tensorforge::swap<64>(v659_el);
        ir5[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v668_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v664_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v662_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v756_sw, v756_sw))))))));
        float v768_sw = tensorforge::swap<64>(v671_el);
        ir5[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v680_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v676_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v674_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v768_sw, v768_sw))))))));
        float v780_sw = tensorforge::swap<64>(v683_el);
        ir5[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v692_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v688_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v686_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v780_sw, v780_sw))))))));
        float v793_sw = tensorforge::swap<64>(v696_sw);
        ir5[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v655_el, (tensorforge::dppUpdate<228, 4, 15, false>(v701_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v649_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v793_sw, v793_sw))))))));
        float v805_sw = tensorforge::swap<64>(v708_sw);
        ir5[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v667_el, (tensorforge::dppUpdate<228, 4, 15, false>(v713_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v661_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v805_sw, v805_sw))))))));
        float v817_sw = tensorforge::swap<64>(v720_sw);
        ir5[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v679_el, (tensorforge::dppUpdate<228, 4, 15, false>(v725_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v673_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v817_sw, v817_sw))))))));
        float v829_sw = tensorforge::swap<64>(v732_sw);
        ir5[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v691_el, (tensorforge::dppUpdate<228, 4, 15, false>(v737_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v685_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v829_sw, v829_sw))))))));
        // r5 = ir5
        // glb_m5 = store{r>g}(r5);
        #pragma unroll
        for (int32_t v843_i0 = 0; v843_i0 < 1; ++v843_i0) {
          #pragma unroll
          for (int32_t v844_i1 = 0; v844_i1 < 16; ++v844_i1) {
            float v846_data = ir5[v844_i1];
            if (batchIdActive0) {
              glb_m5[((v35_lead + (v843_i0 * 16)) + (v844_i1 * 16))] = v846_data;
            }
          }
        }
      }
    }
  }
}

