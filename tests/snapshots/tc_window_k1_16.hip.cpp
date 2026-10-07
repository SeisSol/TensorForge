// === base name ===
kernel_d0cf47ab58ac3fd7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d0cf47ab58ac3fd7 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d0cf47ab58ac3fd7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d0cf47ab58ac3fd7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d0cf47ab58ac3fd7(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d0cf47ab58ac3fd7, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d0cf47ab58ac3fd7, block.x * block.y * block.z, 0));
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
void launcher_kernel_d0cf47ab58ac3fd7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d0cf47ab58ac3fd7(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d0cf47ab58ac3fd7), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_d0cf47ab58ac3fd7, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d0cf47ab58ac3fd7(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×16) {0..16}×{1..17} none
    //   m2 20×9(16×9) {1..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 4;
      int32_t v27_lead = threadIdx.x % 16;
      bool v28_g = v27_lead >= 1;
      bool v38_g = v27_lead < 1;
      int32_t v70_a = v27_lead + ((threadIdx.y % 4) * 16);
      int32_t v77_a = v70_a + 64;
      int32_t v83_a = v70_a + 128;
      int32_t v89_a = v70_a + 192;
      int32_t v153_a = v27_lead + 16;
      int32_t v155_a = v27_lead + 32;
      int32_t v157_a = v27_lead + 48;
      int32_t v159_a = v27_lead + 64;
      int32_t v161_a = v27_lead + 80;
      int32_t v163_a = v27_lead + 96;
      int32_t v165_a = v27_lead + 112;
      int32_t v167_a = v27_lead + 128;
      int32_t v169_a = v27_lead + 144;
      int32_t v171_a = v27_lead + 160;
      int32_t v173_a = v27_lead + 176;
      int32_t v175_a = v27_lead + 192;
      int32_t v177_a = v27_lead + 208;
      int32_t v179_a = v27_lead + 224;
      int32_t v181_a = v27_lead + 240;
      for (size_t v12_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v13_row = v12_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
        size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 144 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v15_batchId0 * 144 + 0 + m2_extraOffset];
        float r0[18]{};
        // r0 = load{g>r}(glb_m2);
        if (v28_g) {
          int32_t v32_a = v27_lead - 1;
          #pragma unroll
          for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
            float v35_data = __builtin_nontemporal_load(&glb_m2[(v32_a + (v29_i1 * 16))]);
            r0[(v29_i1 * 2)] = v35_data;
          }
        }
        if (v38_g) {
          int32_t v42_a = (v27_lead + 16_i32) - 1;
          #pragma unroll
          for (int32_t v39_i1 = 0; v39_i1 < 9; ++v39_i1) {
            float v45_data = __builtin_nontemporal_load(&glb_m2[(v42_a + (v39_i1 * 16))]);
            r0[(1 + (v39_i1 * 2))] = v45_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[9]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 9)] [(1, 17)]
        float v49_data = r0[0];
        float v50_data = r0[2];
        float v51_data = r0[4];
        float v52_data = r0[6];
        float v53_tp{};
        float v54_tp{};
        float v55_tp{};
        float v56_tp{};
        tensorforge::transpose4x4b32(v53_tp, v54_tp, v55_tp, v56_tp, v49_data, v50_data, v51_data, v52_data);
        float v57_data = r0[1];
        float v58_data = r0[3];
        float v59_data = r0[5];
        float v60_data = r0[7];
        float v61_tp{};
        float v62_tp{};
        float v63_tp{};
        float v64_tp{};
        tensorforge::transpose4x4b32(v61_tp, v62_tp, v63_tp, v64_tp, v57_data, v58_data, v59_data, v60_data);
        tensorforge::VectorT<float, 4> v65_acc{};
        float v72_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v72_data, v65_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v72_data, v73_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v72_data, v74_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v72_data, v75_acc, 2, 1, 7);
        float v78_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v78_data, v76_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v78_data, v79_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v78_data, v80_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v78_data, v81_acc, 2, 2, 7);
        float v84_data = glb_m1[v83_a];
        tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v84_data, v82_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v85_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v84_data, v86_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v84_data, v87_acc, 2, 3, 7);
        float v90_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v90_data, v88_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v90_data, v91_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v90_data, v92_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v90_data, v93_acc, 2, 0, 7);
        r1[0] = (v94_acc[0]);
        r1[1] = (v94_acc[1]);
        r1[2] = (v94_acc[2]);
        r1[3] = (v94_acc[3]);
        float v99_data = r0[8];
        float v100_data = r0[10];
        float v101_data = r0[12];
        float v102_data = r0[14];
        float v103_tp{};
        float v104_tp{};
        float v105_tp{};
        float v106_tp{};
        tensorforge::transpose4x4b32(v103_tp, v104_tp, v105_tp, v106_tp, v99_data, v100_data, v101_data, v102_data);
        float v107_data = r0[9];
        float v108_data = r0[11];
        float v109_data = r0[13];
        float v110_data = r0[15];
        float v111_tp{};
        float v112_tp{};
        float v113_tp{};
        float v114_tp{};
        tensorforge::transpose4x4b32(v111_tp, v112_tp, v113_tp, v114_tp, v107_data, v108_data, v109_data, v110_data);
        tensorforge::VectorT<float, 4> v115_acc{};
        float v122_data = glb_m1[v70_a];
        tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v122_data, v115_acc, 2, 0, 4);
        tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v122_data, v123_acc, 2, 0, 5);
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v122_data, v124_acc, 2, 0, 6);
        tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v122_data, v125_acc, 2, 1, 7);
        float v128_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v128_data, v126_acc, 2, 1, 4);
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v128_data, v129_acc, 2, 1, 5);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v128_data, v130_acc, 2, 1, 6);
        tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v128_data, v131_acc, 2, 2, 7);
        float v134_data = glb_m1[v83_a];
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v134_data, v132_acc, 2, 2, 4);
        tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v134_data, v135_acc, 2, 2, 5);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v134_data, v136_acc, 2, 2, 6);
        tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v134_data, v137_acc, 2, 3, 7);
        float v140_data = glb_m1[v89_a];
        tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v140_data, v138_acc, 2, 3, 4);
        tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v140_data, v141_acc, 2, 3, 5);
        tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v140_data, v142_acc, 2, 3, 6);
        tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v140_data, v143_acc, 2, 0, 7);
        r1[4] = (v144_acc[0]);
        r1[5] = (v144_acc[1]);
        r1[6] = (v144_acc[2]);
        r1[7] = (v144_acc[3]);
        float v152_data = glb_m1[v27_lead];
        float v154_data = glb_m1[v153_a];
        float v156_data = glb_m1[v155_a];
        float v158_data = glb_m1[v157_a];
        float v160_data = glb_m1[v159_a];
        float v162_data = glb_m1[v161_a];
        float v164_data = glb_m1[v163_a];
        float v166_data = glb_m1[v165_a];
        float v168_data = glb_m1[v167_a];
        float v170_data = glb_m1[v169_a];
        float v172_data = glb_m1[v171_a];
        float v174_data = glb_m1[v173_a];
        float v176_data = glb_m1[v175_a];
        float v178_data = glb_m1[v177_a];
        float v180_data = glb_m1[v179_a];
        float v182_data = glb_m1[v181_a];
        float v183_acc{};
        float v184_data = r0[16];
        float v185_data = r0[17];
        tensorforge::fmacdpp16<1>(v183_acc, v184_data, v152_data);
        tensorforge::fmacdpp16<2>(v183_acc, v184_data, v154_data);
        tensorforge::fmacdpp16<3>(v183_acc, v184_data, v156_data);
        tensorforge::fmacdpp16<4>(v183_acc, v184_data, v158_data);
        tensorforge::fmacdpp16<5>(v183_acc, v184_data, v160_data);
        tensorforge::fmacdpp16<6>(v183_acc, v184_data, v162_data);
        tensorforge::fmacdpp16<7>(v183_acc, v184_data, v164_data);
        tensorforge::fmacdpp16<8>(v183_acc, v184_data, v166_data);
        tensorforge::fmacdpp16<9>(v183_acc, v184_data, v168_data);
        tensorforge::fmacdpp16<10>(v183_acc, v184_data, v170_data);
        tensorforge::fmacdpp16<11>(v183_acc, v184_data, v172_data);
        tensorforge::fmacdpp16<12>(v183_acc, v184_data, v174_data);
        tensorforge::fmacdpp16<13>(v183_acc, v184_data, v176_data);
        tensorforge::fmacdpp16<14>(v183_acc, v184_data, v178_data);
        tensorforge::fmacdpp16<15>(v183_acc, v184_data, v180_data);
        tensorforge::fmacdpp16<0>(v183_acc, v185_data, v182_data);
        r1[8] = v183_acc;
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v186_i0 = 0; v186_i0 < 1; ++v186_i0) {
          #pragma unroll
          for (int32_t v187_i1 = 0; v187_i1 < 9; ++v187_i1) {
            float v189_data = r1[(v186_i0 + v187_i1)];
            if (batchIdActive0) {
              glb_m0[((v27_lead + (v186_i0 * 16)) + (v187_i1 * 16))] = v189_data;
            }
          }
        }
      }
    }
  }
}

