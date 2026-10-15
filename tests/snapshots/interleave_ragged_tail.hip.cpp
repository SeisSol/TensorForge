// === base name ===
kernel_e0dc2083499797f6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e0dc2083499797f6 = {{32, 8, 1}, 32, 40, 1, 8, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e0dc2083499797f6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e0dc2083499797f6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e0dc2083499797f6(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e0dc2083499797f6, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_e0dc2083499797f6, block.x * block.y * block.z, 0));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e0dc2083499797f6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e0dc2083499797f6(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e0dc2083499797f6), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_e0dc2083499797f6, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e0dc2083499797f6(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (40 active) x 8 per block = block 32x8x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none ordered
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 512];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 2> v9_ld = __builtin_nontemporal_load(&*(tensorforge::VectorRelaxedT<float, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 2;
      size_t v14_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v30_lead = threadIdx.x % 32;
      bool v36_g = v30_lead < 8;
      int32_t v50_r = (threadIdx.y % 2) * 2;
      int32_t v54_vlead = (v30_lead + (v50_r * 32)) * 4;
      int32_t v83_vlead = (v30_lead + ((v50_r + 1) * 32)) * 4;
      float r0[6]{};
      size_t v188_row = v14_batchIdGroup0First + v11_batchIdLane0;
      const bool peel_batchIdActive0 = v188_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v188_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(peel_batchIdActive0 ? v188_row : v14_batchIdGroup0First) * 48 + 0 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v196_i1 = 0; v196_i1 < 6; ++v196_i1) {
        float v199_data = v36_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v30_lead + (v196_i1 * 8))])) : (0.0f);
        r0[v196_i1] = v199_data;
      }
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v185_groupNextAhead = v15_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v187_groupNext = (v185_groupNextAhead < numElements0) ? v185_groupNextAhead : v15_batchIdGroup0;
        size_t v192_row = v187_groupNext + v11_batchIdLane0;
        const bool wrap_batchIdActive0 = v192_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v192_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[(wrap_batchIdActive0 ? v192_row : v187_groupNext) * 48 + 0 + m2_extraOffset];
        size_t v16_row = v15_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 240 + 0 + m0_extraOffset];
        float r1[12]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 40), (0, 6)] [(0, 8)]
        float v40_data = r0[0];
        float v41_data = r0[1];
        float v42_data = r0[2];
        float v43_data = r0[3];
        float v44_tp{};
        float v45_tp{};
        float v46_tp{};
        float v47_tp{};
        tensorforge::transpose4x4b32(v44_tp, v45_tp, v46_tp, v47_tp, v40_data, v41_data, v42_data, v43_data);
        tensorforge::VectorT<float, 4> v48_acc{};
        tensorforge::VectorT<float, 4> v56_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v54_vlead];
        float v57_el = v56_data[0];
        tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_el, v48_acc, 3, 0, 1);
        float v59_el = v56_data[1];
        tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v59_el, v58_acc, 3, 0, 1);
        float v61_el = v56_data[2];
        tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v61_el, v60_acc, 3, 0, 1);
        float v63_el = v56_data[3];
        tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v63_el, v62_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_el, v64_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v59_el, v66_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v61_el, v68_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v63_el, v70_acc, 3, 1, 2);
        r1[0] = (v72_acc[0]);
        r1[2] = (v72_acc[1]);
        r1[4] = (v72_acc[2]);
        r1[6] = (v72_acc[3]);
        tensorforge::VectorT<float, 4> v77_acc{};
        tensorforge::VectorT<float, 4> v85_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v83_vlead];
        float v86_el = v85_data[0];
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v86_el, v77_acc, 3, 0, 1);
        float v88_el = v85_data[1];
        tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v88_el, v87_acc, 3, 0, 1);
        float v90_el = v85_data[2];
        tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v90_el, v89_acc, 3, 0, 1);
        float v92_el = v85_data[3];
        tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v92_el, v91_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v86_el, v93_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v88_el, v95_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v90_el, v97_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v92_el, v99_acc, 3, 1, 2);
        r1[1] = (v101_acc[0]);
        r1[3] = (v101_acc[1]);
        r1[5] = (v101_acc[2]);
        r1[7] = (v101_acc[3]);
        float v106_data = r0[4];
        float v107_data = r0[5];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v200_i1 = 0; v200_i1 < 6; ++v200_i1) {
          float v203_data = v36_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v30_lead + (v200_i1 * 8))])) : (0.0f);
          r0[v200_i1] = v203_data;
        }
        float v109_tp{};
        float v110_tp{};
        float v111_tp{};
        float v112_tp{};
        tensorforge::transpose4x4b32(v109_tp, v110_tp, v111_tp, v112_tp, v106_data, v107_data, 0.0f, 0.0f);
        tensorforge::VectorT<float, 4> v113_acc{};
        tensorforge::VectorT<float, 4> v121_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v54_vlead];
        float v122_el = v121_data[0];
        tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v122_el, v113_acc, 3, 0, 1);
        float v124_el = v121_data[1];
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v124_el, v123_acc, 3, 0, 1);
        float v126_el = v121_data[2];
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v126_el, v125_acc, 3, 0, 1);
        float v128_el = v121_data[3];
        tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v128_el, v127_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v122_el, v129_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v124_el, v131_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v126_el, v133_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v128_el, v135_acc, 3, 1, 2);
        r1[8] = (v137_acc[0]);
        r1[10] = (v137_acc[1]);
        tensorforge::VectorT<float, 4> v140_acc{};
        tensorforge::VectorT<float, 4> v148_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v83_vlead];
        float v149_el = v148_data[0];
        tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v149_el, v140_acc, 3, 0, 1);
        float v151_el = v148_data[1];
        tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v151_el, v150_acc, 3, 0, 1);
        float v153_el = v148_data[2];
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v153_el, v152_acc, 3, 0, 1);
        float v155_el = v148_data[3];
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v155_el, v154_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v149_el, v156_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v151_el, v158_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v153_el, v160_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v155_el, v162_acc, 3, 1, 2);
        r1[9] = (v164_acc[0]);
        r1[11] = (v164_acc[1]);
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v167_i0 = 0; v167_i0 < 1; ++v167_i0) {
          #pragma unroll
          for (int32_t v168_i1 = 0; v168_i1 < 6; ++v168_i1) {
            float v171_data = r1[(v167_i0 + (v168_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v30_lead + (v167_i0 * 32)) + (v168_i1 * 40))] = v171_data;
            }
          }
        }
        if (v36_g) {
          #pragma unroll
          for (int32_t v177_i1 = 0; v177_i1 < 6; ++v177_i1) {
            float v180_data = r1[(1 + (v177_i1 * 2))];
            if (batchIdActive0) {
              glb_m0[((v30_lead + 32_i32) + (v177_i1 * 40))] = v180_data;
            }
          }
        }
      }
    }
  }
}

