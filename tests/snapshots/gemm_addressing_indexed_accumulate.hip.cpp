// === base name ===
kernel_069574ef97177c9c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_069574ef97177c9c = {{32, 8, 1}, 32, 20, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_069574ef97177c9c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_069574ef97177c9c(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_069574ef97177c9c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_069574ef97177c9c, block.x * block.y * block.z, 384 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (384 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_069574ef97177c9c, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (384 * sizeof(float)));
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
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_069574ef97177c9c(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_069574ef97177c9c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_069574ef97177c9c), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  float *const * m0Arg = (float *const *)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_069574ef97177c9c, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_069574ef97177c9c(float *const * __restrict__ m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 8 per block = block 32x8x1, 1536 B shared, occupancy grid
    // operands:
    //   m0 20×9(20×9) {0..20}×{0..9} indexed_pointer(slots=2, slot=1)
    //   m1 20×9(20×9) {0..20}×{0..9} none ordered
    //   m2 9×9(9×9) {0..9}×{0..9} indexed(offset=12)
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"indexed_pointer(slots=2, slot=1)","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[20,9]],"name":"m1","ordered":true,"parts":1,"shape":[20,9],"variant":false},{"addressing":"indexed(offset=12)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"indexed_pointer(slots=2, slot=1)","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},{"addressing":"indexed(offset=12)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 384];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      if ((threadIdx.x + (threadIdx.y * 32)) < 128) {
        float v15_ld = ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256];
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 256] = v15_ld;
      }
      __syncthreads();
      size_t v17_batchIdLane0 = threadIdx.y % 2;
      size_t v20_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v36_lead = threadIdx.x % 32;
      bool v42_g = v36_lead < 9;
      bool v174_g = v36_lead < 20;
      int32_t v61_vlead = (v36_lead + ((threadIdx.y % 2) * 32)) * 4;
      int32_t v82_vlead = (v36_lead + 64_i32) * 4;
      int32_t v137_vlead = v36_lead * 4;
      int32_t v149_vlead = (v36_lead + 32_i32) * 4;
      float r0[9]{};
      size_t v194_row = v20_batchIdGroup0First + v17_batchIdLane0;
      const bool peel_batchIdActive0 = v194_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v194_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[(peel_batchIdActive0 ? v194_row : v20_batchIdGroup0First)]) * 81 + 0 + 12 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v202_i1 = 0; v202_i1 < 9; ++v202_i1) {
        float v205_data = v42_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v36_lead + (v202_i1 * 9))])) : (0.0f);
        r0[v202_i1] = v205_data;
      }
      for (size_t v21_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v21_batchIdGroup0 < numElements0; v21_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v191_groupNextAhead = v21_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v193_groupNext = (v191_groupNextAhead < numElements0) ? v191_groupNextAhead : v21_batchIdGroup0;
        size_t v198_row = v193_groupNext + v17_batchIdLane0;
        const bool wrap_batchIdActive0 = v198_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v198_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[static_cast<size_t>(elementIndices[(wrap_batchIdActive0 ? v198_row : v193_groupNext)]) * 81 + 0 + 12 + m2_extraOffset];
        size_t v22_row = v21_batchIdGroup0 + v17_batchIdLane0;
        const bool batchIdActive0 = v22_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v22_row]));
        size_t v24_batchId0 = batchIdActive0 ? v22_row : v21_batchIdGroup0;
        size_t v25_ahead1 = v24_batchId0 + (gridDim.x * blockDim.y);
        size_t v27_batchId1 = (v25_ahead1 < numElements0) ? v25_ahead1 : v24_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[static_cast<size_t>(elementIndices[v24_batchId0]) * 2 + 1][0 + m0_extraOffset];
        float r1[9]{};
        // r1 = load{g>r}(glb_m0);
        float r2[9]{};
        // ir2 = +(glb_m1 * r0)
        // [(0, 20), (0, 9)] [(0, 9)]
        float ir2[9]{};
        float v47_data = r0[0];
        float v48_data = r0[1];
        float v49_data = r0[2];
        float v50_data = r0[3];
        float v51_tp{};
        float v52_tp{};
        float v53_tp{};
        float v54_tp{};
        tensorforge::transpose4x4b32Fused(v51_tp, v52_tp, v53_tp, v54_tp, v47_data, v48_data, v49_data, v50_data);
        tensorforge::VectorT<float, 4> v55_acc{};
        tensorforge::VectorT<float, 4> v63_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v61_vlead];
        float v64_el = v63_data[0];
        tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_el, v55_acc, 3, 0, 1);
        float v66_el = v63_data[1];
        tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v66_el, v65_acc, 3, 0, 1);
        float v68_el = v63_data[2];
        tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v68_el, v67_acc, 3, 0, 1);
        float v70_el = v63_data[3];
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v70_el, v69_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_el, v71_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v66_el, v73_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v68_el, v75_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v70_el, v77_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v84_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v82_vlead];
        float v85_el = v84_data[0];
        tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v85_el, v79_acc, 3, 2, 0);
        ir2[0] = (v86_acc[0]);
        ir2[1] = (v86_acc[1]);
        ir2[2] = (v86_acc[2]);
        ir2[3] = (v86_acc[3]);
        float v91_data = r0[4];
        float v92_data = r0[5];
        float v93_data = r0[6];
        float v94_data = r0[7];
        float v95_tp{};
        float v96_tp{};
        float v97_tp{};
        float v98_tp{};
        tensorforge::transpose4x4b32Fused(v95_tp, v96_tp, v97_tp, v98_tp, v91_data, v92_data, v93_data, v94_data);
        tensorforge::VectorT<float, 4> v99_acc{};
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v64_el, v99_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v66_el, v109_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v68_el, v111_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v70_el, v113_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v64_el, v115_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v66_el, v117_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v68_el, v119_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v70_el, v121_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v85_el, v123_acc, 3, 2, 0);
        ir2[4] = (v130_acc[0]);
        ir2[5] = (v130_acc[1]);
        ir2[6] = (v130_acc[2]);
        ir2[7] = (v130_acc[3]);
        tensorforge::VectorT<float, 4> v139_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v137_vlead];
        float v140_el = v139_data[0];
        float v142_el = v139_data[1];
        float v144_el = v139_data[2];
        float v146_el = v139_data[3];
        tensorforge::VectorT<float, 4> v151_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v149_vlead];
        float v152_el = v151_data[0];
        float v154_el = v151_data[1];
        float v156_el = v151_data[2];
        float v158_el = v151_data[3];
        float v165_acc{};
        float v166_data = r0[8];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v206_i1 = 0; v206_i1 < 9; ++v206_i1) {
          float v209_data = v42_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v36_lead + (v206_i1 * 9))])) : (0.0f);
          r0[v206_i1] = v209_data;
        }
        float v167_bc = tensorforge::broadcast<32, 16, 0>(v166_data);
        asm("s_nop 1\n\t"
        "v_fmac_f32_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_fmac_f32_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "s_nop 1\n\t"
        : "+&v"(v165_acc)
        : "v"(v167_bc), "v"(v140_el), "v"(v142_el), "v"(v144_el), "v"(v146_el), "v"(v152_el), "v"(v154_el), "v"(v156_el), "v"(v158_el), "v"(v85_el)
        );
        ir2[8] = v165_acc;
        // r2 = ir2 + r1
        // glb_m0 = store{r>g}(r2);
        if (v174_g) {
          #pragma unroll
          for (int32_t v184_i1 = 0; v184_i1 < 9; ++v184_i1) {
            float v186_data = ir2[v184_i1];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[(v36_lead + (v184_i1 * 20))], v186_data);
            }
          }
        }
      }
    }
  }
}

