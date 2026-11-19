// === base name ===
kernel_331ed82199555ccf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_331ed82199555ccf = {{32, 8, 1}, 32, 20, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_331ed82199555ccf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_331ed82199555ccf(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_331ed82199555ccf(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_331ed82199555ccf, block.x * block.y * block.z, 384 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (384 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_331ed82199555ccf, block.x * block.y * block.z, 0));
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
void launcher_kernel_331ed82199555ccf(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_331ed82199555ccf(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_331ed82199555ccf), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  float *const * m0Arg = (float *const *)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> elementIndicesArg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)elementIndices;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_331ed82199555ccf, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_331ed82199555ccf(float *const * __restrict__ m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> elementIndices, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
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
      int32_t v37_lead = threadIdx.x % 32;
      bool v43_g = v37_lead < 9;
      bool v175_g = v37_lead < 20;
      int32_t v62_vlead = (v37_lead + ((threadIdx.y % 2) * 32)) * 4;
      int32_t v83_vlead = (v37_lead + 64_i32) * 4;
      int32_t v138_vlead = v37_lead * 4;
      int32_t v150_vlead = (v37_lead + 32_i32) * 4;
      float r0[9]{};
      size_t v195_row = v20_batchIdGroup0First + v17_batchIdLane0;
      const bool peel_batchIdActive0 = v195_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v195_row]));
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[(peel_batchIdActive0 ? v195_row : v20_batchIdGroup0First)]);
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[peel_elementIndex * 81 + 0 + 12 + m2_extraOffset];
      // r0 = load{g>r}(peel_glb_m2);
      #pragma unroll
      for (int32_t v205_i1 = 0; v205_i1 < 9; ++v205_i1) {
        float v208_data = v43_g ? (__builtin_nontemporal_load(&peel_glb_m2[(v37_lead + (v205_i1 * 9))])) : (0.0f);
        r0[v205_i1] = v208_data;
      }
      size_t v213_elementIndex = peel_elementIndex;
      for (size_t v21_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v21_batchIdGroup0 < numElements0; v21_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v192_groupNextAhead = v21_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v194_groupNext = (v192_groupNextAhead < numElements0) ? v192_groupNextAhead : v21_batchIdGroup0;
        size_t v200_row = v194_groupNext + v17_batchIdLane0;
        const bool wrap_batchIdActive0 = v200_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v200_row]));
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[(wrap_batchIdActive0 ? v200_row : v194_groupNext)]);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[wrap_elementIndex * 81 + 0 + 12 + m2_extraOffset];
        size_t v22_row = v21_batchIdGroup0 + v17_batchIdLane0;
        const bool batchIdActive0 = v22_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v22_row]));
        size_t v24_batchId0 = batchIdActive0 ? v22_row : v21_batchIdGroup0;
        size_t v25_ahead1 = v24_batchId0 + (gridDim.x * blockDim.y);
        size_t v27_batchId1 = (v25_ahead1 < numElements0) ? v25_ahead1 : v24_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v213_elementIndex * 2 + 1][0 + m0_extraOffset];
        float r1[9]{};
        // r1 = load{g>r}(glb_m0);
        float r2[9]{};
        // ir2 = +(glb_m1 * r0)
        // [(0, 20), (0, 9)] [(0, 9)]
        float ir2[9]{};
        float v48_data = r0[0];
        float v49_data = r0[1];
        float v50_data = r0[2];
        float v51_data = r0[3];
        float v52_tp{};
        float v53_tp{};
        float v54_tp{};
        float v55_tp{};
        tensorforge::transpose4x4b32Fused(v52_tp, v53_tp, v54_tp, v55_tp, v48_data, v49_data, v50_data, v51_data);
        tensorforge::VectorT<float, 4> v56_acc{};
        tensorforge::VectorT<float, 4> v64_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v62_vlead];
        float v65_el = v64_data[0];
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_el, v56_acc, 3, 0, 1);
        float v67_el = v64_data[1];
        tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v67_el, v66_acc, 3, 0, 1);
        float v69_el = v64_data[2];
        tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v69_el, v68_acc, 3, 0, 1);
        float v71_el = v64_data[3];
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v71_el, v70_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_el, v72_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v67_el, v74_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v69_el, v76_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v71_el, v78_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v85_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v83_vlead];
        float v86_el = v85_data[0];
        tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v86_el, v80_acc, 3, 2, 0);
        ir2[0] = (v87_acc[0]);
        ir2[1] = (v87_acc[1]);
        ir2[2] = (v87_acc[2]);
        ir2[3] = (v87_acc[3]);
        float v92_data = r0[4];
        float v93_data = r0[5];
        float v94_data = r0[6];
        float v95_data = r0[7];
        float v96_tp{};
        float v97_tp{};
        float v98_tp{};
        float v99_tp{};
        tensorforge::transpose4x4b32Fused(v96_tp, v97_tp, v98_tp, v99_tp, v92_data, v93_data, v94_data, v95_data);
        tensorforge::VectorT<float, 4> v100_acc{};
        tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v65_el, v100_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v67_el, v110_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v69_el, v112_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v71_el, v114_acc, 3, 0, 1);
        tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v65_el, v116_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v97_tp, v67_el, v118_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v69_el, v120_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v71_el, v122_acc, 3, 1, 2);
        tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v86_el, v124_acc, 3, 2, 0);
        ir2[4] = (v131_acc[0]);
        ir2[5] = (v131_acc[1]);
        ir2[6] = (v131_acc[2]);
        ir2[7] = (v131_acc[3]);
        tensorforge::VectorT<float, 4> v140_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v138_vlead];
        float v141_el = v140_data[0];
        float v143_el = v140_data[1];
        float v145_el = v140_data[2];
        float v147_el = v140_data[3];
        tensorforge::VectorT<float, 4> v152_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v150_vlead];
        float v153_el = v152_data[0];
        float v155_el = v152_data[1];
        float v157_el = v152_data[2];
        float v159_el = v152_data[3];
        float v166_acc{};
        float v167_data = r0[8];
        // r0 = load{g>r}(wrap_glb_m2);
        #pragma unroll
        for (int32_t v209_i1 = 0; v209_i1 < 9; ++v209_i1) {
          float v212_data = v43_g ? (__builtin_nontemporal_load(&wrap_glb_m2[(v37_lead + (v209_i1 * 9))])) : (0.0f);
          r0[v209_i1] = v212_data;
        }
        float v168_bc = tensorforge::broadcast<32, 16, 0>(v167_data);
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
        : "+&v"(v166_acc)
        : "v"(v168_bc), "v"(v141_el), "v"(v143_el), "v"(v145_el), "v"(v147_el), "v"(v153_el), "v"(v155_el), "v"(v157_el), "v"(v159_el), "v"(v86_el)
        );
        ir2[8] = v166_acc;
        // r2 = ir2 + r1
        // glb_m0 = store{r>g}(r2);
        if (v175_g) {
          #pragma unroll
          for (int32_t v185_i1 = 0; v185_i1 < 9; ++v185_i1) {
            float v187_data = ir2[v185_i1];
            if (batchIdActive0) {
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[(v37_lead + (v185_i1 * 20))], v187_data);
            }
          }
        }
        v213_elementIndex = wrap_elementIndex;
      }
    }
  }
}

